from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from football_attention.bootstrap import seed_match_bootstrap_mean
from football_attention.faithful_eval import safe_spearman


DEFAULT_SEEDS = (42, 43, 44, 45, 46)


REFERENCE_COLUMNS = {
    'occlusion': 'occlusion_abs',
    'integrated_gradients': 'ig_abs',
    'permutation_shapley': 'permutation_shapley_abs',
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    results_dir: Path
    expected_matches: int
    expected_sequences: int
    expected_token_sequences: int

    @property
    def token_path(self) -> Path:
        return self.results_dir / 'token_attributions.parquet'

    @property
    def deletion_path(self) -> Path:
        return self.results_dir / 'deletion_predictions.parquet'

    @property
    def existing_deletion_bootstrap_path(self) -> Path:
        return self.results_dir / 'deletion_bootstrap.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Aggregate saved recency diagnostics; no inference.')
    parser.add_argument('--project-dir', type=Path, default=Path.cwd())
    parser.add_argument('--source-results-name', default='results/explanations')
    parser.add_argument('--results-name', default='results/recency')
    parser.add_argument('--development-prepared-name', default='development')
    parser.add_argument('--external-prepared-name', default='holdout')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--datasets', nargs='+', choices=['development_test', 'frozen_holdout'],
                        default=['development_test', 'frozen_holdout'])
    parser.add_argument('--n-bootstrap', type=int, default=10_000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f'{label}: missing columns {sorted(missing)}')


def validate_token_rows(
    tokens: pd.DataFrame,
    *,
    spec: DatasetSpec,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    required = {
        'dataset',
        'model_type',
        'seed',
        'shot_seq_id',
        'match_id',
        'token_position',
        'attention',
        'occlusion_abs',
        'ig_abs',
        'permutation_shapley_abs',
        'recency',
    }
    require_columns(tokens, required, f'{spec.name} token attributions')
    tokens = tokens[tokens['seed'].isin(seeds)].copy()
    observed_seeds = tuple(sorted(tokens['seed'].unique().astype(int)))
    if observed_seeds != seeds:
        raise RuntimeError(
            f'{spec.name}: expected seeds {seeds}, found {observed_seeds}'
        )
    datasets = set(tokens['dataset'].astype(str).unique())
    if datasets != {spec.name}:
        raise RuntimeError(f'{spec.name}: unexpected dataset labels {datasets}')
    models = set(tokens['model_type'].astype(str).unique())
    if models != {'gru_attention_faithful'}:
        raise RuntimeError(f'{spec.name}: unexpected models {models}')
    key = ['seed', 'shot_seq_id', 'token_position']
    if tokens.duplicated(key).any():
        raise RuntimeError(f'{spec.name}: duplicate token attribution keys')
    if tokens['match_id'].nunique() != spec.expected_matches:
        raise RuntimeError(
            f'{spec.name}: expected {spec.expected_matches} matches, found '
            f'{tokens["match_id"].nunique()}'
        )
    if tokens['shot_seq_id'].nunique() != spec.expected_token_sequences:
        raise RuntimeError(
            f'{spec.name}: expected {spec.expected_token_sequences} nonempty '
            f'sequences, found {tokens["shot_seq_id"].nunique()}'
        )
    for _, group in tokens.groupby(['seed', 'shot_seq_id'], sort=False):
        ordered = group.sort_values('token_position')
        expected = np.arange(1, len(ordered) + 1, dtype=float)
        if not np.allclose(ordered['recency'].to_numpy(float), expected):
            raise RuntimeError(
                f'{spec.name}: recency is inconsistent with token order for '
                f'seed={int(ordered["seed"].iloc[0])}, '
                f'shot_seq_id={int(ordered["shot_seq_id"].iloc[0])}'
            )
    return tokens


def sequence_rank_diagnostics(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    grouping = ['seed', 'shot_seq_id', 'match_id']
    for keys, group in tokens.groupby(grouping, sort=False):
        row: dict[str, int | float | str] = {
            'dataset': str(group['dataset'].iloc[0]),
            'seed': int(keys[0]),
            'shot_seq_id': int(keys[1]),
            'match_id': int(keys[2]),
            'n_events': int(len(group)),
        }
        attention = group['attention'].to_numpy(float)
        recency = group['recency'].to_numpy(float)
        for reference, column in REFERENCE_COLUMNS.items():
            values = group[column].to_numpy(float)
            attention_rho = safe_spearman(attention, values)
            recency_rho = safe_spearman(recency, values)
            row[f'attention_vs_{reference}'] = attention_rho
            row[f'recency_vs_{reference}'] = recency_rho
            row[f'attention_minus_recency_vs_{reference}'] = (
                attention_rho - recency_rho
                if np.isfinite(attention_rho) and np.isfinite(recency_rho)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_rank_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in diagnostics.columns
        if column.startswith('attention_vs_')
        or column.startswith('recency_vs_')
        or column.startswith('attention_minus_recency_vs_')
    ]
    rows = []
    for metric in metric_columns:
        values = diagnostics[
            ['seed', 'match_id', 'shot_seq_id', metric]
        ].rename(columns={metric: 'difference'})
        result = seed_match_bootstrap_mean(
            values,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                'dataset': str(diagnostics['dataset'].iloc[0]),
                'metric': metric,
                **result,
            }
        )
    return pd.DataFrame(rows)


def summarize_rank_diagnostics_by_seed(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in diagnostics.columns
        if column.startswith('attention_vs_')
        or column.startswith('recency_vs_')
        or column.startswith('attention_minus_recency_vs_')
    ]
    summary = diagnostics.groupby('seed', as_index=False)[metric_columns].mean()
    summary.insert(0, 'dataset', str(diagnostics['dataset'].iloc[0]))
    return summary


def deletion_attention_minus_recency(
    deletions: pd.DataFrame,
    *,
    spec: DatasetSpec,
    seeds: tuple[int, ...],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    required = {
        'dataset',
        'seed',
        'shot_seq_id',
        'match_id',
        'method',
        'k',
        'target_xg',
        'base_prediction',
        'deleted_prediction',
    }
    require_columns(deletions, required, f'{spec.name} deletion predictions')
    deletions = deletions[
        deletions['seed'].isin(seeds)
        & deletions['method'].isin(['attention', 'recency'])
        & deletions['k'].isin([1, 3, 5])
    ].copy()
    observed_seeds = tuple(sorted(deletions['seed'].unique().astype(int)))
    if observed_seeds != seeds:
        raise RuntimeError(
            f'{spec.name}: expected deletion seeds {seeds}, found '
            f'{observed_seeds}'
        )
    key = ['seed', 'shot_seq_id', 'method', 'k']
    if deletions.duplicated(key).any():
        raise RuntimeError(f'{spec.name}: duplicate deletion prediction keys')
    if deletions['match_id'].nunique() != spec.expected_matches:
        raise RuntimeError(f'{spec.name}: incomplete deletion match coverage')
    if deletions['shot_seq_id'].nunique() != spec.expected_sequences:
        raise RuntimeError(f'{spec.name}: incomplete deletion sequence coverage')
    deletions['base_squared_error'] = (
        deletions['target_xg'] - deletions['base_prediction']
    ) ** 2
    deletions['deleted_squared_error'] = (
        deletions['target_xg'] - deletions['deleted_prediction']
    ) ** 2
    deletions['delta_squared_error'] = (
        deletions['deleted_squared_error'] - deletions['base_squared_error']
    )
    rows = []
    keys = ['seed', 'shot_seq_id']
    for k in (1, 3, 5):
        attention = deletions[
            deletions['method'].eq('attention') & deletions['k'].eq(k)
        ][[*keys, 'match_id', 'delta_squared_error']]
        recency = deletions[
            deletions['method'].eq('recency') & deletions['k'].eq(k)
        ][[*keys, 'delta_squared_error']]
        paired = attention.merge(
            recency,
            on=keys,
            suffixes=('_attention', '_recency'),
            validate='one_to_one',
        )
        paired['difference'] = (
            paired['delta_squared_error_attention']
            - paired['delta_squared_error_recency']
        )
        result = seed_match_bootstrap_mean(
            paired,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                'dataset': spec.name,
                'comparison': 'attention_minus_recency',
                'k': int(k),
                **result,
            }
        )
    return pd.DataFrame(rows)


def verify_existing_deletion_bootstrap(
    calculated: pd.DataFrame,
    existing_path: Path,
) -> None:
    if not existing_path.exists():
        raise RuntimeError(f'Missing existing deletion summary: {existing_path}')
    existing = pd.read_csv(existing_path)
    required = {
        'dataset',
        'comparison',
        'k',
        'difference_mean',
        'ci_025',
        'ci_975',
        'n_seeds',
        'n_matches',
        'n_sequences',
        'n_rows',
        'n_bootstrap',
    }
    require_columns(existing, required, 'existing deletion bootstrap')
    existing = existing[
        existing['comparison'].eq('attention_minus_recency')
        & existing['k'].isin([1, 3, 5])
    ]
    merged = calculated.merge(
        existing,
        on=['dataset', 'comparison', 'k'],
        suffixes=('_calculated', '_existing'),
        validate='one_to_one',
    )
    if len(merged) != 3:
        raise RuntimeError('Existing deletion summary lacks k=1,3,5 rows')
    float_columns = ['difference_mean', 'ci_025', 'ci_975']
    integer_columns = [
        'n_seeds',
        'n_matches',
        'n_sequences',
        'n_rows',
        'n_bootstrap',
    ]
    for column in float_columns:
        if not np.allclose(
            merged[f'{column}_calculated'],
            merged[f'{column}_existing'],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f'Deletion bootstrap mismatch in {column}')
    for column in integer_columns:
        if not np.array_equal(
            merged[f'{column}_calculated'].to_numpy(int),
            merged[f'{column}_existing'].to_numpy(int),
        ):
            raise RuntimeError(f'Deletion bootstrap mismatch in {column}')


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.project_dir.resolve()
    output = root / args.results_name
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(sorted(set(args.seeds)))
    summaries = {}
    for name, prepared_name, split in (
        ('development_test', args.development_prepared_name, 'test'),
        ('frozen_holdout', args.external_prepared_name, 'external_test'),
    ):
        if name not in args.datasets:
            continue
        rows = pd.read_parquet(root / 'data' / prepared_name / 'sequences_raw.parquet')
        rows = rows[rows['split'].eq(split)]
        spec = DatasetSpec(
            name=name,
            results_dir=root / args.source_results_name / name,
            expected_matches=int(rows['match_id'].nunique()),
            expected_sequences=len(rows),
            expected_token_sequences=int(rows['n_events'].gt(1).sum()),
        )
        tokens = validate_token_rows(pd.read_parquet(spec.token_path), spec=spec, seeds=seeds)
        expected_ids = set(rows.loc[rows['n_events'].gt(1), 'shot_seq_id'])
        for seed in seeds:
            if set(tokens.loc[tokens['seed'].eq(seed), 'shot_seq_id']) != expected_ids:
                raise RuntimeError(f'{name}: incomplete token coverage for seed {seed}')
        diagnostics = sequence_rank_diagnostics(tokens)
        correlations = bootstrap_rank_diagnostics(
            diagnostics, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
        )
        deletions = deletion_attention_minus_recency(
            pd.read_parquet(spec.deletion_path), spec=spec, seeds=seeds,
            n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
        )
        verify_existing_deletion_bootstrap(deletions, spec.existing_deletion_bootstrap_path)
        destination = output / name
        destination.mkdir(exist_ok=True)
        diagnostics.to_parquet(destination / 'sequence_rank_diagnostics.parquet', index=False)
        summarize_rank_diagnostics_by_seed(diagnostics).to_csv(
            destination / 'correlation_summary_by_seed.csv', index=False,
        )
        correlations.to_csv(destination / 'correlation_bootstrap.csv', index=False)
        deletions.to_csv(destination / 'deletion_attention_minus_recency.csv', index=False)
        summaries[name] = {
            'n_sequences': len(rows),
            'n_matches': spec.expected_matches,
            'inputs': {path.name: sha256_file(path) for path in (spec.token_path, spec.deletion_path)},
        }
        print(f'Aggregated {name}: {len(rows)} sequences', flush=True)
    write_json(output / 'aggregation_complete.json', {
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'seeds': seeds, 'n_bootstrap': args.n_bootstrap, 'datasets': summaries,
    })


if __name__ == '__main__':
    main()
