from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from football_attention.bootstrap import seed_match_bootstrap_mean
from football_attention.config import ExperimentConfig
from football_attention.faithful_eval import (
    gradient_x_input,
    permutation_shapley,
    recency_importance,
    safe_spearman,
    top_k_mask,
)
from football_attention.faithful_model import build_faithful_model
from football_attention.train import make_loader, move_batch
from football_attention.variants import ShotSequenceDataset, Vocabularies
from football_attention.tasks.common import select_device


MODEL = 'gru_attention_faithful'


DATASET_NAMES = ('development_test', 'frozen_holdout')


DEFAULT_EXTERNAL_PREPARED_NAME = 'holdout'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument('--frozen-results-name', default='results/final')
    parser.add_argument('--development-prepared-name', default='development')
    parser.add_argument(
        '--external-prepared-name',
        default=DEFAULT_EXTERNAL_PREPARED_NAME,
    )
    parser.add_argument(
        '--external-results-name',
        default='results/holdout',
    )
    parser.add_argument(
        '--results-name',
        default='results/explanations',
    )
    parser.add_argument(
        '--stage',
        choices=['evaluate', 'aggregate', 'all'],
        default='all',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'mps', 'cuda'],
        default='auto',
    )
    parser.add_argument('--shapley-permutations', type=int, default=64)
    parser.add_argument('--coalition-batch-size', type=int, default=512)
    parser.add_argument('--n-bootstrap', type=int, default=10_000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def load_rows(
    project_dir: Path,
    dataset_name: str,
    *,
    quick: bool,
    development_prepared_name: str = 'prepared',
    external_prepared_name: str = DEFAULT_EXTERNAL_PREPARED_NAME,
) -> pd.DataFrame:
    if dataset_name == 'development_test':
        path = (
            project_dir
            / 'data'
            / development_prepared_name
            / 'sequences_raw.parquet'
        )
        rows = pd.read_parquet(path, filters=[('split', '=', 'test')])
        rows = rows[rows['split'].eq('test')].copy()
    else:
        rows = pd.read_parquet(
            project_dir
            / 'data'
            / external_prepared_name
            / 'sequences_raw.parquet'
        )
    if quick:
        rows = rows.head(16).copy()
    if rows.empty:
        raise RuntimeError(f'No rows for dataset={dataset_name}')
    return rows.reset_index(drop=True)


def checkpoint_path(project_dir: Path, frozen_results_name: str, seed: int) -> Path:
    return (
        project_dir
        / frozen_results_name
        / 'faithful'
        / MODEL
        / f'seed_{seed}'
        / 'model.pt'
    )


def existing_evaluation_dir(
    project_dir: Path,
    dataset_name: str,
    frozen_results_name: str,
    external_results_name: str,
    seed: int,
) -> Path:
    if dataset_name == 'development_test':
        return (
            project_dir
            / frozen_results_name
            / 'faithful'
            / MODEL
            / f'seed_{seed}'
        )
    return (
        project_dir
        / external_results_name
        / 'faithfulness'
        / MODEL
        / f'seed_{seed}'
    )


def load_model(
    path: Path,
    project_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ExperimentConfig, Vocabularies]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = replace(
        ExperimentConfig(**checkpoint['config']),
        project_dir=project_dir,
        num_workers=0,
    )
    vocabularies = Vocabularies(**checkpoint['vocabularies'])
    model = build_faithful_model(
        checkpoint['model_type'],
        vocabularies,
        config.sequence_length,
        config,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval(), config, vocabularies


def seed_output(root: Path, dataset_name: str, seed: int) -> Path:
    return root / dataset_name / f'seed_{seed}'


def evaluate_seed(
    rows: pd.DataFrame,
    *,
    project_dir: Path,
    root: Path,
    dataset_name: str,
    frozen_results_name: str,
    seed: int,
    device: torch.device,
    n_permutations: int,
    coalition_batch_size: int,
    force: bool,
) -> None:
    output = seed_output(root, dataset_name, seed)
    required = [
        output / 'token_attributions.parquet',
        output / 'deletion_predictions.parquet',
        output / 'shapley_diagnostics.parquet',
    ]
    if not force and all(path.exists() for path in required):
        print(f'Using cached baselines dataset={dataset_name}, seed={seed}')
        return

    model, config, vocabularies = load_model(
        checkpoint_path(project_dir, frozen_results_name, seed),
        project_dir,
        device,
    )
    dataset = ShotSequenceDataset(
        rows,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=config.sequence_length,
    )
    loader = make_loader(dataset, config, shuffle=False, seed=seed)
    token_rows: list[dict[str, object]] = []
    deletion_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    processed = 0

    for cpu_batch, cpu_targets in loader:
        batch = move_batch(cpu_batch, device)
        with torch.no_grad():
            reference = model(batch)
        gradient = gradient_x_input(model, batch)
        shapley, shapley_se = permutation_shapley(
            model,
            batch,
            n_permutations=n_permutations,
            seed=seed,
            coalition_batch_size=coalition_batch_size,
        )
        recency = recency_importance(batch['valid_mask'])
        empty_batch = dict(batch)
        empty_batch['valid_mask'] = torch.zeros_like(batch['valid_mask'])
        with torch.no_grad():
            empty_prediction = model(empty_batch)

        arrays = {
            'gradient_x_input_signed': gradient.detach().cpu().numpy(),
            'permutation_shapley_signed': shapley.detach().cpu().numpy(),
            'permutation_shapley_se': shapley_se.detach().cpu().numpy(),
            'recency': recency.detach().cpu().numpy(),
        }
        valid = cpu_batch['valid_mask'].numpy()
        for row in range(len(cpu_targets)):
            shot_seq_id = int(cpu_batch['shot_seq_id'][row])
            match_id = int(cpu_batch['match_id'][row])
            positions = np.flatnonzero(valid[row])
            local_error = abs(
                float(shapley[row].sum().cpu())
                - float((reference[row] - empty_prediction[row]).cpu())
            )
            diagnostic_rows.append(
                {
                    'dataset': dataset_name,
                    'model_type': MODEL,
                    'seed': seed,
                    'shot_seq_id': shot_seq_id,
                    'match_id': match_id,
                    'n_events': int(len(positions)),
                    'local_accuracy_error': local_error,
                    'mean_shapley_se': float(
                        shapley_se[row, batch['valid_mask'][row]].mean().cpu()
                    )
                    if len(positions)
                    else 0.0,
                    'max_shapley_se': float(
                        shapley_se[row, batch['valid_mask'][row]].max().cpu()
                    )
                    if len(positions)
                    else 0.0,
                }
            )
            for position in positions:
                gradient_signed = float(
                    arrays['gradient_x_input_signed'][row, position]
                )
                shapley_signed = float(
                    arrays['permutation_shapley_signed'][row, position]
                )
                token_rows.append(
                    {
                        'dataset': dataset_name,
                        'model_type': MODEL,
                        'seed': seed,
                        'shot_seq_id': shot_seq_id,
                        'match_id': match_id,
                        'token_position': int(position),
                        'target_xg': float(cpu_targets[row]),
                        'gradient_x_input_signed': gradient_signed,
                        'gradient_x_input_abs': abs(gradient_signed),
                        'permutation_shapley_signed': shapley_signed,
                        'permutation_shapley_abs': abs(shapley_signed),
                        'permutation_shapley_se': float(
                            arrays['permutation_shapley_se'][row, position]
                        ),
                        'recency': float(arrays['recency'][row, position]),
                    }
                )

        methods = {
            'gradient_x_input': gradient.abs(),
            'permutation_shapley': shapley.abs(),
            'recency': recency,
        }
        for method, importance in methods.items():
            for k in (1, 3, 5):
                deleted_valid, n_removed = top_k_mask(
                    batch['valid_mask'],
                    importance,
                    k,
                )
                deleted_batch = dict(batch)
                deleted_batch['valid_mask'] = deleted_valid
                with torch.no_grad():
                    deleted_prediction = model(deleted_batch)
                for row in range(len(cpu_targets)):
                    deletion_rows.append(
                        {
                            'dataset': dataset_name,
                            'model_type': MODEL,
                            'seed': seed,
                            'shot_seq_id': int(
                                cpu_batch['shot_seq_id'][row]
                            ),
                            'match_id': int(cpu_batch['match_id'][row]),
                            'method': method,
                            'k': k,
                            'n_removed': int(n_removed[row].cpu()),
                            'target_xg': float(cpu_targets[row]),
                            'base_prediction': float(reference[row].cpu()),
                            'deleted_prediction': float(
                                deleted_prediction[row].cpu()
                            ),
                        }
                    )

        processed += len(cpu_targets)
        print(
            f'Baseline evaluation dataset={dataset_name}, seed={seed}: '
            f'{processed}/{len(dataset)} sequences',
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(token_rows).to_parquet(
        output / 'token_attributions.parquet',
        index=False,
    )
    pd.DataFrame(deletion_rows).to_parquet(
        output / 'deletion_predictions.parquet',
        index=False,
    )
    pd.DataFrame(diagnostic_rows).to_parquet(
        output / 'shapley_diagnostics.parquet',
        index=False,
    )
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def read_new_seed_files(
    root: Path,
    dataset_name: str,
    seeds: list[int],
    filename: str,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = seed_output(root, dataset_name, seed) / filename
        if not path.exists():
            raise RuntimeError(f'Missing baseline output: {path}')
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def read_existing_seed_files(
    project_dir: Path,
    dataset_name: str,
    frozen_results_name: str,
    external_results_name: str,
    seeds: list[int],
    filename: str,
    sequence_ids: set[int],
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = existing_evaluation_dir(
            project_dir,
            dataset_name,
            frozen_results_name,
            external_results_name,
            seed,
        ) / filename
        if not path.exists():
            raise RuntimeError(f'Missing existing faithfulness output: {path}')
        frame = pd.read_parquet(path)
        frames.append(frame[frame['shot_seq_id'].isin(sequence_ids)])
    return pd.concat(frames, ignore_index=True)


def paired_deletion_bootstrap(
    values: pd.DataFrame,
    *,
    left_method: str,
    right_method: str,
    k: int,
    n_bootstrap: int,
) -> dict[str, object]:
    keys = ['seed', 'shot_seq_id']
    columns = [*keys, 'match_id', 'delta_squared_error']
    left = values[
        values['method'].eq(left_method) & values['k'].eq(k)
    ][columns]
    right = values[
        values['method'].eq(right_method) & values['k'].eq(k)
    ][[*keys, 'delta_squared_error']]
    paired = left.merge(
        right,
        on=keys,
        suffixes=('_left', '_right'),
        validate='one_to_one',
    )
    paired['difference'] = (
        paired['delta_squared_error_left']
        - paired['delta_squared_error_right']
    )
    result = seed_match_bootstrap_mean(
        paired,
        n_bootstrap=n_bootstrap,
        seed=2026,
    )
    return {
        'comparison': f'{left_method}_minus_{right_method}',
        'k': int(k),
        **result,
    }


def aggregate_dataset(
    *,
    project_dir: Path,
    root: Path,
    dataset_name: str,
    frozen_results_name: str,
    external_results_name: str,
    development_prepared_name: str,
    external_prepared_name: str,
    seeds: list[int],
    n_bootstrap: int,
    quick: bool,
) -> dict[str, int]:
    rows = load_rows(
        project_dir,
        dataset_name,
        quick=quick,
        development_prepared_name=development_prepared_name,
        external_prepared_name=external_prepared_name,
    )
    sequence_ids = set(rows['shot_seq_id'].astype(int))
    match_map = rows[['shot_seq_id', 'match_id']].drop_duplicates()

    new_tokens = read_new_seed_files(
        root,
        dataset_name,
        seeds,
        'token_attributions.parquet',
    )
    existing_tokens = read_existing_seed_files(
        project_dir,
        dataset_name,
        frozen_results_name,
        external_results_name,
        seeds,
        'token_attributions.parquet',
        sequence_ids,
    )
    existing_columns = [
        'model_type',
        'seed',
        'shot_seq_id',
        'match_id',
        'token_position',
        'attention',
        'occlusion_abs',
        'ig_abs',
    ]
    tokens = new_tokens.merge(
        existing_tokens[existing_columns],
        on=[
            'model_type',
            'seed',
            'shot_seq_id',
            'match_id',
            'token_position',
        ],
        validate='one_to_one',
    )
    output = root / dataset_name
    tokens.to_parquet(output / 'token_attributions.parquet', index=False)

    comparisons = {
        'attention_vs_permutation_shapley': (
            'attention',
            'permutation_shapley_abs',
        ),
        'attention_vs_gradient_x_input': (
            'attention',
            'gradient_x_input_abs',
        ),
        'attention_vs_recency': ('attention', 'recency'),
        'permutation_shapley_vs_occlusion': (
            'permutation_shapley_abs',
            'occlusion_abs',
        ),
        'permutation_shapley_vs_integrated_gradients': (
            'permutation_shapley_abs',
            'ig_abs',
        ),
        'gradient_x_input_vs_integrated_gradients': (
            'gradient_x_input_abs',
            'ig_abs',
        ),
    }
    correlation_rows = []
    grouping = ['seed', 'shot_seq_id', 'match_id']
    for keys, group in tokens.groupby(grouping):
        row = {
            'dataset': dataset_name,
            'model_type': MODEL,
            'seed': int(keys[0]),
            'shot_seq_id': int(keys[1]),
            'match_id': int(keys[2]),
            'n_events': int(len(group)),
        }
        for label, (left, right) in comparisons.items():
            row[label] = safe_spearman(
                group[left].to_numpy(float),
                group[right].to_numpy(float),
            )
        correlation_rows.append(row)
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_parquet(output / 'sequence_correlations.parquet', index=False)
    by_seed = (
        correlations.groupby('seed', as_index=False)
        .agg(
            **{
                label: (label, 'mean')
                for label in comparisons
            }
        )
    )
    by_seed.insert(0, 'dataset', dataset_name)
    by_seed.to_csv(output / 'correlation_summary_by_seed.csv', index=False)

    bootstrap_rows = []
    for label in comparisons:
        values = correlations[
            ['seed', 'match_id', 'shot_seq_id', label]
        ].dropna()
        values = values.rename(columns={label: 'difference'})
        result = seed_match_bootstrap_mean(
            values,
            n_bootstrap=n_bootstrap,
            seed=2026,
        )
        bootstrap_rows.append(
            {'dataset': dataset_name, 'metric': label, **result}
        )
    pd.DataFrame(bootstrap_rows).to_csv(
        output / 'correlation_bootstrap.csv',
        index=False,
    )

    new_deletions = read_new_seed_files(
        root,
        dataset_name,
        seeds,
        'deletion_predictions.parquet',
    )
    existing_deletions = read_existing_seed_files(
        project_dir,
        dataset_name,
        frozen_results_name,
        external_results_name,
        seeds,
        'deletion_predictions.parquet',
        sequence_ids,
    )
    if 'match_id' not in existing_deletions:
        existing_deletions = existing_deletions.merge(
            match_map,
            on='shot_seq_id',
            validate='many_to_one',
        )
    existing_deletions.insert(0, 'dataset', dataset_name)
    deletions = pd.concat(
        [existing_deletions, new_deletions],
        ignore_index=True,
    )
    deletions.to_parquet(output / 'deletion_predictions.parquet', index=False)
    deletions['base_squared_error'] = (
        deletions['target_xg'] - deletions['base_prediction']
    ) ** 2
    deletions['deleted_squared_error'] = (
        deletions['target_xg'] - deletions['deleted_prediction']
    ) ** 2
    deletions['delta_squared_error'] = (
        deletions['deleted_squared_error'] - deletions['base_squared_error']
    )
    deletion_summary = (
        deletions.groupby(['seed', 'method', 'k'], as_index=False)
        .agg(
            n_sequences=('shot_seq_id', 'nunique'),
            delta_mse=('delta_squared_error', 'mean'),
        )
    )
    deletion_summary.insert(0, 'dataset', dataset_name)
    deletion_summary.to_csv(output / 'deletion_summary_by_seed.csv', index=False)

    deletion_bootstrap = []
    methods = sorted(deletions['method'].unique())
    for k in (1, 3, 5):
        for method in methods:
            if method == 'random':
                continue
            deletion_bootstrap.append(
                {
                    'dataset': dataset_name,
                    **paired_deletion_bootstrap(
                        deletions,
                        left_method=method,
                        right_method='random',
                        k=k,
                        n_bootstrap=n_bootstrap,
                    ),
                }
            )
        for method in [
            'gradient_x_input',
            'permutation_shapley',
            'recency',
        ]:
            deletion_bootstrap.append(
                {
                    'dataset': dataset_name,
                    **paired_deletion_bootstrap(
                        deletions,
                        left_method='attention',
                        right_method=method,
                        k=k,
                        n_bootstrap=n_bootstrap,
                    ),
                }
            )
    pd.DataFrame(deletion_bootstrap).to_csv(
        output / 'deletion_bootstrap.csv',
        index=False,
    )

    diagnostics = read_new_seed_files(
        root,
        dataset_name,
        seeds,
        'shapley_diagnostics.parquet',
    )
    diagnostics.to_parquet(output / 'shapley_diagnostics.parquet', index=False)
    diagnostic_summary = (
        diagnostics.groupby('seed', as_index=False)
        .agg(
            n_sequences=('shot_seq_id', 'nunique'),
            max_local_accuracy_error=('local_accuracy_error', 'max'),
            mean_shapley_se=('mean_shapley_se', 'mean'),
            max_shapley_se=('max_shapley_se', 'max'),
        )
    )
    diagnostic_summary.insert(0, 'dataset', dataset_name)
    diagnostic_summary.to_csv(
        output / 'shapley_diagnostics_summary.csv',
        index=False,
    )
    return {
        'n_sequences': int(len(rows)),
        'n_matches': int(rows['match_id'].nunique()),
        'n_tokens_per_seed': int(len(tokens) / len(seeds)),
    }


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    root = project_dir / args.results_name
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds]
    n_permutations = (
        min(4, args.shapley_permutations)
        if args.quick
        else args.shapley_permutations
    )
    n_bootstrap = min(100, args.n_bootstrap) if args.quick else args.n_bootstrap
    device = select_device(args.device)
    protocol = {
        'model': MODEL,
        'checkpoint_root': str(
            (project_dir / args.frozen_results_name).resolve()
        ),
        'external_prepared_name': args.external_prepared_name,
        'development_prepared_name': args.development_prepared_name,
        'datasets': list(args.datasets),
        'seeds': seeds,
        'methods': [
            'permutation_shapley',
            'gradient_x_input',
            'recency',
        ],
        'shapley_permutations': int(n_permutations),
        'antithetic_permutations': True,
        'coalition_baseline': 'learned_empty_sequence',
        'coalition_batch_size': int(args.coalition_batch_size),
        'deletion_k': [1, 3, 5],
        'n_bootstrap': int(n_bootstrap),
        'quick': bool(args.quick),
        'training_performed': False,
    }
    (root / 'evaluation_protocol.json').write_text(
        json.dumps(protocol, indent=2),
        encoding='utf-8',
    )

    if args.stage in {'evaluate', 'all'}:
        for dataset_name in args.datasets:
            rows = load_rows(
                project_dir,
                dataset_name,
                quick=args.quick,
                development_prepared_name=args.development_prepared_name,
                external_prepared_name=args.external_prepared_name,
            )
            for seed in seeds:
                evaluate_seed(
                    rows,
                    project_dir=project_dir,
                    root=root,
                    dataset_name=dataset_name,
                    frozen_results_name=args.frozen_results_name,
                    seed=seed,
                    device=device,
                    n_permutations=n_permutations,
                    coalition_batch_size=args.coalition_batch_size,
                    force=args.force,
                )

    summaries = {}
    if args.stage in {'aggregate', 'all'}:
        for dataset_name in args.datasets:
            summaries[dataset_name] = aggregate_dataset(
                project_dir=project_dir,
                root=root,
                dataset_name=dataset_name,
                frozen_results_name=args.frozen_results_name,
                external_results_name=args.external_results_name,
                development_prepared_name=args.development_prepared_name,
                external_prepared_name=args.external_prepared_name,
                seeds=seeds,
                n_bootstrap=n_bootstrap,
                quick=args.quick,
            )
        marker = {
            **protocol,
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'summaries': summaries,
        }
        (root / 'evaluation_complete.json').write_text(
            json.dumps(marker, indent=2),
            encoding='utf-8',
        )
        print('[COMPLETE] Stronger explanation baselines evaluated.')


if __name__ == '__main__':
    main()
