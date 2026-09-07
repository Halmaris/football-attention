from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from football_attention.bootstrap import seed_match_bootstrap_mean
from football_attention.config import ExperimentConfig
from football_attention.faithful_eval import (
    aggregate_event_attributions,
    aggregate_player_occlusion,
    compare_player_attribution_methods,
    summarize_deletion,
)
from football_attention.faithful_model import build_faithful_model
from football_attention.frozen_final import (
    load_classical_models,
    predict_classical_models,
)
from football_attention.sequence_baselines import build_sequence_baseline
from football_attention.train import make_loader, predict, prediction_metrics
from football_attention.variants import ShotSequenceDataset, Vocabularies
from football_attention.tasks.common import (
    LENGTH_LABELS,
    calibration_metrics,
    novelty_table,
    select_device,
)
from football_attention.tasks.sequence import evaluate_run
from football_attention.tasks.refit import run_dir, verify_training_complete


PRIMARY_MODEL = 'gru_attention_faithful'


REFERENCE_MODEL = 'gradient_boosting'


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
        '--results-name',
        default='results/holdout',
    )
    parser.add_argument(
        '--stage',
        choices=['prepare', 'predict', 'faithfulness', 'aggregate', 'all'],
        default='all',
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'mps', 'cuda'],
        default='auto',
    )
    parser.add_argument('--n-bootstrap', type=int, default=10_000)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def read_protocol(frozen_dir: Path) -> dict[str, object]:
    path = frozen_dir / 'frozen_protocol.json'
    if not path.exists():
        raise RuntimeError(f'Missing frozen protocol: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def load_external(
    project_dir: Path,
    external_prepared_name: str = DEFAULT_EXTERNAL_PREPARED_NAME,
) -> pd.DataFrame:
    path = project_dir / 'data' / external_prepared_name / 'sequences_raw.parquet'
    if not path.exists():
        raise RuntimeError(f'Missing prepared external holdout: {path}')
    rows = pd.read_parquet(path)
    if rows.empty:
        raise RuntimeError('External holdout is empty.')
    return rows


def prepare_metadata(
    project_dir: Path,
    output_dir: Path,
    frozen_dir: Path,
    protocol: dict[str, object],
    *,
    development_prepared_name: str,
    external_prepared_name: str,
) -> None:
    external = load_external(project_dir, external_prepared_name)
    historical = pd.read_parquet(
        project_dir
        / 'data'
        / development_prepared_name
        / 'sequences_raw.parquet'
    )
    development = historical[
        historical['split'].isin(protocol['fit_splits'])
    ]
    novelty, novelty_summary = novelty_table(external, development)
    novelty.to_parquet(output_dir / 'sequence_novelty.parquet', index=False)
    prepared_dir = project_dir / 'data' / external_prepared_name
    manifest_path = prepared_dir / 'match_manifest.parquet'
    manifest = pd.read_parquet(manifest_path)
    week_counts = manifest.groupby('match_week').size() if 'match_week' in manifest else pd.Series(dtype=int)
    requested_match_weeks = sorted(int(value) for value in week_counts.index)
    summary = {
        'dataset': external_prepared_name,
        'requested_match_weeks': requested_match_weeks,
        'n_matches': int(external['match_id'].nunique()),
        'n_sequences': int(len(external)),
        'date_start': str(external['match_date'].min()),
        'date_end': str(external['match_date'].max()),
        'matches_by_week': {
            str(int(key)): int(value)
            for key, value in week_counts.items()
        },
        'empty_pre_shot_sequences': int(
            novelty['n_pre_shot_events'].eq(0).sum()
        ),
        'length_bin_counts': {
            str(key): int(value)
            for key, value in novelty['length_bin'].value_counts().items()
        },
        'novelty_reference_splits': list(protocol['fit_splits']),
        **novelty_summary,
    }
    (output_dir / 'dataset_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (output_dir / 'evaluation_protocol.json').write_text(
        json.dumps(
            {
                'source_frozen_results': str(frozen_dir.resolve()),
                'source_protocol': str(
                    (frozen_dir / 'frozen_protocol.json').resolve()
                ),
                'source_external_prepared': str(prepared_dir.resolve()),
                'source_development_prepared': str(
                    (
                        project_dir / 'data' / development_prepared_name
                    ).resolve()
                ),
                'fit_splits': protocol['fit_splits'],
                'external_data_used_for_training': False,
                'external_data_used_for_tuning': False,
                'neural_models': list(protocol['neural_models']),
                'classical_models': list(protocol['classical_models']),
                'seeds': protocol['seeds'],
            },
            indent=2,
        ),
        encoding='utf-8',
    )


def checkpoint_config(
    checkpoint: dict[str, object],
    project_dir: Path,
) -> tuple[ExperimentConfig, Vocabularies]:
    config = ExperimentConfig(**checkpoint['config'])
    config = replace(config, project_dir=project_dir)
    vocabularies = Vocabularies(**checkpoint['vocabularies'])
    return config, vocabularies


def build_checkpoint_model(
    checkpoint: dict[str, object],
    vocabularies: Vocabularies,
    config: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    if checkpoint['model_family'] == 'sequence':
        model = build_sequence_baseline(
            checkpoint['model_type'],
            vocabularies,
            config,
        )
    else:
        model = build_faithful_model(
            checkpoint['model_type'],
            vocabularies,
            config.sequence_length,
            config,
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)


def prediction_path(
    output_dir: Path,
    model_type: str,
    seed: int,
) -> Path:
    return output_dir / 'predictions' / model_type / f'seed_{seed}.parquet'


def predict_classical(
    external: pd.DataFrame,
    frozen_dir: Path,
    output_dir: Path,
    *,
    force: bool,
) -> None:
    fitted = load_classical_models(frozen_dir / 'classical/models.pkl')
    target = external['xg'].to_numpy(float)
    for model_type, values in predict_classical_models(
        external,
        fitted,
    ).items():
        path = prediction_path(output_dir, model_type, -1)
        if path.exists() and not force:
            print(f'Using cached external prediction model={model_type}')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                'model_type': model_type,
                'seed': -1,
                'shot_seq_id': external['shot_seq_id'].to_numpy(),
                'match_id': external['match_id'].to_numpy(),
                'target_xg': target,
                'prediction': values,
            }
        ).to_parquet(path, index=False)
        print(f'Predicted external holdout model={model_type}')


def predict_neural(
    project_dir: Path,
    external: pd.DataFrame,
    frozen_dir: Path,
    output_dir: Path,
    protocol: dict[str, object],
    *,
    device: torch.device,
    force: bool,
) -> None:
    for model_type in protocol['neural_models']:
        for seed in protocol['seeds']:
            seed = int(seed)
            path = prediction_path(output_dir, model_type, seed)
            if path.exists() and not force:
                print(
                    f'Using cached external prediction model={model_type}, '
                    f'seed={seed}'
                )
                continue
            source = run_dir(frozen_dir, model_type, seed) / 'model.pt'
            checkpoint = torch.load(
                source,
                map_location=device,
                weights_only=False,
            )
            config, vocabularies = checkpoint_config(checkpoint, project_dir)
            dataset = ShotSequenceDataset(
                external,
                vocabularies=vocabularies,
                variant='pre_shot',
                sequence_length=config.sequence_length,
            )
            model = build_checkpoint_model(
                checkpoint,
                vocabularies,
                config,
                device,
            )
            loader = make_loader(dataset, config, shuffle=False)
            sequence_ids, targets, values = predict(model, loader, device)
            match_map = external.set_index('shot_seq_id')['match_id']
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    'model_type': model_type,
                    'seed': seed,
                    'shot_seq_id': sequence_ids,
                    'match_id': pd.Series(sequence_ids)
                    .map(match_map)
                    .to_numpy(),
                    'target_xg': targets,
                    'prediction': values,
                }
            ).to_parquet(path, index=False)
            print(
                f'Predicted external holdout model={model_type}, seed={seed}'
            )
            del model
            if device.type == 'mps':
                torch.mps.empty_cache()


def run_predictions(
    project_dir: Path,
    frozen_dir: Path,
    output_dir: Path,
    protocol: dict[str, object],
    *,
    external_prepared_name: str,
    device: torch.device,
    force: bool,
) -> None:
    external = load_external(project_dir, external_prepared_name)
    predict_classical(external, frozen_dir, output_dir, force=force)
    predict_neural(
        project_dir,
        external,
        frozen_dir,
        output_dir,
        protocol,
        device=device,
        force=force,
    )


def read_predictions(
    output_dir: Path,
    protocol: dict[str, object],
) -> pd.DataFrame:
    paths = [
        prediction_path(output_dir, model_type, -1)
        for model_type in protocol['classical_models']
    ]
    paths.extend(
        prediction_path(output_dir, model_type, int(seed))
        for model_type in protocol['neural_models']
        for seed in protocol['seeds']
    )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f'External predictions are incomplete ({len(missing)} missing).'
        )
    return pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )


def scope_masks(values: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        'all': pd.Series(True, index=values.index),
        'nonempty': values['n_pre_shot_events'].gt(0),
        'empty': values['n_pre_shot_events'].eq(0),
        **{
            f'length_{label}': values['length_bin'].eq(label)
            for label in LENGTH_LABELS
        },
        'known_team': values['team_status'].eq('known_team'),
        'new_team': values['team_status'].eq('new_team'),
        'known_players_only': values['player_status'].eq('known_players_only'),
        'contains_new_player': values['player_status'].eq(
            'contains_new_player'
        ),
    }


def metric_rows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    calibration = []
    for (model_type, seed), group in predictions.groupby(
        ['model_type', 'seed']
    ):
        for scope, mask in scope_masks(group).items():
            selected = group[mask]
            if selected.empty:
                continue
            metrics = prediction_metrics(
                selected['target_xg'].to_numpy(),
                selected['prediction'].to_numpy(),
            )
            intercept, slope = calibration_metrics(
                selected['target_xg'].to_numpy(),
                selected['prediction'].to_numpy(),
            )
            rows.append(
                {
                    'model_type': model_type,
                    'seed': int(seed),
                    'scope': scope,
                    'n_matches': int(selected['match_id'].nunique()),
                    'n_sequences': int(len(selected)),
                    **asdict(metrics),
                }
            )
            calibration.append(
                {
                    'model_type': model_type,
                    'seed': int(seed),
                    'scope': scope,
                    'n_sequences': int(len(selected)),
                    'calibration_intercept': intercept,
                    'calibration_slope': slope,
                    'prediction_mean': float(selected['prediction'].mean()),
                    'target_mean': float(selected['target_xg'].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def bootstrap_model_mse(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for index, (model_type, group) in enumerate(
        predictions.groupby('model_type')
    ):
        for scope_index, scope in enumerate(['all', 'nonempty']):
            selected = group[scope_masks(group)[scope]].copy()
            selected['seed'] = selected['seed'].replace(-1, 42)
            selected['squared_error'] = (
                selected['target_xg'] - selected['prediction']
            ) ** 2
            result = seed_match_bootstrap_mean(
                selected,
                value_column='squared_error',
                n_bootstrap=n_bootstrap,
                seed=20_260 + index * 10 + scope_index,
            )
            rows.append(
                {
                    'model_type': model_type,
                    'scope': scope,
                    'mse_mean': result.pop('difference_mean'),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def paired_mse_difference(
    predictions: pd.DataFrame,
    model_left: str,
    model_right: str,
    scope: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    left = predictions[predictions['model_type'].eq(model_left)].copy()
    right = predictions[predictions['model_type'].eq(model_right)].copy()
    neural_pair = left['seed'].nunique() > 1 and right['seed'].nunique() > 1
    keys = ['seed', 'shot_seq_id'] if neural_pair else ['shot_seq_id']
    right_columns = [*keys, 'prediction']
    paired = left.merge(
        right[right_columns].rename(columns={'prediction': 'prediction_right'}),
        on=keys,
        validate='one_to_one' if neural_pair else 'many_to_one',
    )
    selected = paired[scope_masks(paired)[scope]].copy()
    selected['seed'] = selected['seed'].replace(-1, 42)
    selected['difference'] = (
        (selected['target_xg'] - selected['prediction']) ** 2
        - (selected['target_xg'] - selected['prediction_right']) ** 2
    )
    if selected.empty:
        return {
            'model_left': model_left, 'model_right': model_right, 'scope': scope,
            'difference_mean': np.nan, 'ci_025': np.nan, 'ci_975': np.nan,
            'n_seeds': 0, 'n_matches': 0, 'n_sequences': 0, 'n_rows': 0,
            'n_bootstrap': n_bootstrap,
        }
    return {
        'model_left': model_left,
        'model_right': model_right,
        'scope': scope,
        **seed_match_bootstrap_mean(
            selected,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }


def aggregate_predictions(
    project_dir: Path,
    frozen_dir: Path,
    output_dir: Path,
    protocol: dict[str, object],
    *,
    n_bootstrap: int,
) -> None:
    predictions = read_predictions(output_dir, protocol)
    novelty = pd.read_parquet(output_dir / 'sequence_novelty.parquet')
    predictions = predictions.merge(
        novelty,
        on=['shot_seq_id', 'match_id'],
        validate='many_to_one',
    )
    predictions.to_parquet(output_dir / 'predictions.parquet', index=False)
    metrics, calibration = metric_rows(predictions)
    metrics.to_csv(output_dir / 'prediction_metrics_by_seed.csv', index=False)
    calibration.to_csv(output_dir / 'calibration_by_seed.csv', index=False)
    metric_summary = (
        metrics.groupby(['model_type', 'scope'], as_index=False)
        .agg(
            n_seeds=('seed', 'nunique'),
            n_matches=('n_matches', 'max'),
            n_sequences=('n_sequences', 'max'),
            mse_mean=('mse', 'mean'),
            mse_sd=('mse', 'std'),
            mae_mean=('mae', 'mean'),
            mae_sd=('mae', 'std'),
            r2_mean=('r2', 'mean'),
            r2_sd=('r2', 'std'),
        )
    )
    metric_summary.to_csv(
        output_dir / 'prediction_metrics_summary.csv',
        index=False,
    )
    (
        calibration.groupby(['model_type', 'scope'], as_index=False)
        .agg(
            n_seeds=('seed', 'nunique'),
            n_sequences=('n_sequences', 'max'),
            calibration_intercept_mean=('calibration_intercept', 'mean'),
            calibration_intercept_sd=('calibration_intercept', 'std'),
            calibration_slope_mean=('calibration_slope', 'mean'),
            calibration_slope_sd=('calibration_slope', 'std'),
            prediction_mean=('prediction_mean', 'mean'),
            target_mean=('target_mean', 'mean'),
        )
        .to_csv(output_dir / 'calibration_summary.csv', index=False)
    )
    bootstrap_model_mse(
        predictions,
        n_bootstrap=n_bootstrap,
    ).to_csv(output_dir / 'prediction_mse_bootstrap.csv', index=False)

    comparisons = []
    scopes = [
        'all',
        'nonempty',
        'known_team',
        'new_team',
        'known_players_only',
        'contains_new_player',
    ]
    models = sorted(predictions['model_type'].unique())
    counter = 0
    for model_type in models:
        if model_type == REFERENCE_MODEL:
            continue
        for scope in scopes:
            comparisons.append(
                paired_mse_difference(
                    predictions,
                    model_type,
                    REFERENCE_MODEL,
                    scope,
                    n_bootstrap=n_bootstrap,
                    seed=21_260 + counter,
                )
            )
            counter += 1
    for scope in scopes:
        comparisons.append(
            paired_mse_difference(
                predictions,
                PRIMARY_MODEL,
                'gru_attention_predictive',
                scope,
                n_bootstrap=n_bootstrap,
                seed=22_260 + counter,
            )
        )
        counter += 1
    pd.DataFrame(comparisons).to_csv(
        output_dir / 'prediction_differences_bootstrap.csv',
        index=False,
    )

    historical = pd.read_csv(frozen_dir / 'test_metrics_summary.csv')
    historical = historical[
        ['model_type', 'mse_mean', 'mse_sd', 'mae_mean', 'mae_sd']
    ].rename(
        columns={
            'mse_mean': 'historical_mse_mean',
            'mse_sd': 'historical_mse_sd',
            'mae_mean': 'historical_mae_mean',
            'mae_sd': 'historical_mae_sd',
        }
    )
    external_all = metric_summary[metric_summary['scope'].eq('all')][
        ['model_type', 'mse_mean', 'mse_sd', 'mae_mean', 'mae_sd']
    ].rename(
        columns={
            'mse_mean': 'external_mse_mean',
            'mse_sd': 'external_mse_sd',
            'mae_mean': 'external_mae_mean',
            'mae_sd': 'external_mae_sd',
        }
    )
    historical.merge(
        external_all,
        on='model_type',
        validate='one_to_one',
    ).to_csv(output_dir / 'historical_external_metrics.csv', index=False)


def run_faithfulness(
    project_dir: Path,
    frozen_dir: Path,
    output_dir: Path,
    protocol: dict[str, object],
    *,
    external_prepared_name: str,
    device: torch.device,
    force: bool,
) -> None:
    external = load_external(project_dir, external_prepared_name)
    root = output_dir / 'faithfulness' / PRIMARY_MODEL
    for seed in protocol['seeds']:
        seed = int(seed)
        target = root / f'seed_{seed}'
        required = [
            target / 'token_attributions.parquet',
            target / 'sequence_faithfulness.parquet',
            target / 'deletion_predictions.parquet',
            target / 'player_occlusion.parquet',
            target / 'randomization_per_sequence.parquet',
        ]
        if not force and all(path.exists() for path in required):
            print(f'Using cached external faithfulness seed={seed}')
            continue
        source = run_dir(frozen_dir, PRIMARY_MODEL, seed) / 'model.pt'
        checkpoint = torch.load(
            source,
            map_location=device,
            weights_only=False,
        )
        config, vocabularies = checkpoint_config(checkpoint, project_dir)
        dataset = ShotSequenceDataset(
            external,
            vocabularies=vocabularies,
            variant='pre_shot',
            sequence_length=config.sequence_length,
        )
        model = build_checkpoint_model(
            checkpoint,
            vocabularies,
            config,
            device,
        )
        target.mkdir(parents=True, exist_ok=True)
        evaluate_run(
            model,
            PRIMARY_MODEL,
            seed,
            run_dir=target,
            dataset=dataset,
            config=config,
            force=force or not all(path.exists() for path in required),
        )
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()


def read_faithfulness_files(
    output_dir: Path,
    protocol: dict[str, object],
    filename: str,
) -> pd.DataFrame:
    paths = [
        output_dir
        / 'faithfulness'
        / PRIMARY_MODEL
        / f'seed_{int(seed)}'
        / filename
        for seed in protocol['seeds']
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f'External faithfulness is incomplete ({len(missing)} missing).'
        )
    return pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )


def attach_match_id(
    values: pd.DataFrame,
    match_map: pd.DataFrame,
) -> pd.DataFrame:
    if 'match_id' in values.columns:
        return values
    return values.merge(
        match_map,
        on='shot_seq_id',
        validate='many_to_one',
    )


def aggregate_faithfulness(
    project_dir: Path,
    output_dir: Path,
    protocol: dict[str, object],
    *,
    external_prepared_name: str,
    n_bootstrap: int,
) -> None:
    tokens = read_faithfulness_files(
        output_dir, protocol, 'token_attributions.parquet'
    )
    sequences = read_faithfulness_files(
        output_dir, protocol, 'sequence_faithfulness.parquet'
    )
    deletions = read_faithfulness_files(
        output_dir, protocol, 'deletion_predictions.parquet'
    )
    players = read_faithfulness_files(
        output_dir, protocol, 'player_occlusion.parquet'
    )
    randomization = read_faithfulness_files(
        output_dir, protocol, 'randomization_per_sequence.parquet'
    )
    external = load_external(project_dir, external_prepared_name)
    match_map = external[['shot_seq_id', 'match_id']]
    sequences = attach_match_id(sequences, match_map)
    deletions = attach_match_id(deletions, match_map)
    randomization = attach_match_id(randomization, match_map)

    by_seed = (
        sequences.groupby(['model_type', 'seed'], as_index=False)
        .agg(
            n_occlusion=('attention_vs_occlusion_rho', 'count'),
            attention_vs_occlusion=('attention_vs_occlusion_rho', 'mean'),
            n_ig=('attention_vs_ig_rho', 'count'),
            attention_vs_ig=('attention_vs_ig_rho', 'mean'),
        )
    )
    by_seed.to_csv(output_dir / 'faithfulness_by_seed.csv', index=False)
    (
        by_seed.groupby('model_type', as_index=False)
        .agg(
            attention_vs_occlusion_mean=(
                'attention_vs_occlusion',
                'mean',
            ),
            attention_vs_occlusion_sd=('attention_vs_occlusion', 'std'),
            attention_vs_ig_mean=('attention_vs_ig', 'mean'),
            attention_vs_ig_sd=('attention_vs_ig', 'std'),
        )
        .to_csv(output_dir / 'faithfulness_summary.csv', index=False)
    )
    deletion_by_seed = summarize_deletion(deletions)
    deletion_by_seed.to_csv(output_dir / 'deletion_by_seed.csv', index=False)
    (
        deletion_by_seed.groupby(['model_type', 'method', 'k'], as_index=False)
        .agg(
            delta_mse_mean=('delta_mse', 'mean'),
            delta_mse_sd=('delta_mse', 'std'),
            prediction_change_mean=(
                'mean_absolute_prediction_change',
                'mean',
            ),
        )
        .to_csv(output_dir / 'deletion_summary.csv', index=False)
    )
    randomization_by_seed = (
        randomization.groupby(['model_type', 'seed'], as_index=False)
        .agg(
            n_sequences=('trained_vs_randomized_rho', 'count'),
            rho_mean=('trained_vs_randomized_rho', 'mean'),
        )
    )
    randomization_by_seed.to_csv(
        output_dir / 'randomization_by_seed.csv', index=False
    )
    (
        randomization_by_seed.groupby('model_type', as_index=False)
        .agg(rho_mean=('rho_mean', 'mean'), rho_sd=('rho_mean', 'std'))
        .to_csv(output_dir / 'randomization_summary.csv', index=False)
    )

    event_report = aggregate_event_attributions(tokens)
    comparison = compare_player_attribution_methods(event_report)
    (
        comparison.groupby(['model_type', 'comparison'], as_index=False)
        .agg(
            spearman_mean=('spearman_rho', 'mean'),
            spearman_sd=('spearman_rho', 'std'),
            jaccard_at_10_mean=('jaccard_at_10', 'mean'),
            jaccard_at_10_sd=('jaccard_at_10', 'std'),
        )
        .to_csv(
            output_dir / 'attribution_method_comparison_summary.csv',
            index=False,
        )
    )
    player_report = aggregate_player_occlusion(players)
    names = json.loads(
        (
            project_dir
            / 'data'
            / external_prepared_name
            / 'player_id_to_name.json'
        ).read_text(encoding='utf-8')
    )
    player_report.insert(
        3,
        'player_name',
        player_report['player_id'].map(
            lambda value: names.get(str(int(value)), f'pid{int(value)}')
        ),
    )
    player_report.to_csv(
        output_dir / 'player_contribution_descriptive.csv', index=False
    )

    bootstrap_rows = []
    for column in ['attention_vs_occlusion_rho', 'attention_vs_ig_rho']:
        result = seed_match_bootstrap_mean(
            sequences,
            value_column=column,
            n_bootstrap=n_bootstrap,
            seed=23_260,
        )
        bootstrap_rows.append(
            {
                'metric': column,
                'estimate': result.pop('difference_mean'),
                **result,
            }
        )
    deletion_values = deletions.copy()
    deletion_values['base_squared_error'] = (
        deletion_values['target_xg'] - deletion_values['base_prediction']
    ) ** 2
    deletion_values['deleted_squared_error'] = (
        deletion_values['target_xg'] - deletion_values['deleted_prediction']
    ) ** 2
    deletion_values['delta_squared_error'] = (
        deletion_values['deleted_squared_error']
        - deletion_values['base_squared_error']
    )
    for k in sorted(deletion_values['k'].unique()):
        attention = deletion_values[
            deletion_values['method'].eq('attention')
            & deletion_values['k'].eq(k)
        ][
            [
                'seed',
                'shot_seq_id',
                'match_id',
                'delta_squared_error',
            ]
        ]
        random = deletion_values[
            deletion_values['method'].eq('random')
            & deletion_values['k'].eq(k)
        ][['seed', 'shot_seq_id', 'delta_squared_error']]
        paired = attention.merge(
            random,
            on=['seed', 'shot_seq_id'],
            suffixes=('_attention', '_random'),
            validate='one_to_one',
        )
        paired['difference'] = (
            paired['delta_squared_error_attention']
            - paired['delta_squared_error_random']
        )
        result = seed_match_bootstrap_mean(
            paired,
            n_bootstrap=n_bootstrap,
            seed=24_260 + int(k),
        )
        bootstrap_rows.append(
            {
                'metric': f'attention_minus_random:k={int(k)}',
                'estimate': result.pop('difference_mean'),
                **result,
            }
        )
    result = seed_match_bootstrap_mean(
        randomization,
        value_column='trained_vs_randomized_rho',
        n_bootstrap=n_bootstrap,
        seed=25_260,
    )
    bootstrap_rows.append(
        {
            'metric': 'trained_vs_randomized_rho',
            'estimate': result.pop('difference_mean'),
            **result,
        }
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / 'faithfulness_bootstrap.csv', index=False
    )


def write_completion_marker(
    project_dir: Path,
    frozen_dir: Path,
    output_dir: Path,
    protocol: dict[str, object],
    *,
    external_prepared_name: str,
) -> None:
    predictions = read_predictions(output_dir, protocol)
    read_faithfulness_files(
        output_dir, protocol, 'randomization_per_sequence.parquet'
    )
    external = load_external(project_dir, external_prepared_name)
    marker = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_frozen_results': str(frozen_dir.resolve()),
        'models': sorted(predictions['model_type'].unique()),
        'faithfulness_model': PRIMARY_MODEL,
        'seeds': protocol['seeds'],
        'n_matches': int(external['match_id'].nunique()),
        'n_sequences': int(len(external)),
        'training_performed': False,
        'tuning_performed': False,
    }
    (output_dir / 'external_evaluation_complete.json').write_text(
        json.dumps(marker, indent=2), encoding='utf-8'
    )


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    frozen_dir = project_dir / args.frozen_results_name
    output_dir = project_dir / args.results_name
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = read_protocol(frozen_dir)
    verify_training_complete(frozen_dir, protocol)
    device = select_device(args.device)

    if args.stage in {'prepare', 'all'}:
        prepare_metadata(
            project_dir,
            output_dir,
            frozen_dir,
            protocol,
            development_prepared_name=args.development_prepared_name,
            external_prepared_name=args.external_prepared_name,
        )
        print('[PREPARED] External holdout metadata.')
    if args.stage in {'predict', 'all'}:
        run_predictions(
            project_dir,
            frozen_dir,
            output_dir,
            protocol,
            external_prepared_name=args.external_prepared_name,
            device=device,
            force=args.force,
        )
        print('[PREDICTIONS COMPLETE] All frozen models evaluated.')
    if args.stage in {'faithfulness', 'all'}:
        run_faithfulness(
            project_dir,
            frozen_dir,
            output_dir,
            protocol,
            external_prepared_name=args.external_prepared_name,
            device=device,
            force=args.force,
        )
        print('[FAITHFULNESS COMPLETE] Final attention model evaluated.')
    if args.stage in {'aggregate', 'all'}:
        aggregate_predictions(
            project_dir,
            frozen_dir,
            output_dir,
            protocol,
            n_bootstrap=args.n_bootstrap,
        )
        aggregate_faithfulness(
            project_dir,
            output_dir,
            protocol,
            external_prepared_name=args.external_prepared_name,
            n_bootstrap=args.n_bootstrap,
        )
        write_completion_marker(
            project_dir,
            frozen_dir,
            output_dir,
            protocol,
            external_prepared_name=args.external_prepared_name,
        )
        print('[COMPLETE] Frozen external holdout evaluation completed.')


if __name__ == '__main__':
    main()
