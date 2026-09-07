from __future__ import annotations
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from football_attention.config import ExperimentConfig
from football_attention.data import split_rows
from football_attention.exposure import build_exposure_table
from football_attention.faithful_eval import (
    aggregate_event_attributions,
    aggregate_player_occlusion,
    bootstrap_player_occlusion,
    compare_player_attribution_methods,
    cross_seed_player_stability,
    evaluate_faithfulness,
    randomized_attention_test,
    summarize_deletion,
)
from football_attention.faithful_model import FaithfulModel, build_faithful_model
from football_attention.faithful_train import fit_faithful_transformer, save_faithful_run
from football_attention.train import make_loader, predict, prediction_metrics, select_device
from football_attention.serialization import config_from_dict
from football_attention.variants import ShotSequenceDataset, fit_vocabularies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        '--stage',
        choices=['train', 'evaluate', 'aggregate', 'all'],
        default='all',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=[
            'gru_attention',
            'gru_attention_shuffled',
        ],
        default=['gru_attention'],
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--results-name', default=None)
    parser.add_argument('--prepared-name', default='prepared')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    default_results_name = (
        'results_faithful_quick' if args.quick else 'results_faithful'
    )
    config = ExperimentConfig(
        project_dir=args.project_dir.resolve(),
        results_name=args.results_name or default_results_name,
        prepared_name=args.prepared_name,
    )
    updates = {}
    if args.models:
        updates['faithful_models'] = tuple(args.models)
    if args.seeds:
        updates['seeds'] = tuple(args.seeds)
    if args.quick:
        updates.update(
            {
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 2,
                'max_epochs': 2,
                'patience': 2,
                'alignment_lambda': 0.01,
                'alignment_warmup_epochs': 0,
                'alignment_ramp_epochs': 1,
                'occlusion_samples': 2,
                'integrated_gradients_steps': 4,
                'bootstrap_samples': 100,
            }
        )
    return replace(config, **updates)


def load_data(
    config: ExperimentConfig,
    *,
    quick: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, ShotSequenceDataset], object]:
    sequences = pd.read_parquet(config.prepared_dir / 'sequences_raw.parquet')
    if quick:
        sequences = (
            sequences.groupby('split', group_keys=False)
            .head(160)
            .reset_index(drop=True)
        )
    partitions = split_rows(sequences)
    vocabularies = fit_vocabularies(partitions['train'])
    datasets = build_datasets(
        partitions,
        vocabularies=vocabularies,
        config=config,
    )
    return partitions, datasets, vocabularies


def build_datasets(
    partitions: dict[str, pd.DataFrame],
    *,
    vocabularies: object,
    config: ExperimentConfig,
    shuffle_events: bool = False,
) -> dict[str, ShotSequenceDataset]:
    return {
        split: ShotSequenceDataset(
            rows,
            vocabularies=vocabularies,
            variant='pre_shot',
            sequence_length=config.sequence_length,
            shuffle_events=shuffle_events,
        )
        for split, rows in partitions.items()
    }


def load_model(
    run_dir: Path,
    *,
    vocabularies: object,
    config: ExperimentConfig,
    device: torch.device,
) -> FaithfulModel:
    checkpoint = torch.load(
        run_dir / 'model.pt',
        map_location=device,
        weights_only=False,
    )
    config = config_from_dict(config, checkpoint['config'])
    model = build_faithful_model(
        checkpoint['model_type'],
        vocabularies,
        config.sequence_length,
        config,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def train_run(
    model_type: str,
    seed: int,
    *,
    run_dir: Path,
    datasets: dict[str, ShotSequenceDataset],
    vocabularies: object,
    config: ExperimentConfig,
    force: bool,
) -> FaithfulModel:
    device = select_device()
    if not force and (run_dir / 'model.pt').exists() and (run_dir / 'metrics.csv').exists():
        print(f'Using cached faithful run model={model_type}, seed={seed}')
        return load_model(
            run_dir,
            vocabularies=vocabularies,
            config=config,
            device=device,
        )

    print(f'Training faithful model={model_type}, seed={seed}, device={device}')
    model, history, validation_metrics = fit_faithful_transformer(
        datasets['train'],
        datasets['validation'],
        vocabularies=vocabularies,
        config=config,
        model_type=model_type,
        seed=seed,
    )
    save_faithful_run(
        run_dir,
        model=model,
        history=history,
        validation_metrics=validation_metrics,
        vocabularies=vocabularies,
        config=config,
        model_type=model_type,
        seed=seed,
    )
    best_epoch = int(
        history.loc[history['validation_mse'].idxmin(), 'epoch']
    )
    metric_rows = []
    prediction_frames = []
    for split in ['train', 'validation', 'test']:
        loader = make_loader(datasets[split], config, shuffle=False)
        sequence_ids, targets, predictions = predict(model, loader, device)
        metrics = prediction_metrics(targets, predictions)
        metric_rows.append(
            {
                'model_type': model_type,
                'seed': seed,
                'split': split,
                **asdict(metrics),
                'best_epoch': best_epoch,
            }
        )
        if split == 'test':
            prediction_frames.append(
                pd.DataFrame(
                    {
                        'model_type': model_type,
                        'seed': seed,
                        'shot_seq_id': sequence_ids,
                        'target_xg': targets,
                        'prediction': predictions,
                    }
                )
            )
    pd.DataFrame(metric_rows).to_csv(run_dir / 'metrics.csv', index=False)
    pd.concat(prediction_frames).to_parquet(
        run_dir / 'test_predictions.parquet',
        index=False,
    )
    return model


def evaluate_run(
    model: FaithfulModel,
    model_type: str,
    seed: int,
    *,
    run_dir: Path,
    dataset: ShotSequenceDataset,
    config: ExperimentConfig,
    force: bool,
) -> None:
    required = [
        run_dir / 'token_attributions.parquet',
        run_dir / 'player_occlusion.parquet',
        run_dir / 'deletion_summary.csv',
        run_dir / 'randomization_summary.csv',
    ]
    if not force and all(path.exists() for path in required):
        print(f'Using cached evaluation model={model_type}, seed={seed}')
        return

    print(f'Evaluating faithful model={model_type}, seed={seed}')
    device = select_device()
    loader = make_loader(dataset, config, shuffle=False)
    tokens, sequences, deletions, player_rows = evaluate_faithfulness(
        model,
        loader,
        device=device,
        config=config,
        model_type=model_type,
        seed=seed,
    )
    randomization = randomized_attention_test(
        model,
        loader,
        tokens,
        device=device,
        model_type=model_type,
        seed=seed,
    )
    event_player_report = aggregate_event_attributions(tokens)
    method_comparison = compare_player_attribution_methods(event_player_report)

    tokens.to_parquet(run_dir / 'token_attributions.parquet', index=False)
    sequences.to_parquet(run_dir / 'sequence_faithfulness.parquet', index=False)
    deletions.to_parquet(run_dir / 'deletion_predictions.parquet', index=False)
    summarize_deletion(deletions).to_csv(
        run_dir / 'deletion_summary.csv',
        index=False,
    )
    player_rows.to_parquet(run_dir / 'player_occlusion.parquet', index=False)
    aggregate_player_occlusion(player_rows).to_csv(
        run_dir / 'player_occlusion_report.csv',
        index=False,
    )
    event_player_report.to_csv(
        run_dir / 'event_player_report.csv',
        index=False,
    )
    method_comparison.to_csv(
        run_dir / 'attribution_method_comparison.csv',
        index=False,
    )
    randomization.to_parquet(
        run_dir / 'randomization_per_sequence.parquet',
        index=False,
    )
    (
        randomization.groupby(['model_type', 'seed'], as_index=False)
        .agg(
            n_sequences=('trained_vs_randomized_rho', 'count'),
            rho_mean=('trained_vs_randomized_rho', 'mean'),
            rho_median=('trained_vs_randomized_rho', 'median'),
        )
        .to_csv(run_dir / 'randomization_summary.csv', index=False)
    )


def correlation_table(
    report: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    grouping: list[str],
    scores: list[str],
) -> pd.DataFrame:
    exposure_columns = [
        'sequence_count_exposure',
        'event_count_exposure',
        'sum_sequence_xg_exposure',
        'minutes',
        'team_xg',
        'goals_plus_assists',
    ]
    merged = report.merge(exposures, on='player_id', how='left')
    rows = []
    for keys, group in merged.groupby(grouping):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for score in scores:
            for exposure in exposure_columns:
                complete = group[[score, exposure]].dropna()
                if (
                    len(complete) < 3
                    or complete[score].nunique() < 2
                    or complete[exposure].nunique() < 2
                ):
                    rho = np.nan
                    p_value = np.nan
                else:
                    result = spearmanr(complete[score], complete[exposure])
                    rho = float(result.statistic)
                    p_value = float(result.pvalue)
                rows.append(
                    {
                        **dict(zip(grouping, keys)),
                        'score': score,
                        'exposure': exposure,
                        'n_players': len(complete),
                        'spearman_rho': rho,
                        'p_value': p_value,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_results(
    config: ExperimentConfig,
    partitions: dict[str, pd.DataFrame],
) -> None:
    root = config.results_dir / 'faithful'
    run_dirs = sorted(root.glob('*/seed_*'))
    metric_files = [path / 'metrics.csv' for path in run_dirs if (path / 'metrics.csv').exists()]
    if metric_files:
        metrics = pd.concat(map(pd.read_csv, metric_files), ignore_index=True)
        metrics.to_csv(root / 'metrics.csv', index=False)
        metrics_summary = (
            metrics.groupby(['model_type', 'split'], as_index=False)
            .agg(
                mse_mean=('mse', 'mean'),
                mse_sd=('mse', 'std'),
                mae_mean=('mae', 'mean'),
                mae_sd=('mae', 'std'),
                r2_mean=('r2', 'mean'),
                r2_sd=('r2', 'std'),
            )
        )
        metrics_summary.to_csv(root / 'metrics_summary.csv', index=False)

    evaluated = [path for path in run_dirs if (path / 'token_attributions.parquet').exists()]
    if not evaluated:
        return
    tokens = pd.concat(
        [pd.read_parquet(path / 'token_attributions.parquet') for path in evaluated],
        ignore_index=True,
    )
    sequence_faithfulness = pd.concat(
        [pd.read_parquet(path / 'sequence_faithfulness.parquet') for path in evaluated],
        ignore_index=True,
    )
    deletions = pd.concat(
        [pd.read_parquet(path / 'deletion_predictions.parquet') for path in evaluated],
        ignore_index=True,
    )
    player_rows = pd.concat(
        [pd.read_parquet(path / 'player_occlusion.parquet') for path in evaluated],
        ignore_index=True,
    )
    player_report = aggregate_player_occlusion(player_rows)
    event_report = aggregate_event_attributions(tokens)
    method_comparison = compare_player_attribution_methods(event_report)
    randomization = pd.concat(
        [pd.read_parquet(path / 'randomization_per_sequence.parquet') for path in evaluated],
        ignore_index=True,
    )

    player_names = json.loads(
        (config.prepared_dir / 'player_id_to_name.json').read_text(encoding='utf-8')
    )
    name_map = lambda player_id: player_names.get(  # noqa: E731
        str(int(player_id)),
        f'pid{player_id}',
    )
    player_report.insert(
        player_report.columns.get_loc('player_id') + 1,
        'player_name',
        player_report['player_id'].map(name_map),
    )
    event_report.insert(
        event_report.columns.get_loc('player_id') + 1,
        'player_name',
        event_report['player_id'].map(name_map),
    )

    exposures = build_exposure_table(partitions['test'], config.cache_dir)
    occlusion_correlations = correlation_table(
        player_report,
        exposures,
        grouping=['model_type', 'seed'],
        scores=[
            'total_contribution',
            'mean_contribution',
            'mean_abs_contribution',
        ],
    )
    event_correlations = correlation_table(
        event_report,
        exposures,
        grouping=['model_type', 'seed', 'method'],
        scores=['xg_weighted_contribution', 'xg_contribution_per_sequence'],
    )
    stability = cross_seed_player_stability(
        player_report,
        score='mean_contribution',
        min_sequences=config.min_player_sequences,
    )
    bootstrap = pd.concat(
        [
            bootstrap_player_occlusion(
                player_rows,
                model_type=model_type,
                min_sequences=config.min_player_sequences,
                n_bootstrap=config.bootstrap_samples,
            )
            for model_type in sorted(player_rows['model_type'].unique())
        ],
        ignore_index=True,
    )
    bootstrap.insert(
        bootstrap.columns.get_loc('player_id') + 1,
        'player_name',
        bootstrap['player_id'].map(name_map),
    )
    bootstrap = bootstrap.merge(exposures, on='player_id', how='left')
    bootstrap['rank'] = bootstrap.groupby('model_type')['mean_contribution'].rank(
        method='min',
        ascending=False,
    )

    tokens.to_parquet(root / 'test_token_attributions.parquet', index=False)
    sequence_faithfulness.to_parquet(
        root / 'test_sequence_faithfulness.parquet',
        index=False,
    )
    summarize_deletion(deletions).to_csv(
        root / 'deletion_summary_by_seed.csv',
        index=False,
    )
    (
        summarize_deletion(deletions)
        .groupby(['model_type', 'method', 'k'], as_index=False)
        .agg(
            delta_mse_mean=('delta_mse', 'mean'),
            delta_mse_sd=('delta_mse', 'std'),
            prediction_change_mean=('mean_absolute_prediction_change', 'mean'),
            prediction_change_sd=('mean_absolute_prediction_change', 'std'),
        )
        .to_csv(root / 'deletion_summary.csv', index=False)
    )
    faithfulness_by_seed = (
        sequence_faithfulness.groupby(['model_type', 'seed'], as_index=False)
        .agg(
            n_occlusion=('attention_vs_occlusion_rho', 'count'),
            attention_vs_occlusion_mean=('attention_vs_occlusion_rho', 'mean'),
            attention_vs_occlusion_median=('attention_vs_occlusion_rho', 'median'),
            n_ig=('attention_vs_ig_rho', 'count'),
            attention_vs_ig_mean=('attention_vs_ig_rho', 'mean'),
            attention_vs_ig_median=('attention_vs_ig_rho', 'median'),
        )
    )
    faithfulness_by_seed.to_csv(
        root / 'faithfulness_summary_by_seed.csv',
        index=False,
    )
    (
        faithfulness_by_seed.groupby('model_type', as_index=False)
        .agg(
            attention_vs_occlusion_mean=(
                'attention_vs_occlusion_mean',
                'mean',
            ),
            attention_vs_occlusion_sd=(
                'attention_vs_occlusion_mean',
                'std',
            ),
            attention_vs_ig_mean=('attention_vs_ig_mean', 'mean'),
            attention_vs_ig_sd=('attention_vs_ig_mean', 'std'),
        )
        .to_csv(root / 'faithfulness_summary.csv', index=False)
    )
    randomization_by_seed = (
        randomization.groupby(['model_type', 'seed'], as_index=False)
        .agg(
            n_sequences=('trained_vs_randomized_rho', 'count'),
            rho_mean=('trained_vs_randomized_rho', 'mean'),
            rho_median=('trained_vs_randomized_rho', 'median'),
        )
    )
    randomization_by_seed.to_csv(
        root / 'randomization_summary_by_seed.csv',
        index=False,
    )
    (
        randomization_by_seed.groupby('model_type', as_index=False)
        .agg(
            rho_mean=('rho_mean', 'mean'),
            rho_sd=('rho_mean', 'std'),
        )
        .to_csv(root / 'randomization_summary.csv', index=False)
    )
    player_rows.to_parquet(root / 'test_player_occlusion.parquet', index=False)
    player_report.to_csv(root / 'player_occlusion_report_by_seed.csv', index=False)
    event_report.to_csv(root / 'event_attribution_player_report.csv', index=False)
    method_comparison.to_csv(root / 'attribution_method_comparison.csv', index=False)
    (
        method_comparison.groupby(['model_type', 'comparison'], as_index=False)
        .agg(
            spearman_mean=('spearman_rho', 'mean'),
            spearman_sd=('spearman_rho', 'std'),
            jaccard_at_10_mean=('jaccard_at_10', 'mean'),
            jaccard_at_10_sd=('jaccard_at_10', 'std'),
        )
        .to_csv(root / 'attribution_method_comparison_summary.csv', index=False)
    )
    occlusion_correlations.to_csv(
        root / 'occlusion_exposure_correlations.csv',
        index=False,
    )
    event_correlations.to_csv(
        root / 'event_attribution_exposure_correlations.csv',
        index=False,
    )
    (
        occlusion_correlations.groupby(
            ['model_type', 'score', 'exposure'],
            as_index=False,
        )
        .agg(
            spearman_mean=('spearman_rho', 'mean'),
            spearman_sd=('spearman_rho', 'std'),
        )
        .to_csv(root / 'occlusion_exposure_summary.csv', index=False)
    )
    stability.to_csv(root / 'occlusion_ranking_stability.csv', index=False)
    bootstrap.sort_values(['model_type', 'rank']).to_csv(
        root / 'player_occlusion_report.csv',
        index=False,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    partitions, datasets, vocabularies = load_data(config, quick=args.quick)
    root = config.results_dir / 'faithful'
    root.mkdir(parents=True, exist_ok=True)
    shuffled_datasets = None

    if args.stage != 'aggregate':
        for model_type in config.faithful_models:
            model_datasets = datasets
            if model_type == 'gru_attention_shuffled':
                if shuffled_datasets is None:
                    shuffled_datasets = build_datasets(
                        partitions,
                        vocabularies=vocabularies,
                        config=config,
                        shuffle_events=True,
                    )
                model_datasets = shuffled_datasets
            for seed in config.seeds:
                run_dir = root / model_type / f'seed_{seed}'
                if args.stage in {'train', 'all'}:
                    model = train_run(
                        model_type,
                        seed,
                        run_dir=run_dir,
                        datasets=model_datasets,
                        vocabularies=vocabularies,
                        config=config,
                        force=args.force,
                    )
                else:
                    model = load_model(
                        run_dir,
                        vocabularies=vocabularies,
                        config=config,
                        device=select_device(),
                    )
                if args.stage in {'evaluate', 'all'}:
                    evaluate_run(
                        model,
                        model_type,
                        seed,
                        run_dir=run_dir,
                        dataset=model_datasets['test'],
                        config=config,
                        force=args.force,
                    )
    aggregate_results(config, partitions)


if __name__ == '__main__':
    main()
