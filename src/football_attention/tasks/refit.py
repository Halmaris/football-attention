from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from football_attention.bootstrap import seed_match_bootstrap_mean
from football_attention.config import ExperimentConfig
from football_attention.faithful_model import build_faithful_model
from football_attention.frozen_final import (
    build_frozen_protocol,
    config_from_spec,
    fit_classical_models,
    fit_neural_fixed_epochs,
    load_classical_models,
    load_or_create_protocol,
    predict_classical_models,
    save_classical_models,
    save_neural_refit,
)
from football_attention.sequence_baselines import build_sequence_baseline
from football_attention.train import make_loader, predict, prediction_metrics, select_device
from football_attention.variants import ShotSequenceDataset, Vocabularies, fit_vocabularies
from football_attention.tasks.sequence import aggregate_results, evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        '--stage',
        choices=['train', 'evaluate', 'all'],
        default='all',
    )
    parser.add_argument('--results-name', default='results_frozen_final')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepared-name', default='prepared')
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
    )
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    return parser.parse_args()


def load_rows(
    project_dir: Path,
    splits: list[str],
    *,
    prepared_name: str = 'prepared',
    quick: bool,
) -> pd.DataFrame:
    path = project_dir / 'data' / prepared_name / 'sequences_raw.parquet'
    rows = pd.read_parquet(path, filters=[('split', 'in', splits)])
    rows = rows[rows['split'].isin(splits)].copy()
    if quick:
        rows = (
            rows.groupby('split', group_keys=False)
            .head(160)
            .reset_index(drop=True)
        )
    if rows.empty:
        raise RuntimeError(f'No rows for splits: {splits}')
    return rows


def run_dir(results_dir: Path, model_label: str, seed: int) -> Path:
    group = 'faithful' if model_label == 'gru_attention_faithful' else 'neural'
    return results_dir / group / model_label / f'seed_{seed}'


def train_all(
    project_dir: Path,
    results_dir: Path,
    results_name: str,
    protocol: dict[str, object],
    *,
    prepared_name: str,
    force: bool,
    quick: bool,
) -> None:
    # The historical test is deliberately not loaded in this function.
    development = load_rows(
        project_dir,
        ['train', 'validation'],
        prepared_name=prepared_name,
        quick=quick,
    )
    vocabularies = fit_vocabularies(development)
    classical_path = results_dir / 'classical' / 'models.pkl'
    if force or not classical_path.exists():
        print('Training frozen classical baselines on train+validation')
        fitted = fit_classical_models(
            development,
            protocol['classical_models'],
            seed=int(protocol['seeds'][0]),
        )
        save_classical_models(
            classical_path,
            fitted=fitted,
        )
        (classical_path.parent / 'feature_names.json').write_text(
            json.dumps(
                fitted['vectorizer'].get_feature_names_out().tolist(),
                indent=2,
            ),
            encoding='utf-8',
        )
        (classical_path.parent / 'last_event_feature_names.json').write_text(
            json.dumps(
                fitted['last_event_vectorizer']
                .get_feature_names_out()
                .tolist(),
                indent=2,
            ),
            encoding='utf-8',
        )
    else:
        print('Using cached frozen classical baselines')

    dataset = ShotSequenceDataset(
        development,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=20,
    )
    for model_label, spec in protocol['neural_models'].items():
        config = config_from_spec(
            project_dir,
            results_name,
            spec,
            prepared_name=prepared_name,
        )
        for seed in protocol['seeds']:
            output = run_dir(results_dir, model_label, int(seed))
            required = [output / 'model.pt', output / 'history.csv']
            if not force and all(path.exists() for path in required):
                print(f'Using cached model={model_label}, seed={seed}')
                continue
            print(
                f'Training model={model_label}, seed={seed}, '
                f'epochs={config.max_epochs}, device={select_device()}'
            )
            model, history = fit_neural_fixed_epochs(
                dataset,
                vocabularies=vocabularies,
                config=config,
                family=spec['family'],
                architecture=spec['architecture'],
                seed=int(seed),
            )
            save_neural_refit(
                output,
                model=model,
                history=history,
                vocabularies=vocabularies,
                config=config,
                model_label=model_label,
                family=spec['family'],
                architecture=spec['architecture'],
                seed=int(seed),
            )
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print('[TRAIN COMPLETE] All frozen models are fitted on train+validation.')


def verify_training_complete(
    results_dir: Path,
    protocol: dict[str, object],
) -> None:
    missing = []
    if not (results_dir / 'classical' / 'models.pkl').exists():
        missing.append('classical/models.pkl')
    for model_label in protocol['neural_models']:
        for seed in protocol['seeds']:
            path = run_dir(results_dir, model_label, int(seed)) / 'model.pt'
            if not path.exists():
                missing.append(str(path.relative_to(results_dir)))
    if missing:
        preview = ', '.join(missing[:5])
        raise RuntimeError(
            f'Frozen training is incomplete ({len(missing)} missing): {preview}'
        )


def load_neural_model(
    checkpoint_path: Path,
    *,
    config: ExperimentConfig,
    vocabularies: Vocabularies,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=select_device(),
        weights_only=False,
    )
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
    return model.to(select_device())


def classical_prediction_frames(
    test_rows: pd.DataFrame,
    results_dir: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    fitted = load_classical_models(results_dir / 'classical' / 'models.pkl')
    targets = test_rows['xg'].to_numpy(float)
    frames = []
    metric_rows = []
    for label, predictions in predict_classical_models(
        test_rows,
        fitted,
    ).items():
        metrics = prediction_metrics(targets, predictions)
        frames.append(
            pd.DataFrame(
                {
                    'model_type': label,
                    'seed': -1,
                    'shot_seq_id': test_rows['shot_seq_id'].to_numpy(),
                    'match_id': test_rows['match_id'].to_numpy(),
                    'target_xg': targets,
                    'prediction': predictions,
                }
            )
        )
        metric_rows.append(
            {
                'model_type': label,
                'seed': -1,
                'split': 'test',
                **asdict(metrics),
                'fixed_epochs': np.nan,
            }
        )
    return frames, metric_rows


def prediction_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[
        predictions['model_type'].eq('gradient_boosting')
    ][['shot_seq_id', 'prediction']].rename(
        columns={'prediction': 'baseline_prediction'}
    )
    rows = []
    for model_type, values in predictions.groupby('model_type'):
        if model_type == 'gradient_boosting':
            continue
        paired = values.merge(baseline, on='shot_seq_id', validate='many_to_one')
        paired['seed'] = paired['seed'].replace(-1, 42)
        paired['difference'] = (
            (paired['target_xg'] - paired['prediction']) ** 2
            - (paired['target_xg'] - paired['baseline_prediction']) ** 2
        )
        result = seed_match_bootstrap_mean(
            paired,
            n_bootstrap=10_000,
            seed=2026,
        )
        rows.append(
            {
                'model_type': model_type,
                'reference_model': 'gradient_boosting',
                **result,
            }
        )
    return pd.DataFrame(rows)


def evaluate_test_once(
    project_dir: Path,
    results_dir: Path,
    results_name: str,
    protocol: dict[str, object],
    *,
    prepared_name: str,
    quick: bool,
) -> None:
    marker = results_dir / 'test_evaluation_complete.json'
    if marker.exists():
        print('Using cached one-time historical-test evaluation.')
        return
    verify_training_complete(results_dir, protocol)

    # This is the only point at which the historical test is loaded.
    test_rows = load_rows(
        project_dir,
        ['test'],
        prepared_name=prepared_name,
        quick=quick,
    )
    frames, metric_rows = classical_prediction_frames(test_rows, results_dir)

    for model_label, spec in protocol['neural_models'].items():
        config = config_from_spec(
            project_dir,
            results_name,
            spec,
            prepared_name=prepared_name,
        )
        for seed in protocol['seeds']:
            output = run_dir(results_dir, model_label, int(seed))
            checkpoint = torch.load(
                output / 'model.pt',
                map_location='cpu',
                weights_only=False,
            )
            vocabularies = Vocabularies(**checkpoint['vocabularies'])
            dataset = ShotSequenceDataset(
                test_rows,
                vocabularies=vocabularies,
                variant='pre_shot',
                sequence_length=config.sequence_length,
            )
            model = load_neural_model(
                output / 'model.pt',
                config=config,
                vocabularies=vocabularies,
            )
            loader = make_loader(dataset, config, shuffle=False)
            sequence_ids, targets, predictions = predict(
                model,
                loader,
                select_device(),
            )
            match_map = test_rows.set_index('shot_seq_id')['match_id']
            prediction_frame = pd.DataFrame(
                {
                    'model_type': model_label,
                    'seed': int(seed),
                    'shot_seq_id': sequence_ids,
                    'match_id': pd.Series(sequence_ids)
                    .map(match_map)
                    .to_numpy(),
                    'target_xg': targets,
                    'prediction': predictions,
                }
            )
            prediction_frame.to_parquet(
                output / 'test_predictions.parquet',
                index=False,
            )
            frames.append(prediction_frame)
            metrics = prediction_metrics(targets, predictions)
            metric_row = {
                'model_type': model_label,
                'seed': int(seed),
                'split': 'test',
                **asdict(metrics),
                'fixed_epochs': int(spec['fixed_epochs']),
            }
            metric_rows.append(metric_row)
            pd.DataFrame([metric_row]).to_csv(
                output / 'metrics.csv',
                index=False,
            )
            if model_label in {
                'gru_attention_faithful',
                'faithful_no_player_id',
            }:
                evaluate_run(
                    model,
                    model_label,
                    int(seed),
                    run_dir=output,
                    dataset=dataset,
                    config=config,
                    force=False,
                )
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    predictions = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_parquet(results_dir / 'test_predictions.parquet', index=False)
    metrics.to_csv(results_dir / 'test_metrics_by_seed.csv', index=False)
    (
        metrics.groupby('model_type', as_index=False)
        .agg(
            n_seeds=('seed', 'nunique'),
            mse_mean=('mse', 'mean'),
            mse_sd=('mse', 'std'),
            mae_mean=('mae', 'mean'),
            mae_sd=('mae', 'std'),
            r2_mean=('r2', 'mean'),
            r2_sd=('r2', 'std'),
        )
        .to_csv(results_dir / 'test_metrics_summary.csv', index=False)
    )
    prediction_bootstrap(predictions).to_csv(
        results_dir / 'test_mse_vs_gradient_boosting_bootstrap.csv',
        index=False,
    )

    all_rows = load_rows(
        project_dir,
        ['train', 'validation', 'test'],
        prepared_name=prepared_name,
        quick=quick,
    )
    partitions = {
        split: all_rows[all_rows['split'].eq(split)].copy()
        for split in ['train', 'validation', 'test']
    }
    aggregate_config = ExperimentConfig(
        project_dir=project_dir,
        results_name=results_name,
        prepared_name=prepared_name,
        seeds=tuple(int(value) for value in protocol['seeds']),
        min_player_sequences=20,
        bootstrap_samples=100 if quick else 1000,
    )
    aggregate_results(aggregate_config, partitions)

    marker.write_text(
        json.dumps(
            {
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'n_test_sequences': int(len(test_rows)),
                'models': sorted(predictions['model_type'].unique()),
                'seeds': protocol['seeds'],
                'protocol_file': str(
                    (results_dir / 'frozen_protocol.json').resolve()
                ),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print('[TEST COMPLETE] Historical test evaluated with frozen models.')


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    results_dir = project_dir / args.results_name
    results_dir.mkdir(parents=True, exist_ok=True)
    candidate = build_frozen_protocol(
        project_dir,
        config_path=args.config,
        seeds=tuple(args.seeds) if args.seeds else None,
        quick=args.quick,
    )
    candidate['prepared_name'] = args.prepared_name
    protocol = load_or_create_protocol(
        results_dir / 'frozen_protocol.json',
        candidate,
    )
    if args.stage in {'train', 'all'}:
        train_all(
            project_dir,
            results_dir,
            args.results_name,
            protocol,
            prepared_name=args.prepared_name,
            force=args.force_train,
            quick=args.quick,
        )
    if args.stage in {'evaluate', 'all'}:
        evaluate_test_once(
            project_dir,
            results_dir,
            args.results_name,
            protocol,
            prepared_name=args.prepared_name,
            quick=args.quick,
        )


if __name__ == '__main__':
    main()
