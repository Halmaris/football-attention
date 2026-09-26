from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
import torch
from football_attention.bootstrap import seed_match_bootstrap_mean
from football_attention.faithful_eval import summarize_deletion
from football_attention.frozen_final import (
    config_from_spec,
    fit_neural_fixed_epochs,
    save_neural_refit,
)
from football_attention.train import make_loader, predict, prediction_metrics, select_device
from football_attention.variants import ShotSequenceDataset, Vocabularies
from football_attention.tasks.contrasts import paired_deletion_rows, paired_models
from football_attention.tasks.holdout import (
    DEFAULT_EXTERNAL_PREPARED_NAME,
    build_checkpoint_model,
    checkpoint_config,
    load_external,
)
from football_attention.tasks.sequence import evaluate_run
from football_attention.tasks.refit import load_rows


MODEL_ALIGNED = 'gru_attention_faithful'


MODEL_PREDICTIVE = 'gru_attention_predictive'


MODEL_MATCHED = 'gru_attention_matched_control'


MODELS = (MODEL_ALIGNED, MODEL_PREDICTIVE, MODEL_MATCHED)


COMPARISONS = (
    (MODEL_ALIGNED, MODEL_MATCHED),
    (MODEL_ALIGNED, MODEL_PREDICTIVE),
)


EVALUATION_FILES = (
    'token_attributions.parquet',
    'sequence_faithfulness.parquet',
    'deletion_predictions.parquet',
    'player_occlusion.parquet',
    'randomization_per_sequence.parquet',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument('--frozen-results-name', default='results/final')
    parser.add_argument('--source-controls-name', help='Read existing matched checkpoints from this directory.')
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
        '--results-name', default='results/alignment'
    )
    parser.add_argument(
        '--stage',
        choices=['train', 'development', 'external', 'aggregate', 'aggregate-external', 'all'],
        default='all',
    )
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--n-bootstrap', type=int, default=10_000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    parser.add_argument('--force-evaluation', action='store_true')
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise RuntimeError(f'Missing required file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def build_matched_control_spec(
    frozen_protocol: dict[str, object],
    *,
    quick: bool,
) -> dict[str, object]:
    aligned = frozen_protocol['neural_models'][MODEL_ALIGNED]
    control = copy.deepcopy(aligned)
    control['fixed_epochs'] = min(2, int(aligned['fixed_epochs'])) if quick else int(
        aligned['fixed_epochs']
    )
    control['design'] = 'matched_alignment_control'
    control['source_model'] = MODEL_ALIGNED
    control['controlled_change'] = {'alignment_lambda': 0.0}
    control['config']['alignment_lambda'] = 0.0
    return control


def verify_matched_spec(
    aligned: dict[str, object],
    control: dict[str, object],
) -> None:
    if int(aligned['fixed_epochs']) != int(control['fixed_epochs']):
        raise RuntimeError('Matched control must use the aligned epoch count.')
    aligned_config = dict(aligned['config'])
    control_config = dict(control['config'])
    changed = {
        key
        for key in set(aligned_config) | set(control_config)
        if aligned_config.get(key) != control_config.get(key)
    }
    if changed != {'alignment_lambda'}:
        raise RuntimeError(
            'Matched control differs from aligned configuration in: '
            f'{sorted(changed)}'
        )
    if float(control_config['alignment_lambda']) != 0.0:
        raise RuntimeError('Matched control must set alignment_lambda to zero.')


def build_control_protocol(
    frozen_dir: Path,
    frozen_protocol: dict[str, object],
    *,
    seeds: list[int],
    quick: bool,
    development_prepared_name: str = 'prepared',
) -> dict[str, object]:
    control = build_matched_control_spec(frozen_protocol, quick=quick)
    aligned = frozen_protocol['neural_models'][MODEL_ALIGNED]
    if not quick:
        verify_matched_spec(aligned, control)
    return {
        'protocol_version': 1,
        'design': 'posthoc_matched_alignment_control',
        'source_frozen_protocol': str(
            (frozen_dir / 'frozen_protocol.json').resolve()
        ),
        'fit_splits': list(frozen_protocol['fit_splits']),
        'development_prepared_name': development_prepared_name,
        'development_evaluation_split': 'test',
        'external_training_performed': False,
        'external_tuning_performed': False,
        'seeds': seeds,
        'quick': quick,
        'models': {
            MODEL_ALIGNED: copy.deepcopy(aligned),
            MODEL_PREDICTIVE: copy.deepcopy(
                frozen_protocol['neural_models'][MODEL_PREDICTIVE]
            ),
            MODEL_MATCHED: control,
        },
        'primary_comparison': f'{MODEL_ALIGNED} - {MODEL_MATCHED}',
        'external_baseline_comparison': (
            f'{MODEL_ALIGNED} - {MODEL_PREDICTIVE}'
        ),
    }


def load_or_create_control_protocol(
    path: Path,
    candidate: dict[str, object],
) -> dict[str, object]:
    if path.exists():
        existing = read_json(path)
        if existing != candidate:
            raise RuntimeError(
                'Alignment-control protocol differs from the existing file. '
                'Use a new results directory.'
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    return candidate


def checkpoint_path(
    frozen_dir: Path,
    results_dir: Path,
    model_type: str,
    seed: int,
) -> Path:
    if model_type == MODEL_ALIGNED:
        return frozen_dir / 'faithful' / model_type / f'seed_{seed}' / 'model.pt'
    if model_type == MODEL_PREDICTIVE:
        return frozen_dir / 'neural' / model_type / f'seed_{seed}' / 'model.pt'
    if model_type == MODEL_MATCHED:
        return results_dir / 'checkpoints' / model_type / f'seed_{seed}' / 'model.pt'
    raise ValueError(f'Unknown model: {model_type}')


def control_checkpoint_dir(results_dir: Path, seed: int) -> Path:
    return results_dir / 'checkpoints' / MODEL_MATCHED / f'seed_{seed}'


def evaluation_dir(
    frozen_dir: Path,
    external_dir: Path,
    results_dir: Path,
    dataset_name: str,
    model_type: str,
    seed: int,
) -> Path:
    if dataset_name == 'development_test':
        if model_type == MODEL_ALIGNED:
            return frozen_dir / 'faithful' / model_type / f'seed_{seed}'
        if model_type == MODEL_PREDICTIVE:
            return (
                frozen_dir
                / 'attention_ablation'
                / model_type
                / f'seed_{seed}'
            )
    elif dataset_name == 'frozen_holdout' and model_type == MODEL_ALIGNED:
        return external_dir / 'faithfulness' / model_type / f'seed_{seed}'
    return (
        results_dir
        / dataset_name
        / 'faithfulness'
        / model_type
        / f'seed_{seed}'
    )


def prediction_path(
    results_dir: Path,
    dataset_name: str,
    model_type: str,
    seed: int,
) -> Path:
    return (
        results_dir
        / dataset_name
        / 'predictions'
        / model_type
        / f'seed_{seed}.parquet'
    )


def train_matched_control(
    project_dir: Path,
    frozen_dir: Path,
    results_dir: Path,
    protocol: dict[str, object],
    *,
    development_prepared_name: str,
    force: bool,
    quick: bool,
) -> None:
    rows = load_rows(
        project_dir,
        list(protocol['fit_splits']),
        prepared_name=development_prepared_name,
        quick=quick,
    )
    seeds = [int(seed) for seed in protocol['seeds']]
    reference = torch.load(
        checkpoint_path(
            frozen_dir,
            results_dir,
            MODEL_ALIGNED,
            seeds[0],
        ),
        map_location='cpu',
        weights_only=False,
    )
    vocabularies = Vocabularies(**reference['vocabularies'])
    spec = protocol['models'][MODEL_MATCHED]
    config = config_from_spec(
        project_dir,
        results_dir.name,
        spec,
        prepared_name=development_prepared_name,
    )
    dataset = ShotSequenceDataset(
        rows,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=config.sequence_length,
    )
    for seed in seeds:
        output = control_checkpoint_dir(results_dir, seed)
        required = [output / 'model.pt', output / 'history.csv']
        if not force and all(path.exists() for path in required):
            print(f'Using cached matched control seed={seed}')
            continue
        print(
            f'Training matched control seed={seed}, '
            f'epochs={config.max_epochs}, lambda=0, device={select_device()}'
        )
        model, history = fit_neural_fixed_epochs(
            dataset,
            vocabularies=vocabularies,
            config=config,
            family=str(spec['family']),
            architecture=str(spec['architecture']),
            seed=seed,
        )
        save_neural_refit(
            output,
            model=model,
            history=history,
            vocabularies=vocabularies,
            config=config,
            model_label=MODEL_MATCHED,
            family=str(spec['family']),
            architecture=str(spec['architecture']),
            seed=seed,
        )
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    missing = [
        control_checkpoint_dir(results_dir, seed) / 'model.pt'
        for seed in seeds
        if not (control_checkpoint_dir(results_dir, seed) / 'model.pt').exists()
    ]
    if missing:
        raise RuntimeError(f'Matched-control training incomplete: {missing[:3]}')
    (results_dir / 'training_complete.json').write_text(
        json.dumps(
            {
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'model': MODEL_MATCHED,
                'seeds': seeds,
                'fixed_epochs': int(spec['fixed_epochs']),
                'alignment_lambda': 0.0,
                'fit_splits': protocol['fit_splits'],
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print('[TRAIN COMPLETE] Matched alignment control fitted.')


def evaluate_dataset(
    project_dir: Path,
    frozen_dir: Path,
    external_dir: Path,
    results_dir: Path,
    protocol: dict[str, object],
    *,
    dataset_name: str,
    rows: pd.DataFrame,
    force: bool,
    source_controls_dir: Path | None = None,
) -> None:
    device = select_device()
    for model_type in MODELS:
        for seed_value in protocol['seeds']:
            seed = int(seed_value)
            source = checkpoint_path(
                frozen_dir,
                source_controls_dir if source_controls_dir is not None else results_dir,
                model_type,
                seed,
            )
            if not source.exists():
                raise RuntimeError(f'Missing checkpoint: {source}')
            checkpoint = torch.load(
                source,
                map_location=device,
                weights_only=False,
            )
            config, vocabularies = checkpoint_config(checkpoint, project_dir)
            dataset = ShotSequenceDataset(
                rows,
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
            predictions_file = prediction_path(
                results_dir,
                dataset_name,
                model_type,
                seed,
            )
            if force or not predictions_file.exists():
                loader = make_loader(dataset, config, shuffle=False)
                sequence_ids, targets, values = predict(model, loader, device)
                match_map = rows.set_index('shot_seq_id')['match_id']
                predictions_file.parent.mkdir(parents=True, exist_ok=True)
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
                ).to_parquet(predictions_file, index=False)
            output = evaluation_dir(
                frozen_dir,
                external_dir,
                results_dir,
                dataset_name,
                model_type,
                seed,
            )
            output.mkdir(parents=True, exist_ok=True)
            output_is_new = results_dir in output.parents
            evaluate_run(
                model,
                model_type,
                seed,
                run_dir=output,
                dataset=dataset,
                config=config,
                force=force and output_is_new,
            )
            missing = [
                output / filename
                for filename in EVALUATION_FILES
                if not (output / filename).exists()
            ]
            if missing:
                raise RuntimeError(
                    f'Incomplete evaluation model={model_type}, seed={seed}: '
                    f'{missing}'
                )
            del model
            if device.type == 'mps':
                torch.mps.empty_cache()
    print(f'[EVALUATION COMPLETE] {dataset_name}.')


def attach_match_id(values: pd.DataFrame, match_map: pd.DataFrame) -> pd.DataFrame:
    if 'match_id' in values.columns:
        return values
    return values.merge(
        match_map,
        on='shot_seq_id',
        validate='many_to_one',
    )


def read_evaluation_files(
    frozen_dir: Path,
    external_dir: Path,
    results_dir: Path,
    protocol: dict[str, object],
    dataset_name: str,
    filename: str,
) -> pd.DataFrame:
    paths = [
        evaluation_dir(
            frozen_dir,
            external_dir,
            results_dir,
            dataset_name,
            model_type,
            int(seed),
        )
        / filename
        for model_type in MODELS
        for seed in protocol['seeds']
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f'{dataset_name} evaluation incomplete ({len(missing)} missing).'
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def read_predictions(
    results_dir: Path,
    protocol: dict[str, object],
    dataset_name: str,
) -> pd.DataFrame:
    paths = [
        prediction_path(results_dir, dataset_name, model_type, int(seed))
        for model_type in MODELS
        for seed in protocol['seeds']
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f'{dataset_name} predictions incomplete ({len(missing)} missing).'
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def individual_bootstrap_rows(
    sequences: pd.DataFrame,
    randomization: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    rows = []
    metrics = [
        ('attention_vs_occlusion_rho', sequences),
        ('attention_vs_ig_rho', sequences),
        ('trained_vs_randomized_rho', randomization),
    ]
    for metric_index, (metric, values) in enumerate(metrics):
        for model_index, model_type in enumerate(MODELS):
            selected = values[values['model_type'].eq(model_type)]
            result = seed_match_bootstrap_mean(
                selected,
                value_column=metric,
                n_bootstrap=n_bootstrap,
                seed=30_260 + metric_index * 10 + model_index,
            )
            rows.append(
                {
                    'metric': metric,
                    'comparison': model_type,
                    **result,
                }
            )
    return rows


def aggregate_dataset(
    frozen_dir: Path,
    external_dir: Path,
    results_dir: Path,
    protocol: dict[str, object],
    *,
    dataset_name: str,
    rows: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    output = results_dir / dataset_name
    output.mkdir(parents=True, exist_ok=True)
    match_map = rows[['shot_seq_id', 'match_id']]
    predictions = read_predictions(results_dir, protocol, dataset_name)
    sequences = attach_match_id(
        read_evaluation_files(
            frozen_dir,
            external_dir,
            results_dir,
            protocol,
            dataset_name,
            'sequence_faithfulness.parquet',
        ),
        match_map,
    )
    deletions = attach_match_id(
        read_evaluation_files(
            frozen_dir,
            external_dir,
            results_dir,
            protocol,
            dataset_name,
            'deletion_predictions.parquet',
        ),
        match_map,
    )
    randomization = attach_match_id(
        read_evaluation_files(
            frozen_dir,
            external_dir,
            results_dir,
            protocol,
            dataset_name,
            'randomization_per_sequence.parquet',
        ),
        match_map,
    )

    predictions = predictions.merge(
        rows[['shot_seq_id', 'n_events']],
        on='shot_seq_id',
        validate='many_to_one',
    )
    metric_rows = []
    for (model_type, seed), group in predictions.groupby(['model_type', 'seed']):
        scopes = {
            'all': group,
            'nonempty': group[group['n_events'].gt(1)],
        }
        for scope, selected in scopes.items():
            metrics = prediction_metrics(
                selected['target_xg'].to_numpy(float),
                selected['prediction'].to_numpy(float),
            )
            metric_rows.append(
                {
                    'model_type': model_type,
                    'seed': int(seed),
                    'scope': scope,
                    **asdict(metrics),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / 'prediction_metrics_by_seed.csv', index=False)
    (
        metrics.groupby(['model_type', 'scope'], as_index=False)
        .agg(
            mse_mean=('mse', 'mean'),
            mse_sd=('mse', 'std'),
            mae_mean=('mae', 'mean'),
            mae_sd=('mae', 'std'),
            r2_mean=('r2', 'mean'),
            r2_sd=('r2', 'std'),
        )
        .to_csv(output / 'prediction_metrics_summary.csv', index=False)
    )

    faithfulness = (
        sequences.groupby(['model_type', 'seed'], as_index=False)
        .agg(
            n_occlusion=('attention_vs_occlusion_rho', 'count'),
            attention_vs_occlusion=('attention_vs_occlusion_rho', 'mean'),
            n_ig=('attention_vs_ig_rho', 'count'),
            attention_vs_ig=('attention_vs_ig_rho', 'mean'),
        )
    )
    faithfulness.to_csv(output / 'faithfulness_by_seed.csv', index=False)
    (
        faithfulness.groupby('model_type', as_index=False)
        .agg(
            attention_vs_occlusion_mean=(
                'attention_vs_occlusion',
                'mean',
            ),
            attention_vs_occlusion_sd=('attention_vs_occlusion', 'std'),
            attention_vs_ig_mean=('attention_vs_ig', 'mean'),
            attention_vs_ig_sd=('attention_vs_ig', 'std'),
        )
        .to_csv(output / 'faithfulness_summary.csv', index=False)
    )
    deletion_by_seed = summarize_deletion(deletions)
    deletion_by_seed.to_csv(output / 'deletion_by_seed.csv', index=False)
    (
        deletion_by_seed.groupby(['model_type', 'method', 'k'], as_index=False)
        .agg(
            delta_mse_mean=('delta_mse', 'mean'),
            delta_mse_sd=('delta_mse', 'std'),
        )
        .to_csv(output / 'deletion_summary.csv', index=False)
    )
    randomization_by_seed = (
        randomization.groupby(['model_type', 'seed'], as_index=False)
        .agg(
            n_sequences=('trained_vs_randomized_rho', 'count'),
            rho_mean=('trained_vs_randomized_rho', 'mean'),
        )
    )
    randomization_by_seed.to_csv(
        output / 'randomization_by_seed.csv',
        index=False,
    )

    bootstrap_rows = individual_bootstrap_rows(
        sequences,
        randomization,
        n_bootstrap=n_bootstrap,
    )
    predictions_for_pairing = predictions.copy()
    predictions_for_pairing['squared_error'] = (
        predictions_for_pairing['target_xg']
        - predictions_for_pairing['prediction']
    ) ** 2
    for model_left, model_right in COMPARISONS:
        bootstrap_rows.append(
            paired_models(
                predictions_for_pairing,
                value_column='squared_error',
                keys=[],
                label='mse',
                n_bootstrap=n_bootstrap,
                model_left=model_left,
                model_right=model_right,
            )
        )
        for column in ['attention_vs_occlusion_rho', 'attention_vs_ig_rho']:
            bootstrap_rows.append(
                paired_models(
                    sequences,
                    value_column=column,
                    keys=[],
                    label=column,
                    n_bootstrap=n_bootstrap,
                    model_left=model_left,
                    model_right=model_right,
                )
            )
        comparison = f'{model_left} - {model_right}'
        bootstrap_rows.extend(
            row
            for row in paired_deletion_rows(
                deletions,
                n_bootstrap=n_bootstrap,
                model_left=model_left,
                model_right=model_right,
            )
            if row['comparison'] == comparison
        )
        bootstrap_rows.append(
            paired_models(
                randomization,
                value_column='trained_vs_randomized_rho',
                keys=[],
                label='trained_vs_randomized_rho',
                n_bootstrap=n_bootstrap,
                model_left=model_left,
                model_right=model_right,
            )
        )
    pd.DataFrame(bootstrap_rows).to_csv(
        output / 'paired_seed_match_bootstrap.csv',
        index=False,
    )
    (output / 'aggregation_complete.json').write_text(
        json.dumps(
            {
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': dataset_name,
                'models': list(MODELS),
                'comparisons': [f'{left} - {right}' for left, right in COMPARISONS],
                'seeds': protocol['seeds'],
                'n_matches': int(rows['match_id'].nunique()),
                'n_sequences': int(len(rows)),
                'n_bootstrap': int(n_bootstrap),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f'[AGGREGATION COMPLETE] {dataset_name}.')


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    frozen_dir = project_dir / args.frozen_results_name
    external_dir = project_dir / args.external_results_name
    results_dir = project_dir / args.results_name
    source_controls = project_dir / args.source_controls_name if args.source_controls_name else None
    if source_controls is not None and args.stage not in {'external', 'aggregate-external'}:
        raise ValueError('--source-controls-name is restricted to inference-only external stages')
    frozen_protocol = read_json(frozen_dir / 'frozen_protocol.json')
    seeds = (
        [int(seed) for seed in args.seeds]
        if args.seeds
        else [int(seed) for seed in frozen_protocol['seeds']]
    )
    candidate = build_control_protocol(
        frozen_dir,
        frozen_protocol,
        seeds=seeds,
        quick=args.quick,
        development_prepared_name=args.development_prepared_name,
    )
    protocol = load_or_create_control_protocol(
        results_dir / 'alignment_control_protocol.json',
        candidate,
    )
    n_bootstrap = 100 if args.quick else args.n_bootstrap

    if args.stage in {'train', 'all'}:
        train_matched_control(
            project_dir,
            frozen_dir,
            results_dir,
            protocol,
            development_prepared_name=args.development_prepared_name,
            force=args.force_train,
            quick=args.quick,
        )
    if args.stage in {'development', 'all'}:
        development = load_rows(
            project_dir,
            ['test'],
            prepared_name=args.development_prepared_name,
            quick=args.quick,
        )
        evaluate_dataset(
            project_dir,
            frozen_dir,
            external_dir,
            results_dir,
            protocol,
            dataset_name='development_test',
            rows=development,
            force=args.force_evaluation,
        )
    if args.stage in {'external', 'all'}:
        external = load_external(project_dir, args.external_prepared_name)
        if args.quick:
            external = external.head(160).reset_index(drop=True)
        evaluate_dataset(
            project_dir,
            frozen_dir,
            external_dir,
            results_dir,
            protocol,
            dataset_name='frozen_holdout',
            rows=external,
            force=args.force_evaluation,
            source_controls_dir=source_controls,
        )
    if args.stage == 'aggregate-external':
        external = load_external(project_dir, args.external_prepared_name)
        aggregate_dataset(
            frozen_dir, external_dir, results_dir, protocol,
            dataset_name='frozen_holdout', rows=external, n_bootstrap=n_bootstrap,
        )
        (results_dir / 'external_evaluation_complete.json').write_text(json.dumps({
            'training_performed': False, 'tuning_performed': False,
            'n_sequences': len(external), 'n_matches': int(external['match_id'].nunique()),
            'models': list(MODELS), 'seeds': seeds,
        }, indent=2), encoding='utf-8')
    if args.stage in {'aggregate', 'all'}:
        development = load_rows(
            project_dir,
            ['test'],
            prepared_name=args.development_prepared_name,
            quick=args.quick,
        )
        external = load_external(project_dir, args.external_prepared_name)
        if args.quick:
            external = external.head(160).reset_index(drop=True)
        aggregate_dataset(
            frozen_dir,
            external_dir,
            results_dir,
            protocol,
            dataset_name='development_test',
            rows=development,
            n_bootstrap=n_bootstrap,
        )
        aggregate_dataset(
            frozen_dir,
            external_dir,
            results_dir,
            protocol,
            dataset_name='frozen_holdout',
            rows=external,
            n_bootstrap=n_bootstrap,
        )
        (results_dir / 'pipeline_complete.json').write_text(
            json.dumps(
                {
                    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'matched_control_trained': True,
                    'external_models_trained': False,
                    'external_prepared_name': args.external_prepared_name,
                    'development_prepared_name': (
                        args.development_prepared_name
                    ),
                    'models': list(MODELS),
                    'seeds': protocol['seeds'],
                },
                indent=2,
            ),
            encoding='utf-8',
        )
        print('[COMPLETE] Alignment controls evaluated on both test sets.')


if __name__ == '__main__':
    main()
