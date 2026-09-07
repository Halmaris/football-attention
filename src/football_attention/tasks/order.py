from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from football_attention.config import ExperimentConfig
from football_attention.data import split_rows
from football_attention.train import make_loader, predict, prediction_metrics, select_device
from football_attention.variants import fit_vocabularies
from football_attention.tasks.sequence import build_datasets, evaluate_run, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument('--results-name', default='results_order_followup')
    parser.add_argument('--prepared-name', default='prepared')
    parser.add_argument(
        '--ordered-results-name',
        default='results_gru_attention',
    )
    parser.add_argument(
        '--shuffled-results-name',
        default='results_order_ablation',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument('--max-test-sequences', type=int)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def evaluate_cross_condition(
    specification: dict[str, str],
    seed: int,
    *,
    project_dir: Path,
    output_root: Path,
    datasets: dict[str, object],
    vocabularies: object,
    config: ExperimentConfig,
    force: bool,
) -> None:
    label = specification['label']
    run_dir = output_root / label / f'seed_{seed}'
    required = [
        run_dir / 'test_predictions.parquet',
        run_dir / 'token_attributions.parquet',
        run_dir / 'sequence_faithfulness.parquet',
        run_dir / 'deletion_predictions.parquet',
        run_dir / 'player_occlusion.parquet',
        run_dir / 'randomization_summary.csv',
    ]
    if not force and all(path.exists() for path in required):
        print(f'Using cached cross-evaluation={label}, seed={seed}')
        return

    source_run = (
        project_dir
        / specification['source_results']
        / 'faithful'
        / specification['source_model']
        / f'seed_{seed}'
    )
    if not (source_run / 'model.pt').exists():
        raise FileNotFoundError(f'Missing source checkpoint: {source_run}')

    run_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f'Evaluating cross-condition={label}, seed={seed}, device={device}')
    model = load_model(
        source_run,
        vocabularies=vocabularies,
        config=config,
        device=device,
    )
    dataset = datasets[specification['test_order']]
    loader = make_loader(dataset, config, shuffle=False)
    sequence_ids, targets, predictions = predict(model, loader, device)
    metrics = prediction_metrics(targets, predictions)
    pd.DataFrame(
        [
            {
                'model_type': label,
                'seed': seed,
                'split': 'test',
                **asdict(metrics),
            }
        ]
    ).to_csv(run_dir / 'metrics.csv', index=False)
    pd.DataFrame(
        {
            'model_type': label,
            'seed': seed,
            'shot_seq_id': sequence_ids,
            'target_xg': targets,
            'prediction': predictions,
        }
    ).to_parquet(run_dir / 'test_predictions.parquet', index=False)
    metadata = {
        **specification,
        'seed': seed,
        'n_test_sequences': len(dataset),
        'retrained': False,
    }
    (run_dir / 'run_metadata.json').write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )
    evaluate_run(
        model,
        label,
        seed,
        run_dir=run_dir,
        dataset=dataset,
        config=config,
        force=force,
    )


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    config = ExperimentConfig(
        project_dir=project_dir,
        results_name=args.results_name,
        prepared_name=args.prepared_name,
        seeds=tuple(args.seeds),
    )
    sequences = pd.read_parquet(
        config.prepared_dir / 'sequences_raw.parquet'
    )
    partitions = split_rows(sequences)
    if args.max_test_sequences is not None:
        partitions['test'] = partitions['test'].head(
            args.max_test_sequences
        ).reset_index(drop=True)
    vocabularies = fit_vocabularies(partitions['train'])
    ordered = build_datasets(
        partitions,
        vocabularies=vocabularies,
        config=config,
    )['test']
    shuffled = build_datasets(
        partitions,
        vocabularies=vocabularies,
        config=config,
        shuffle_events=True,
    )['test']
    datasets = {'ordered': ordered, 'shuffled': shuffled}
    output_root = config.results_dir / 'cross_evaluation'
    output_root.mkdir(parents=True, exist_ok=True)

    cross_evaluations = (
        {
            'label': 'gru_attention_ordered_to_shuffled',
            'source_results': args.ordered_results_name,
            'source_model': 'gru_attention',
            'test_order': 'shuffled',
        },
        {
            'label': 'gru_attention_shuffled_to_ordered',
            'source_results': args.shuffled_results_name,
            'source_model': 'gru_attention_shuffled',
            'test_order': 'ordered',
        },
    )
    for specification in cross_evaluations:
        for seed in config.seeds:
            evaluate_cross_condition(
                specification,
                seed,
                project_dir=project_dir,
                output_root=output_root,
                datasets=datasets,
                vocabularies=vocabularies,
                config=config,
                force=args.force,
            )


if __name__ == '__main__':
    main()
