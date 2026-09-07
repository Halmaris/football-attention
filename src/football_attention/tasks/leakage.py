from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from football_attention.baselines import fit_predictive_baselines
from football_attention.data import split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare pre-shot, shot-only and full inputs.')
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--prepared-name', default='development')
    parser.add_argument('--results-name', default='results/leakage')
    args = parser.parse_args()
    partitions = split_rows(pd.read_parquet(
        args.project_dir / 'data' / args.prepared_name / 'sequences_raw.parquet',
    ))
    output = args.project_dir / args.results_name
    output.mkdir(parents=True, exist_ok=True)
    metrics, predictions = [], []
    for variant in ('pre_shot', 'shot_only', 'full'):
        metric, prediction = fit_predictive_baselines(
            partitions['train'], partitions['validation'], partitions['test'], variant=variant,
        )
        metrics.append(metric)
        predictions.append(prediction)
    pd.concat(metrics).to_csv(output / 'metrics.csv', index=False)
    pd.concat(predictions).to_parquet(output / 'predictions.parquet', index=False)


if __name__ == '__main__':
    main()
