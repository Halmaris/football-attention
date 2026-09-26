from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


CALIBRATION_MODELS = {
    'gradient_boosting': 'Histogram boosting',
    'gru_attention_faithful': 'Aligned GRU attention',
}


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame['target_xg'].to_numpy(float)
    prediction = frame['prediction'].to_numpy(float)
    residual = target - prediction
    denominator = np.square(target - target.mean()).sum()
    return {
        'mse': float(np.mean(np.square(residual))),
        'mae': float(np.mean(np.abs(residual))),
        'r2': float(1.0 - np.square(residual).sum() / denominator),
    }


def equal_count_calibration(frame: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    ranked = frame.sort_values('prediction', kind='mergesort').reset_index(drop=True)
    ranked['bin'] = np.floor(np.arange(len(ranked)) * n_bins / len(ranked)).astype(int)
    ranked['bin'] = ranked['bin'].clip(upper=n_bins - 1)
    return (
        ranked.groupby('bin', as_index=False)
        .agg(
            n=('target_xg', 'size'),
            mean_prediction=('prediction', 'mean'),
            mean_target=('target_xg', 'mean'),
        )
    )


def build_summaries(
    project: Path,
    output_dir: Path,
    *,
    development_predictions: Path | str = Path(
        'results/final/test_predictions.parquet'
    ),
    holdout_predictions: Path | str = Path(
        'results/holdout/predictions.parquet'
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    calibration_fit_rows: list[dict[str, object]] = []
    range_rows: list[dict[str, object]] = []

    period_files = {
        'Development test': Path(development_predictions),
        'Frozen holdout': Path(holdout_predictions),
    }
    for period, relative_path in period_files.items():
        predictions = pd.read_parquet(project / relative_path)
        targets = (
            predictions[['shot_seq_id', 'target_xg']]
            .drop_duplicates('shot_seq_id')
            .sort_values('shot_seq_id')
        )
        values = targets['target_xg'].to_numpy(float)
        quantiles = np.quantile(values, [0.25, 0.50, 0.75])
        target_rows.append({
            'period': period,
            'n': len(values),
            'mean': values.mean(),
            'variance': values.var(ddof=1),
            'sd': values.std(ddof=1),
            'q25': quantiles[0],
            'median': quantiles[1],
            'q75': quantiles[2],
            'minimum': values.min(),
            'maximum': values.max(),
        })

        for (model_type, seed), frame in predictions.groupby(['model_type', 'seed']):
            metric_rows.append({
                'period': period,
                'model_type': model_type,
                'seed': int(seed),
                **regression_metrics(frame),
            })
            range_rows.append({
                'period': period,
                'model_type': model_type,
                'seed': int(seed),
                'minimum': frame['prediction'].min(),
                'maximum': frame['prediction'].max(),
                'outside_0_1': int(
                    ((frame['prediction'] < 0) | (frame['prediction'] > 1)).sum()
                ),
            })

        for model_type, model_label in CALIBRATION_MODELS.items():
            selected = predictions.loc[predictions['model_type'].eq(model_type)]
            ensemble = (
                selected.groupby(['shot_seq_id', 'match_id'], as_index=False)
                .agg(target_xg=('target_xg', 'first'), prediction=('prediction', 'mean'))
            )
            design = np.column_stack([np.ones(len(ensemble)), ensemble['prediction']])
            intercept, slope = np.linalg.lstsq(
                design,
                ensemble['target_xg'].to_numpy(float),
                rcond=None,
            )[0]
            calibration_fit_rows.append({
                'period': period,
                'model_type': model_type,
                'model': model_label,
                'intercept': intercept,
                'slope': slope,
                'mean_prediction': ensemble['prediction'].mean(),
                'mean_target': ensemble['target_xg'].mean(),
            })
            bins = equal_count_calibration(ensemble)
            bins.insert(0, 'model', model_label)
            bins.insert(0, 'model_type', model_type)
            bins.insert(0, 'period', period)
            calibration_rows.extend(bins.to_dict('records'))

    metrics = pd.DataFrame(metric_rows)
    metrics_summary = (
        metrics.groupby(['period', 'model_type'], as_index=False)
        .agg(
            n_seeds=('seed', 'size'),
            mse=('mse', 'mean'),
            mse_sd=('mse', 'std'),
            mae=('mae', 'mean'),
            mae_sd=('mae', 'std'),
            r2=('r2', 'mean'),
            r2_sd=('r2', 'std'),
        )
    )
    metrics_summary.to_csv(output_dir / 'predictive_metrics.csv', index=False)
    pd.DataFrame(target_rows).to_csv(output_dir / 'target_distribution.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / 'calibration_bins.csv', index=False)
    pd.DataFrame(calibration_fit_rows).to_csv(
        output_dir / 'calibration_fits.csv', index=False
    )
    pd.DataFrame(range_rows).to_csv(output_dir / 'prediction_ranges.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize predictions and target distributions.')
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--holdout-results-name', default='results/holdout')
    parser.add_argument('--results-name', default='results/summaries')
    args = parser.parse_args()
    build_summaries(args.project_dir, args.project_dir / args.results_name,
                    holdout_predictions=Path(args.holdout_results_name) / 'predictions.parquet')


if __name__ == '__main__':
    main()
