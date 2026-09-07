from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

from football_attention.baselines import feature_matrix
from football_attention.frozen_final import load_classical_models


PERIODS = ('Development test', 'Next-season holdout')
COLORS = {'Development test': '#D55E00', 'Next-season holdout': '#1F3A93'}
LENGTH_BINS = ('1-2', '3-5', '6-10', '11-20')
GROUPS = (
    'Final-event features',
    'Spatial summaries',
    'Temporal summaries',
    'Event-type counts',
    'Outcome counts',
    'Sequence scale',
)


def match_bootstrap(
    differences: np.ndarray,
    match_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    grouped = (
        pd.DataFrame({'match_id': match_ids, 'difference': differences})
        .groupby('match_id')['difference']
        .agg(['sum', 'count'])
    )
    sums = grouped['sum'].to_numpy(float)
    counts = grouped['count'].to_numpy(float)
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        selected = generator.integers(
            0,
            len(grouped),
            size=(stop - start, len(grouped)),
        )
        estimates[start:stop] = (
            sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
        )
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def sequence_lengths(path: Path, split: str | None = None) -> pd.DataFrame:
    columns = ['shot_seq_id', 'match_id', 'is_shot']
    if split is not None:
        columns.append('split')
    rows = pd.read_parquet(path, columns=columns)
    if split is not None:
        rows = rows.loc[rows['split'].eq(split)].drop(columns='split')
    rows['n_pre_shot_events'] = rows['is_shot'].map(
        lambda values: int((~np.asarray(values, dtype=bool)).sum())
    )
    if not rows['n_pre_shot_events'].between(1, 20).all():
        raise RuntimeError('Expected between 1 and 20 visible pre-shot events')
    return rows.drop(columns='is_shot')


def paired_predictions(
    predictions: pd.DataFrame,
    lengths: pd.DataFrame,
) -> pd.DataFrame:
    columns = ['shot_seq_id', 'match_id', 'target_xg', 'prediction']
    last = predictions.loc[
        predictions['model_type'].eq('gradient_boosting_last_event'),
        columns,
    ]
    full = predictions.loc[
        predictions['model_type'].eq('gradient_boosting'),
        columns,
    ]
    paired = last.merge(
        full,
        on=['shot_seq_id', 'match_id'],
        suffixes=('_last', '_full'),
        validate='one_to_one',
    ).merge(lengths, on=['shot_seq_id', 'match_id'], validate='one_to_one')
    if len(paired) != len(lengths):
        raise RuntimeError('Boosting predictions do not cover every sequence')
    paired['difference'] = (
        np.square(paired['target_xg_last'] - paired['prediction_last'])
        - np.square(paired['target_xg_full'] - paired['prediction_full'])
    )
    paired['length_bin'] = pd.cut(
        paired['n_pre_shot_events'],
        [0, 2, 5, 10, 20],
        labels=LENGTH_BINS,
    )
    return paired


def buildup_length_summary(
    project: Path,
    *,
    bootstrap_samples: int = 10_000,
) -> pd.DataFrame:
    inputs = (
        (
            PERIODS[0],
            project / 'results/final/test_predictions.parquet',
            project / 'data/development/sequences_raw.parquet',
            'test',
        ),
        (
            PERIODS[1],
            project / 'results/holdout/predictions.parquet',
            project / 'data/holdout/sequences_raw.parquet',
            None,
        ),
    )
    output = []
    for period_index, (period, prediction_path, data_path, split) in enumerate(
        inputs
    ):
        predictions = pd.read_parquet(prediction_path)
        paired = paired_predictions(
            predictions,
            sequence_lengths(data_path, split),
        )
        for bin_index, length_bin in enumerate(LENGTH_BINS):
            selected = paired.loc[paired['length_bin'].eq(length_bin)]
            if selected.empty:
                low, high = np.nan, np.nan
            else:
                low, high = match_bootstrap(
                    selected['difference'].to_numpy(float),
                    selected['match_id'].to_numpy(),
                    samples=bootstrap_samples,
                    seed=2027 + period_index * 100 + bin_index,
                )
            output.append(
                {
                    'period': period,
                    'length_bin': length_bin,
                    'n_sequences': len(selected),
                    'n_matches': selected['match_id'].nunique(),
                    'difference_mean': selected['difference'].mean(),
                    'ci_025': low,
                    'ci_975': high,
                }
            )
    return pd.DataFrame(output)


def plot_buildup_length(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(3.55, 2.45))
    positions = np.arange(len(LENGTH_BINS), dtype=float)
    counts = {}
    for period, offset, marker in (
        (PERIODS[0], -0.12, 'o'),
        (PERIODS[1], 0.12, 's'),
    ):
        values = summary.loc[summary['period'].eq(period)].set_index(
            'length_bin'
        ).loc[list(LENGTH_BINS)]
        estimate = 1000 * values['difference_mean'].to_numpy(float)
        error = np.vstack(
            [
                estimate - 1000 * values['ci_025'].to_numpy(float),
                1000 * values['ci_975'].to_numpy(float) - estimate,
            ]
        )
        axis.errorbar(
            estimate,
            positions + offset,
            xerr=error,
            fmt=marker,
            color=COLORS[period],
            markersize=4,
            capsize=2.2,
            label=period,
        )
        counts[period] = values['n_sequences'].astype(int).tolist()
    labels = [
        f'{label}  [{dev}/{holdout}]'
        for label, dev, holdout in zip(
            LENGTH_BINS,
            counts[PERIODS[0]],
            counts[PERIODS[1]],
            strict=True,
        )
    ]
    axis.axvline(0, color='#666666', linewidth=0.75, linestyle='--')
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_ylabel('Pre-shot events [n: dev/holdout]')
    axis.set_xlabel(r'Last-event MSE $-$ full-buildup MSE ($\times 10^{-3}$)')
    axis.grid(axis='x', linewidth=0.4, alpha=0.35)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(0, 1.03),
        ncol=2,
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.8)},
    )
    axis.text(
        0.99,
        0.02,
        'Positive favors full buildup',
        transform=axis.transAxes,
        ha='right',
        va='bottom',
        fontsize=6.2,
        color='#4A4A4A',
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches='tight')
    plt.close(figure)


def feature_group(name: str) -> str:
    if name in {'n_tokens', 'n_unique_players'}:
        return 'Sequence scale'
    if name.endswith('_last') or name.startswith('last_'):
        return 'Final-event features'
    if name.startswith('type_count='):
        return 'Event-type counts'
    if name.startswith('outcome_count='):
        return 'Outcome counts'
    if name.startswith('delta_time_'):
        return 'Temporal summaries'
    if name.startswith(('start_x_', 'start_y_', 'end_x_', 'end_y_')):
        return 'Spatial summaries'
    raise ValueError(f'Unassigned feature: {name}')


def importance_summary(
    project: Path,
    *,
    permutations: int = 100,
    bootstrap_samples: int = 10_000,
) -> pd.DataFrame:
    fitted = load_classical_models(project / 'results/final/classical/models.pkl')
    vectorizer = fitted['vectorizer']
    model = fitted['models']['gradient_boosting']
    names = vectorizer.get_feature_names_out()
    columns = {
        group: np.flatnonzero([feature_group(str(name)) == group for name in names])
        for group in GROUPS
    }
    inputs = (
        (PERIODS[0], project / 'data/development/sequences_raw.parquet', 'test'),
        (PERIODS[1], project / 'data/holdout/sequences_raw.parquet', None),
    )
    output = []
    for period_index, (period, path, split) in enumerate(inputs):
        rows = pd.read_parquet(path)
        if split is not None:
            rows = rows.loc[rows['split'].eq(split)].copy()
        x = vectorizer.transform(feature_matrix(rows, 'pre_shot'))
        target = rows['xg'].to_numpy(float)
        original_loss = np.square(target - model.predict(x))
        for group_index, group in enumerate(GROUPS):
            generator = np.random.default_rng(2027 + 1000 * period_index + group_index)
            differences = np.empty((permutations, len(rows)), dtype=float)
            for repeat in range(permutations):
                order = generator.permutation(len(rows))
                permuted = x.copy()
                permuted[:, columns[group]] = x[order][:, columns[group]]
                differences[repeat] = (
                    np.square(target - model.predict(permuted)) - original_loss
                )
            per_sequence = differences.mean(axis=0)
            low, high = match_bootstrap(
                per_sequence,
                rows['match_id'].to_numpy(),
                samples=bootstrap_samples,
                seed=42027 + 1000 * period_index + group_index,
            )
            output.append(
                {
                    'period': period,
                    'feature_group': group,
                    'n_features': len(columns[group]),
                    'delta_mse': per_sequence.mean(),
                    'ci_025': low,
                    'ci_975': high,
                }
            )
    return pd.DataFrame(output)


def plot_importance(summary: pd.DataFrame, output: Path) -> None:
    figure, (detail, final) = plt.subplots(
        1,
        2,
        figsize=(3.55, 2.65),
        sharey=True,
        gridspec_kw={'width_ratios': [3.2, 1.05], 'wspace': 0.08},
    )
    positions = np.arange(len(GROUPS), dtype=float)
    for period, offset, marker in (
        (PERIODS[0], -0.12, 'o'),
        (PERIODS[1], 0.12, 's'),
    ):
        values = summary.loc[summary['period'].eq(period)].set_index(
            'feature_group'
        ).loc[list(GROUPS)]
        estimate = 1000 * values['delta_mse'].to_numpy(float)
        error = np.vstack(
            [
                estimate - 1000 * values['ci_025'].to_numpy(float),
                1000 * values['ci_975'].to_numpy(float) - estimate,
            ]
        )
        for axis in (detail, final):
            axis.errorbar(
                estimate,
                positions + offset,
                xerr=error,
                fmt=marker,
                color=COLORS[period],
                markersize=4,
                capsize=2.2,
                label=period,
            )
    detail.axvline(0, color='#666666', linewidth=0.75, linestyle='--')
    detail.set_yticks(positions, GROUPS)
    detail.set_ylabel('Feature group')
    detail.invert_yaxis()
    detail.set_xlim(-0.35, 5.05)
    final.set_xlim(17, 25)
    for axis in (detail, final):
        axis.grid(axis='x', linewidth=0.4, alpha=0.35)
        axis.spines['top'].set_visible(False)
    detail.spines['right'].set_visible(False)
    final.spines[['left', 'right']].set_visible(False)
    final.tick_params(axis='y', left=False, labelleft=False)
    break_style = {
        'marker': [(-1, -0.8), (1, 0.8)],
        'markersize': 5.5,
        'linestyle': 'none',
        'color': 'black',
        'clip_on': False,
    }
    detail.plot([1], [0], transform=detail.transAxes, **break_style)
    final.plot([0], [0], transform=final.transAxes, **break_style)
    handles, labels = detail.get_legend_handles_labels()
    figure.legend(
        handles[:2],
        labels[:2],
        frameon=False,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.235),
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.8)},
    )
    figure.supxlabel(r'MSE increase after group permutation ($\times 10^{-3}$)')
    figure.subplots_adjust(left=0.43, right=0.985, top=0.96, bottom=0.2)
    figure.savefig(output, bbox_inches='tight')
    plt.close(figure)


def build_boosting_figures(project: Path, output_dir: Path) -> None:
    length = buildup_length_summary(project)
    length.to_csv(output_dir / 'buildup_value_by_length.csv', index=False)
    plot_buildup_length(length, output_dir / 'buildup_value_by_length.pdf')
    importance = importance_summary(project)
    importance.to_csv(
        output_dir / 'boosting_group_permutation_importance.csv',
        index=False,
    )
    plot_importance(
        importance,
        output_dir / 'boosting_group_permutation_importance.pdf',
    )
