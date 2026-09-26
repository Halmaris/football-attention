from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
import numpy as np
import pandas as pd


def build_summary_figure(
    project: Path,
    output: Path,
    *,
    frozen_results_name: str = 'results/final',
    external_results_name: str = (
        'results/holdout'
    ),
    explanation_results_name: str = (
        'results/explanations'
    ),
) -> None:
    """Render the summary figure from the frozen result tables."""
    model_types = [
        'faithful_no_player_id',
        'gru',
        'gru_attention_faithful',
        'gru_attention_predictive',
        'deepsets',
    ]
    model_labels = [
        'Transformer',
        'GRU',
        'Aligned\nGRU attention',
        'Predictive\nGRU attention',
        'DeepSets',
    ]
    development = (
        pd.read_csv(
            project
            / frozen_results_name
            / 'test_mse_vs_gradient_boosting_bootstrap.csv'
        )
        .set_index('model_type')
        .loc[model_types]
    )
    holdout = (
        pd.read_csv(
            project
            / external_results_name
            / 'prediction_differences_bootstrap.csv'
        )
        .query("model_right == 'gradient_boosting' and scope == 'all'")
        .set_index('model_left')
        .loc[model_types]
    )
    deletion = pd.read_csv(
        project
        / explanation_results_name
        / 'frozen_holdout'
        / 'deletion_summary_by_seed.csv'
    )
    deletion = (
        deletion.groupby(['method', 'k'], as_index=False)['delta_mse']
        .mean()
        .pivot(index='k', columns='method', values='delta_mse')
    )
    deletion_k = np.array([1, 3, 5])
    plt.rcParams.update({
        'font.family': 'serif',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 6.7,
        'axes.labelsize': 6.7,
        'axes.titlesize': 7.2,
        'legend.fontsize': 5.9,
        'xtick.labelsize': 6.2,
        'ytick.labelsize': 6.2,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55))
    mse_axis, deletion_axis = axes

    y = np.arange(len(model_types), dtype=float)
    offset = 0.12
    for values, positions, label, color in [
        (development, y - offset, 'Development test', '#D55E00'),
        (holdout, y + offset, 'Next-season holdout', '#1F3A93'),
    ]:
        estimate = 1000 * values['difference_mean'].to_numpy(float)
        errors = np.vstack([
            estimate - 1000 * values['ci_025'].to_numpy(float),
            1000 * values['ci_975'].to_numpy(float) - estimate,
        ])
        mse_axis.errorbar(
            estimate,
            positions,
            xerr=errors,
            fmt='o',
            markersize=3.8,
            capsize=2.2,
            elinewidth=1.05,
            capthick=0.8,
            label=label,
            color=color,
        )
    mse_axis.axvline(0.0, color='#5A5A5A', linewidth=0.7, linestyle='--')
    mse_axis.set_yticks(y, model_labels)
    mse_axis.invert_yaxis()
    mse_axis.set_xlabel(r'$\Delta$MSE vs boosting ($\times 10^{-3}$)')
    mse_axis.set_xlim(min(-0.55, 1.06 * 1000 * min(development['ci_025'].min(), holdout['ci_025'].min())),
                      max(3.7, 1.06 * 1000 * max(development['ci_975'].max(), holdout['ci_975'].max())))
    mse_axis.set_xticks([-0.5, 0.0, 1.0, 2.0, 3.0])
    mse_axis.legend(
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(0.0, 1.085),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.5,
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.9)},
    )
    mse_axis.grid(axis='x', linewidth=0.35, alpha=0.35)

    for method, label, marker, color in [
        ('attention', 'Attention', 'o', '#4e79a7'),
        ('recency', 'Recency', 's', '#f28e2b'),
        ('gradient_x_input', r'Gradient $\times$ input', '^', '#59a14f'),
        ('permutation_shapley', 'Permutation Shapley', 'D', '#e15759'),
    ]:
        deletion_axis.plot(
            deletion_k,
            deletion.loc[deletion_k, method],
            marker=marker,
            linewidth=1.2,
            markersize=4,
            label=label,
            color=color,
        )
    deletion_axis.plot(
        deletion_k,
        deletion.loc[deletion_k, 'random'],
        marker='x',
        linewidth=1.0,
        linestyle='--',
        label='Random',
        color='#606060',
    )
    deletion_axis.set_xticks(deletion_k)
    deletion_axis.set_xlabel(r'Deleted events ($k$)')
    deletion_axis.set_ylabel('Increase in MSE')
    deletion_max = deletion.loc[
        deletion_k,
        [
            'attention',
            'recency',
            'gradient_x_input',
            'permutation_shapley',
            'random',
        ],
    ].to_numpy(float).max()
    deletion_axis.set_ylim(0, 1.08 * deletion_max)
    deletion_axis.legend(
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(0.08, 0.985),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    deletion_axis.grid(linewidth=0.35, alpha=0.35)

    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=1.8)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
