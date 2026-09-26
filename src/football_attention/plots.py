from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

from .plotting.attention import extract_seed_attention
from .plotting.boosting import build_boosting_figures
from .plotting.heatmap import load_attributions, player_event_frame, smoothed_relative_attention
from .plotting.pitch import draw_pitch, sequence_figure
from .plotting.summary import build_summary_figure


def player_responses(players: pd.DataFrame, output: Path, highlight: int | None = None) -> None:
    players = players[players['sequence_count'].ge(20)].copy()
    if players.empty:
        raise ValueError('No players have at least 20 development-test sequences')
    players['response'] = 1000 * players['mean_contribution']
    figure, axis = plt.subplots(figsize=(6.0, 4.4), layout='constrained')
    for positive, color, label in ((False, '#1F3A93', 'Negative response'),
                                   (True, '#D55E00', 'Positive response')):
        part = players[players['response'].ge(0).eq(positive)]
        axis.scatter(part['sequence_count'], part['response'],
                     s=(11 + 5 * (part['match_count'] - 3)).clip(lower=4),
                     color=color, alpha=0.5, edgecolors='none', label=label)
    low, high = players['response'].min(), players['response'].max()
    span = max(high - low, 1.0)
    for positive in (False, True):
        part = players[players['response'].ge(0).eq(positive)]
        extremes = part.nlargest(5, 'response') if positive else part.nsmallest(5, 'response')
        extremes = pd.concat([extremes, part[part['player_id'].eq(highlight)]]).drop_duplicates('player_id')
        extremes = extremes.sort_values('response')
        ys = np.linspace(max(0, low), high, len(extremes)) if positive else np.linspace(low, min(0, high), len(extremes))
        for row, label_y in zip(extremes.itertuples(), ys):
            x = players['sequence_count'].max() if positive else players['sequence_count'].min()
            axis.annotate(row.player_name, (row.sequence_count, row.response), (x, label_y),
                          ha='right' if positive else 'left', fontsize=8,
                          weight='bold' if row.player_id == highlight else 'normal',
                          arrowprops={'arrowstyle': '-', 'color': '#888888', 'lw': 0.5})
    axis.axhline(0, color='#666666', linestyle='--', linewidth=0.7)
    axis.set(xlabel='Development-test sequences per player',
             ylabel=r'Mean signed model response ($\times 10^{-3}$)',
             ylim=(low - 0.12 * span, high + 0.12 * span))
    axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)
    axis.grid(alpha=0.2)
    figure.savefig(output, bbox_inches='tight')
    plt.close(figure)


def player_maps(raw, tokens, players, *, team, splits, support, output):
    n_columns = min(3, len(players))
    n_rows = math.ceil(len(players) / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(12.2, 3.15 * n_rows + 0.55), squeeze=False)
    figure.subplots_adjust(left=0.025, right=0.985, top=0.9, bottom=0.22, wspace=0.075, hspace=0.3)
    color_map = LinearSegmentedColormap.from_list('relative_attention', ['#1F3A93', 'white', '#D55E00'])
    for axis, (player_id, label) in zip(axes.flat, players):
        events, n_sequences, n_matches, _ = player_event_frame(
            raw, tokens, team_name=team, player_id=player_id, splits=tuple(splits),
        )
        values, opacity, extent = smoothed_relative_attention(events, minimum_local_events=support)
        image = axis.imshow(values, extent=extent, origin='lower', cmap=color_map,
                            norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
                            interpolation='bicubic', alpha=opacity)
        axis.scatter(events['end_x'], events['end_y'], s=4, facecolors='white',
                     edgecolors='#252525', linewidths=0.22, alpha=0.34, zorder=4)
        draw_pitch(axis)
        axis.set_title(label, loc='left', fontsize=10, weight='bold', pad=16)
        axis.text(0, 1.015, f'{n_sequences} sequences · {n_matches} matches · {len(events)} events',
                  transform=axis.transAxes, fontsize=7, va='bottom')
        axis.text(0.78, 0.055, 'Attack', transform=axis.transAxes, fontsize=7, ha='right', va='center')
        axis.add_patch(FancyArrowPatch((0.8, 0.055), (0.97, 0.055), transform=axis.transAxes,
                                      arrowstyle='-|>', mutation_scale=10, color='#222222'))
    for axis in list(axes.flat)[len(players):]:
        axis.set_visible(False)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), orientation='horizontal',
                              fraction=0.055, pad=0.075, shrink=0.38, aspect=32, extend='both')
    colorbar.set_ticks([-1, 0, 1], labels=['0.5×', '1×', '2×'])
    colorbar.set_label('Mean attention relative to uniform')
    figure.savefig(output, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot current locally generated results.')
    parser.add_argument('--work-dir', type=Path, default=Path('local/run'))
    parser.add_argument('--holdout-name', default='holdout')
    parser.add_argument('--holdout-results-name', default='results/holdout')
    parser.add_argument('--explanation-results-name', default='results/explanations')
    parser.add_argument('--output-name', default='results/figures')
    parser.add_argument('--team')
    parser.add_argument('--players', type=int, nargs='+')
    parser.add_argument('--player-labels', nargs='+')
    parser.add_argument('--splits', nargs='+', choices=('train', 'validation', 'test'), default=['train', 'validation'])
    parser.add_argument('--minimum-local-events', type=int, default=8)
    parser.add_argument('--sequence-id', type=int)
    parser.add_argument('--highlight-player', type=int)
    args = parser.parse_args()
    if args.players and not args.team:
        parser.error('--team is required with --players')
    if args.player_labels and len(args.player_labels) != len(args.players or []):
        parser.error('Provide one label per player')
    root = args.work_dir.resolve()
    final = root / 'results/final'
    output = root / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    build_summary_figure(root, output / 'summary.pdf',
                         external_results_name=args.holdout_results_name,
                         explanation_results_name=args.explanation_results_name)
    build_boosting_figures(root, output, holdout_name=args.holdout_name,
                          holdout_results_name=args.holdout_results_name)
    players = pd.read_csv(final / 'faithful/player_occlusion_report.csv')
    player_responses(players, output / 'player_responses.pdf', args.highlight_player)
    prepared = root / 'data/development'
    raw = pd.read_parquet(prepared / 'sequences_raw.parquet')
    names = {int(key): value for key, value in json.loads((prepared / 'player_id_to_name.json').read_text()).items()}
    if args.sequence_id is not None:
        tokens = load_attributions(final)
        tokens = tokens.groupby(['shot_seq_id', 'token_position'], as_index=False)[['attention', 'occlusion_abs']].mean()
        sequence_figure(raw, tokens, names, output / 'sequence.pdf', sequence_id=args.sequence_id)
    if args.players:
        selected = raw[raw['split'].isin(args.splits) & raw['team_name'].eq(args.team)]
        protocol = json.loads((final / 'frozen_protocol.json').read_text())
        tokens = pd.concat([
            extract_seed_attention(selected, checkpoint_path=final / 'faithful/gru_attention_faithful' / f'seed_{seed}/model.pt',
                                   project_dir=root, results_name='results/final', prepared_name='development',
                                   model_label='gru_attention_faithful', seed=int(seed))
            for seed in protocol['seeds']
        ], ignore_index=True)
        labels = args.player_labels or [names[player] for player in args.players]
        player_maps(raw, tokens, list(zip(args.players, labels)), team=args.team, splits=args.splits,
                    support=args.minimum_local_events, output=output / 'player_attention.pdf')
    print(f'Figures written to {output}')


if __name__ == '__main__':
    main()
