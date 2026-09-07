from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


PITCH_X = 120.0


PITCH_Y = 80.0


def draw_pitch(
    axis: plt.Axes,
    *,
    line_color: str = '#303030',
    face_color: str = 'none',
) -> None:
    axis.set_facecolor(face_color)
    axis.add_patch(Rectangle((0, 0), PITCH_X, PITCH_Y, fill=False,
                             color=line_color, linewidth=0.8))
    axis.plot([60, 60], [0, 80], color=line_color, linewidth=0.55)
    axis.add_patch(plt.Circle((60, 40), 10, fill=False,
                              color=line_color, linewidth=0.55))
    axis.add_patch(Rectangle((102, 18), 18, 44, fill=False,
                             color=line_color, linewidth=0.7))
    axis.add_patch(Rectangle((114, 30), 6, 20, fill=False,
                             color=line_color, linewidth=0.7))
    axis.add_patch(Arc((108, 40), 20, 20, theta1=128, theta2=232,
                       color=line_color, linewidth=0.55))
    axis.scatter([60, 108], [40, 40], s=3, color=line_color, zorder=5)
    axis.set_xlim(0, PITCH_X)
    axis.set_ylim(PITCH_Y, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def sequence_figure(
    raw: pd.DataFrame,
    attributions: pd.DataFrame,
    names: dict[int, str],
    output: Path,
    *,
    sequence_id: int,
) -> None:
    row = raw.loc[raw['shot_seq_id'].eq(sequence_id)].iloc[0]
    sequence = (
        attributions.loc[attributions['shot_seq_id'].eq(sequence_id)]
        .sort_values('token_position')
        .reset_index(drop=True)
    )
    n_pre = int(row['n_events']) - 1
    if len(sequence) != n_pre:
        raise RuntimeError('Attribution and event counts do not match.')

    attention = sequence['attention'].to_numpy(float)
    occlusion = sequence['occlusion_abs'].to_numpy(float)
    norm = Normalize(vmin=0.0, vmax=max(0.40, float(attention.max())))
    type_colors = {
        'Pass': '#1F3A93',
        'Carry': '#D55E00',
        'Ball Receipt': '#59a14f',
        'Ball Recovery': '#e15759',
        'Interception': '#4e79a7',
    }
    figure = plt.figure(figsize=(3.48, 5.02), facecolor='white')
    grid = figure.add_gridspec(2, 1, height_ratios=[1.35, 2.15], hspace=0.04)
    pitch = figure.add_subplot(grid[0, 0])
    legend = figure.add_subplot(grid[1, 0])
    draw_pitch(pitch, line_color='#555555', face_color='white')

    legend_step = min(0.066, 0.84 / max(n_pre, 1))
    for event_index in range(n_pre):
        coordinates = np.asarray(row['coordinates'][event_index], dtype=float)
        start_x, start_y, end_x, end_y = coordinates
        weight = float(attention[event_index])
        event_type = str(row['type_name'][event_index]).replace('*', '')
        color = type_colors.get(event_type, '#616A73')
        linewidth = 0.8 + 4.2 * norm(weight)
        distance = float(np.hypot(end_x - start_x, end_y - start_y))
        if distance > 0.8:
            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='-|>', mutation_scale=6.5 + 4.5 * norm(weight),
                linewidth=linewidth, color=color, alpha=0.92,
                shrinkA=0.4, shrinkB=0.4, zorder=4,
            )
            pitch.add_patch(arrow)
            label_x = 0.5 * (start_x + end_x)
            label_y = 0.5 * (start_y + end_y)
        else:
            pitch.scatter([start_x], [start_y], s=23 + 65 * norm(weight),
                          color=color, edgecolor='white', linewidth=0.5,
                          zorder=5)
            label_x, label_y = start_x, start_y
        offset = (0, 5.5 if event_index % 2 == 0 else -6.5)
        pitch.annotate(
            str(event_index + 1),
            (label_x, label_y),
            xytext=offset,
            textcoords='offset points',
            ha='center',
            va='center',
            fontsize=5.9,
            fontweight='bold',
            color='white',
            zorder=7,
            bbox={
                'boxstyle': 'circle,pad=0.14',
                'facecolor': color,
                'edgecolor': 'white',
                'linewidth': 0.45,
            },
        )

    shot = np.asarray(row['coordinates'][-1], dtype=float)
    pitch.add_patch(FancyArrowPatch(
        (shot[0], shot[1]), (shot[2], shot[3]), arrowstyle='-|>',
        mutation_scale=8, linewidth=1.3, linestyle='--',
        color='#333333', zorder=6,
    ))
    pitch.text(
        0.78,
        0.055,
        'Attack',
        transform=pitch.transAxes,
        ha='right',
        va='center',
        fontsize=5.4,
        fontweight='bold',
        color='#202020',
        bbox={
            'boxstyle': 'round,pad=0.16',
            'facecolor': 'white',
            'edgecolor': 'none',
            'alpha': 0.86,
        },
        zorder=9,
    )
    pitch.add_patch(FancyArrowPatch(
        (0.80, 0.055),
        (0.975, 0.055),
        transform=pitch.transAxes,
        arrowstyle='-|>',
        mutation_scale=9,
        linewidth=1.15,
        color='#202020',
        zorder=9,
    ))
    legend.axis('off')
    legend.text(0.0, 0.98, 'Player / action',
                fontsize=7.2, fontweight='bold', va='top')
    legend.text(0.80, 0.98, 'Attention', fontsize=6.8,
                fontweight='bold', va='top', ha='right')
    legend.text(0.99, 0.98, r'$|\Delta\hat y|$', fontsize=7.0,
                fontweight='bold', va='top', ha='right')
    for event_index in range(n_pre):
        player_id = int(row['player_id'][event_index])
        player = names.get(player_id, str(player_id))
        if len(player) > 18:
            parts = player.split()
            player = f'{parts[0][0]}. {parts[-1]}'
        event_type = str(row['type_name'][event_index]).replace('*', '')
        compact_type = {
            'Ball Receipt': 'Receipt',
            'Ball Recovery': 'Recovery',
        }.get(event_type, event_type)
        text = f'{event_index + 1:>2}  {player} — {compact_type}'
        y = 0.91 - event_index * legend_step
        color = type_colors.get(event_type, '#616A73')
        legend.text(0.0, y, text, fontsize=5.7, va='top', color=color)
        legend.text(0.80, y, f'{100 * attention[event_index]:.1f}%',
                    fontsize=5.8, va='top', ha='right', color='#262626',
                    fontweight='bold')
        legend.text(0.99, y, f'{occlusion[event_index]:.3f}',
                    fontsize=5.8, va='top', ha='right', color='#262626')
    legend.text(
        0.0,
        0.025,
        r'Width/size = attention; $|\Delta\hat y|$ = occlusion',
        fontsize=6.1,
        color='#39434A',
        va='bottom',
    )
    figure.savefig(output, bbox_inches='tight', facecolor='white')
    plt.close(figure)
