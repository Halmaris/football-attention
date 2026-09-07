from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter
from .pitch import PITCH_X, PITCH_Y


SEQUENCE_LENGTH = 20


def load_attributions(
    results_dir: Path,
    source_file: Path | None = None,
) -> pd.DataFrame:
    if source_file is not None and not source_file.resolve().exists():
        raise RuntimeError(
            f'Attribution file does not exist: {source_file.resolve()}'
        )
    combined = (
        source_file.resolve()
        if source_file is not None
        else results_dir / 'faithful' / 'test_token_attributions.parquet'
    )
    if combined.exists():
        rows = pd.read_parquet(combined)
    else:
        paths = sorted(
            (
                results_dir
                / 'faithful'
                / 'gru_attention_faithful'
            ).glob('seed_*/token_attributions.parquet')
        )
        if not paths:
            raise RuntimeError(f'No attribution files found in {results_dir}.')
        rows = pd.concat(
            [pd.read_parquet(path) for path in paths],
            ignore_index=True,
        )
    required = {
        'seed',
        'shot_seq_id',
        'token_position',
        'player_id',
        'attention',
        'uniform',
    }
    missing = required.difference(rows.columns)
    if missing:
        raise RuntimeError(f'Missing attribution columns: {sorted(missing)}.')
    return rows


def player_event_frame(
    sequences: pd.DataFrame,
    attributions: pd.DataFrame,
    *,
    team_name: str,
    player_id: int,
    splits: tuple[str, ...],
) -> tuple[pd.DataFrame, int, int, int]:
    selected = sequences[
        sequences['split'].isin(splits)
        & sequences['team_name'].eq(team_name)
    ].copy()
    if selected.empty:
        raise RuntimeError(
            f'No sequences found for team={team_name!r}, splits={splits!r}.'
        )

    seed_count = int(attributions['seed'].nunique())
    mean_attributions = (
        attributions.groupby(
            ['shot_seq_id', 'token_position', 'player_id'],
            as_index=False,
        )[['attention', 'uniform']]
        .mean()
    )
    lookup = mean_attributions.set_index(['shot_seq_id', 'token_position'])
    records: list[dict[str, object]] = []
    for sequence in selected.itertuples(index=False):
        n_pre_shot = int(sequence.n_events) - 1
        first_position = SEQUENCE_LENGTH - n_pre_shot
        for event_index in range(n_pre_shot):
            if int(sequence.player_id[event_index]) != player_id:
                continue
            key = (int(sequence.shot_seq_id), first_position + event_index)
            if key not in lookup.index:
                raise RuntimeError(f'Missing token attribution for {key}.')
            attribution = lookup.loc[key]
            if isinstance(attribution, pd.DataFrame):
                raise RuntimeError(f'Duplicate token attribution for {key}.')
            if int(attribution['player_id']) != player_id:
                raise RuntimeError(
                    f'Player mismatch for sequence/token {key}: '
                    f'{int(attribution["player_id"])} != {player_id}.'
                )
            coordinates = np.asarray(
                sequence.coordinates[event_index],
                dtype=float,
            )
            uniform = float(attribution['uniform'])
            attention = float(attribution['attention'])
            records.append(
                {
                    'shot_seq_id': int(sequence.shot_seq_id),
                    'match_id': int(sequence.match_id),
                    'event_type': str(sequence.type_name[event_index]),
                    'start_x': float(coordinates[0]),
                    'start_y': float(coordinates[1]),
                    'end_x': float(coordinates[2]),
                    'end_y': float(coordinates[3]),
                    'attention': attention,
                    'uniform': uniform,
                    'attention_ratio': attention / uniform,
                }
            )
    events = pd.DataFrame(records)
    if events.empty:
        raise RuntimeError(f'Player {player_id} has no attributed events.')
    return (
        events,
        int(events['shot_seq_id'].nunique()),
        int(events['match_id'].nunique()),
        seed_count,
    )


def smoothed_relative_attention(
    events: pd.DataFrame,
    *,
    minimum_local_events: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    x_edges = np.linspace(0.0, PITCH_X, 61)
    y_edges = np.linspace(0.0, PITCH_Y, 41)
    weighted, _, _ = np.histogram2d(
        events['end_x'],
        events['end_y'],
        bins=[x_edges, y_edges],
        weights=events['attention_ratio'],
    )
    counts, _, _ = np.histogram2d(
        events['end_x'],
        events['end_y'],
        bins=[x_edges, y_edges],
    )
    numerator = gaussian_filter(weighted, sigma=2.5, mode='constant')
    denominator = gaussian_filter(counts, sigma=2.5, mode='constant')
    mean_ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-8,
    )
    local_support = uniform_filter(
        counts,
        size=(11, 11),
        mode='constant',
    ) * 121
    opacity = np.clip(
        (local_support - (minimum_local_events - 1))
        / max(2, minimum_local_events),
        0.0,
        0.92,
    )
    opacity = gaussian_filter(opacity, sigma=1.4, mode='constant')
    opacity[denominator < 1e-6] = 0.0
    log_ratio = np.log2(np.clip(mean_ratio, 1e-6, None)).T
    extent = (
        float(x_edges[0]),
        float(x_edges[-1]),
        float(y_edges[0]),
        float(y_edges[-1]),
    )
    return log_ratio, opacity.T, extent
