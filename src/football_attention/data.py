from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


PITCH_X = 120.0


PITCH_Y = 80.0


EVENT_OUTCOME_COLUMNS = {
    'Ball Receipt*': 'ball_receipt_outcome',
    'Dribble': 'dribble_outcome',
    'Duel': 'duel_outcome',
    'Goal Keeper': 'goalkeeper_outcome',
    'Interception': 'interception_outcome',
    'Pass': 'pass_outcome',
    'Shot': 'shot_outcome',
    'Substitution': 'substitution_outcome',
}


IMPLICIT_SUCCESS_EVENT_TYPES = {'Pass', 'Carry', 'Dribble'}


def object_name(value: object) -> str:
    if isinstance(value, dict):
        for key in ['name', 'team_name', 'player_name', 'type_name']:
            if value.get(key) is not None:
                return str(value[key])
    return str(value)


def parse_location(value: object) -> tuple[float, float]:
    if value is None:
        return np.nan, np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan, np.nan
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return np.nan, np.nan
    if isinstance(value, dict):
        value = [value.get('x'), value.get('y')]
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return np.nan, np.nan
    return np.nan, np.nan


def event_end_location(row: pd.Series) -> tuple[float, float]:
    for column in [
        'pass_end_location',
        'carry_end_location',
        'shot_end_location',
        'dribble_end_location',
        'goalkeeper_end_location',
    ]:
        if column in row.index:
            x, y = parse_location(row[column])
            if np.isfinite(x) and np.isfinite(y):
                return x, y
    return parse_location(row.get('location'))


def _nonmissing_name(value: object) -> str | None:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    name = object_name(value).strip()
    if not name or name.casefold() in {'nan', 'none', '<na>'}:
        return None
    return name


def event_outcome(row: pd.Series) -> str:
    event_type = object_name(row.get('type', '')).strip()
    specific_column = EVENT_OUTCOME_COLUMNS.get(event_type)
    if specific_column is not None:
        name = _nonmissing_name(row.get(specific_column))
        if name is not None:
            return name

    name = _nonmissing_name(row.get('outcome'))
    if name is not None:
        return name
    if event_type in IMPLICIT_SUCCESS_EVENT_TYPES:
        return 'Success'
    return 'NA'


def timestamp_seconds(value: object) -> float:
    if not isinstance(value, str) or ':' not in value:
        return np.nan
    try:
        hours, minutes, seconds = value.split(':')
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return np.nan


def event_times(events: pd.DataFrame) -> np.ndarray:
    if 'timestamp' in events.columns:
        values = events['timestamp'].map(timestamp_seconds).to_numpy(float)
        if np.isfinite(values).any():
            return values
    minutes = events.get('minute', pd.Series(0, index=events.index)).to_numpy(float)
    seconds = events.get('second', pd.Series(0, index=events.index)).to_numpy(float)
    return minutes * 60 + seconds


def _safe_int(value: object, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def build_match_sequences(
    events: pd.DataFrame,
    *,
    match_id: int,
    match_date: str,
    sequence_length: int,
) -> tuple[list[dict[str, object]], dict[int, str]]:
    required = {'type', 'team', 'possession'}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise RuntimeError(f'Event columns missing: {missing}')

    events = events.copy()
    events['_time_seconds'] = event_times(events)
    sort_columns = [column for column in ['period', 'index'] if column in events]
    if not sort_columns:
        sort_columns = ['_time_seconds']
    events = events.sort_values(sort_columns, kind='mergesort').reset_index(drop=True)

    type_names = events['type'].map(object_name).astype(str)
    team_names = events['team'].map(object_name).astype(str)
    shot_indices = np.flatnonzero(type_names.str.casefold().eq('shot').to_numpy())
    player_names: dict[int, str] = {}
    if {'player_id', 'player'}.issubset(events.columns):
        for player_id, player in zip(events['player_id'], events['player']):
            parsed_id = _safe_int(player_id)
            if parsed_id != -1 and pd.notna(player):
                player_names[parsed_id] = object_name(player)

    rows: list[dict[str, object]] = []
    for shot_index in shot_indices:
        shot = events.iloc[shot_index]
        xg = shot.get('shot_statsbomb_xg')
        if pd.isna(xg):
            continue

        possession = shot['possession']
        team_name = team_names.iloc[shot_index]
        mask = (
            events['possession'].eq(possession)
            & team_names.eq(team_name)
            & (events.index <= shot_index)
        )
        indices = np.flatnonzero(mask.to_numpy())[-sequence_length:]
        if len(indices) == 0:
            continue
        chunk = events.iloc[indices]

        player_ids = [
            _safe_int(value)
            for value in chunk.get(
                'player_id',
                pd.Series(-1, index=chunk.index),
            )
        ]
        event_ids = [str(value) for value in chunk.get('id', chunk.index)]
        chunk_types = chunk['type'].map(object_name).astype(str).tolist()
        outcomes = [event_outcome(row) for _, row in chunk.iterrows()]
        times = chunk['_time_seconds'].to_numpy(np.float32)
        deltas = np.zeros(len(chunk), dtype=np.float32)
        if len(chunk) > 1:
            deltas[1:] = np.maximum(0.0, np.diff(times))

        starts = [parse_location(value) for value in chunk.get('location')]
        ends = [event_end_location(row) for _, row in chunk.iterrows()]
        coordinates = []
        for (start_x, start_y), (end_x, end_y) in zip(starts, ends):
            if not np.isfinite(start_x):
                start_x = end_x
            if not np.isfinite(start_y):
                start_y = end_y
            if not np.isfinite(end_x):
                end_x = start_x
            if not np.isfinite(end_y):
                end_y = start_y
            values = [start_x, start_y, end_x, end_y]
            coordinates.append([
                float(np.clip(value, 0.0, limit)) if np.isfinite(value) else 0.0
                for value, limit in zip(values, [PITCH_X, PITCH_Y, PITCH_X, PITCH_Y])
            ])

        rows.append(
            {
                'match_id': int(match_id),
                'match_date': str(match_date),
                'shot_event_index': int(shot_index),
                'shot_event_id': event_ids[-1],
                'possession': _safe_int(possession),
                'team_name': team_name,
                'shooter_id': player_ids[-1],
                'xg': float(np.clip(float(xg), 0.0, 1.0)),
                'player_id': player_ids,
                'event_id': event_ids,
                'type_name': chunk_types,
                'outcome_name': outcomes,
                'coordinates': coordinates,
                'delta_seconds': deltas.astype(float).tolist(),
                'is_shot': [False] * (len(chunk) - 1) + [True],
                'n_events': int(len(chunk)),
            }
        )
    return rows, player_names


def _nearest_date_boundary(
    date_counts: pd.Series,
    target_matches: int,
    minimum_index: int = 0,
) -> int:
    cumulative = date_counts.cumsum().to_numpy()
    candidates = np.arange(len(cumulative))
    candidates = candidates[candidates >= minimum_index]
    return int(candidates[np.argmin(np.abs(cumulative[candidates] - target_matches))])


def temporal_match_split(
    matches: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    required = {'match_id', 'match_date'}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise RuntimeError(f'Match columns missing: {missing}')

    manifest = matches[['match_id', 'match_date']].drop_duplicates().copy()
    manifest['match_date'] = pd.to_datetime(manifest['match_date'])
    manifest = manifest.sort_values(['match_date', 'match_id']).reset_index(drop=True)
    counts = manifest.groupby('match_date', sort=True).size()
    n_matches = len(manifest)
    train_target = round(train_fraction * n_matches)
    validation_target = round((train_fraction + validation_fraction) * n_matches)
    train_boundary = _nearest_date_boundary(counts, train_target)
    validation_boundary = _nearest_date_boundary(
        counts,
        validation_target,
        minimum_index=train_boundary + 1,
    )
    dates = counts.index
    train_end = dates[train_boundary]
    validation_end = dates[validation_boundary]
    manifest['split'] = np.select(
        [
            manifest['match_date'].le(train_end),
            manifest['match_date'].le(validation_end),
        ],
        ['train', 'validation'],
        default='test',
    )
    manifest['match_date'] = manifest['match_date'].dt.strftime('%Y-%m-%d')
    return manifest


def build_sequence_dataset(
    cache_dir: Path,
    output_dir: Path,
    *,
    competition_id: int = 38,
    season_id: int = 318,
    sequence_length: int = 20,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    matches_path = cache_dir / f'matches_c{competition_id}_s{season_id}.parquet'
    matches = pd.read_parquet(matches_path)
    manifest = temporal_match_split(
        matches,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    split_by_match = manifest.set_index('match_id')['split'].to_dict()
    date_by_match = manifest.set_index('match_id')['match_date'].to_dict()

    rows: list[dict[str, object]] = []
    player_names: dict[int, str] = {}
    for events_path in sorted(cache_dir.glob('events_m*.parquet')):
        match_id = int(events_path.stem.split('m')[-1])
        if match_id not in split_by_match:
            continue
        match_rows, names = build_match_sequences(
            pd.read_parquet(events_path),
            match_id=match_id,
            match_date=date_by_match[match_id],
            sequence_length=sequence_length,
        )
        for row in match_rows:
            row['split'] = split_by_match[match_id]
        rows.extend(match_rows)
        player_names.update(names)

    sequences = pd.DataFrame(rows)
    sequences.insert(0, 'shot_seq_id', np.arange(len(sequences), dtype=np.int64))
    sequences = sequences.sort_values(
        ['match_date', 'match_id', 'shot_event_index'],
    ).reset_index(drop=True)
    sequences['shot_seq_id'] = np.arange(len(sequences), dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    sequences.to_parquet(output_dir / 'sequences_raw.parquet', index=False)
    manifest.to_parquet(output_dir / 'split_manifest.parquet', index=False)
    manifest.to_csv(output_dir / 'split_manifest.csv', index=False)
    (output_dir / 'player_id_to_name.json').write_text(
        json.dumps(
            {str(key): value for key, value in sorted(player_names.items())},
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    return sequences


def build_external_sequence_dataset(
    cache_dir: Path,
    output_dir: Path,
    *,
    competition_id: int,
    season_id: int,
    sequence_length: int = 20,
) -> pd.DataFrame:
    '''Build an external holdout from the event files available in a cache.'''
    matches_path = cache_dir / f'matches_c{competition_id}_s{season_id}.parquet'
    matches = pd.read_parquet(matches_path)
    event_paths = sorted(cache_dir.glob('events_m*.parquet'))
    if not event_paths:
        raise RuntimeError(f'No event files found in {cache_dir}.')
    event_match_ids = {
        int(path.stem.removeprefix('events_m')) for path in event_paths
    }
    manifest_columns = [
        column
        for column in [
            'match_id',
            'match_date',
            'match_week',
            'home_team',
            'away_team',
            'home_score',
            'away_score',
            'match_status',
            'play_status',
            'collection_status',
        ]
        if column in matches
    ]
    manifest = matches[
        matches['match_id'].isin(event_match_ids)
    ][manifest_columns].drop_duplicates('match_id').copy()
    missing_metadata = event_match_ids - set(manifest['match_id'].astype(int))
    if missing_metadata:
        raise RuntimeError(
            f'No match metadata for event files: {sorted(missing_metadata)}'
        )
    manifest['match_date'] = pd.to_datetime(manifest['match_date'])
    manifest = manifest.sort_values(
        ['match_date', 'match_id']
    ).reset_index(drop=True)
    manifest['match_date'] = manifest['match_date'].dt.strftime('%Y-%m-%d')
    manifest['split'] = 'external_test'
    date_by_match = manifest.set_index('match_id')['match_date'].to_dict()

    rows: list[dict[str, object]] = []
    player_names: dict[int, str] = {}
    for events_path in event_paths:
        match_id = int(events_path.stem.removeprefix('events_m'))
        match_rows, names = build_match_sequences(
            pd.read_parquet(events_path),
            match_id=match_id,
            match_date=date_by_match[match_id],
            sequence_length=sequence_length,
        )
        for row in match_rows:
            row['split'] = 'external_test'
        rows.extend(match_rows)
        player_names.update(names)

    sequences = pd.DataFrame(rows)
    if sequences.empty:
        raise RuntimeError('No shot sequences were built for external holdout.')
    sequences = sequences.sort_values(
        ['match_date', 'match_id', 'shot_event_index'],
    ).reset_index(drop=True)
    sequences.insert(0, 'shot_seq_id', np.arange(len(sequences), dtype=np.int64))

    output_dir.mkdir(parents=True, exist_ok=True)
    sequences.to_parquet(output_dir / 'sequences_raw.parquet', index=False)
    manifest.to_parquet(output_dir / 'match_manifest.parquet', index=False)
    manifest.to_csv(output_dir / 'match_manifest.csv', index=False)
    (output_dir / 'player_id_to_name.json').write_text(
        json.dumps(
            {str(key): value for key, value in sorted(player_names.items())},
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    return sequences


def split_rows(
    sequences: pd.DataFrame,
    names: Iterable[str] = ('train', 'validation', 'test'),
) -> dict[str, pd.DataFrame]:
    return {
        name: sequences.loc[sequences['split'].eq(name)].reset_index(drop=True)
        for name in names
    }
