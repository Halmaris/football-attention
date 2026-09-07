from __future__ import annotations
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import object_name, timestamp_seconds


def _parse_nested(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    for parser in [json.loads, ast.literal_eval]:
        try:
            parsed = parser(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return []


def _match_end_seconds(events: pd.DataFrame) -> float:
    timestamps = events.get('timestamp', pd.Series(dtype=object)).map(
        timestamp_seconds
    )
    finite = timestamps[np.isfinite(timestamps)]
    return float(finite.max()) if len(finite) else 90 * 60.0


def player_minutes(
    cache_dir: Path,
    test_match_ids: set[int],
) -> pd.DataFrame:
    minutes: dict[int, float] = {}
    for lineup_path in sorted(cache_dir.glob('lineups_m*.parquet')):
        match_id = int(lineup_path.stem.split('m')[-1])
        if match_id not in test_match_ids:
            continue
        events_path = cache_dir / f'events_m{match_id}.parquet'
        if not events_path.exists():
            continue
        match_end = _match_end_seconds(pd.read_parquet(events_path))
        lineup = pd.read_parquet(lineup_path)
        for _, player in lineup.iterrows():
            player_id = int(player['player_id'])
            seconds = 0.0
            for stint in _parse_nested(player.get('positions')):
                start = timestamp_seconds(stint.get('from'))
                end = timestamp_seconds(stint.get('to'))
                if not np.isfinite(start):
                    continue
                if not np.isfinite(end):
                    end = match_end
                seconds += max(0.0, end - start)
            minutes[player_id] = minutes.get(player_id, 0.0) + seconds / 60.0
    return pd.DataFrame(
        [{'player_id': player_id, 'minutes': value} for player_id, value in minutes.items()],
        columns=['player_id', 'minutes'],
    )


def event_outcomes(
    cache_dir: Path,
    test_match_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_rows = []
    team_rows = []
    for events_path in sorted(cache_dir.glob('events_m*.parquet')):
        match_id = int(events_path.stem.split('m')[-1])
        if match_id not in test_match_ids:
            continue
        events = pd.read_parquet(events_path)
        player_ids = events.get('player_id', pd.Series(-1, index=events.index))
        types = events['type'].map(object_name).astype(str)
        shot_outcomes = events.get(
            'shot_outcome',
            pd.Series('', index=events.index),
        ).map(object_name).astype(str)
        goals = types.eq('Shot') & shot_outcomes.eq('Goal')
        assists = events.get(
            'pass_goal_assist',
            pd.Series(False, index=events.index),
        ).eq(True)
        for player_id in set(player_ids[goals | assists].dropna().astype(int)):
            player_mask = player_ids.fillna(-1).astype(int).eq(player_id)
            player_rows.append(
                {
                    'player_id': int(player_id),
                    'goals': int((goals & player_mask).sum()),
                    'assists': int((assists & player_mask).sum()),
                }
            )
        shots = events.loc[types.eq('Shot')].copy()
        if not shots.empty:
            shots['team_name'] = shots['team'].map(object_name).astype(str)
            shots['xg'] = pd.to_numeric(shots['shot_statsbomb_xg'], errors='coerce')
            team_rows.extend(
                shots.groupby('team_name', as_index=False)['xg']
                .sum()
                .to_dict('records')
            )
    players = pd.DataFrame(player_rows)
    if players.empty:
        players = pd.DataFrame(columns=['player_id', 'goals', 'assists'])
    else:
        players = players.groupby('player_id', as_index=False)[['goals', 'assists']].sum()
    players['goals_plus_assists'] = players.get('goals', 0) + players.get('assists', 0)
    teams = pd.DataFrame(team_rows)
    if teams.empty:
        teams = pd.DataFrame(columns=['team_name', 'team_xg'])
    else:
        teams = (
            teams.groupby('team_name', as_index=False)['xg']
            .sum()
            .rename(columns={'xg': 'team_xg'})
        )
    return players, teams


def build_exposure_table(
    test_sequences: pd.DataFrame,
    cache_dir: Path,
) -> pd.DataFrame:
    sequence_rows = []
    event_rows = []
    player_teams = []
    for _, sequence in test_sequences.iterrows():
        players = [int(value) for value in sequence['player_id'] if int(value) != -1]
        for player_id in set(players):
            sequence_rows.append(
                {
                    'player_id': player_id,
                    'shot_seq_id': int(sequence['shot_seq_id']),
                    'xg': float(sequence['xg']),
                }
            )
            player_teams.append(
                {'player_id': player_id, 'team_name': sequence['team_name']}
            )
        event_rows.extend({'player_id': player_id} for player_id in players)

    sequence_frame = pd.DataFrame(sequence_rows)
    event_frame = pd.DataFrame(event_rows)
    exposure = (
        sequence_frame.groupby('player_id', as_index=False)
        .agg(
            sequence_count_exposure=('shot_seq_id', 'nunique'),
            sum_sequence_xg_exposure=('xg', 'sum'),
        )
        .merge(
            event_frame.groupby('player_id').size().rename('event_count_exposure'),
            on='player_id',
            how='outer',
        )
    )

    test_match_ids = set(test_sequences['match_id'].astype(int))
    minutes = player_minutes(cache_dir, test_match_ids)
    outcomes, team_xg = event_outcomes(cache_dir, test_match_ids)
    teams = pd.DataFrame(player_teams).drop_duplicates()
    player_team_xg = (
        teams.merge(team_xg, on='team_name', how='left')
        .groupby('player_id', as_index=False)['team_xg']
        .sum(min_count=1)
    )
    result = (
        exposure.merge(minutes, on='player_id', how='left')
        .merge(outcomes, on='player_id', how='left')
        .merge(player_team_xg, on='player_id', how='left')
    )
    count_columns = ['goals', 'assists', 'goals_plus_assists']
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    return result
