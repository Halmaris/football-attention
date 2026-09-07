from __future__ import annotations
import numpy as np
import pandas as pd


LENGTH_LABELS = ('0', '1-2', '3-5', '6-10', '11-20')


def length_bin(n_events: pd.Series) -> pd.Series:
    return pd.cut(
        n_events,
        bins=[-1, 0, 2, 5, 10, 20],
        labels=LENGTH_LABELS,
    ).astype(str)


def calibration_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(predictions)), predictions])
    intercept, slope = np.linalg.lstsq(
        design,
        targets,
        rcond=None,
    )[0]
    return float(intercept), float(slope)


def pre_shot_values(row: pd.Series, column: str) -> list[object]:
    return [
        value
        for value, is_shot in zip(row[column], row['is_shot'])
        if not bool(is_shot)
    ]


def novelty_table(
    external: pd.DataFrame,
    historical_train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    known_players = {
        int(value)
        for values in historical_train['player_id']
        for value in values
        if int(value) != -1
    }
    known_teams = set(historical_train['team_name'].astype(str))
    known_types = {
        str(value)
        for values in historical_train['type_name']
        for value in values
    }
    known_outcomes = {
        str(value)
        for values in historical_train['outcome_name']
        for value in values
    }

    rows = []
    unseen_types: set[str] = set()
    unseen_outcomes: set[str] = set()
    unseen_players: set[int] = set()
    for _, row in external.iterrows():
        players = {
            int(value)
            for value in pre_shot_values(row, 'player_id')
            if int(value) != -1
        }
        types = {
            str(value) for value in pre_shot_values(row, 'type_name')
        }
        outcomes = {
            str(value) for value in pre_shot_values(row, 'outcome_name')
        }
        new_players = players - known_players
        new_types = types - known_types
        new_outcomes = outcomes - known_outcomes
        unseen_players.update(new_players)
        unseen_types.update(new_types)
        unseen_outcomes.update(new_outcomes)
        rows.append(
            {
                'shot_seq_id': int(row['shot_seq_id']),
                'match_id': int(row['match_id']),
                'n_pre_shot_events': int(sum(not value for value in row['is_shot'])),
                'team_status': (
                    'known_team'
                    if str(row['team_name']) in known_teams
                    else 'new_team'
                ),
                'player_status': (
                    'contains_new_player'
                    if new_players
                    else 'known_players_only'
                ),
                'new_pre_shot_player_count': len(new_players),
                'unknown_event_type_count': len(new_types),
                'unknown_outcome_count': len(new_outcomes),
            }
        )
    table = pd.DataFrame(rows)
    table['length_bin'] = length_bin(table['n_pre_shot_events'])
    new_teams = sorted(set(external['team_name'].astype(str)) - known_teams)
    summary = {
        'known_team_count': len(known_teams),
        'external_team_count': int(external['team_name'].nunique()),
        'new_teams': new_teams,
        'unseen_player_count': len(unseen_players),
        'unseen_event_types': sorted(unseen_types),
        'unseen_outcomes': sorted(unseen_outcomes),
        'sequences_with_new_player': int(
            table['player_status'].eq('contains_new_player').sum()
        ),
        'sequences_from_new_team': int(
            table['team_status'].eq('new_team').sum()
        ),
    }
    return table, summary

from football_attention.train import select_device
