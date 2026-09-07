import pandas as pd
import pytest

from football_attention.data import (
    build_external_sequence_dataset,
    event_outcome,
    temporal_match_split,
)


@pytest.mark.parametrize(
    ('event_type', 'column', 'value', 'expected'),
    [
        ('Ball Receipt*', 'ball_receipt_outcome', 'Incomplete', 'Incomplete'),
        ('Dribble', 'dribble_outcome', 'Incomplete', 'Incomplete'),
        ('Dribble', 'dribble_outcome', 'Complete', 'Complete'),
        ('Duel', 'duel_outcome', 'Won', 'Won'),
        ('Goal Keeper', 'goalkeeper_outcome', 'Claim', 'Claim'),
        ('Interception', 'interception_outcome', 'Success In Play', 'Success In Play'),
        ('Pass', 'pass_outcome', 'Out', 'Out'),
        ('Shot', 'shot_outcome', 'Saved', 'Saved'),
        ('Substitution', 'substitution_outcome', 'Injury', 'Injury'),
    ],
)
def test_event_outcome_reads_event_specific_field(
    event_type: str,
    column: str,
    value: str,
    expected: str,
) -> None:
    row = pd.Series({'type': event_type, column: value})

    assert event_outcome(row) == expected


@pytest.mark.parametrize('event_type', ['Pass', 'Carry', 'Dribble'])
def test_event_outcome_maps_implicit_success(event_type: str) -> None:
    assert event_outcome(pd.Series({'type': event_type})) == 'Success'


def test_event_outcome_maps_other_missing_values_to_na() -> None:
    assert event_outcome(pd.Series({'type': 'Ball Receipt*'})) == 'NA'


def test_event_outcome_prefers_specific_field_over_generic_outcome() -> None:
    row = pd.Series(
        {
            'type': 'Dribble',
            'dribble_outcome': 'Incomplete',
            'outcome': 'Generic value',
        }
    )

    assert event_outcome(row) == 'Incomplete'


def test_temporal_split_keeps_match_dates_together() -> None:
    matches = pd.DataFrame(
        {
            'match_id': range(12),
            'match_date': [
                '2025-01-01',
                '2025-01-01',
                '2025-01-08',
                '2025-01-08',
                '2025-01-15',
                '2025-01-15',
                '2025-01-22',
                '2025-01-22',
                '2025-01-29',
                '2025-01-29',
                '2025-02-05',
                '2025-02-05',
            ],
        }
    )
    manifest = temporal_match_split(
        matches,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    assert manifest.groupby('match_date')['split'].nunique().max() == 1
    date_ranges = manifest.groupby('split')['match_date'].agg(['min', 'max'])
    assert date_ranges.loc['train', 'max'] < date_ranges.loc['validation', 'min']
    assert date_ranges.loc['validation', 'max'] < date_ranges.loc['test', 'min']


def test_external_dataset_uses_only_matches_with_event_files(tmp_path) -> None:
    cache_dir = tmp_path / 'cache'
    output_dir = tmp_path / 'prepared'
    cache_dir.mkdir()
    matches = pd.DataFrame(
        {
            'match_id': [101, 102],
            'match_date': ['2026-07-24', '2026-07-25'],
            'match_week': [1, 1],
            'home_team': ['A', 'C'],
            'away_team': ['B', 'D'],
        }
    )
    matches.to_parquet(cache_dir / 'matches_c38_s351.parquet', index=False)
    events = pd.DataFrame(
        {
            'id': ['shot-1'],
            'index': [1],
            'period': [1],
            'timestamp': ['00:01:00.000'],
            'type': ['Shot'],
            'team': ['A'],
            'possession': [1],
            'player_id': [7],
            'player': ['Player'],
            'location': [[100.0, 40.0]],
            'shot_statsbomb_xg': [0.2],
            'shot_outcome': ['Saved'],
        }
    )
    events.to_parquet(cache_dir / 'events_m101.parquet', index=False)
    sequences = build_external_sequence_dataset(
        cache_dir,
        output_dir,
        competition_id=38,
        season_id=351,
    )
    assert sequences['match_id'].tolist() == [101]
    assert sequences['split'].tolist() == ['external_test']
    manifest = pd.read_parquet(output_dir / 'match_manifest.parquet')
    assert manifest['match_id'].tolist() == [101]
