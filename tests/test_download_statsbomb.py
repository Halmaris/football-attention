import pandas as pd
import pytest

from football_attention.download import (
    build_metadata_report,
    select_competition,
    select_matches,
)


def test_select_competition_resolves_expected_season() -> None:
    df = pd.DataFrame(
        {
            'competition_id': [38, 38],
            'competition_name': ['Ekstraklasa', 'Ekstraklasa'],
            'season_id': [318, 400],
            'season_name': ['2025/2026', '2026/2027'],
        }
    )
    selected = select_competition(
        df,
        competition_id=38,
        season_id=318,
        expected_season='2026/2027',
        auto_season=True,
    )
    assert selected.iloc[0]['season_id'] == 400


def test_select_matches_filters_complete_round_block() -> None:
    df = pd.DataFrame(
        {
            'match_id': [4, 2, 1, 3, 5],
            'match_date': pd.to_datetime(
                ['2026-08-01', '2026-07-20', '2026-07-19', '2026-07-27',
                 '2026-08-08']
            ),
            'match_week': [3, 1, 1, 2, 5],
        }
    )
    selected = select_matches(
        df,
        match_week_start=1,
        match_week_end=4,
        limit=None,
        min_matches=4,
    )
    assert selected['match_id'].tolist() == [1, 2, 3, 4]


def test_select_matches_rejects_incomplete_block() -> None:
    df = pd.DataFrame(
        {
            'match_id': [1],
            'match_date': ['2026-07-19'],
            'match_week': [1],
        }
    )
    with pytest.raises(RuntimeError, match='expected at least 30'):
        select_matches(
            df,
            match_week_start=1,
            match_week_end=4,
            limit=None,
            min_matches=30,
        )


def test_select_matches_filters_available_matches_before_cutoff() -> None:
    df = pd.DataFrame(
        {
            'match_id': [1, 2, 3, 4],
            'match_date': [
                '2026-08-21',
                '2026-08-22',
                '2026-08-23',
                '2026-09-03',
            ],
            'match_week': [5, 5, 5, 5],
            'match_status': [
                'available',
                'scheduled',
                'available',
                'available',
            ],
        }
    )
    selected = select_matches(
        df,
        match_week_start=5,
        match_week_end=5,
        match_date_end='2026-08-23',
        available_only=True,
        limit=None,
        min_matches=2,
    )
    assert selected['match_id'].tolist() == [1, 3]


def test_build_metadata_report_counts_availability() -> None:
    df = pd.DataFrame(
        {
            'match_id': [1, 2, 3],
            'match_week': [6, 6, 6],
            'match_date': ['2026-08-28', '2026-08-29', '2026-08-30'],
            'home_team': ['A', 'B', 'C'],
            'away_team': ['D', 'E', 'F'],
            'match_status': ['available', 'scheduled', 'AVAILABLE'],
        }
    )
    report = build_metadata_report(
        df,
        competition_id=38,
        competition_name='Ekstraklasa',
        season_id=351,
        season_name='2026/2027',
    )
    assert report['match_count'] == 3
    assert report['available_count'] == 2
    assert report['status_counts'] == {'available': 2, 'scheduled': 1}
    assert [match['match_id'] for match in report['matches']] == [1, 2, 3]
