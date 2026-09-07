import pandas as pd
import pytest

from football_attention.baselines import last_event_features, sequence_features
from football_attention.plotting.boosting import feature_group, match_bootstrap


def sequence_row(first_type: str = 'Pass') -> pd.Series:
    return pd.Series(
        {
            'shot_seq_id': 1,
            'player_id': [10, 11, 12],
            'type_name': [first_type, 'Carry', 'Shot'],
            'outcome_name': ['Complete', 'NA', 'Saved'],
            'coordinates': [
                [10.0, 20.0, 30.0, 20.0],
                [30.0, 20.0, 80.0, 40.0],
                [102.0, 40.0, 120.0, 40.0],
            ],
            'delta_seconds': [0.0, 1.5, 0.5],
            'is_shot': [False, False, True],
        }
    )


def test_last_event_features_exclude_shot_and_sequence_summaries() -> None:
    features = last_event_features(sequence_row())
    assert features['last_type=Carry'] == 1.0
    assert features['last_outcome=NA'] == 1.0
    assert features['last_start_x'] == pytest.approx(30 / 120)
    assert features['last_end_x'] == pytest.approx(80 / 120)
    assert set(features) == {
        'last_start_x',
        'last_start_y',
        'last_end_x',
        'last_end_y',
        'last_delta_time',
        'last_type=Carry',
        'last_outcome=NA',
    }


def test_last_event_features_ignore_earlier_events() -> None:
    first = sequence_row('Pass')
    second = sequence_row('Pressure')
    assert last_event_features(first) == last_event_features(second)
    assert sequence_features(first, 'pre_shot') != sequence_features(
        second,
        'pre_shot',
    )


def test_last_event_features_require_visible_buildup() -> None:
    row = sequence_row()
    for column in ('player_id', 'type_name', 'outcome_name', 'coordinates'):
        row[column] = [row[column][-1]]
    row['delta_seconds'] = [0.0]
    row['is_shot'] = [True]
    with pytest.raises(ValueError, match='no visible pre-shot event'):
        last_event_features(row)


@pytest.mark.parametrize(
    ('name', 'group'),
    [
        ('last_type=Carry', 'Final-event features'),
        ('start_x_mean', 'Spatial summaries'),
        ('delta_time_std', 'Temporal summaries'),
        ('type_count=Pass', 'Event-type counts'),
        ('outcome_count=Complete', 'Outcome counts'),
        ('n_tokens', 'Sequence scale'),
    ],
)
def test_boosting_feature_groups(name: str, group: str) -> None:
    assert feature_group(name) == group


def test_match_bootstrap_is_reproducible() -> None:
    values = [0.1, 0.2, -0.1, 0.3]
    matches = [1, 1, 2, 2]
    first = match_bootstrap(values, matches, samples=100, seed=7)
    second = match_bootstrap(values, matches, samples=100, seed=7)
    assert first == second
