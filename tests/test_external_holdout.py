import numpy as np
import pandas as pd

from football_attention.tasks.common import calibration_metrics, novelty_table


def test_calibration_metrics_recovers_linear_relation() -> None:
    predictions = np.array([0.1, 0.2, 0.3])
    targets = 0.05 + 1.5 * predictions
    intercept, slope = calibration_metrics(targets, predictions)
    assert np.isclose(intercept, 0.05)
    assert np.isclose(slope, 1.5)


def test_novelty_table_uses_pre_shot_players_only() -> None:
    historical = pd.DataFrame(
        {
            'player_id': [[1, 9]],
            'type_name': [['Pass', 'Shot']],
            'outcome_name': [['Success', 'Saved']],
            'team_name': ['Known'],
        }
    )
    external = pd.DataFrame(
        {
            'shot_seq_id': [0, 1],
            'match_id': [10, 10],
            'player_id': [[1, 100], [2, 101]],
            'type_name': [['Pass', 'Shot'], ['Carry', 'Shot']],
            'outcome_name': [['Success', 'Saved'], ['Success', 'Saved']],
            'is_shot': [[False, True], [False, True]],
            'team_name': ['Known', 'New'],
        }
    )
    table, summary = novelty_table(external, historical)
    assert table['player_status'].tolist() == [
        'known_players_only',
        'contains_new_player',
    ]
    assert table['team_status'].tolist() == ['known_team', 'new_team']
    assert summary['unseen_player_count'] == 1
    assert summary['unseen_event_types'] == ['Carry']
