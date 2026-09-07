import numpy as np
import pandas as pd

from football_attention.tasks.order_summary import sequence_length_table


def test_sequence_length_bins_use_visible_pre_shot_events() -> None:
    df = pd.DataFrame(
        {
            'shot_seq_id': [1, 2, 3, 4, 5, 6],
            'match_id': [10] * 6,
            'split': ['test'] * 6,
            'is_shot': [
                np.array([True]),
                np.array([False, True]),
                np.array([False] * 3 + [True]),
                np.array([False] * 6 + [True]),
                np.array([False] * 11 + [True]),
                np.array([False] * 25 + [True]),
            ],
        }
    )
    lengths = sequence_length_table(df)
    assert lengths['length_bin'].tolist() == [
        '0',
        '1-2',
        '3-5',
        '6-10',
        '11-20',
        '11-20',
    ]
    assert lengths['n_pre_shot_events'].tolist()[-1] == 20
    assert lengths['n_pre_shot_events_raw'].tolist()[-1] == 25
