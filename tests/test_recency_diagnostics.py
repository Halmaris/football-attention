import numpy as np
import pandas as pd

from football_attention.tasks.recency import sequence_rank_diagnostics


def test_sequence_rank_diagnostics_returns_paired_recency_contrast() -> None:
    tokens = pd.DataFrame(
        {
            'dataset': ['development_test'] * 3,
            'seed': [42] * 3,
            'shot_seq_id': [10] * 3,
            'match_id': [100] * 3,
            'attention': [0.1, 0.3, 0.2],
            'recency': [1.0, 2.0, 3.0],
            'occlusion_abs': [0.1, 0.3, 0.2],
            'ig_abs': [0.1, 0.3, 0.2],
            'permutation_shapley_abs': [0.1, 0.3, 0.2],
        }
    )

    result = sequence_rank_diagnostics(tokens).iloc[0]

    assert np.isclose(result['attention_vs_occlusion'], 1.0)
    assert np.isclose(result['recency_vs_occlusion'], 0.5)
    assert np.isclose(
        result['attention_minus_recency_vs_occlusion'],
        0.5,
    )


def test_sequence_rank_diagnostics_excludes_short_sequences() -> None:
    tokens = pd.DataFrame(
        {
            'dataset': ['development_test'] * 2,
            'seed': [42] * 2,
            'shot_seq_id': [10] * 2,
            'match_id': [100] * 2,
            'attention': [0.4, 0.6],
            'recency': [1.0, 2.0],
            'occlusion_abs': [0.2, 0.8],
            'ig_abs': [0.2, 0.8],
            'permutation_shapley_abs': [0.2, 0.8],
        }
    )

    result = sequence_rank_diagnostics(tokens).iloc[0]

    assert np.isnan(result['attention_vs_occlusion'])
    assert np.isnan(result['recency_vs_occlusion'])
    assert np.isnan(result['attention_minus_recency_vs_occlusion'])
