import numpy as np
import pandas as pd

from football_attention.bootstrap import (
    seed_match_bootstrap_distribution,
    seed_match_bootstrap_mean,
)


def test_seed_match_bootstrap_is_deterministic_and_paired() -> None:
    rows = []
    for seed in [42, 43]:
        for match_id in [1, 2, 3]:
            for sequence in [0, 1]:
                rows.append(
                    {
                        'seed': seed,
                        'match_id': match_id,
                        'shot_seq_id': match_id * 10 + sequence,
                        'difference': seed - 42 + match_id / 10,
                    }
                )
    df = pd.DataFrame(rows)
    first = seed_match_bootstrap_mean(df, n_bootstrap=200, seed=7)
    second = seed_match_bootstrap_mean(df, n_bootstrap=200, seed=7)
    assert first == second
    assert np.isclose(first['difference_mean'], df['difference'].mean())
    assert first['n_seeds'] == 2
    assert first['n_matches'] == 3
    assert first['n_sequences'] == 6

    estimate, samples, metadata = seed_match_bootstrap_distribution(
        df,
        n_bootstrap=200,
        seed=7,
    )
    assert np.isclose(estimate, first['difference_mean'])
    assert np.isclose(np.quantile(samples, 0.025), first['ci_025'])
    assert metadata['n_rows'] == len(df)


def test_seed_match_bootstrap_rejects_incomplete_schema() -> None:
    df = pd.DataFrame({'seed': [42], 'difference': [0.1]})
    try:
        seed_match_bootstrap_mean(df, n_bootstrap=10)
    except ValueError as error:
        assert 'Missing bootstrap columns' in str(error)
    else:
        raise AssertionError('Expected a schema validation error')
