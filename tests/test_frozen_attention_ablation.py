import pandas as pd
import pytest

from football_attention.tasks.contrasts import paired_deletion_rows, paired_models


def test_paired_models_preserves_seed_and_match_pairing() -> None:
    rows = []
    for seed in [42, 43]:
        for shot_seq_id, match_id in [(1, 10), (2, 20)]:
            rows.extend(
                [
                    {
                        'model_type': 'gru_attention_faithful',
                        'seed': seed,
                        'shot_seq_id': shot_seq_id,
                        'match_id': match_id,
                        'rho': 0.7,
                    },
                    {
                        'model_type': 'gru_attention_predictive',
                        'seed': seed,
                        'shot_seq_id': shot_seq_id,
                        'match_id': match_id,
                        'rho': 0.5,
                    },
                ]
            )
    result = paired_models(
        pd.DataFrame(rows),
        value_column='rho',
        keys=[],
        label='rho',
        n_bootstrap=100,
    )
    assert result['difference_mean'] == pytest.approx(0.2)
    assert result['n_seeds'] == 2
    assert result['n_matches'] == 2


def test_paired_deletion_rows_compares_attention_random_contrasts() -> None:
    rows = []
    deleted_predictions = {
        'gru_attention_faithful': {
            'attention': 0.4**0.5,
            'random': 0.1**0.5,
            'occlusion': 0.3**0.5,
            'integrated_gradients': 0.3**0.5,
        },
        'gru_attention_predictive': {
            'attention': 0.25**0.5,
            'random': 0.1**0.5,
            'occlusion': 0.2**0.5,
            'integrated_gradients': 0.2**0.5,
        },
    }
    for model_type, methods in deleted_predictions.items():
        for seed in [42, 43]:
            for shot_seq_id, match_id in [(1, 10), (2, 20)]:
                for method, deleted_prediction in methods.items():
                    rows.append(
                        {
                            'model_type': model_type,
                            'seed': seed,
                            'shot_seq_id': shot_seq_id,
                            'match_id': match_id,
                            'method': method,
                            'k': 1,
                            'target_xg': 0.0,
                            'base_prediction': 0.0,
                            'deleted_prediction': deleted_prediction,
                        }
                    )
    results = paired_deletion_rows(
        pd.DataFrame(rows),
        n_bootstrap=100,
    )
    contrast = next(
        row
        for row in results
        if row['metric'] == 'deletion_attention_minus_random:k=1'
    )
    assert contrast['difference_mean'] == pytest.approx(0.15)
