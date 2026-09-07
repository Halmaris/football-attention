from pathlib import Path

import pandas as pd
import pytest

from football_attention.tasks.holdout import (
    attach_match_id,
    load_external,
    paired_mse_difference,
    prediction_path,
)


def test_prediction_path_separates_model_and_seed() -> None:
    assert prediction_path(Path('/tmp/results'), 'gru', 42) == Path(
        '/tmp/results/predictions/gru/seed_42.parquet'
    )


def test_attach_match_id_does_not_duplicate_existing_column() -> None:
    values = pd.DataFrame({'shot_seq_id': [1], 'match_id': [10]})
    match_map = pd.DataFrame({'shot_seq_id': [1], 'match_id': [10]})
    result = attach_match_id(values, match_map)
    assert result.columns.tolist() == ['shot_seq_id', 'match_id']


def test_load_external_uses_requested_prepared_directory(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / 'data' / 'prepared_external_custom'
    prepared.mkdir(parents=True)
    expected = pd.DataFrame({'shot_seq_id': [1], 'match_id': [10]})
    expected.to_parquet(prepared / 'sequences_raw.parquet', index=False)
    result = load_external(tmp_path, 'prepared_external_custom')
    pd.testing.assert_frame_equal(result, expected)


def test_paired_mse_difference_pairs_neural_seeds_and_matches() -> None:
    rows = []
    for seed in [42, 43]:
        for shot_seq_id, match_id, target in [
            (1, 10, 0.2),
            (2, 20, 0.4),
        ]:
            common = {
                'seed': seed,
                'shot_seq_id': shot_seq_id,
                'match_id': match_id,
                'target_xg': target,
                'n_pre_shot_events': 3,
                'length_bin': '3-5',
                'team_status': 'known_team',
                'player_status': 'known_players_only',
            }
            rows.append(
                {
                    **common,
                    'model_type': 'gru_attention_faithful',
                    'prediction': target + 0.1,
                }
            )
            rows.append(
                {
                    **common,
                    'model_type': 'gru_attention_predictive',
                    'prediction': target + 0.2,
                }
            )
    result = paired_mse_difference(
        pd.DataFrame(rows),
        'gru_attention_faithful',
        'gru_attention_predictive',
        'all',
        n_bootstrap=100,
        seed=2026,
    )
    assert result['difference_mean'] == pytest.approx(-0.03)
    assert result['n_seeds'] == 2
    assert result['n_matches'] == 2
