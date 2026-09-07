from pathlib import Path

import pandas as pd
import torch

from football_attention.config import ExperimentConfig
from football_attention.sequence_baselines import (
    DeepSetsRegressor,
    GRURegressor,
)
from football_attention.variants import (
    ShotSequenceDataset,
    collate_sequences,
    fit_vocabularies,
)


def baseline_batch() -> tuple[dict[str, torch.Tensor], object, ExperimentConfig]:
    df = pd.DataFrame(
        [
            {
                'shot_seq_id': 1,
                'match_id': 100,
                'shooter_id': 12,
                'xg': 0.2,
                'player_id': [10, 11, 12],
                'type_name': ['Pass', 'Carry', 'Shot'],
                'outcome_name': ['Success', 'Success', 'Saved'],
                'coordinates': [
                    [10.0, 20.0, 30.0, 20.0],
                    [30.0, 20.0, 80.0, 40.0],
                    [102.0, 40.0, 120.0, 40.0],
                ],
                'delta_seconds': [0.0, 1.0, 1.0],
                'is_shot': [False, False, True],
            },
            {
                'shot_seq_id': 2,
                'match_id': 100,
                'shooter_id': 13,
                'xg': 0.1,
                'player_id': [13],
                'type_name': ['Shot'],
                'outcome_name': ['Saved'],
                'coordinates': [[100.0, 40.0, 120.0, 40.0]],
                'delta_seconds': [0.0],
                'is_shot': [True],
            },
        ]
    )
    vocabularies = fit_vocabularies(df)
    dataset = ShotSequenceDataset(
        df,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=5,
    )
    batch, _ = collate_sequences([dataset[0], dataset[1]])
    config = ExperimentConfig(
        project_dir=Path('.'),
        sequence_length=5,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )
    return batch, vocabularies, config


def reverse_valid_events(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = {key: value.clone() for key, value in batch.items()}
    sequence_keys = [
        'player',
        'event_type',
        'outcome',
        'continuous',
        'valid_mask',
        'original_player_id',
        'is_shot',
    ]
    positions = torch.nonzero(batch['valid_mask'][0], as_tuple=False).flatten()
    reversed_positions = positions.flip(0)
    for key in sequence_keys:
        output[key][0, positions] = batch[key][0, reversed_positions]
    return output


def test_sequence_baselines_ignore_player_ids_and_handle_empty() -> None:
    batch, vocabularies, config = baseline_batch()
    changed = dict(batch)
    changed['player'] = batch['player'].clone()
    changed['player'][batch['valid_mask']] = 1
    for model_class in [DeepSetsRegressor, GRURegressor]:
        model = model_class(vocabularies, config).eval()
        with torch.no_grad():
            prediction = model(batch)
            changed_prediction = model(changed)
        assert torch.isfinite(prediction).all()
        assert torch.allclose(prediction, changed_prediction)


def test_deepsets_is_permutation_invariant() -> None:
    batch, vocabularies, config = baseline_batch()
    model = DeepSetsRegressor(vocabularies, config).eval()
    reversed_batch = reverse_valid_events(batch)
    with torch.no_grad():
        original = model(batch)
        reversed_prediction = model(reversed_batch)
    assert torch.allclose(original, reversed_prediction, atol=1e-6)


def test_gru_uses_event_order() -> None:
    batch, vocabularies, config = baseline_batch()
    model = GRURegressor(vocabularies, config).eval()
    reversed_batch = reverse_valid_events(batch)
    with torch.no_grad():
        original = model(batch)
        reversed_prediction = model(reversed_batch)
    assert not torch.allclose(original[0], reversed_prediction[0])
