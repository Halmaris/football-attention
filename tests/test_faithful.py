from pathlib import Path

import pandas as pd
import torch

from football_attention.config import ExperimentConfig
from football_attention.faithful_eval import (
    gradient_x_input,
    integrated_gradients,
    permutation_shapley,
    recency_importance,
    top_k_mask,
)
from football_attention.faithful_model import (
    FaithfulPreShotGRUAttention,
    FaithfulPreShotTransformer,
)
from football_attention.faithful_train import (
    apply_event_dropout,
    sampled_occlusion_alignment_loss,
)
from football_attention.variants import (
    ShotSequenceDataset,
    collate_sequences,
    fit_vocabularies,
)


def faithful_batch() -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    object,
    ExperimentConfig,
]:
    rows = pd.DataFrame(
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
    vocabularies = fit_vocabularies(rows)
    dataset = ShotSequenceDataset(
        rows,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=5,
    )
    batch, targets = collate_sequences([dataset[0], dataset[1]])
    config = ExperimentConfig(
        project_dir=Path('.'),
        sequence_length=5,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        integrated_gradients_steps=2,
    )
    return batch, targets, vocabularies, config


def reverse_first_sequence(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    reversed_batch = {key: value.clone() for key, value in batch.items()}
    positions = torch.nonzero(batch['valid_mask'][0]).flatten()
    for key in [
        'player',
        'event_type',
        'outcome',
        'continuous',
        'valid_mask',
        'original_player_id',
        'is_shot',
    ]:
        reversed_batch[key][0, positions] = batch[key][0, positions.flip(0)]
    return reversed_batch


def test_bottleneck_handles_empty_pre_shot_sequence() -> None:
    batch, _, vocabularies, config = faithful_batch()
    model = FaithfulPreShotTransformer(vocabularies, 5, config)
    predictions, attention = model(batch, return_attention=True)
    assert torch.isfinite(predictions).all()
    assert torch.allclose(attention[0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(attention[1], torch.zeros(5))


def test_no_player_model_is_invariant_to_player_ids() -> None:
    batch, _, vocabularies, config = faithful_batch()
    model = FaithfulPreShotTransformer(
        vocabularies,
        5,
        config,
        use_player_id=False,
    ).eval()
    changed = dict(batch)
    changed['player'] = batch['player'].clone()
    changed['player'][batch['valid_mask']] = 1
    with torch.no_grad():
        original_prediction, original_attention = model(
            batch,
            return_attention=True,
        )
        changed_prediction, changed_attention = model(
            changed,
            return_attention=True,
        )
    assert torch.allclose(original_prediction, changed_prediction)
    assert torch.allclose(original_attention, changed_attention)


def test_event_dropout_keeps_one_observed_event() -> None:
    batch, _, _, _ = faithful_batch()
    dropped = apply_event_dropout(batch, 1.0)
    assert dropped['valid_mask'][0].sum() == 1
    assert dropped['valid_mask'][1].sum() == 0


def test_alignment_and_integrated_gradients_are_finite() -> None:
    batch, _, vocabularies, config = faithful_batch()
    model = FaithfulPreShotTransformer(vocabularies, 5, config)
    _, attention = model(batch, return_attention=True)
    loss = sampled_occlusion_alignment_loss(
        model,
        batch,
        attention,
        n_samples=2,
        temperature=0.01,
    )
    attributions = integrated_gradients(model, batch, steps=2)
    assert torch.isfinite(loss)
    assert torch.isfinite(attributions).all()
    assert torch.equal(attributions[1], torch.zeros(5))


def test_stronger_attribution_baselines_are_finite_and_local() -> None:
    batch, _, vocabularies, config = faithful_batch()
    torch.manual_seed(0)
    model = FaithfulPreShotGRUAttention(vocabularies, 5, config).eval()
    gradient = gradient_x_input(model, batch)
    shapley, shapley_se = permutation_shapley(
        model,
        batch,
        n_permutations=4,
        seed=42,
        coalition_batch_size=8,
    )
    repeated, _ = permutation_shapley(
        model,
        batch,
        n_permutations=4,
        seed=42,
        coalition_batch_size=8,
    )
    empty_batch = dict(batch)
    empty_batch['valid_mask'] = torch.zeros_like(batch['valid_mask'])
    with torch.no_grad():
        full_prediction = model(batch)
        empty_prediction = model(empty_batch)

    assert torch.isfinite(gradient).all()
    assert torch.isfinite(shapley).all()
    assert torch.isfinite(shapley_se).all()
    assert torch.equal(gradient[1], torch.zeros(5))
    assert torch.equal(shapley[1], torch.zeros(5))
    assert torch.allclose(shapley, repeated)
    assert torch.allclose(
        shapley.sum(dim=1),
        full_prediction - empty_prediction,
        atol=1e-6,
    )


def test_recency_importance_uses_observed_event_order() -> None:
    batch, _, _, _ = faithful_batch()
    recency = recency_importance(batch['valid_mask'])
    assert recency[0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert recency[1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_top_k_mask_never_deletes_padding() -> None:
    batch, _, _, _ = faithful_batch()
    importance = torch.arange(10, dtype=torch.float32).view(2, 5)
    deleted, counts = top_k_mask(batch['valid_mask'], importance, 3)
    assert counts.tolist() == [2, 0]
    assert deleted.sum() == 0


def test_gru_attention_is_finite_and_ignores_player_ids() -> None:
    batch, _, vocabularies, config = faithful_batch()
    model = FaithfulPreShotGRUAttention(vocabularies, 5, config).eval()
    changed = dict(batch)
    changed['player'] = batch['player'].clone()
    changed['player'][batch['valid_mask']] = 1
    with torch.no_grad():
        predictions, attention = model(batch, return_attention=True)
        changed_predictions, changed_attention = model(
            changed,
            return_attention=True,
        )
    assert torch.isfinite(predictions).all()
    assert torch.allclose(attention[0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(attention[1], torch.zeros(5))
    assert torch.allclose(predictions, changed_predictions)
    assert torch.allclose(attention, changed_attention)


def test_gru_attention_uses_order_and_supports_alignment() -> None:
    batch, _, vocabularies, config = faithful_batch()
    torch.manual_seed(0)
    model = FaithfulPreShotGRUAttention(vocabularies, 5, config).eval()
    reversed_batch = reverse_first_sequence(batch)
    with torch.no_grad():
        original, attention = model(batch, return_attention=True)
        reversed_prediction = model(reversed_batch)
    assert not torch.allclose(original[0], reversed_prediction[0])
    loss = sampled_occlusion_alignment_loss(
        model,
        batch,
        attention,
        n_samples=2,
        temperature=0.01,
    )
    attributions = integrated_gradients(model, batch, steps=2)
    assert torch.isfinite(loss)
    assert torch.isfinite(attributions).all()


def test_transformer_without_positions_is_permutation_invariant() -> None:
    batch, _, vocabularies, config = faithful_batch()
    model = FaithfulPreShotTransformer(
        vocabularies,
        5,
        config,
        use_player_id=False,
        use_position_encoding=False,
    ).eval()
    reversed_batch = reverse_first_sequence(batch)
    positions = torch.nonzero(batch['valid_mask'][0]).flatten()
    with torch.no_grad():
        prediction, attention = model(batch, return_attention=True)
        reversed_prediction, reversed_attention = model(
            reversed_batch,
            return_attention=True,
        )
    assert torch.allclose(prediction, reversed_prediction, atol=1e-6)
    assert torch.allclose(
        attention[0, positions],
        reversed_attention[0, positions.flip(0)],
        atol=1e-6,
    )
