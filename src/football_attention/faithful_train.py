from __future__ import annotations
import copy
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.nn import functional as functional
from .config import ExperimentConfig
from .faithful_model import FaithfulModel, build_faithful_model
from .train import (
    PredictionMetrics,
    make_loader,
    move_batch,
    predict,
    prediction_metrics,
    select_device,
    set_seed,
)
from .variants import ShotSequenceDataset, Vocabularies


FAITHFUL_MODEL_TYPES = {
    'faithful',
    'faithful_no_player_id',
    'faithful_no_position',
    'gru_attention',
    'gru_attention_shuffled',
}


def apply_event_dropout(
    batch: dict[str, torch.Tensor],
    probability: float,
) -> dict[str, torch.Tensor]:
    if probability <= 0:
        return batch
    output = dict(batch)
    valid = batch['valid_mask']
    keep = valid & (torch.rand(valid.shape, device=valid.device) >= probability)
    lost_all = valid.any(dim=1) & ~keep.any(dim=1)
    if lost_all.any():
        positions = torch.arange(valid.shape[1], device=valid.device)
        last_valid = positions.masked_fill(~valid, -1).max(dim=1).values
        keep[lost_all, last_valid[lost_all]] = True
    output['valid_mask'] = keep
    return output


def sampled_occlusion_alignment_loss(
    model: FaithfulModel,
    batch: dict[str, torch.Tensor],
    attention: torch.Tensor,
    *,
    n_samples: int,
    temperature: float,
) -> torch.Tensor:
    valid = batch['valid_mask']
    n_samples = min(n_samples, valid.shape[1])
    counts = valid.sum(dim=1)
    useful_rows = counts >= 2
    if n_samples < 2 or not useful_rows.any():
        return attention.sum() * 0.0

    random_scores = torch.rand(valid.shape, device=valid.device)
    random_scores = random_scores.masked_fill(~valid, 2.0)
    positions = random_scores.topk(n_samples, dim=1, largest=False).indices
    active = (
        torch.arange(n_samples, device=valid.device)[None, :]
        < counts[:, None].clamp(max=n_samples)
    )
    safe_active = active.clone()
    safe_active[counts.eq(0), 0] = True

    was_training = model.training
    model.eval()
    with torch.no_grad():
        reference = model(batch)
        effects = []
        row_indices = torch.arange(valid.shape[0], device=valid.device)
        for sample_index in range(n_samples):
            masked_batch = dict(batch)
            masked_valid = valid.clone()
            selected = positions[:, sample_index]
            selected_rows = active[:, sample_index]
            masked_valid[
                row_indices[selected_rows],
                selected[selected_rows],
            ] = False
            masked_batch['valid_mask'] = masked_valid
            masked_prediction = model(masked_batch)
            effects.append((reference - masked_prediction).abs())
    model.train(was_training)

    effects_tensor = torch.stack(effects, dim=1)
    effect_logits = effects_tensor / max(temperature, 1e-8)
    effect_logits = effect_logits.masked_fill(~safe_active, float('-inf'))
    target = torch.softmax(effect_logits, dim=1).detach()

    selected_attention = attention.gather(1, positions).masked_fill(
        ~safe_active,
        0.0,
    )
    predicted = selected_attention / selected_attention.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)
    kl_rows = (
        target
        * (
            target.clamp_min(1e-8).log()
            - predicted.clamp_min(1e-8).log()
        )
    ).sum(dim=1)
    return kl_rows[useful_rows].mean()


def train_faithful_epoch(
    model: FaithfulModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ExperimentConfig,
    model_type: str,
    alignment_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {'mse': 0.0, 'alignment': 0.0, 'loss': 0.0}
    n_samples = 0
    for batch, targets in loader:
        batch = move_batch(batch, device)
        targets = targets.to(device)
        if model_type in FAITHFUL_MODEL_TYPES:
            batch = apply_event_dropout(batch, config.event_dropout)
        predictions, attention = model(batch, return_attention=True)
        mse_loss = functional.mse_loss(predictions, targets)
        if model_type in FAITHFUL_MODEL_TYPES and alignment_weight > 0:
            alignment_loss = sampled_occlusion_alignment_loss(
                model,
                batch,
                attention,
                n_samples=config.occlusion_samples,
                temperature=config.alignment_temperature,
            )
        else:
            alignment_loss = attention.sum() * 0.0
        loss = mse_loss + alignment_weight * alignment_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        batch_size = len(targets)
        totals['mse'] += float(mse_loss.detach().cpu()) * batch_size
        totals['alignment'] += float(alignment_loss.detach().cpu()) * batch_size
        totals['loss'] += float(loss.detach().cpu()) * batch_size
        n_samples += batch_size
    return {key: value / max(1, n_samples) for key, value in totals.items()}


def fit_faithful_transformer(
    train_dataset: ShotSequenceDataset,
    validation_dataset: ShotSequenceDataset,
    *,
    vocabularies: Vocabularies,
    config: ExperimentConfig,
    model_type: str,
    seed: int,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> tuple[FaithfulModel, pd.DataFrame, PredictionMetrics]:
    if model_type not in {'bottleneck', *FAITHFUL_MODEL_TYPES}:
        raise ValueError(f'Unknown faithful model type: {model_type}')
    set_seed(seed)
    device = select_device()
    model = build_faithful_model(
        model_type,
        vocabularies,
        config.sequence_length,
        config,
    ).to(device)
    train_loader = make_loader(train_dataset, config, shuffle=True, seed=seed)
    validation_loader = make_loader(validation_dataset, config, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = np.inf
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        ramp_position = max(0, epoch - config.alignment_warmup_epochs)
        alignment_weight = config.alignment_lambda * min(
            1.0,
            ramp_position / max(1, config.alignment_ramp_epochs),
        )
        train_values = train_faithful_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            model_type,
            alignment_weight,
        )
        _, targets, predictions = predict(model, validation_loader, device)
        validation_mse = float(mean_squared_error(targets, predictions))
        history.append(
            {
                'epoch': epoch,
                'train_mse': train_values['mse'],
                'train_alignment': train_values['alignment'],
                'train_loss': train_values['loss'],
                'alignment_weight': alignment_weight,
                'validation_mse': validation_mse,
            }
        )
        if epoch_callback is not None:
            epoch_callback(epoch, validation_mse)
        if validation_mse < best_loss - config.min_delta:
            best_loss = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break
    if best_state is None:
        raise RuntimeError('Training did not produce a checkpoint.')
    model.load_state_dict(best_state)
    _, targets, predictions = predict(model, validation_loader, device)
    return (
        model,
        pd.DataFrame(history),
        prediction_metrics(targets, predictions),
    )


def save_faithful_run(
    output_dir: Path,
    *,
    model: FaithfulModel,
    history: pd.DataFrame,
    validation_metrics: PredictionMetrics,
    vocabularies: Vocabularies,
    config: ExperimentConfig,
    model_type: str,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / 'history.csv', index=False)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'use_player_id': model.use_player_id,
            'use_position_encoding': getattr(
                model,
                'use_position_encoding',
                False,
            ),
            'shuffle_events': model_type == 'gru_attention_shuffled',
            'input_variant': 'pre_shot',
            'seed': seed,
            'vocabularies': asdict(vocabularies),
            'config': asdict(config),
        },
        output_dir / 'model.pt',
    )
    (output_dir / 'validation_metrics.json').write_text(
        json.dumps(asdict(validation_metrics), indent=2),
        encoding='utf-8',
    )
    (output_dir / 'run_metadata.json').write_text(
        json.dumps(
            {
                'model_type': model_type,
                'seed': seed,
                'parameter_count': sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                'use_player_id': model.use_player_id,
                'use_position_encoding': getattr(
                    model,
                    'use_position_encoding',
                    False,
                ),
                'shuffle_events': model_type == 'gru_attention_shuffled',
                'input_variant': 'pre_shot',
                'event_dropout': config.event_dropout,
                'alignment_lambda': (
                    config.alignment_lambda
                    if model_type in FAITHFUL_MODEL_TYPES
                    else 0.0
                ),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
