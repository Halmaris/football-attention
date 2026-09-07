from __future__ import annotations
import torch
from .config import ExperimentConfig
from .faithful_train import apply_event_dropout
from .train import move_batch


def train_sequence_baseline_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ExperimentConfig,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch, targets in loader:
        batch = move_batch(batch, device)
        batch = apply_event_dropout(batch, config.event_dropout)
        targets = targets.to(device)
        predictions = model(batch)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(targets)
        n_samples += len(targets)
    return total_loss / max(1, n_samples)
