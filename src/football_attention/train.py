from __future__ import annotations
import random
import os
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import ExperimentConfig
from .variants import ShotSequenceDataset, collate_sequences


@dataclass(frozen=True)
class PredictionMetrics:
    mse: float
    mae: float
    r2: float


def select_device(name: str = 'auto') -> torch.device:
    name = os.environ.get('FOOTBALL_ATTENTION_DEVICE', name) if name == 'auto' else name
    available = {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': torch.backends.mps.is_available(),
    }
    if name == 'auto':
        name = next(device for device in ('cuda', 'mps', 'cpu') if available[device])
    if not available.get(name, False):
        raise ValueError(f'Device is unavailable: {name}')
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: ShotSequenceDataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator().manual_seed(
        config.seeds[0] if seed is None else seed
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_sequences,
        generator=generator,
    )


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    sequence_ids = []
    for batch, batch_targets in loader:
        sequence_ids.append(batch['shot_seq_id'].numpy())
        batch = move_batch(batch, device)
        predictions.append(model(batch).detach().cpu().numpy())
        targets.append(batch_targets.numpy())
    return (
        np.concatenate(sequence_ids),
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def prediction_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> PredictionMetrics:
    return PredictionMetrics(
        mse=float(mean_squared_error(targets, predictions)),
        mae=float(mean_absolute_error(targets, predictions)),
        r2=float(r2_score(targets, predictions)),
    )
