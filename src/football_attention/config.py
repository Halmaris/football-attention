from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class ExperimentConfig:
    project_dir: Path
    results_name: str = 'results'
    prepared_name: str = 'prepared'
    sequence_length: int = 20
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    variants: tuple[str, ...] = (
        'full',
        'masked_shot',
        'pre_shot',
        'shot_only',
    )
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-5
    grad_clip: float = 1.0
    attention_layers: int = 2
    num_workers: int = 0
    faithful_models: tuple[str, ...] = ('bottleneck', 'faithful')
    event_dropout: float = 0.15
    alignment_lambda: float = 0.01
    alignment_warmup_epochs: int = 5
    alignment_ramp_epochs: int = 5
    alignment_temperature: float = 0.01
    occlusion_samples: int = 3
    integrated_gradients_steps: int = 16
    deletion_k: tuple[int, ...] = (1, 3, 5)
    min_player_sequences: int = 20
    bootstrap_samples: int = 1000
    contribution_variants: tuple[str, ...] = (
        'full',
        'pre_shot',
        'shooter_excluded',
    )
    baseline_models: tuple[str, ...] = (
        'train_mean',
        'linear',
        'ridge',
        'gradient_boosting',
    )
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def cache_dir(self) -> Path:
        manifest = self.prepared_dir / 'preparation.json'
        if manifest.exists():
            return Path(json.loads(manifest.read_text())['source_cache'])
        return self.project_dir / 'cache' / 'development'

    @property
    def prepared_dir(self) -> Path:
        return self.project_dir / 'data' / self.prepared_name

    @property
    def results_dir(self) -> Path:
        return self.project_dir / self.results_name
