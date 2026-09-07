from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from football_attention.config import ExperimentConfig
from football_attention.faithful_model import build_faithful_model
from football_attention.serialization import config_from_dict
from football_attention.train import make_loader, move_batch, select_device
from football_attention.variants import ShotSequenceDataset, Vocabularies


def extract_seed_attention(
    rows: pd.DataFrame,
    *,
    checkpoint_path: Path,
    project_dir: Path,
    results_name: str,
    prepared_name: str,
    model_label: str,
    seed: int,
) -> pd.DataFrame:
    device = select_device()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    vocabularies = Vocabularies(**checkpoint['vocabularies'])
    base_config = ExperimentConfig(
        project_dir=project_dir,
        results_name=results_name,
        prepared_name=prepared_name,
    )
    config = config_from_dict(base_config, checkpoint['config'])
    dataset = ShotSequenceDataset(
        rows,
        vocabularies=vocabularies,
        variant='pre_shot',
        sequence_length=config.sequence_length,
    )
    loader = make_loader(dataset, config, shuffle=False)
    model = build_faithful_model(
        checkpoint['model_type'],
        vocabularies,
        config.sequence_length,
        config,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    split_by_sequence = rows.set_index('shot_seq_id')['split'].to_dict()
    token_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for cpu_batch, cpu_targets in loader:
            batch = move_batch(cpu_batch, device)
            _, attention = model(batch, return_attention=True)
            attention_values = attention.detach().cpu().numpy()
            valid_values = cpu_batch['valid_mask'].numpy()
            for row_index in range(len(cpu_targets)):
                sequence_id = int(cpu_batch['shot_seq_id'][row_index])
                positions = np.flatnonzero(valid_values[row_index])
                uniform = 1.0 / len(positions) if len(positions) else 0.0
                for position in positions:
                    token_rows.append(
                        {
                            'model_type': model_label,
                            'seed': seed,
                            'split': split_by_sequence[sequence_id],
                            'shot_seq_id': sequence_id,
                            'match_id': int(
                                cpu_batch['match_id'][row_index]
                            ),
                            'token_position': int(position),
                            'player_id': int(
                                cpu_batch['original_player_id'][
                                    row_index,
                                    position,
                                ]
                            ),
                            'target_xg': float(cpu_targets[row_index]),
                            'attention': float(
                                attention_values[row_index, position]
                            ),
                            'uniform': uniform,
                        }
                    )
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return pd.DataFrame(token_rows)
