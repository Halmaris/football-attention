from pathlib import Path

import pytest

from football_attention.config import ExperimentConfig
from football_attention.serialization import resolved_config_dict
from football_attention.tasks.alignment import (
    MODEL_ALIGNED,
    MODEL_MATCHED,
    build_control_protocol,
    build_matched_control_spec,
    verify_matched_spec,
)


def frozen_protocol(tmp_path: Path) -> dict[str, object]:
    config = resolved_config_dict(ExperimentConfig(project_dir=tmp_path))
    aligned_config = {
        **config,
        'alignment_lambda': 0.003,
        'alignment_temperature': 0.006,
        'alignment_warmup_epochs': 5,
        'alignment_ramp_epochs': 1,
        'occlusion_samples': 5,
    }
    return {
        'fit_splits': ['train', 'validation'],
        'seeds': [42, 43],
        'neural_models': {
            MODEL_ALIGNED: {
                'family': 'faithful',
                'architecture': 'gru_attention',
                'fixed_epochs': 25,
                'config': aligned_config,
            },
            'gru_attention_predictive': {
                'family': 'faithful',
                'architecture': 'gru_attention',
                'fixed_epochs': 30,
                'config': {**config, 'alignment_lambda': 0.0},
            },
        },
    }


def test_matched_control_changes_only_alignment_lambda(tmp_path: Path) -> None:
    protocol = frozen_protocol(tmp_path)
    aligned = protocol['neural_models'][MODEL_ALIGNED]
    control = build_matched_control_spec(protocol, quick=False)
    verify_matched_spec(aligned, control)
    assert control['fixed_epochs'] == 25
    assert control['config']['alignment_lambda'] == 0.0
    assert control['config']['alignment_warmup_epochs'] == 5
    assert control['config']['alignment_ramp_epochs'] == 1
    assert control['config']['alignment_temperature'] == 0.006
    assert control['config']['occlusion_samples'] == 5


def test_verify_matched_spec_rejects_second_change(tmp_path: Path) -> None:
    protocol = frozen_protocol(tmp_path)
    aligned = protocol['neural_models'][MODEL_ALIGNED]
    control = build_matched_control_spec(protocol, quick=False)
    control['config']['learning_rate'] = 0.5
    with pytest.raises(RuntimeError, match='learning_rate'):
        verify_matched_spec(aligned, control)


def test_control_protocol_records_both_comparisons(tmp_path: Path) -> None:
    protocol = build_control_protocol(
        tmp_path / 'results_frozen_final',
        frozen_protocol(tmp_path),
        seeds=[42, 43],
        quick=False,
    )
    assert MODEL_MATCHED in protocol['models']
    assert protocol['primary_comparison'].endswith(MODEL_MATCHED)
    assert protocol['external_baseline_comparison'].endswith(
        'gru_attention_predictive'
    )
