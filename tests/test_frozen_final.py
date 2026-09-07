from dataclasses import replace
from pathlib import Path
import pytest
from football_attention.config import ExperimentConfig
from football_attention.frozen_final import (
    alignment_weight,
    config_from_spec,
    load_or_create_protocol,
)
from football_attention.serialization import resolved_config_dict


def test_existing_protocol_cannot_change(tmp_path: Path) -> None:
    path = tmp_path / 'frozen_protocol.json'
    load_or_create_protocol(path, {'seeds': [42]})
    with pytest.raises(RuntimeError, match='differs'):
        load_or_create_protocol(path, {'seeds': [43]})


def test_alignment_schedule() -> None:
    config = replace(
        ExperimentConfig(project_dir=Path('.')),
        alignment_lambda=0.003,
        alignment_warmup_epochs=5,
        alignment_ramp_epochs=2,
    )
    assert alignment_weight(config, 5) == 0.0
    assert alignment_weight(config, 6) == pytest.approx(0.0015)
    assert alignment_weight(config, 7) == pytest.approx(0.003)


def test_frozen_config_uses_explicit_prepared_input(tmp_path: Path) -> None:
    spec = {
        'fixed_epochs': 3,
        'config': resolved_config_dict(
            ExperimentConfig(project_dir=tmp_path, prepared_name='prepared')
        ),
    }

    config = config_from_spec(
        tmp_path,
        'results_nonempty',
        spec,
        prepared_name='prepared_nonempty',
    )

    assert config.prepared_name == 'prepared_nonempty'
    assert config.prepared_dir == tmp_path / 'data' / 'prepared_nonempty'
