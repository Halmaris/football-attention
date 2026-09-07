from __future__ import annotations
from dataclasses import asdict, replace
from typing import Any
from .config import ExperimentConfig


def resolved_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    values['project_dir'] = str(values['project_dir'])
    values['seeds'] = list(values['seeds'])
    values['variants'] = list(values['variants'])
    values['faithful_models'] = list(values['faithful_models'])
    values['deletion_k'] = list(values['deletion_k'])
    values['contribution_variants'] = list(values['contribution_variants'])
    values['baseline_models'] = list(values['baseline_models'])
    return values


def config_from_dict(
    base: ExperimentConfig,
    values: dict[str, Any],
) -> ExperimentConfig:
    allowed = set(asdict(base)).difference({'project_dir', 'results_name'})
    updates = {key: value for key, value in values.items() if key in allowed}
    tuple_fields = {
        'seeds',
        'variants',
        'faithful_models',
        'deletion_k',
        'contribution_variants',
        'baseline_models',
    }
    for key in tuple_fields.intersection(updates):
        updates[key] = tuple(updates[key])
    return replace(base, **updates)
