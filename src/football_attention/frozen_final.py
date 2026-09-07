from __future__ import annotations
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .baselines import feature_matrix, last_event_feature_matrix
from .config import ExperimentConfig
from .faithful_model import build_faithful_model
from .faithful_train import train_faithful_epoch
from .serialization import config_from_dict, resolved_config_dict
from .sequence_baseline_train import train_sequence_baseline_epoch
from .sequence_baselines import build_sequence_baseline
from .train import make_loader, select_device, set_seed
from .variants import ShotSequenceDataset, Vocabularies


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_frozen_protocol(
    project_dir: Path,
    *,
    config_path: Path,
    seeds: tuple[int, ...] | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    settings = read_json(config_path)
    seeds = tuple(settings['seeds'] if seeds is None else seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Provide at least one seed, without duplicates')
    neural_models = {}
    for label, spec in settings['neural_models'].items():
        config = config_from_dict(ExperimentConfig(project_dir=project_dir), spec['config'])
        if config.sequence_length != 20:
            raise ValueError('This protocol uses 20 recorded tokens, including the reference shot')
        epochs = int(spec['fixed_epochs'])
        if epochs < 1:
            raise ValueError(f'{label}: fixed_epochs must be positive')
        values = resolved_config_dict(config)
        values.pop('project_dir')
        values.pop('results_name')
        neural_models[label] = {
            'family': spec['family'],
            'architecture': spec['architecture'],
            'fixed_epochs': min(2, epochs) if quick else epochs,
            'config': values,
        }
    return {
        'protocol_version': 1,
        'input_variant': 'pre_shot',
        'fit_splits': ['train', 'validation'],
        'evaluation_split': 'test',
        'seeds': list(seeds),
        'quick': quick,
        'neural_models': neural_models,
        'classical_models': settings['classical_models'],
    }


def alignment_weight(config: ExperimentConfig, epoch: int) -> float:
    ramp_position = max(0, epoch - config.alignment_warmup_epochs)
    return config.alignment_lambda * min(
        1.0,
        ramp_position / max(1, config.alignment_ramp_epochs),
    )


def load_or_create_protocol(
    path: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if path.exists():
        frozen = read_json(path)
        if frozen != candidate:
            raise RuntimeError(
                'Frozen protocol already exists and differs from the requested '
                'configuration. Use a new results directory.'
            )
        return frozen
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    return candidate


def config_from_spec(
    project_dir: Path,
    results_name: str,
    spec: dict[str, Any],
    *,
    prepared_name: str = 'prepared',
) -> ExperimentConfig:
    base = ExperimentConfig(
        project_dir=project_dir,
        results_name=results_name,
        prepared_name=prepared_name,
    )
    config = config_from_dict(base, spec['config'])
    return replace(
        config,
        max_epochs=int(spec['fixed_epochs']),
        prepared_name=prepared_name,
    )


def fit_neural_fixed_epochs(
    dataset: ShotSequenceDataset,
    *,
    vocabularies: Vocabularies,
    config: ExperimentConfig,
    family: str,
    architecture: str,
    seed: int,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    set_seed(seed)
    device = select_device()
    if family == 'sequence':
        model = build_sequence_baseline(
            architecture,
            vocabularies,
            config,
        ).to(device)
    elif family == 'faithful':
        model = build_faithful_model(
            architecture,
            vocabularies,
            config.sequence_length,
            config,
        ).to(device)
    else:
        raise ValueError(f'Unknown model family: {family}')
    loader = make_loader(dataset, config, shuffle=True, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []
    for epoch in range(1, config.max_epochs + 1):
        if family == 'sequence':
            train_mse = train_sequence_baseline_epoch(
                model,
                loader,
                optimizer,
                device,
                config,
            )
            row = {'epoch': epoch, 'train_mse': train_mse}
        else:
            weight = alignment_weight(config, epoch)
            values = train_faithful_epoch(
                model,
                loader,
                optimizer,
                device,
                config,
                architecture,
                weight,
            )
            row = {
                'epoch': epoch,
                'train_mse': values['mse'],
                'train_alignment': values['alignment'],
                'train_loss': values['loss'],
                'alignment_weight': weight,
            }
        history.append(row)
    return model, pd.DataFrame(history)


def save_neural_refit(
    run_dir: Path,
    *,
    model: torch.nn.Module,
    history: pd.DataFrame,
    vocabularies: Vocabularies,
    config: ExperimentConfig,
    model_label: str,
    family: str,
    architecture: str,
    seed: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / 'history.csv', index=False)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_label': model_label,
            'model_family': family,
            'model_type': architecture,
            'input_variant': 'pre_shot',
            'fit_splits': ['train', 'validation'],
            'seed': seed,
            'vocabularies': asdict(vocabularies),
            'config': resolved_config_dict(config),
            'fixed_epochs': int(config.max_epochs),
        },
        run_dir / 'model.pt',
    )
    metadata = {
        'model_label': model_label,
        'model_family': family,
        'model_type': architecture,
        'seed': seed,
        'fixed_epochs': int(config.max_epochs),
        'parameter_count': int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        'fit_splits': ['train', 'validation'],
        'evaluation_split': 'test',
        'input_variant': 'pre_shot',
    }
    (run_dir / 'run_metadata.json').write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )


def fit_classical_models(
    rows: pd.DataFrame,
    params: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    full_vectorizer = DictVectorizer(sparse=False)
    full_x = full_vectorizer.fit_transform(feature_matrix(rows, 'pre_shot'))
    last_vectorizer = DictVectorizer(sparse=False)
    last_x = last_vectorizer.fit_transform(last_event_feature_matrix(rows))
    y = rows['xg'].to_numpy(float)
    models: dict[str, object | float] = {
        'train_mean': float(y.mean()),
        'linear': make_pipeline(StandardScaler(), LinearRegression()),
        'ridge': make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(params['ridge']['ridge_alpha'])),
        ),
        'gradient_boosting': HistGradientBoostingRegressor(
            **params['gradient_boosting'],
            early_stopping=False,
            random_state=seed,
        ),
    }
    for model in models.values():
        if not isinstance(model, float):
            model.fit(full_x, y)
    last_model = HistGradientBoostingRegressor(
        **params['gradient_boosting_last_event'],
        early_stopping=False,
        random_state=seed,
    ).fit(last_x, y)
    return {
        'vectorizer': full_vectorizer,
        'models': models,
        'last_event_vectorizer': last_vectorizer,
        'last_event_model': last_model,
    }


def save_classical_models(
    path: Path,
    *,
    fitted: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(fitted, handle)


def load_classical_models(path: Path) -> dict[str, Any]:
    with path.open('rb') as handle:
        return pickle.load(handle)


def predict_classical_models(
    rows: pd.DataFrame,
    fitted: dict[str, Any],
) -> dict[str, Any]:
    full_x = fitted['vectorizer'].transform(feature_matrix(rows, 'pre_shot'))
    predictions = {
        name: (
            np.full(len(rows), model, dtype=float)
            if isinstance(model, float)
            else np.asarray(model.predict(full_x), dtype=float)
        )
        for name, model in fitted['models'].items()
    }
    last_x = fitted['last_event_vectorizer'].transform(
        last_event_feature_matrix(rows)
    )
    predictions['gradient_boosting_last_event'] = np.asarray(
        fitted['last_event_model'].predict(last_x),
        dtype=float,
    )
    return predictions
