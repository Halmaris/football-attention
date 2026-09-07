from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FrozenTrial, TrialState
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer

from football_attention.baselines import feature_matrix, last_event_feature_matrix


REPRESENTATIONS = ('full_summary', 'last_pre_shot_event')
INITIAL_PARAMETERS = {
    'learning_rate': 0.013503193939180041,
    'max_iter': 400,
    'max_leaf_nodes': 31,
    'max_depth': None,
    'min_samples_leaf': 10,
    'l2_regularization': 0.3192960621472932,
    'max_bins': 255,
}


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.20, log=True
        ),
        'max_iter': trial.suggest_int('max_iter', 100, 600, step=50),
        'max_leaf_nodes': trial.suggest_categorical(
            'max_leaf_nodes', [7, 15, 31, 63]
        ),
        'max_depth': trial.suggest_categorical(
            'max_depth', [None, 3, 5, 7]
        ),
        'min_samples_leaf': trial.suggest_categorical(
            'min_samples_leaf', [10, 20, 40, 80]
        ),
        'l2_regularization': trial.suggest_float(
            'l2_regularization', 1e-4, 10.0, log=True
        ),
        'max_bins': trial.suggest_categorical('max_bins', [63, 127, 255]),
    }


def representation_features(
    rows: pd.DataFrame,
    representation: str,
) -> list[dict[str, float]]:
    if representation == 'full_summary':
        return feature_matrix(rows, 'pre_shot')
    if representation == 'last_pre_shot_event':
        return last_event_feature_matrix(rows)
    raise ValueError(f'Unknown representation: {representation}')


def completed_trials(study: optuna.Study) -> int:
    return sum(trial.state == TrialState.COMPLETE for trial in study.trials)


def sampler_callback(path: Path):
    def callback(study: optuna.Study, _: FrozenTrial) -> None:
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(study.sampler, handle)
        temporary.replace(path)

    return callback


def tune_representation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    representation: str,
    output_dir: Path,
    trials: int,
    sampler_seed: int,
    model_seed: int,
) -> dict[str, Any]:
    vectorizer = DictVectorizer(sparse=False)
    train_x = vectorizer.fit_transform(
        representation_features(train, representation)
    )
    validation_x = vectorizer.transform(
        representation_features(validation, representation)
    )
    train_y = train['xg'].to_numpy(float)
    validation_y = validation['xg'].to_numpy(float)
    sampler_path = output_dir / f'sampler_{representation}.pkl'
    if sampler_path.exists():
        with sampler_path.open('rb') as handle:
            sampler = pickle.load(handle)
    else:
        sampler = optuna.samplers.TPESampler(
            seed=sampler_seed,
            n_startup_trials=15,
            multivariate=True,
            group=True,
        )
    study = optuna.create_study(
        study_name=f'boosting_{representation}',
        storage=f'sqlite:///{output_dir / "optuna.db"}',
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        direction='minimize',
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(INITIAL_PARAMETERS)

    def objective(trial: optuna.Trial) -> float:
        model = HistGradientBoostingRegressor(
            **suggest_parameters(trial),
            early_stopping=False,
            random_state=model_seed,
        ).fit(train_x, train_y)
        prediction = np.asarray(model.predict(validation_x), dtype=float)
        return float(np.mean(np.square(validation_y - prediction)))

    remaining = trials - completed_trials(study)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            callbacks=[sampler_callback(sampler_path)],
            gc_after_trial=True,
        )
    study.trials_dataframe().to_csv(
        output_dir / f'trials_{representation}.csv',
        index=False,
    )
    best = {
        'representation': representation,
        'trial_number': int(study.best_trial.number),
        'validation_mse': float(study.best_value),
        'params': study.best_params,
        'n_trials_complete': completed_trials(study),
    }
    (output_dir / f'best_{representation}.json').write_text(
        json.dumps(best, indent=2),
        encoding='utf-8',
    )
    return best


def run(
    project_dir: Path,
    *,
    trials: int,
    sampler_seed: int,
    model_seed: int,
) -> None:
    if trials < 1:
        raise ValueError('--trials must be positive')
    rows = pd.read_parquet(
        project_dir / 'data/development/sequences_raw.parquet',
        filters=[('split', 'in', ['train', 'validation'])],
    )
    train = rows.loc[rows['split'].eq('train')].copy()
    validation = rows.loc[rows['split'].eq('validation')].copy()
    if train.empty or validation.empty:
        raise RuntimeError('Both train and validation splits are required')
    output_dir = project_dir / 'results/boosting_hpo'
    output_dir.mkdir(parents=True, exist_ok=True)
    for representation in REPRESENTATIONS:
        best = tune_representation(
            train,
            validation,
            representation=representation,
            output_dir=output_dir,
            trials=trials,
            sampler_seed=sampler_seed,
            model_seed=model_seed,
        )
        print(
            f'{representation}: MSE={best["validation_mse"]:.8f}, '
            f'trial={best["trial_number"]}'
        )
