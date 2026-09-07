from __future__ import annotations
from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .train import PredictionMetrics, prediction_metrics
from .variants import transform_sequence


def sequence_features(row: pd.Series, variant: str) -> dict[str, float]:
    sequence = transform_sequence(row, variant)
    n_tokens = len(sequence['player_id'])
    features: dict[str, float] = {
        'n_tokens': float(n_tokens),
        'n_unique_players': float(len(set(sequence['player_id']).difference({-1}))),
    }
    if not n_tokens:
        return features

    continuous = np.asarray(sequence['continuous'], dtype=float)
    feature_names = ['start_x', 'start_y', 'end_x', 'end_y', 'delta_time']
    for index, name in enumerate(feature_names):
        values = continuous[:, index]
        features[f'{name}_mean'] = float(values.mean())
        features[f'{name}_std'] = float(values.std())
        features[f'{name}_last'] = float(values[-1])
        features[f'{name}_max'] = float(values.max())
    for event_type in sequence['type_name']:
        key = f'type_count={event_type}'
        features[key] = features.get(key, 0.0) + 1.0
    for outcome in sequence['outcome_name']:
        key = f'outcome_count={outcome}'
        features[key] = features.get(key, 0.0) + 1.0
    features[f'last_type={sequence["type_name"][-1]}'] = 1.0
    features[f'last_outcome={sequence["outcome_name"][-1]}'] = 1.0
    return features


def feature_matrix(
    rows: pd.DataFrame,
    variant: str,
) -> list[dict[str, float]]:
    return [sequence_features(row, variant) for _, row in rows.iterrows()]


def last_event_features(row: pd.Series) -> dict[str, float]:
    sequence = transform_sequence(row, 'pre_shot')
    if not sequence['type_name']:
        raise ValueError(
            f'Sequence {row["shot_seq_id"]} has no visible pre-shot event'
        )
    continuous = np.asarray(sequence['continuous'][-1], dtype=float)
    names = ('start_x', 'start_y', 'end_x', 'end_y', 'delta_time')
    features = {
        f'last_{name}': float(value)
        for name, value in zip(names, continuous, strict=True)
    }
    features[f'last_type={sequence["type_name"][-1]}'] = 1.0
    features[f'last_outcome={sequence["outcome_name"][-1]}'] = 1.0
    return features


def last_event_feature_matrix(rows: pd.DataFrame) -> list[dict[str, float]]:
    return [last_event_features(row) for _, row in rows.iterrows()]


def fit_baseline_models(
    train_rows: pd.DataFrame,
    *,
    variant: str,
    seed: int = 42,
) -> tuple[DictVectorizer, dict[str, object | float]]:
    '''Fit the fixed classical baseline specification on training rows.'''
    vectorizer = DictVectorizer(sparse=False)
    train_features = vectorizer.fit_transform(
        feature_matrix(train_rows, variant)
    )
    train_targets = train_rows['xg'].to_numpy(float)
    models: dict[str, object | float] = {
        'train_mean': float(train_targets.mean()),
        'linear': make_pipeline(StandardScaler(), LinearRegression()),
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'gradient_boosting': HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }
    for model in models.values():
        if not isinstance(model, float):
            model.fit(train_features, train_targets)
    return vectorizer, models


def predict_baseline_models(
    rows: pd.DataFrame,
    *,
    variant: str,
    vectorizer: DictVectorizer,
    models: dict[str, object | float],
) -> dict[str, np.ndarray]:
    '''Predict with a fitted fixed baseline specification.'''
    features = vectorizer.transform(feature_matrix(rows, variant))
    return {
        name: (
            np.full(len(rows), model, dtype=float)
            if isinstance(model, float)
            else np.asarray(model.predict(features), dtype=float)
        )
        for name, model in models.items()
    }


def fit_predictive_baselines(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    variant: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer, models = fit_baseline_models(
        train_rows,
        variant=variant,
        seed=seed,
    )
    validation_targets = validation_rows['xg'].to_numpy(float)
    test_targets = test_rows['xg'].to_numpy(float)
    validation_by_model = predict_baseline_models(
        validation_rows,
        variant=variant,
        vectorizer=vectorizer,
        models=models,
    )
    test_by_model = predict_baseline_models(
        test_rows,
        variant=variant,
        vectorizer=vectorizer,
        models=models,
    )
    metric_rows = []
    prediction_rows = []
    for name in models:
        validation_predictions = validation_by_model[name]
        test_predictions = test_by_model[name]
        for split, targets, predictions in [
            ('validation', validation_targets, validation_predictions),
            ('test', test_targets, test_predictions),
        ]:
            metrics: PredictionMetrics = prediction_metrics(targets, predictions)
            metric_rows.append(
                {
                    'variant': variant,
                    'model': name,
                    'split': split,
                    **asdict(metrics),
                }
            )
        prediction_rows.extend(
            {
                'shot_seq_id': int(sequence_id),
                'variant': variant,
                'model': name,
                'split': 'test',
                'target_xg': float(target),
                'prediction': float(prediction),
            }
            for sequence_id, target, prediction in zip(
                test_rows['shot_seq_id'],
                test_targets,
                test_predictions,
            )
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)
