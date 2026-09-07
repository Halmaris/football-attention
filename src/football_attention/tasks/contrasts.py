from __future__ import annotations
import pandas as pd
from football_attention.bootstrap import seed_match_bootstrap_mean


MODEL_PREDICTIVE = 'gru_attention_predictive'


MODEL_FAITHFUL = 'gru_attention_faithful'


def paired_models(
    values: pd.DataFrame,
    *,
    value_column: str,
    keys: list[str],
    label: str,
    n_bootstrap: int,
    model_left: str = MODEL_FAITHFUL,
    model_right: str = MODEL_PREDICTIVE,
) -> dict[str, object]:
    columns = ['seed', 'match_id', 'shot_seq_id', value_column, *keys]
    left = values[values['model_type'].eq(model_left)][columns]
    right = values[values['model_type'].eq(model_right)][columns]
    paired = left.merge(
        right,
        on=['seed', 'match_id', 'shot_seq_id', *keys],
        suffixes=('_left', '_right'),
        validate='one_to_one',
    )
    paired['difference'] = (
        paired[f'{value_column}_left'] - paired[f'{value_column}_right']
    )
    result = seed_match_bootstrap_mean(
        paired,
        n_bootstrap=n_bootstrap,
        seed=2026,
    )
    return {
        'metric': label,
        'comparison': f'{model_left} - {model_right}',
        **{key: paired[key].iloc[0] for key in keys},
        **result,
    }


def paired_deletion_rows(
    deletions: pd.DataFrame,
    *,
    n_bootstrap: int,
    model_left: str = MODEL_FAITHFUL,
    model_right: str = MODEL_PREDICTIVE,
) -> list[dict[str, object]]:
    values = deletions.copy()
    values['base_squared_error'] = (
        values['target_xg'] - values['base_prediction']
    ) ** 2
    values['deleted_squared_error'] = (
        values['target_xg'] - values['deleted_prediction']
    ) ** 2
    values['delta_squared_error'] = (
        values['deleted_squared_error'] - values['base_squared_error']
    )
    rows = []
    for method in ['attention', 'occlusion', 'integrated_gradients', 'random']:
        for k in sorted(values['k'].unique()):
            subset = values[
                values['method'].eq(method) & values['k'].eq(k)
            ]
            rows.append(
                paired_models(
                    subset,
                    value_column='delta_squared_error',
                    keys=[],
                    label=f'deletion_delta_mse:{method}:k={int(k)}',
                    n_bootstrap=n_bootstrap,
                    model_left=model_left,
                    model_right=model_right,
                )
            )

    for model_type in [model_right, model_left]:
        model_values = values[values['model_type'].eq(model_type)]
        for k in sorted(model_values['k'].unique()):
            attention = model_values[
                model_values['method'].eq('attention')
                & model_values['k'].eq(k)
            ][['seed', 'shot_seq_id', 'match_id', 'delta_squared_error']]
            random = model_values[
                model_values['method'].eq('random')
                & model_values['k'].eq(k)
            ][['seed', 'shot_seq_id', 'delta_squared_error']]
            paired = attention.merge(
                random,
                on=['seed', 'shot_seq_id'],
                suffixes=('_attention', '_random'),
                validate='one_to_one',
            )
            paired['difference'] = (
                paired['delta_squared_error_attention']
                - paired['delta_squared_error_random']
            )
            result = seed_match_bootstrap_mean(
                paired,
                n_bootstrap=n_bootstrap,
                seed=2026,
            )
            rows.append(
                {
                    'metric': f'attention_minus_random:k={int(k)}',
                    'comparison': model_type,
                    **result,
                }
            )

    # Compare the attention advantage over the same random-deletion control.
    # This removes model-wide differences in sensitivity to event removal.
    for k in sorted(values['k'].unique()):
        model_contrasts = []
        for model_type in [model_right, model_left]:
            model_values = values[values['model_type'].eq(model_type)]
            attention = model_values[
                model_values['method'].eq('attention')
                & model_values['k'].eq(k)
            ][
                [
                    'seed',
                    'shot_seq_id',
                    'match_id',
                    'delta_squared_error',
                ]
            ]
            random = model_values[
                model_values['method'].eq('random')
                & model_values['k'].eq(k)
            ][['seed', 'shot_seq_id', 'delta_squared_error']]
            contrast = attention.merge(
                random,
                on=['seed', 'shot_seq_id'],
                suffixes=('_attention', '_random'),
                validate='one_to_one',
            )
            contrast['model_type'] = model_type
            contrast['attention_minus_random'] = (
                contrast['delta_squared_error_attention']
                - contrast['delta_squared_error_random']
            )
            model_contrasts.append(contrast)
        rows.append(
            paired_models(
                pd.concat(model_contrasts, ignore_index=True),
                value_column='attention_minus_random',
                keys=[],
                label=f'deletion_attention_minus_random:k={int(k)}',
                n_bootstrap=n_bootstrap,
                model_left=model_left,
                model_right=model_right,
            )
        )
    return rows
