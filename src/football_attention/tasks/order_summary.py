from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from football_attention.bootstrap import seed_match_bootstrap_mean


FACTORIAL_CONTRASTS = {
    'ordered_model_test_shuffle': {
        'ordered_to_shuffled': 1.0,
        'ordered_to_ordered': -1.0,
    },
    'shuffled_model_test_order': {
        'shuffled_to_ordered': 1.0,
        'shuffled_to_shuffled': -1.0,
    },
    'matching_shuffled_vs_matching_ordered': {
        'shuffled_to_shuffled': 1.0,
        'ordered_to_ordered': -1.0,
    },
    'train_test_order_interaction': {
        'ordered_to_shuffled': 1.0,
        'ordered_to_ordered': -1.0,
        'shuffled_to_shuffled': -1.0,
        'shuffled_to_ordered': 1.0,
    },
}


LENGTH_ORDER = ['all', '0', '1-2', '3-5', '6-10', '11-20']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument('--results-name', default='results_order_followup')
    parser.add_argument('--prepared-name', default='prepared')
    parser.add_argument(
        '--ordered-results-name',
        default='results_gru_attention',
    )
    parser.add_argument(
        '--shuffled-results-name',
        default='results_order_ablation',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument('--n-bootstrap', type=int, default=10_000)
    return parser.parse_args()


def sequence_length_table(sequences: pd.DataFrame) -> pd.DataFrame:
    test = sequences[sequences['split'].eq('test')].copy()
    test['n_pre_shot_events_raw'] = test['is_shot'].map(
        lambda values: int((~np.asarray(values, dtype=bool)).sum())
    )
    test['n_pre_shot_events'] = test['n_pre_shot_events_raw'].clip(upper=20)
    test['length_bin'] = pd.cut(
        test['n_pre_shot_events'],
        bins=[-1, 0, 2, 5, 10, 20],
        labels=LENGTH_ORDER[1:],
    ).astype(str)
    return test[
        [
            'shot_seq_id',
            'match_id',
            'n_pre_shot_events_raw',
            'n_pre_shot_events',
            'length_bin',
        ]
    ]


def read_condition_files(
    project_dir: Path,
    filename: str,
    *,
    results_name: str,
    ordered_results_name: str,
    shuffled_results_name: str,
    seeds: list[int],
) -> pd.DataFrame:
    frames = []
    conditions = {
        'ordered_to_ordered': (
            f'{ordered_results_name}/faithful/gru_attention'
        ),
        'ordered_to_shuffled': (
            f'{results_name}/cross_evaluation/'
            'gru_attention_ordered_to_shuffled'
        ),
        'shuffled_to_ordered': (
            f'{results_name}/cross_evaluation/'
            'gru_attention_shuffled_to_ordered'
        ),
        'shuffled_to_shuffled': (
            f'{shuffled_results_name}/faithful/gru_attention_shuffled'
        ),
    }
    for condition, relative_root in conditions.items():
        root = project_dir / relative_root
        files = [root / f'seed_{seed}' / filename for seed in seeds]
        files = [path for path in files if path.exists()]
        if len(files) != len(seeds):
            raise RuntimeError(
                f'Expected {len(seeds)} files for {condition}/{filename}, '
                f'found {len(files)}'
            )
        values = pd.concat(map(
            pd.read_parquet if filename.endswith('.parquet') else pd.read_csv,
            files,
        ), ignore_index=True)
        values['condition'] = condition
        frames.append(values)
    return pd.concat(frames, ignore_index=True)


def add_all_length_bin(values: pd.DataFrame) -> pd.DataFrame:
    all_lengths = values.copy()
    all_lengths['length_bin'] = 'all'
    return pd.concat([all_lengths, values], ignore_index=True)


def prediction_tables(
    predictions: pd.DataFrame,
    lengths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = add_all_length_bin(predictions.merge(lengths, on='shot_seq_id'))
    values['squared_error'] = (
        values['target_xg'] - values['prediction']
    ) ** 2
    values['absolute_error'] = (
        values['target_xg'] - values['prediction']
    ).abs()
    rows = []
    for keys, group in values.groupby(['condition', 'seed', 'length_bin']):
        denominator = ((group['target_xg'] - group['target_xg'].mean()) ** 2).sum()
        rows.append(
            {
                'condition': keys[0],
                'seed': int(keys[1]),
                'length_bin': keys[2],
                'n_sequences': int(group['shot_seq_id'].nunique()),
                'mse': float(group['squared_error'].mean()),
                'mae': float(group['absolute_error'].mean()),
                'r2': float(1.0 - group['squared_error'].sum() / denominator)
                if denominator > 0
                else np.nan,
            }
        )
    by_seed = pd.DataFrame(rows)
    summary = (
        by_seed.groupby(['condition', 'length_bin'], as_index=False)
        .agg(
            n_sequences=('n_sequences', 'max'),
            mse_mean=('mse', 'mean'),
            mse_sd=('mse', 'std'),
            mae_mean=('mae', 'mean'),
            mae_sd=('mae', 'std'),
            r2_mean=('r2', 'mean'),
            r2_sd=('r2', 'std'),
        )
    )
    return by_seed, summary


def faithfulness_tables(
    faithfulness: pd.DataFrame,
    lengths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = add_all_length_bin(faithfulness.merge(lengths, on='shot_seq_id'))
    by_seed = (
        values.groupby(['condition', 'seed', 'length_bin'], as_index=False)
        .agg(
            n_occlusion=('attention_vs_occlusion_rho', 'count'),
            attention_vs_occlusion=('attention_vs_occlusion_rho', 'mean'),
            n_ig=('attention_vs_ig_rho', 'count'),
            attention_vs_ig=('attention_vs_ig_rho', 'mean'),
        )
    )
    summary = (
        by_seed.groupby(['condition', 'length_bin'], as_index=False)
        .agg(
            n_occlusion=('n_occlusion', 'sum'),
            attention_vs_occlusion_mean=(
                'attention_vs_occlusion',
                'mean',
            ),
            attention_vs_occlusion_sd=('attention_vs_occlusion', 'std'),
            n_ig=('n_ig', 'sum'),
            attention_vs_ig_mean=('attention_vs_ig', 'mean'),
            attention_vs_ig_sd=('attention_vs_ig', 'std'),
        )
    )
    return by_seed, summary


def deletion_contrasts(
    deletions: pd.DataFrame,
    lengths: pd.DataFrame,
) -> pd.DataFrame:
    values = deletions.copy()
    values['delta_squared_error'] = (
        (values['target_xg'] - values['deleted_prediction']) ** 2
        - (values['target_xg'] - values['base_prediction']) ** 2
    )
    selected = values[values['method'].isin(['attention', 'random'])]
    paired = selected.pivot_table(
        index=['condition', 'seed', 'shot_seq_id', 'k'],
        columns='method',
        values='delta_squared_error',
    ).reset_index()
    paired['attention_minus_random'] = paired['attention'] - paired['random']
    return add_all_length_bin(paired.merge(lengths, on='shot_seq_id'))


def deletion_tables(
    contrasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = (
        contrasts.groupby(
            ['condition', 'seed', 'length_bin', 'k'],
            as_index=False,
        )
        .agg(
            n_sequences=('shot_seq_id', 'nunique'),
            attention_minus_random=('attention_minus_random', 'mean'),
        )
    )
    summary = (
        by_seed.groupby(['condition', 'length_bin', 'k'], as_index=False)
        .agg(
            n_sequences=('n_sequences', 'max'),
            difference_mean=('attention_minus_random', 'mean'),
            difference_sd=('attention_minus_random', 'std'),
        )
    )
    return by_seed, summary


def factorial_bootstrap(
    values: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
    n_bootstrap: int,
    seed_offset: int,
) -> pd.DataFrame:
    index = ['seed', 'match_id', 'shot_seq_id', 'length_bin', *group_columns]
    wide = values.pivot_table(
        index=index,
        columns='condition',
        values=value_column,
    ).reset_index()
    rows = []
    grouped = wide.groupby(['length_bin', *group_columns], dropna=False)
    for group_index, (keys, group) in enumerate(grouped):
        keys = keys if isinstance(keys, tuple) else (keys,)
        key_values = dict(zip(['length_bin', *group_columns], keys))
        for contrast_index, (contrast, weights) in enumerate(
            FACTORIAL_CONTRASTS.items()
        ):
            contrast_values = group[['seed', 'match_id', 'shot_seq_id']].copy()
            contrast_values['difference'] = sum(
                group[condition] * weight
                for condition, weight in weights.items()
            )
            if not contrast_values['difference'].notna().any():
                continue
            rows.append(
                {
                    **key_values,
                    'contrast': contrast,
                    **seed_match_bootstrap_mean(
                        contrast_values,
                        n_bootstrap=n_bootstrap,
                        seed=(
                            seed_offset
                            + group_index * len(FACTORIAL_CONTRASTS)
                            + contrast_index
                        ),
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = project_dir / args.results_name / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = pd.read_parquet(
        project_dir
        / 'data'
        / args.prepared_name
        / 'sequences_raw.parquet'
    )
    lengths = sequence_length_table(sequences)
    lengths.to_csv(output_dir / 'test_sequence_lengths.csv', index=False)

    predictions = read_condition_files(
        project_dir,
        'test_predictions.parquet',
        results_name=args.results_name,
        ordered_results_name=args.ordered_results_name,
        shuffled_results_name=args.shuffled_results_name,
        seeds=args.seeds,
    )
    faithfulness = read_condition_files(
        project_dir,
        'sequence_faithfulness.parquet',
        results_name=args.results_name,
        ordered_results_name=args.ordered_results_name,
        shuffled_results_name=args.shuffled_results_name,
        seeds=args.seeds,
    )
    deletions = read_condition_files(
        project_dir,
        'deletion_predictions.parquet',
        results_name=args.results_name,
        ordered_results_name=args.ordered_results_name,
        shuffled_results_name=args.shuffled_results_name,
        seeds=args.seeds,
    )

    prediction_seed, prediction_summary = prediction_tables(
        predictions,
        lengths,
    )
    faithfulness_seed, faithfulness_summary = faithfulness_tables(
        faithfulness,
        lengths,
    )
    deletion_values = deletion_contrasts(deletions, lengths)
    deletion_seed, deletion_summary = deletion_tables(deletion_values)

    prediction_seed.to_csv(
        output_dir / 'prediction_by_length_seed.csv',
        index=False,
    )
    prediction_summary.to_csv(
        output_dir / 'prediction_by_length_summary.csv',
        index=False,
    )
    faithfulness_seed.to_csv(
        output_dir / 'faithfulness_by_length_seed.csv',
        index=False,
    )
    faithfulness_summary.to_csv(
        output_dir / 'faithfulness_by_length_summary.csv',
        index=False,
    )
    deletion_seed.to_csv(
        output_dir / 'deletion_by_length_seed.csv',
        index=False,
    )
    deletion_summary.to_csv(
        output_dir / 'deletion_by_length_summary.csv',
        index=False,
    )

    prediction_values = add_all_length_bin(
        predictions.merge(lengths, on='shot_seq_id')
    )
    prediction_values['squared_error'] = (
        prediction_values['target_xg'] - prediction_values['prediction']
    ) ** 2
    factorial_bootstrap(
        prediction_values,
        value_column='squared_error',
        group_columns=[],
        n_bootstrap=args.n_bootstrap,
        seed_offset=9026,
    ).to_csv(
        output_dir / 'mse_factorial_seed_match_bootstrap.csv',
        index=False,
    )

    faithfulness_values = add_all_length_bin(
        faithfulness.merge(lengths, on='shot_seq_id')
    ).melt(
        id_vars=[
            'condition',
            'seed',
            'shot_seq_id',
            'match_id_x',
            'length_bin',
        ],
        value_vars=[
            'attention_vs_occlusion_rho',
            'attention_vs_ig_rho',
        ],
        var_name='metric',
        value_name='faithfulness',
    ).rename(columns={'match_id_x': 'match_id'})
    factorial_bootstrap(
        faithfulness_values,
        value_column='faithfulness',
        group_columns=['metric'],
        n_bootstrap=args.n_bootstrap,
        seed_offset=10_026,
    ).to_csv(
        output_dir / 'faithfulness_factorial_seed_match_bootstrap.csv',
        index=False,
    )

    factorial_bootstrap(
        deletion_values,
        value_column='attention_minus_random',
        group_columns=['k'],
        n_bootstrap=args.n_bootstrap,
        seed_offset=11_026,
    ).to_csv(
        output_dir / 'deletion_factorial_seed_match_bootstrap.csv',
        index=False,
    )
    print(f'Order follow-up summary written to {output_dir}')


if __name__ == '__main__':
    main()
