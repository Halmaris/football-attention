from __future__ import annotations
import numpy as np
import pandas as pd


def seed_match_bootstrap_distribution(
    values: pd.DataFrame,
    *,
    value_column: str = 'difference',
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> tuple[float, np.ndarray, dict[str, int]]:
    """Return a crossed seed-by-match bootstrap distribution of a mean."""
    required = {'seed', 'match_id', 'shot_seq_id', value_column}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f'Missing bootstrap columns: {sorted(missing)}')
    complete = values[
        ['seed', 'match_id', 'shot_seq_id', value_column]
    ].dropna()
    if complete.empty:
        raise ValueError('No complete rows for seed-match bootstrap')

    seeds = np.sort(complete['seed'].unique())
    matches = np.sort(complete['match_id'].unique())
    cells = (
        complete.groupby(['seed', 'match_id'])[value_column]
        .agg(['sum', 'count'])
    )
    full_index = pd.MultiIndex.from_product(
        [seeds, matches],
        names=['seed', 'match_id'],
    )
    cells = cells.reindex(full_index, fill_value=0)
    sums = cells['sum'].to_numpy().reshape(len(seeds), len(matches))
    counts = cells['count'].to_numpy().reshape(len(seeds), len(matches))

    generator = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_seeds = generator.integers(0, len(seeds), len(seeds))
        sampled_matches = generator.integers(0, len(matches), len(matches))
        selected_sums = sums[np.ix_(sampled_seeds, sampled_matches)].sum()
        selected_counts = counts[np.ix_(sampled_seeds, sampled_matches)].sum()
        samples[index] = selected_sums / selected_counts

    metadata = {
        'n_seeds': int(len(seeds)),
        'n_matches': int(len(matches)),
        'n_sequences': int(complete['shot_seq_id'].nunique()),
        'n_rows': int(len(complete)),
        'n_bootstrap': int(n_bootstrap),
    }
    return float(complete[value_column].mean()), samples, metadata


def seed_match_bootstrap_mean(
    values: pd.DataFrame,
    *,
    value_column: str = 'difference',
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Bootstrap a mean while preserving paired seed and match clusters."""
    estimate, samples, metadata = seed_match_bootstrap_distribution(
        values,
        value_column=value_column,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        'difference_mean': estimate,
        'ci_025': float(np.quantile(samples, 0.025)),
        'ci_975': float(np.quantile(samples, 0.975)),
        **metadata,
    }
