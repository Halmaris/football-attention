from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from football_attention.faithful_eval import (
    aggregate_event_attributions,
    aggregate_player_occlusion,
)
import numpy as np
from scipy.stats import spearmanr


ALIGNED = 'gru_attention_faithful'


TRANSFORMER = 'faithful_no_player_id'


def compare_player_rankings(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    score: str,
    ranking_type: str,
    min_sequences: int = 30,
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []
    for seed in sorted(set(left['seed']) & set(right['seed'])):
        left_seed = left[
            left['seed'].eq(seed)
            & left['sequence_count'].ge(min_sequences)
        ][['player_id', score]]
        right_seed = right[
            right['seed'].eq(seed)
            & right['sequence_count'].ge(min_sequences)
        ][['player_id', score]]
        paired = left_seed.merge(
            right_seed,
            on='player_id',
            suffixes=('_gru_attention', '_transformer'),
        )
        left_top = set(left_seed.nlargest(top_k, score)['player_id'])
        right_top = set(right_seed.nlargest(top_k, score)['player_id'])
        union = left_top | right_top
        rows.append(
            {
                'ranking_type': ranking_type,
                'seed': int(seed),
                'n_players': len(paired),
                'spearman': float(
                    spearmanr(
                        paired[f'{score}_gru_attention'],
                        paired[f'{score}_transformer'],
                    ).statistic
                ),
                f'jaccard_at_{top_k}': (
                    len(left_top & right_top) / len(union)
                    if union
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Compare player rankings from the two frozen attention '
            'architectures. No training or inference is performed.'
        )
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument('--frozen-results-name', default='results_frozen_final')
    parser.add_argument('--results-name', default='results_architecture_diagnostics')
    parser.add_argument('--min-player-sequences', type=int, default=30)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    return parser.parse_args()


def model_root(frozen_dir: Path, model_type: str) -> Path:
    group = 'faithful' if model_type == ALIGNED else 'neural'
    return frozen_dir / group / model_type


def read_seed_outputs(
    root: Path,
    seeds: list[int],
    filename: str,
) -> pd.DataFrame:
    paths = [root / f'seed_{seed}' / filename for seed in seeds]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f'Missing {len(missing)} frozen attribution files: {missing[:3]}'
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def add_player_names(values: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    result = values.copy()
    result.insert(
        result.columns.get_loc('player_id') + 1,
        'player_name',
        result['player_id'].map(
            lambda value: names.get(str(int(value)), f'pid{int(value)}')
        ),
    )
    return result


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    frozen_dir = project_dir / args.frozen_results_name
    output_dir = project_dir / args.results_name
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(
        (frozen_dir / 'frozen_protocol.json').read_text(encoding='utf-8')
    )
    if sorted(args.seeds) != sorted(int(seed) for seed in protocol['seeds']):
        raise RuntimeError('Requested seeds differ from the frozen protocol.')
    prepared_name = str(protocol.get('prepared_name', 'prepared'))
    names = json.loads(
        (
            project_dir
            / 'data'
            / prepared_name
            / 'player_id_to_name.json'
        ).read_text(encoding='utf-8')
    )

    reports: dict[str, dict[str, pd.DataFrame]] = {}
    for model_type in [ALIGNED, TRANSFORMER]:
        root = model_root(frozen_dir, model_type)
        player_rows = read_seed_outputs(
            root,
            args.seeds,
            'player_occlusion.parquet',
        )
        token_rows = read_seed_outputs(
            root,
            args.seeds,
            'token_attributions.parquet',
        )
        occlusion = add_player_names(
            aggregate_player_occlusion(player_rows),
            names,
        )
        attention = add_player_names(
            aggregate_event_attributions(token_rows),
            names,
        )
        attention = attention[attention['method'].eq('attention')].copy()
        reports[model_type] = {
            'occlusion': occlusion,
            'attention': attention,
        }
        occlusion.to_csv(
            output_dir / f'{model_type}_player_occlusion_by_seed.csv',
            index=False,
        )
        attention.to_csv(
            output_dir / f'{model_type}_attention_players_by_seed.csv',
            index=False,
        )

    by_seed = pd.concat(
        [
            compare_player_rankings(
                reports[ALIGNED]['occlusion'],
                reports[TRANSFORMER]['occlusion'],
                score='mean_contribution',
                ranking_type='signed_player_occlusion',
                min_sequences=args.min_player_sequences,
                top_k=args.top_k,
            ),
            compare_player_rankings(
                reports[ALIGNED]['attention'],
                reports[TRANSFORMER]['attention'],
                score='xg_contribution_per_sequence',
                ranking_type='attention',
                min_sequences=args.min_player_sequences,
                top_k=args.top_k,
            ),
        ],
        ignore_index=True,
    )
    by_seed.to_csv(
        output_dir / 'player_ranking_comparison_by_seed.csv',
        index=False,
    )
    summary = (
        by_seed.groupby('ranking_type', as_index=False)
        .agg(
            n_players=('n_players', 'min'),
            spearman_mean=('spearman', 'mean'),
            spearman_sd=('spearman', 'std'),
            jaccard_at_10_mean=('jaccard_at_10', 'mean'),
            jaccard_at_10_sd=('jaccard_at_10', 'std'),
        )
    )
    summary.to_csv(
        output_dir / 'player_ranking_comparison_summary.csv',
        index=False,
    )
    print(f'Frozen architecture diagnostics written to {output_dir}')


if __name__ == '__main__':
    main()
