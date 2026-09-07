from __future__ import annotations
import copy
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from .config import ExperimentConfig
from .faithful_model import FaithfulModel
from .train import move_batch, prediction_metrics, set_seed


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return np.nan
    return float(spearmanr(left, right).statistic)


def integrated_gradients(
    model: FaithfulModel,
    batch: dict[str, torch.Tensor],
    *,
    steps: int,
) -> torch.Tensor:
    embeddings = model.embed_events(batch).detach()
    total_gradient = torch.zeros_like(embeddings)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        interpolated = (alpha * embeddings).requires_grad_(True)
        predictions = model.forward_from_embeddings(
            interpolated,
            batch['valid_mask'],
        )
        # Explicit output gradients avoid an MPS reduction-backward issue that
        # can silently drop gradients for some rows in a batched calculation.
        gradient = torch.autograd.grad(
            predictions,
            interpolated,
            grad_outputs=torch.ones_like(predictions),
            retain_graph=False,
            create_graph=False,
        )[0]
        total_gradient += gradient.detach()
    attributions = (embeddings * total_gradient / steps).sum(dim=-1)
    return attributions.masked_fill(~batch['valid_mask'], 0.0)


def gradient_x_input(
    model: FaithfulModel,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    '''Return signed event-level gradient-times-input attributions.'''
    embeddings = model.embed_events(batch).detach().requires_grad_(True)
    predictions = model.forward_from_embeddings(
        embeddings,
        batch['valid_mask'],
    )
    gradients = torch.autograd.grad(
        predictions,
        embeddings,
        grad_outputs=torch.ones_like(predictions),
        retain_graph=False,
        create_graph=False,
    )[0]
    attributions = (embeddings * gradients).sum(dim=-1)
    return attributions.masked_fill(~batch['valid_mask'], 0.0).detach()


def recency_importance(valid: torch.Tensor) -> torch.Tensor:
    '''Rank later observed events above earlier events.'''
    ranks = valid.to(torch.float32).cumsum(dim=1)
    return ranks.masked_fill(~valid, 0.0)


@torch.no_grad()
def permutation_shapley(
    model: FaithfulModel,
    batch: dict[str, torch.Tensor],
    *,
    n_permutations: int,
    seed: int,
    coalition_batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Estimate signed event Shapley values with antithetic permutations.

    Missing events are represented by setting their validity mask to false.
    Each permutation starts from the model's learned empty-sequence state and
    adds events while preserving the recorded order among the active events.
    '''
    if n_permutations < 2:
        raise ValueError('n_permutations must be at least 2')
    if coalition_batch_size < 1:
        raise ValueError('coalition_batch_size must be positive')

    valid = batch['valid_mask']
    attributions = torch.zeros(valid.shape, device=valid.device)
    standard_errors = torch.zeros(valid.shape, device=valid.device)
    sequence_ids = batch.get('shot_seq_id')

    for row in range(valid.shape[0]):
        positions = torch.nonzero(valid[row], as_tuple=False).flatten()
        n_events = int(len(positions))
        if n_events == 0:
            continue

        sequence_id = (
            int(sequence_ids[row].detach().cpu())
            if sequence_ids is not None
            else row
        )
        sequence_seed = (int(seed) * 1_000_003 + sequence_id) % (2**32)
        generator = np.random.default_rng(sequence_seed)
        orders: list[np.ndarray] = []
        while len(orders) < n_permutations:
            order = generator.permutation(n_events)
            orders.append(order)
            if len(orders) < n_permutations:
                orders.append(order[::-1].copy())
        order_indices = np.stack(orders[:n_permutations])

        coalition_masks = torch.zeros(
            (n_permutations, n_events + 1, valid.shape[1]),
            dtype=torch.bool,
            device=valid.device,
        )
        for permutation_index, order in enumerate(order_indices):
            active = coalition_masks[permutation_index, 0].clone()
            for step, local_position in enumerate(order, start=1):
                active = active.clone()
                active[positions[int(local_position)]] = True
                coalition_masks[permutation_index, step] = active

        flat_masks = coalition_masks.flatten(0, 1)
        predictions = []
        for start in range(0, len(flat_masks), coalition_batch_size):
            masks = flat_masks[start : start + coalition_batch_size]
            count = len(masks)
            expanded = {
                key: value[row : row + 1]
                .expand(count, *value.shape[1:])
                .clone()
                for key, value in batch.items()
            }
            expanded['valid_mask'] = masks
            predictions.append(model(expanded))
        path_predictions = torch.cat(predictions).view(
            n_permutations,
            n_events + 1,
        )
        marginal_values = path_predictions[:, 1:] - path_predictions[:, :-1]

        samples = torch.zeros(
            (n_permutations, valid.shape[1]),
            device=valid.device,
        )
        event_positions = positions[
            torch.as_tensor(order_indices, device=valid.device)
        ]
        samples.scatter_(1, event_positions, marginal_values)
        attributions[row] = samples.mean(dim=0)
        standard_errors[row] = samples.std(dim=0, unbiased=True) / np.sqrt(
            n_permutations
        )

    return attributions, standard_errors


@torch.no_grad()
def event_occlusion_effects(
    model: FaithfulModel,
    batch: dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    valid = batch['valid_mask']
    effects = torch.zeros(valid.shape, device=valid.device)
    for position in range(valid.shape[1]):
        active = valid[:, position]
        if not active.any():
            continue
        masked_batch = dict(batch)
        masked_valid = valid.clone()
        masked_valid[active, position] = False
        masked_batch['valid_mask'] = masked_valid
        masked_prediction = model(masked_batch)
        effects[active, position] = (
            reference[active] - masked_prediction[active]
        )
    return effects


def top_k_mask(
    valid: torch.Tensor,
    importance: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(k, valid.shape[1])
    scores = importance.masked_fill(~valid, float('-inf'))
    positions = scores.topk(k, dim=1).indices
    counts = valid.sum(dim=1).clamp(max=k)
    active = torch.arange(k, device=valid.device)[None, :] < counts[:, None]
    deleted = valid.clone()
    rows = torch.arange(valid.shape[0], device=valid.device)[:, None]
    deleted[rows.expand_as(positions)[active], positions[active]] = False
    return deleted, counts


@torch.no_grad()
def player_occlusion_rows(
    model: FaithfulModel,
    batch: dict[str, torch.Tensor],
    targets: torch.Tensor,
    reference: torch.Tensor,
    *,
    model_type: str,
    seed: int,
) -> list[dict[str, float | int | str]]:
    valid = batch['valid_mask']
    players = batch['original_player_id']
    player_lists = []
    for row in range(valid.shape[0]):
        ids = players[row, valid[row]].detach().cpu().numpy().astype(int)
        player_lists.append(sorted(set(ids) - {-1}))
    rows = []
    for player_slot in range(max(map(len, player_lists), default=0)):
        active_rows = [
            row
            for row, player_ids in enumerate(player_lists)
            if player_slot < len(player_ids)
        ]
        if not active_rows:
            continue
        masked_batch = dict(batch)
        masked_valid = valid.clone()
        selected_players = {
            row: player_lists[row][player_slot]
            for row in active_rows
        }
        for row, player_id in selected_players.items():
            masked_valid[row] &= players[row].ne(player_id)
        masked_batch['valid_mask'] = masked_valid
        masked_predictions = model(masked_batch)
        for row, player_id in selected_players.items():
            rows.append(
                {
                    'model_type': model_type,
                    'seed': seed,
                    'shot_seq_id': int(batch['shot_seq_id'][row].cpu()),
                    'match_id': int(batch['match_id'][row].cpu()),
                    'player_id': int(player_id),
                    'target_xg': float(targets[row]),
                    'base_prediction': float(reference[row].cpu()),
                    'masked_prediction': float(masked_predictions[row].cpu()),
                    'contribution': float(
                        (reference[row] - masked_predictions[row]).cpu()
                    ),
                }
            )
    return rows


def evaluate_faithfulness(
    model: FaithfulModel,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    config: ExperimentConfig,
    model_type: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()
    token_rows = []
    sequence_rows = []
    deletion_rows = []
    player_rows = []
    random_generator = torch.Generator().manual_seed(seed + 100_000)

    for batch_index, (cpu_batch, cpu_targets) in enumerate(loader):
        batch = move_batch(cpu_batch, device)
        targets = cpu_targets.to(device)
        with torch.no_grad():
            reference, attention = model(batch, return_attention=True)
            occlusion = event_occlusion_effects(model, batch, reference)
        ig = integrated_gradients(
            model,
            batch,
            steps=config.integrated_gradients_steps,
        )
        valid = batch['valid_mask']
        attention_np = attention.detach().cpu().numpy()
        occlusion_np = occlusion.detach().cpu().numpy()
        ig_np = ig.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()

        for row in range(valid.shape[0]):
            positions = np.flatnonzero(valid_np[row])
            sequence_rows.append(
                {
                    'model_type': model_type,
                    'seed': seed,
                    'shot_seq_id': int(cpu_batch['shot_seq_id'][row]),
                    'match_id': int(cpu_batch['match_id'][row]),
                    'n_events': int(len(positions)),
                    'attention_vs_occlusion_rho': safe_spearman(
                        attention_np[row, positions],
                        np.abs(occlusion_np[row, positions]),
                    ),
                    'attention_vs_ig_rho': safe_spearman(
                        attention_np[row, positions],
                        np.abs(ig_np[row, positions]),
                    ),
                }
            )
            uniform = 1.0 / len(positions) if len(positions) else 0.0
            for position in positions:
                token_rows.append(
                    {
                        'model_type': model_type,
                        'seed': seed,
                        'shot_seq_id': int(cpu_batch['shot_seq_id'][row]),
                        'match_id': int(cpu_batch['match_id'][row]),
                        'token_position': int(position),
                        'player_id': int(
                            cpu_batch['original_player_id'][row, position]
                        ),
                        'target_xg': float(cpu_targets[row]),
                        'attention': float(attention_np[row, position]),
                        'uniform': uniform,
                        'occlusion_signed': float(occlusion_np[row, position]),
                        'occlusion_abs': float(abs(occlusion_np[row, position])),
                        'ig_signed': float(ig_np[row, position]),
                        'ig_abs': float(abs(ig_np[row, position])),
                    }
                )

        methods = {
            'attention': attention,
            'occlusion': occlusion.abs(),
            'integrated_gradients': ig.abs(),
            'random': torch.rand(
                valid.shape,
                generator=random_generator,
            ).to(device),
        }
        for method, importance in methods.items():
            for k in config.deletion_k:
                deleted_valid, n_removed = top_k_mask(valid, importance, k)
                deleted_batch = dict(batch)
                deleted_batch['valid_mask'] = deleted_valid
                with torch.no_grad():
                    deleted_prediction = model(deleted_batch)
                for row in range(valid.shape[0]):
                    deletion_rows.append(
                        {
                            'model_type': model_type,
                            'seed': seed,
                            'shot_seq_id': int(cpu_batch['shot_seq_id'][row]),
                            'method': method,
                            'k': int(k),
                            'n_removed': int(n_removed[row].cpu()),
                            'target_xg': float(cpu_targets[row]),
                            'base_prediction': float(reference[row].cpu()),
                            'deleted_prediction': float(
                                deleted_prediction[row].cpu()
                            ),
                        }
                    )

        player_rows.extend(
            player_occlusion_rows(
                model,
                batch,
                cpu_targets,
                reference,
                model_type=model_type,
                seed=seed,
            )
        )

    return (
        pd.DataFrame(token_rows),
        pd.DataFrame(sequence_rows),
        pd.DataFrame(deletion_rows),
        pd.DataFrame(player_rows),
    )


def summarize_deletion(deletions: pd.DataFrame) -> pd.DataFrame:
    values = deletions.copy()
    values['base_squared_error'] = (
        values['target_xg'] - values['base_prediction']
    ) ** 2
    values['deleted_squared_error'] = (
        values['target_xg'] - values['deleted_prediction']
    ) ** 2
    values['absolute_prediction_change'] = (
        values['base_prediction'] - values['deleted_prediction']
    ).abs()
    summary = (
        values.groupby(['model_type', 'seed', 'method', 'k'], as_index=False)
        .agg(
            n_sequences=('shot_seq_id', 'nunique'),
            mean_removed=('n_removed', 'mean'),
            base_mse=('base_squared_error', 'mean'),
            deleted_mse=('deleted_squared_error', 'mean'),
            mean_absolute_prediction_change=(
                'absolute_prediction_change',
                'mean',
            ),
        )
    )
    summary['delta_mse'] = summary['deleted_mse'] - summary['base_mse']
    return summary


def aggregate_player_occlusion(player_rows: pd.DataFrame) -> pd.DataFrame:
    report = (
        player_rows.groupby(
            ['model_type', 'seed', 'player_id'],
            as_index=False,
        )
        .agg(
            sequence_count=('shot_seq_id', 'nunique'),
            match_count=('match_id', 'nunique'),
            total_contribution=('contribution', 'sum'),
            mean_contribution=('contribution', 'mean'),
            contribution_sd=('contribution', 'std'),
            mean_abs_contribution=('contribution', lambda x: x.abs().mean()),
        )
    )
    return report


def aggregate_event_attributions(tokens: pd.DataFrame) -> pd.DataFrame:
    method_columns = {
        'attention': 'attention',
        'uniform': 'uniform',
        'occlusion': 'occlusion_abs',
        'integrated_gradients': 'ig_abs',
    }
    frames = []
    grouping = ['model_type', 'seed', 'shot_seq_id']
    for method, column in method_columns.items():
        values = tokens.copy()
        denominator = values.groupby(grouping)[column].transform('sum')
        values['weight'] = values[column] / denominator.clip(lower=1e-12)
        values['weighted_xg'] = values['weight'] * values['target_xg']
        values['method'] = method
        frames.append(values)
    long = pd.concat(frames, ignore_index=True)
    per_sequence = (
        long.groupby(
            [
                'model_type',
                'seed',
                'method',
                'shot_seq_id',
                'player_id',
            ],
            as_index=False,
        )
        .agg(
            event_count=('token_position', 'size'),
            contribution=('weight', 'sum'),
            xg_contribution=('weighted_xg', 'sum'),
        )
    )
    report = (
        per_sequence.groupby(
            ['model_type', 'seed', 'method', 'player_id'],
            as_index=False,
        )
        .agg(
            sequence_count=('shot_seq_id', 'nunique'),
            event_count=('event_count', 'sum'),
            total_contribution=('contribution', 'sum'),
            xg_weighted_contribution=('xg_contribution', 'sum'),
        )
    )
    report['contribution_per_sequence'] = (
        report['total_contribution'] / report['sequence_count']
    )
    report['xg_contribution_per_sequence'] = (
        report['xg_weighted_contribution'] / report['sequence_count']
    )
    return report


def compare_player_attribution_methods(
    report: pd.DataFrame,
    *,
    score: str = 'xg_contribution_per_sequence',
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []
    for keys, group in report.groupby(['model_type', 'seed']):
        attention = group[group['method'].eq('attention')][['player_id', score]]
        for method in ['uniform', 'occlusion', 'integrated_gradients']:
            comparison = group[group['method'].eq(method)][['player_id', score]]
            paired = attention.merge(
                comparison,
                on='player_id',
                suffixes=('_attention', '_comparison'),
            )
            rho = safe_spearman(
                paired[f'{score}_attention'].to_numpy(),
                paired[f'{score}_comparison'].to_numpy(),
            )
            top_attention = set(attention.nlargest(top_k, score)['player_id'])
            top_comparison = set(comparison.nlargest(top_k, score)['player_id'])
            union = top_attention | top_comparison
            rows.append(
                {
                    'model_type': keys[0],
                    'seed': keys[1],
                    'comparison': method,
                    'score': score,
                    'n_players': len(paired),
                    'spearman_rho': rho,
                    f'jaccard_at_{top_k}': (
                        len(top_attention & top_comparison) / len(union)
                        if union else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def randomized_attention_test(
    model: FaithfulModel,
    loader: torch.utils.data.DataLoader,
    trained_tokens: pd.DataFrame,
    *,
    device: torch.device,
    model_type: str,
    seed: int,
) -> pd.DataFrame:
    randomized = copy.deepcopy(model)
    set_seed(seed + 200_000)
    randomized.randomize_pool_attention()
    randomized.to(device).eval()
    rows = []
    for cpu_batch, _ in loader:
        batch = move_batch(cpu_batch, device)
        _, attention = randomized(batch, return_attention=True)
        attention = attention.cpu().numpy()
        valid = cpu_batch['valid_mask'].numpy()
        for row in range(valid.shape[0]):
            for position in np.flatnonzero(valid[row]):
                rows.append(
                    {
                        'shot_seq_id': int(cpu_batch['shot_seq_id'][row]),
                        'token_position': int(position),
                        'randomized_attention': float(attention[row, position]),
                    }
                )
    paired = trained_tokens.merge(
        pd.DataFrame(rows),
        on=['shot_seq_id', 'token_position'],
    )
    results = []
    for shot_seq_id, group in paired.groupby('shot_seq_id'):
        results.append(
            {
                'model_type': model_type,
                'seed': seed,
                'shot_seq_id': int(shot_seq_id),
                'trained_vs_randomized_rho': safe_spearman(
                    group['attention'].to_numpy(),
                    group['randomized_attention'].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(results)


def bootstrap_player_occlusion(
    player_rows: pd.DataFrame,
    *,
    model_type: str,
    min_sequences: int,
    n_bootstrap: int,
    seed: int = 2026,
) -> pd.DataFrame:
    values = player_rows[player_rows['model_type'].eq(model_type)]
    values = (
        values.groupby(
            ['shot_seq_id', 'match_id', 'player_id'],
            as_index=False,
        )['contribution']
        .mean()
    )
    generator = np.random.default_rng(seed)
    rows = []
    for player_id, group in values.groupby('player_id'):
        if group['shot_seq_id'].nunique() < min_sequences:
            continue
        match_values = (
            group.groupby('match_id')['contribution']
            .agg(['sum', 'count'])
            .reset_index(drop=True)
        )
        samples = []
        for _ in range(n_bootstrap):
            indices = generator.integers(0, len(match_values), len(match_values))
            sampled = match_values.iloc[indices]
            samples.append(sampled['sum'].sum() / sampled['count'].sum())
        rows.append(
            {
                'model_type': model_type,
                'player_id': int(player_id),
                'sequence_count': int(group['shot_seq_id'].nunique()),
                'match_count': int(group['match_id'].nunique()),
                'mean_contribution': float(group['contribution'].mean()),
                'ci_025': float(np.quantile(samples, 0.025)),
                'ci_975': float(np.quantile(samples, 0.975)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            'model_type',
            'player_id',
            'sequence_count',
            'match_count',
            'mean_contribution',
            'ci_025',
            'ci_975',
        ],
    )


def cross_seed_player_stability(
    report: pd.DataFrame,
    *,
    score: str,
    min_sequences: int,
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []
    for model_type, group in report.groupby('model_type'):
        pair_values = []
        for seed_a, seed_b in combinations(sorted(group['seed'].unique()), 2):
            left = group[
                group['seed'].eq(seed_a)
                & group['sequence_count'].ge(min_sequences)
            ][['player_id', score]]
            right = group[
                group['seed'].eq(seed_b)
                & group['sequence_count'].ge(min_sequences)
            ][['player_id', score]]
            paired = left.merge(right, on='player_id', suffixes=('_a', '_b'))
            top_left = set(left.nlargest(top_k, score)['player_id'])
            top_right = set(right.nlargest(top_k, score)['player_id'])
            union = top_left | top_right
            pair_values.append(
                {
                    'rho': safe_spearman(
                        paired[f'{score}_a'].to_numpy(),
                        paired[f'{score}_b'].to_numpy(),
                    ),
                    'jaccard': len(top_left & top_right) / len(union),
                }
            )
        pairs = pd.DataFrame(pair_values, columns=['rho', 'jaccard'])
        rows.append(
            {
                'model_type': model_type,
                'score': score,
                'min_sequences': min_sequences,
                'n_seed_pairs': len(pairs),
                'spearman_mean': pairs['rho'].mean(),
                'spearman_sd': pairs['rho'].std(),
                f'jaccard_at_{top_k}_mean': pairs['jaccard'].mean(),
                f'jaccard_at_{top_k}_sd': pairs['jaccard'].std(),
            }
        )
    return pd.DataFrame(rows)


def predictive_metric_row(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    model_type: str,
    seed: int,
    split: str,
    best_epoch: int,
) -> dict[str, float | int | str]:
    metrics = prediction_metrics(targets, predictions)
    return {
        'model_type': model_type,
        'seed': seed,
        'split': split,
        'mse': metrics.mse,
        'mae': metrics.mae,
        'r2': metrics.r2,
        'best_epoch': best_epoch,
    }
