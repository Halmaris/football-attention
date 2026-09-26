from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import pandas as pd

from .prepare import file_hash
from .update_holdout import run_lock, snapshot_name, write_json_atomic


STAGES = ('holdout', 'alignment', 'explanations', 'recency', 'summaries', 'figures')


def commands(root: Path, name: str, seeds: list[int], stages: list[str], *,
             n_bootstrap: int = 10_000, shapley_permutations: int = 64) -> list[list[str]]:
    destination = f'results/{name}'
    evaluation = ['--external-prepared-name', name]
    seed_args = ['--seeds', *map(str, seeds)]
    bootstrap = ['--n-bootstrap', str(n_bootstrap)]
    steps = []
    if 'holdout' in stages:
        steps.append(('tasks.holdout', [*evaluation, '--results-name', f'{destination}/holdout',
                                       *bootstrap]))
    if 'alignment' in stages:
        for stage in ('external', 'aggregate-external'):
            steps.append(('tasks.alignment', [*evaluation, '--stage', stage,
                          '--source-controls-name', 'results/alignment',
                          '--external-results-name', f'{destination}/holdout',
                          '--results-name', f'{destination}/alignment', *seed_args, *bootstrap]))
    if 'explanations' in stages:
        steps.append(('tasks.explain', [*evaluation, '--datasets', 'frozen_holdout',
                      '--external-results-name', f'{destination}/holdout',
                      '--results-name', f'{destination}/explanations', *seed_args, *bootstrap,
                      '--shapley-permutations', str(shapley_permutations)]))
    if 'recency' in stages:
        steps.append(('tasks.recency', [*evaluation, '--datasets', 'frozen_holdout',
                      '--source-results-name', f'{destination}/explanations',
                      '--results-name', f'{destination}/recency', *seed_args, *bootstrap]))
    if 'summaries' in stages:
        steps.append(('tasks.summaries', ['--holdout-results-name', f'{destination}/holdout',
                      '--results-name', f'{destination}/summaries']))
    if 'figures' in stages:
        steps.append(('plots', ['--holdout-name', name,
                      '--holdout-results-name', f'{destination}/holdout',
                      '--explanation-results-name', f'{destination}/explanations',
                      '--output-name', f'{destination}/figures']))
    return [[sys.executable, '-m', f'football_attention.{module}',
             '--work-dir' if module == 'plots' else '--project-dir', str(root), *arguments]
            for module, arguments in steps]


def source_fingerprints(root: Path, name: str, protocol: dict) -> dict[str, str]:
    from .tasks.refit import run_dir
    from .tasks.alignment import MODEL_MATCHED, checkpoint_path

    frozen = root / 'results/final'
    paths = [root / 'data/development/sequences_raw.parquet',
             root / 'data' / name / 'sequences_raw.parquet',
             frozen / 'frozen_protocol.json', frozen / 'classical/models.pkl',
             root / 'results/alignment/alignment_control_protocol.json']
    for seed in protocol['seeds']:
        for model_type in protocol['neural_models']:
            paths.append(run_dir(frozen, model_type, int(seed)) / 'model.pt')
        paths.append(checkpoint_path(frozen, root / 'results/alignment', MODEL_MATCHED, int(seed)))
    paths.extend(path for path in frozen.glob('*') if path.suffix in {'.csv', '.parquet'})
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def validate_inputs(root: Path, name: str) -> None:
    historical = pd.read_parquet(root / 'data/development/sequences_raw.parquet')
    holdout = pd.read_parquet(root / 'data' / name / 'sequences_raw.parquet')
    if (holdout.empty or not holdout['n_events'].gt(1).all()
            or not holdout['xg'].between(0, 1).all()
            or holdout['shot_seq_id'].duplicated().any()
            or holdout['shot_event_id'].duplicated().any()):
        raise ValueError('Invalid or empty holdout sequences.')
    if set(holdout['match_id']) & set(historical['match_id']):
        raise ValueError('Development and holdout matches overlap.')
    if pd.to_datetime(holdout['match_date']).min() <= pd.to_datetime(historical['match_date']).max():
        raise ValueError('Holdout dates must follow development.')
    original = root / 'run_protocol.json'
    if original.exists():
        expected = json.loads(original.read_text())['inputs']['development']
        if expected != file_hash(root / 'data/development/sequences_raw.parquet'):
            raise ValueError('Development data changed since the original run.')


def run(args: argparse.Namespace) -> None:
    from .frozen_final import load_or_create_protocol
    from .tasks.refit import verify_training_complete
    from .tasks.alignment import build_control_protocol
    from .train import select_device

    root = args.work_dir.resolve()
    frozen = root / 'results/final'
    protocol = json.loads((frozen / 'frozen_protocol.json').read_text())
    output = root / 'results' / args.name
    if args.name in {'final', 'holdout', 'alignment', 'explanations', 'recency',
                     'summaries', 'figures', 'order', 'ordered', 'shuffled', 'architecture',
                     'leakage', 'boosting_hpo'}:
        raise ValueError('Use a new snapshot name, not an existing analysis stage.')
    if args.threads < 1 or args.n_bootstrap < 1 or args.shapley_permutations < 2:
        raise ValueError('Threads and bootstrap samples must be positive; use at least two permutations.')
    steps = commands(root, args.name, protocol['seeds'], args.stages,
                     n_bootstrap=args.n_bootstrap, shapley_permutations=args.shapley_permutations)
    if args.dry_run:
        for command in steps:
            print(shlex.join(command))
        return
    with run_lock(root):
        validate_inputs(root, args.name)
        verify_training_complete(frozen, protocol)
        control = json.loads((root / 'results/alignment/alignment_control_protocol.json').read_text())
        expected = build_control_protocol(frozen, protocol, seeds=protocol['seeds'], quick=False,
                                          development_prepared_name='development')
        if control != expected:
            raise ValueError('Matched-control protocol does not match the frozen models.')
        fingerprints = source_fingerprints(root, args.name, protocol)
        source = hashlib.sha256()
        for path in sorted(Path(__file__).parent.rglob('*.py')):
            source.update(str(path.relative_to(Path(__file__).parent)).encode())
            source.update(path.read_bytes())
        device = str(select_device(args.device))
        candidate = {
            'files': fingerprints, 'source_sha256': source.hexdigest(),
            'versions': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'torch')},
            'python': platform.python_version(), 'device': device, 'threads': args.threads,
            'n_bootstrap': args.n_bootstrap, 'shapley_permutations': args.shapley_permutations,
            'training_performed': False, 'tuning_performed': False,
        }
        saved = output / 'reevaluation_protocol.json'
        if output.exists() and not saved.exists() and any(output.iterdir()):
            raise FileExistsError(f'{output} already contains unrelated results.')
        load_or_create_protocol(saved, candidate)
        environment = {**os.environ, 'FOOTBALL_ATTENTION_DEVICE': device, 'PYTHONUNBUFFERED': '1',
                       **{key: str(args.threads) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                          'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}}
        for command in steps:
            print(shlex.join(command), flush=True)
            subprocess.run(command, env=environment, check=True)
        if fingerprints != source_fingerprints(root, args.name, protocol):
            raise RuntimeError('Input data or saved model files changed during evaluation.')
        required = ['holdout/external_evaluation_complete.json',
                    'alignment/external_evaluation_complete.json',
                    'explanations/evaluation_complete.json', 'recency/aggregation_complete.json',
                    'summaries/predictive_metrics.csv', 'figures/summary.pdf',
                    'figures/boosting_group_permutation_importance.pdf']
        if all((output / path).is_file() for path in required):
            write_json_atomic(output / 'reevaluation_complete.json', {
                'completed_at_utc': datetime.now(UTC).isoformat(),
                'training_performed': False, 'tuning_performed': False,
                'source_files_unchanged': True, 'snapshot': args.name,
            })
    print(f'Completed requested inference-only stages: {output}')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--work-dir', type=Path, default=Path('local/run'))
    parser.add_argument('--name', required=True, type=snapshot_name)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--n-bootstrap', type=int, default=10_000)
    parser.add_argument('--shapley-permutations', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true')
