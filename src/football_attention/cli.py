from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

STAGES = ('refit', 'holdout', 'alignment', 'explanations', 'recency', 'leakage', 'order', 'architecture', 'summaries')


def commands(work_dir: Path, config: Path, seeds: list[int], stages: list[str]) -> list[list[str]]:
    seed_args = ['--seeds', *map(str, seeds)]
    development = ['--prepared-name', 'development']
    evaluation = [
        '--frozen-results-name', 'results/final',
        '--development-prepared-name', 'development',
        '--external-prepared-name', 'holdout',
    ]
    tasks: list[tuple[str, list[str]]] = []
    if 'refit' in stages:
        tasks.append(('refit', ['--config', str(config), '--results-name', 'results/final', *development]))
    if 'holdout' in stages:
        tasks.append(('holdout', [*evaluation, '--results-name', 'results/holdout']))
    if 'alignment' in stages:
        tasks.append(('alignment', [*evaluation, '--external-results-name', 'results/holdout',
                                    '--results-name', 'results/alignment', *seed_args]))
    if 'explanations' in stages:
        tasks.append(('explain', [*evaluation, '--external-results-name', 'results/holdout',
                                 '--results-name', 'results/explanations', *seed_args]))
    if 'recency' in stages:
        tasks.append(('recency', ['--source-results-name', 'results/explanations',
                                 '--development-prepared-name', 'development',
                                 '--external-prepared-name', 'holdout', *seed_args]))
    if 'leakage' in stages:
        tasks.append(('leakage', development))
    if 'order' in stages:
        for model, directory in (('gru_attention', 'ordered'), ('gru_attention_shuffled', 'shuffled')):
            tasks.append(('sequence', ['--models', model, '--results-name', f'results/{directory}',
                                      *development, *seed_args]))
        for task in ('order', 'order_summary'):
            tasks.append((task, ['--ordered-results-name', 'results/ordered',
                                 '--shuffled-results-name', 'results/shuffled',
                                 '--results-name', 'results/order', *development, *seed_args]))
    if 'architecture' in stages:
        tasks.append(('architecture', ['--frozen-results-name', 'results/final',
                                      '--results-name', 'results/architecture', *seed_args]))
    if 'summaries' in stages:
        tasks.append(('summaries', []))
    return [[sys.executable, '-m', f'football_attention.tasks.{task}',
             '--project-dir', str(work_dir), *args] for task, args in tasks]


def run(args: argparse.Namespace) -> None:
    from .frozen_final import build_frozen_protocol, load_or_create_protocol
    from .prepare import file_hash
    from .train import select_device

    root, config = args.work_dir.resolve(), args.config.resolve()
    protocol = build_frozen_protocol(root, config_path=config)
    steps = commands(root, config, protocol['seeds'], args.stages)
    if args.dry_run:
        for command in steps:
            print(shlex.join(command))
        return
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    inputs = {name: file_hash(root / 'data' / name / 'sequences_raw.parquet')
              for name in ('development', 'holdout')}
    packages = ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'torch')
    versions = {name: importlib.metadata.version(name) for name in packages}
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        source.update(str(path.relative_to(Path(__file__).parent)).encode())
        source.update(path.read_bytes())
    device = str(select_device(args.device))
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.run.lock'
    try:
        handle = lock.open('x')
    except FileExistsError as error:
        raise RuntimeError(f'{lock} exists. Check for a running process before removing it.') from error
    try:
        with handle:
            handle.write(str(os.getpid()))
        load_or_create_protocol(root / 'run_protocol.json', {
            'inputs': inputs, 'settings': protocol, 'versions': versions,
            'source_sha256': source.hexdigest(),
            'python': platform.python_version(), 'device': device, 'threads': args.threads,
        })
        environment = {
            **os.environ,
            'FOOTBALL_ATTENTION_DEVICE': device,
            'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': str(args.threads),
            'MKL_NUM_THREADS': str(args.threads),
            'OPENBLAS_NUM_THREADS': str(args.threads),
        }
        for index, command in enumerate(steps, 1):
            print(f'[{index}/{len(steps)}] {command[2]}', flush=True)
            subprocess.run(command, env=environment, check=True)
        print(f'Completed requested stages. Local outputs: {root / "results"}')
    finally:
        lock.unlink()


def main() -> None:
    from . import reevaluate_holdout, update_holdout

    parser = argparse.ArgumentParser(description='Football buildup prediction and attention diagnostics.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='Build corrected, nonempty sequence datasets')
    prepare.add_argument('--work-dir', type=Path, default=Path('local/run'))
    prepare.add_argument('--development-cache', type=Path, required=True)
    prepare.add_argument('--holdout-cache', type=Path, required=True)
    update_holdout.add_arguments(subparsers.add_parser(
        'update-holdout', help='Download and prepare a separate enlarged holdout snapshot',
    ))
    reevaluate_holdout.add_arguments(subparsers.add_parser(
        'evaluate-holdout', help='Evaluate saved models on a new holdout without training',
    ))
    tune = subparsers.add_parser(
        'tune-boosting',
        help='Tune full-buildup and final-event-only boosting on validation data',
    )
    tune.add_argument('--work-dir', type=Path, default=Path('local/run'))
    tune.add_argument('--trials', type=int, default=100)
    tune.add_argument('--sampler-seed', type=int, default=2027)
    tune.add_argument('--model-seed', type=int, default=1729)
    execute = subparsers.add_parser('run', help='Run or resume the frozen analysis')
    execute.add_argument('--work-dir', type=Path, default=Path('local/run'))
    execute.add_argument('--config', type=Path, default=Path('configs/frozen.json'))
    execute.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    execute.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    execute.add_argument('--threads', type=int, default=1)
    execute.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        from .prepare import prepare_datasets
        prepare_datasets(args.work_dir.resolve(), args.development_cache.resolve(), args.holdout_cache.resolve())
    elif args.command == 'update-holdout':
        update_holdout.run(args)
    elif args.command == 'evaluate-holdout':
        reevaluate_holdout.run(args)
    elif args.command == 'tune-boosting':
        from .tasks.tune_boosting import run as tune_boosting

        tune_boosting(
            args.work_dir.resolve(),
            trials=args.trials,
            sampler_seed=args.sampler_seed,
            model_seed=args.model_seed,
        )
    else:
        run(args)


if __name__ == '__main__':
    main()
