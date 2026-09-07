import importlib
import json
import pkgutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import football_attention
from football_attention.cli import commands, main
from football_attention.frozen_final import build_frozen_protocol
from football_attention.prepare import prepare_datasets


def synthetic_cache(root, *, offset, count, date):
    root.mkdir()
    matches = pd.DataFrame({
        'match_id': np.arange(offset, offset + count),
        'match_date': pd.date_range(date, periods=count).strftime('%Y-%m-%d'),
    })
    matches.to_parquet(root / 'matches_c1_s1.parquet', index=False)
    for mid in matches['match_id']:
        events = []
        for possession in range(12):
            types = ['Shot'] if possession < 3 else ['Pass', 'Ball Receipt*', 'Carry', 'Pass', 'Shot']
            for position, kind in enumerate(types):
                index = len(events)
                events.append({
                    'id': f'{mid}-{index}', 'index': index, 'period': 1,
                    'minute': index, 'second': 0, 'possession': possession,
                    'type': kind, 'team': 'Team A', 'player_id': 1 + position % 3,
                    'player': f'Player {1 + position % 3}',
                    'location': [20.0 + 15 * position + (mid + possession) % 5, 10.0 + possession],
                    'pass_end_location': [50.0 + position, 30.0],
                    'shot_outcome': 'Goal' if possession % 3 == 0 else 'Saved',
                    'ball_receipt_outcome': 'Incomplete' if possession % 2 else None,
                    'shot_statsbomb_xg': (0.03 + ((mid + possession) % 10) * 0.02) if kind == 'Shot' else None,
                })
        pd.DataFrame(events).to_parquet(root / f'events_m{mid}.parquet', index=False)


@pytest.fixture
def prepared(tmp_path):
    synthetic_cache(tmp_path / 'development', offset=1, count=20, date='2020-01-01')
    synthetic_cache(tmp_path / 'holdout', offset=101, count=4, date='2021-01-01')
    work = tmp_path / 'run'
    prepare_datasets(work, tmp_path / 'development', tmp_path / 'holdout')
    return work


def test_preparation_preserves_ids_and_excludes_shot_only(prepared):
    rows = pd.read_parquet(prepared / 'data/development/sequences_raw.parquet')
    assert len(rows) == 180
    assert rows['shot_seq_id'].min() == 3
    assert rows['n_events'].gt(1).all()
    assert set(rows['split']) == {'train', 'validation', 'test'}
    report = json.loads((prepared / 'data/development/preparation.json').read_text())
    assert report['excluded_sequences'] == 60


def test_preparation_will_not_replace_existing_data(prepared):
    with pytest.raises(FileExistsError):
        prepare_datasets(prepared, prepared.parent / 'development', prepared.parent / 'holdout')


def test_all_modules_import_without_original_project():
    for module in pkgutil.walk_packages(football_attention.__path__, football_attention.__name__ + '.'):
        if not module.name.endswith('.__main__'):
            importlib.import_module(module.name)


def test_config_does_not_need_optuna(tmp_path):
    config = Path(__file__).parents[1] / 'configs/frozen.json'
    protocol = build_frozen_protocol(tmp_path, config_path=config)
    assert len(protocol['neural_models']) == 5
    assert protocol['fit_splits'] == ['train', 'validation']
    assert 'source_files' not in protocol


def test_runner_does_not_start_twice(prepared, monkeypatch):
    (prepared / '.run.lock').write_text('123')
    config = Path(__file__).parents[1] / 'configs/frozen.json'
    monkeypatch.setattr(sys, 'argv', ['football-attention', 'run', '--work-dir', str(prepared),
                                    '--config', str(config), '--stages', 'refit', '--device', 'cpu'])
    with pytest.raises(RuntimeError, match='running process'):
        main()
    assert (prepared / '.run.lock').read_text() == '123'


def test_pipeline_on_synthetic_data(prepared, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv('FOOTBALL_ATTENTION_DEVICE', 'cpu')
    settings = json.loads((Path(__file__).parents[1] / 'configs/frozen.json').read_text())
    settings['seeds'] = [42, 43]
    for spec in settings['neural_models'].values():
        spec['fixed_epochs'] = 1
        spec['config'].update(d_model=8, n_heads=2, n_layers=1, batch_size=32,
                              integrated_gradients_steps=4, occlusion_samples=2)
    settings['classical_models']['gradient_boosting'].update(max_iter=3, min_samples_leaf=2)
    settings['classical_models']['gradient_boosting_last_event'].update(
        max_iter=3,
        min_samples_leaf=2,
    )
    config = prepared / 'small-config.json'
    config.write_text(json.dumps(settings))
    from football_attention.tasks import sequence
    original = sequence.make_config
    monkeypatch.setattr(sequence, 'make_config', lambda args: replace(
        original(args), d_model=8, n_heads=2, n_layers=1, max_epochs=1,
        integrated_gradients_steps=4, occlusion_samples=2, bootstrap_samples=10,
    ))
    stages = ['refit', 'holdout', 'alignment', 'explanations', 'recency', 'leakage', 'order', 'architecture', 'summaries']
    for command in commands(prepared, config, settings['seeds'], stages):
        name, arguments = command[2], command[3:]
        if name.rsplit('.', 1)[1] in {'holdout', 'alignment', 'explain', 'recency', 'order_summary'}:
            arguments = [*arguments, '--n-bootstrap', '20']
        if name.endswith('.explain'):
            arguments += ['--shapley-permutations', '4']
        monkeypatch.setattr(sys, 'argv', [name, *arguments])
        importlib.import_module(name).main()
    assert (prepared / 'results/final/test_evaluation_complete.json').is_file()
    assert (prepared / 'results/recency/aggregation_complete.json').is_file()
    for dataset in ('development_test', 'frozen_holdout'):
        summary = pd.read_csv(prepared / 'results/recency' / dataset / 'correlation_bootstrap.csv')
        assert len(summary) == 9
        assert summary['n_seeds'].between(1, 2).all()
        tokens = pd.read_parquet(prepared / 'results/explanations' / dataset / 'token_attributions.parquet')
        assert set(tokens['seed']) == {42, 43}
    assert (prepared / 'results/order/summary').is_dir()
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    monkeypatch.setattr(sys, 'argv', ['plots', '--work-dir', str(prepared), '--team', 'Team A',
                                    '--players', '1', '2', '3', '--sequence-id', '231'])
    importlib.import_module('football_attention.plots').main()
    assert {p.name for p in (prepared / 'results/figures').glob('*.pdf')} == {
        'summary.pdf',
        'buildup_value_by_length.pdf',
        'boosting_group_permutation_importance.pdf',
        'player_responses.pdf',
        'player_attention.pdf',
        'sequence.pdf',
    }
