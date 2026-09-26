import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from football_attention import update_holdout
from football_attention.download import normalize_match_metadata
from football_attention.prepare import file_hash
from football_attention.reevaluate_holdout import STAGES, commands
from test_pipeline import prepared, synthetic_cache


def expanded_cache(work):
    cache = work.parent / 'expanded'
    synthetic_cache(cache, offset=101, count=5, date='2021-01-01')
    path = cache / 'matches_c1_s1.parquet'
    matches = pd.read_parquet(path)
    matches.loc[matches['match_id'].eq(105), 'match_date'] = '2020-12-31'
    matches['match_status'] = 'available'
    matches['home_score'] = 1
    matches['away_score'] = 0
    matches.to_parquet(path, index=False)
    for mid in matches['match_id']:
        pd.DataFrame({'player_id': [1, 2, 3]}).to_parquet(cache / f'lineups_m{mid}.parquet')
    return cache


def extend(work):
    return update_holdout.prepare_snapshot(
        expanded_cache(work), work / 'data/expanded', work / 'data/holdout',
        work / 'data/development', '2021-01-31', work.parent / 'holdout',
    )


def test_expansion_preserves_old_sequences_and_exclusions(prepared):
    original_path = prepared / 'data/holdout/sequences_raw.parquet'
    before = file_hash(original_path)
    report = extend(prepared)
    previous = pd.read_parquet(original_path)
    rows = pd.read_parquet(prepared / 'data/expanded/sequences_raw.parquet')
    pd.testing.assert_frame_equal(rows.iloc[:len(previous)].reset_index(drop=True), previous)
    assert report['extended_match_count'] == 5
    assert report['extended_sequence_count'] == 45
    assert report['excluded_sequences'] == 15
    assert report['new_sequence_count'] == 9
    assert report['old_sequences_identical']
    assert report['source_data_checks']['raw_shots'] == 60
    assert rows.loc[rows['match_id'].eq(105), 'shot_seq_id'].min() > previous['shot_seq_id'].max()
    assert file_hash(original_path) == before
    with pytest.raises(FileExistsError):
        update_holdout.prepare_snapshot(
            prepared.parent / 'expanded', prepared / 'data/expanded',
            prepared / 'data/holdout', prepared / 'data/development',
            '2021-01-31', prepared.parent / 'holdout',
        )


def test_old_format_preparation_can_be_extended(prepared):
    (prepared / 'data/holdout/sequence_ids.parquet').unlink()
    assert extend(prepared)['old_sequences_identical']


def test_changed_previous_events_are_rejected(prepared):
    cache = expanded_cache(prepared)
    path = cache / 'events_m101.parquet'
    events = pd.read_parquet(path)
    events.loc[events['type'].eq('Shot'), 'shot_statsbomb_xg'] = 0.91
    events.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Previously retained sequences changed'):
        update_holdout.prepare_snapshot(
            cache, prepared / 'data/expanded', prepared / 'data/holdout',
            prepared / 'data/development', '2021-01-31', prepared.parent / 'holdout',
        )
    assert not (prepared / 'data/expanded').exists()


def test_incomplete_cache_is_not_published(prepared):
    cache = expanded_cache(prepared)
    (cache / 'events_m105.parquet').unlink()
    with pytest.raises(ValueError, match='events: missing='):
        update_holdout.validate_cache(cache, '2021-01-31')


def test_missing_xg_is_counted_and_out_of_range_xg_rejected(prepared):
    cache = expanded_cache(prepared)
    path = cache / 'events_m105.parquet'
    events = pd.read_parquet(path)
    events.loc[events['type'].eq('Shot'), 'shot_statsbomb_xg'] = None
    events.to_parquet(path, index=False)
    _, report = update_holdout.validate_cache(cache, '2021-01-31')
    assert report['shots_missing_xg'] == 12
    events.loc[events['type'].eq('Shot'), 'shot_statsbomb_xg'] = 1.01
    events.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='outside'):
        update_holdout.validate_cache(cache, '2021-01-31')


def test_mixed_manager_ids_can_be_saved(tmp_path):
    source = pd.DataFrame({'home_managers_id': [42, '42, 73', None],
                           'managers': [[{'id': 42}], None, []]})
    result = normalize_match_metadata(source)
    result.to_parquet(tmp_path / 'matches.parquet', index=False)
    restored = pd.read_parquet(tmp_path / 'matches.parquet')
    assert restored['home_managers_id'].tolist()[:2] == ['42', '42, 73']
    assert source['home_managers_id'].iloc[0] == 42


def test_downloader_reuses_old_resources_and_fixes_the_selection(prepared, monkeypatch):
    full = expanded_cache(prepared)
    old = prepared.parent / 'holdout'
    for path in full.glob('lineups_m*.parquet'):
        if path.name != 'lineups_m105.parquet':
            shutil.copy2(path, old / path.name)
    seen = []

    def events(*, match_id, creds):
        seen.append(('events', match_id))
        return pd.read_parquet(full / f'events_m{match_id}.parquet')

    def lineups(*, match_id, creds):
        seen.append(('lineups', match_id))
        return pd.read_parquet(full / f'lineups_m{match_id}.parquet')

    fake = SimpleNamespace(
        matches=lambda **kwargs: pd.read_parquet(full / 'matches_c1_s1.parquet'),
        events=events, lineups=lineups,
    )
    monkeypatch.setitem(sys.modules, 'statsbombpy', SimpleNamespace(sb=fake))
    monkeypatch.setattr(update_holdout, 'read_credentials', lambda: {})
    hashes = {path: file_hash(path) for path in old.glob('*.parquet')}
    destination = prepared.parent / 'downloaded'
    update_holdout.download_snapshot(destination, old, '2021-01-31')
    assert seen == [('events', 105), ('lineups', 105)]
    assert all(file_hash(path) == value for path, value in hashes.items())
    fake.matches = lambda **kwargs: pytest.fail('A saved selection must not be refreshed on resume')
    update_holdout.download_snapshot(destination, old, '2021-01-31')
    assert len(seen) == 2
    with pytest.raises(ValueError, match='different selection'):
        update_holdout.download_snapshot(destination, old, '2021-02-01')


def test_reevaluation_plan_never_contains_training():
    plan = commands(Path('local/run'), 'expanded', [42, 43], list(STAGES))
    modules = {command[2] for command in plan}
    assert not any(name.endswith(('.refit', '.sequence', '.tune_boosting', '.order')) for name in modules)
    controls = [command for command in plan if command[2].endswith('.alignment')]
    assert {command[command.index('--stage') + 1] for command in controls} == {
        'external', 'aggregate-external',
    }
    assert all('results/alignment' in command for command in controls)
    assert all('results/expanded/alignment' in command for command in controls)


@pytest.mark.parametrize('name', ['', '..', '../final', '/tmp/data', 'a/b', 'a.b'])
def test_snapshot_name_cannot_escape_local_directories(name):
    with pytest.raises(argparse.ArgumentTypeError):
        update_holdout.snapshot_name(name)
