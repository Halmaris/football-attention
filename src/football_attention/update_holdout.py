from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, date, datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .data import build_external_sequence_dataset, object_name
from .download import as_dataframe, normalize_match_metadata, read_credentials, save_parquet, select_matches
from .prepare import cache_metadata, file_hash


SEQUENCE_FILE = 'sequences_raw.parquet'


@contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.run.lock'
    try:
        handle = lock.open('x')
    except FileExistsError as error:
        raise RuntimeError(f'{lock} exists; check for a running process before removing it.') from error
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def snapshot_name(value: str) -> str:
    if not value or not all(character.isascii() and (character.isalnum() or character in '-_')
                            for character in value):
        raise argparse.ArgumentTypeError('Use a snapshot name containing letters, digits, - or _.')
    return value


def select_available(matches: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    if matches['match_id'].isna().any() or matches['match_id'].duplicated().any():
        raise ValueError('Match IDs must be present and unique.')
    selected = select_matches(
        matches, match_week_start=None, match_week_end=None, limit=None,
        min_matches=None, match_date_end=cutoff, available_only=True,
    )
    scores = selected[['home_score', 'away_score']].apply(pd.to_numeric, errors='coerce')
    if scores.isna().any().any() or scores.lt(0).any().any():
        raise ValueError('An available match has no valid final score.')
    return selected


def download_snapshot(cache: Path, previous_cache: Path, cutoff: str) -> None:
    competition_id, season_id, _ = cache_metadata(previous_cache)
    matches_file = f'matches_c{competition_id}_s{season_id}.parquet'
    if cache.resolve() == previous_cache.resolve():
        raise ValueError('Use a new cache directory; the previous cache is immutable.')
    from statsbombpy import sb

    creds = read_credentials()
    metadata_path = cache / matches_file
    selection_path = cache / 'selection_manifest.json'
    if not metadata_path.exists():
        matches = as_dataframe(
            sb.matches(competition_id=competition_id, season_id=season_id, creds=creds), 'matches',
        )
        selected = select_available(matches, cutoff)
        previous_ids = {
            int(path.stem.removeprefix('events_m'))
            for path in previous_cache.glob('events_m*.parquet')
        }
        missing = previous_ids - set(selected['match_id'].astype(int))
        if missing:
            raise ValueError(f'Previously included matches are no longer eligible: {sorted(missing)}')
        cache.mkdir(parents=True, exist_ok=True)
        manifest = {
            'checked_at_utc': datetime.now(UTC).isoformat(),
            'data_cutoff': cutoff, 'competition_id': competition_id, 'season_id': season_id,
            'match_count': len(selected),
            'match_ids': selected['match_id'].astype(int).tolist(),
            'previous_match_count': len(previous_ids),
            'new_match_ids': sorted(set(selected['match_id'].astype(int)) - previous_ids),
            'selection': 'match_status == available and match_date <= cutoff; final scores present',
        }
        if selection_path.exists():
            saved = json.loads(selection_path.read_text())
            if saved['data_cutoff'] != cutoff:
                raise ValueError('Existing snapshot has a different cutoff.')
            missing = set(saved['match_ids']) - set(selected['match_id'].astype(int))
            if missing:
                raise ValueError(f'Snapshot matches are no longer eligible: {sorted(missing)}')
            selected = selected.set_index('match_id').loc[saved['match_ids']].reset_index()
        else:
            write_json_atomic(selection_path, manifest)
        save_parquet(normalize_match_metadata(selected), metadata_path)
    else:
        selected = select_available(pd.read_parquet(metadata_path), cutoff)
        saved = json.loads(selection_path.read_text())
        if saved['data_cutoff'] != cutoff or saved['match_ids'] != selected['match_id'].astype(int).tolist():
            raise ValueError('Existing snapshot has a different selection; use a new cutoff.')

    print(f'Snapshot: {len(selected)} available matches through {cutoff}.', flush=True)
    for index, match_id in enumerate(selected['match_id'].astype(int), 1):
        for resource, loader in [('events', sb.events), ('lineups', sb.lineups)]:
            destination = cache / f'{resource}_m{match_id}.parquet'
            source = previous_cache / destination.name
            if destination.exists():
                if source.exists() and file_hash(source) != file_hash(destination):
                    raise ValueError(f'Previously cached resource changed: {destination.name}')
                continue
            if source.exists():
                shutil.copy2(source, destination)
                continue
            print(f'[{index}/{len(selected)}] Downloading {resource}: {match_id}', flush=True)
            data = as_dataframe(loader(match_id=match_id, creds=creds), resource)
            save_parquet(data, destination, stringify_nested=resource == 'lineups')


def preserve_sequence_ids(rows: pd.DataFrame, previous: pd.DataFrame,
                          previous_excluded: pd.DataFrame) -> pd.DataFrame:
    keys = ['shot_event_id', 'shot_seq_id']
    mapping = pd.concat([previous[keys], previous_excluded[keys]], ignore_index=True)
    if mapping['shot_event_id'].duplicated().any() or mapping['shot_seq_id'].duplicated().any():
        raise ValueError('Previous sequence identifiers are not unique.')
    if rows['shot_event_id'].duplicated().any():
        raise ValueError('Target shots are duplicated in the expanded dataset.')
    old_matches = set(previous['match_id']) | set(previous_excluded['match_id'])
    old_shots = set(rows.loc[rows['match_id'].isin(old_matches), 'shot_event_id'])
    if old_shots != set(mapping['shot_event_id']):
        raise ValueError('Target shots in previously included matches changed.')
    result = rows.copy()
    identifiers = result['shot_event_id'].map(mapping.set_index('shot_event_id')['shot_seq_id'])
    added = identifiers.isna()
    first_new = int(mapping['shot_seq_id'].max()) + 1
    identifiers.loc[added] = np.arange(first_new, first_new + int(added.sum()))
    result['shot_seq_id'] = identifiers.astype('int64')
    return result.sort_values('shot_seq_id').reset_index(drop=True)


def validate_cache(cache: Path, cutoff: str) -> tuple[pd.DataFrame, dict]:
    _, _, matches = cache_metadata(cache)
    selected = select_available(matches, cutoff)
    expected = set(selected['match_id'].astype(int))
    for resource in ['events', 'lineups']:
        found = {int(path.stem.split('_m')[1]) for path in cache.glob(f'{resource}_m*.parquet')}
        if found != expected:
            raise ValueError(f'{resource}: missing={sorted(expected - found)}, extra={sorted(found - expected)}')
    report = {'raw_shots': 0, 'shots_missing_xg': 0, 'event_count': 0,
              'matches_with_missing_xg': {}}
    event_ids: set[str] = set()
    for match_id in sorted(expected):
        events = pd.read_parquet(cache / f'events_m{match_id}.parquet')
        if events.empty or events['id'].isna().any() or events['id'].duplicated().any():
            raise ValueError(f'Match {match_id}: empty events or invalid event IDs.')
        ids = set(events['id'].astype(str))
        if event_ids & ids:
            raise ValueError(f'Match {match_id}: event IDs overlap another match.')
        event_ids.update(ids)
        if 'match_id' in events and not events['match_id'].eq(match_id).all():
            raise ValueError(f'Match {match_id}: incorrect event match IDs.')
        shots = events.loc[events['type'].map(object_name).eq('Shot')]
        xg = pd.to_numeric(shots['shot_statsbomb_xg'], errors='raise')
        if not xg.dropna().between(0, 1).all():
            raise ValueError(f'Match {match_id}: xG outside [0, 1].')
        missing = int(xg.isna().sum())
        report['raw_shots'] += len(shots)
        report['shots_missing_xg'] += missing
        report['event_count'] += len(events)
        if missing:
            report['matches_with_missing_xg'][str(match_id)] = missing
        if pd.read_parquet(cache / f'lineups_m{match_id}.parquet').empty:
            raise ValueError(f'Match {match_id}: empty lineups.')
    return selected, report


def prepare_snapshot(cache: Path, output: Path, previous_dir: Path,
                     development_dir: Path, cutoff: str, previous_cache: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'Prepared output already exists: {output}')
    competition_id, season_id, _ = cache_metadata(cache)
    previous_competition, previous_season, _ = cache_metadata(previous_cache)
    if (competition_id, season_id) != (previous_competition, previous_season):
        raise ValueError('The expanded holdout must use the same competition and season.')
    selected, quality = validate_cache(cache, cutoff)
    previous = pd.read_parquet(previous_dir / SEQUENCE_FILE)
    development = pd.read_parquet(development_dir / SEQUENCE_FILE,
                                  columns=['match_id', 'match_date'])
    if set(selected['match_id']) & set(development['match_id']):
        raise ValueError('Development and holdout match IDs overlap.')
    if pd.to_datetime(selected['match_date']).min() <= pd.to_datetime(development['match_date']).max():
        raise ValueError('Holdout does not follow the development period.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.holdout-prepare-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'prepared'
        identity_path = previous_dir / 'sequence_ids.parquet'
        if identity_path.exists():
            identities = pd.read_parquet(identity_path)
        else:
            # Support runs prepared before the full identity manifest was introduced.
            old_rows = build_external_sequence_dataset(
                previous_cache, Path(temporary) / 'previous',
                competition_id=competition_id, season_id=season_id,
            )
            old_rows = pd.read_parquet(Path(temporary) / 'previous' / SEQUENCE_FILE)
            if not old_rows.loc[old_rows['n_events'].gt(1)].reset_index(drop=True).equals(
                previous.reset_index(drop=True)
            ):
                raise ValueError('The previous cache no longer reproduces the previous holdout.')
            identities = old_rows[['shot_event_id', 'shot_seq_id', 'match_id']]
        previous_excluded = identities.loc[~identities['shot_event_id'].isin(previous['shot_event_id'])]
        rows = build_external_sequence_dataset(
            cache, staged, competition_id=competition_id, season_id=season_id, sequence_length=20,
        )
        rows = preserve_sequence_ids(rows, previous, previous_excluded)
        retained = rows.loc[rows['n_events'].gt(1)].reset_index(drop=True)
        excluded = rows.loc[rows['n_events'].le(1)]
        if quality['raw_shots'] != quality['shots_missing_xg'] + len(retained) + len(excluded):
            raise ValueError('Shot accounting does not reconcile.')
        if set(retained['match_id']) != set(selected['match_id']):
            raise ValueError('A selected match has no retained sequences; inspect before excluding it.')
        retained.to_parquet(staged / SEQUENCE_FILE, index=False)
        persisted = pd.read_parquet(staged / SEQUENCE_FILE)
        old = persisted.loc[persisted['match_id'].isin(previous['match_id'])].reset_index(drop=True)
        if not old.equals(previous.reset_index(drop=True)):
            raise ValueError('Previously retained sequences changed; no dataset was published.')
        keys = ['shot_seq_id', 'shot_event_id', 'match_id']
        rows[keys].to_parquet(staged / 'sequence_ids.parquet', index=False)
        excluded[[*keys, 'match_date']].to_csv(
            staged / 'excluded_no_pre_shot_sequences.csv', index=False,
        )
        counts = rows.groupby('match_id').size().rename('source_sequences').to_frame()
        counts['retained_sequences'] = retained.groupby('match_id').size()
        counts['excluded_shot_only_sequences'] = excluded.groupby('match_id').size()
        counts.fillna(0).astype(int).to_csv(staged / 'sequence_counts_by_match.csv')
        new_ids = sorted(set(selected['match_id'].astype(int)) - set(identities['match_id'].astype(int)))
        manifest = {
            'status': 'complete', 'prepared_at_utc': datetime.now(UTC).isoformat(),
            'data_cutoff': cutoff, 'competition_id': competition_id, 'season_id': season_id,
            'previous_prepared_name': previous_dir.name, 'output_prepared_name': output.name,
            'source_cache': str(cache.resolve()),
            'previous_match_count': int(identities['match_id'].nunique()),
            'previous_sequence_count': len(previous),
            'extended_match_count': len(selected), 'extended_sequence_count': len(retained),
            'new_match_ids': new_ids, 'new_sequence_count': len(retained) - len(previous),
            'source_sequences': len(rows), 'excluded_sequences': len(excluded),
            'filter_expression': 'n_events > 1',
            'outcome_schema': 'statsbomb-event-specific-outcomes-v2',
            'sequence_length_including_reference_shot': 20,
            'old_sequences_identical': True, 'training_performed': False,
            'previous_sha256': file_hash(previous_dir / SEQUENCE_FILE),
            'output_sha256': file_hash(staged / SEQUENCE_FILE),
            'source_data_checks': quality,
        }
        write_json_atomic(staged / 'extension_manifest.json', manifest)
        staged.rename(output)
    return manifest


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--work-dir', type=Path, default=Path('local/run'))
    parser.add_argument('--name', required=True, type=snapshot_name)
    parser.add_argument('--cutoff', required=True, type=date.fromisoformat)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--previous-cache', required=True, type=Path)
    parser.add_argument('--previous-holdout', default='holdout', type=snapshot_name)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Use an already complete cache without connecting to StatsBomb.')


def run(args: argparse.Namespace) -> None:
    root = args.work_dir.resolve()
    output = root / 'data' / args.name
    previous = root / 'data' / args.previous_holdout
    cutoff = args.cutoff.isoformat()
    if output == previous or args.name == 'development':
        raise ValueError('Use a new name; existing datasets are immutable.')
    with run_lock(root):
        if output.exists():
            manifest = json.loads((output / 'extension_manifest.json').read_text())
            if (manifest['data_cutoff'] != cutoff
                    or manifest['previous_sha256'] != file_hash(previous / SEQUENCE_FILE)
                    or manifest['output_sha256'] != file_hash(output / SEQUENCE_FILE)):
                raise ValueError('Existing snapshot differs; use a new name.')
        else:
            if not args.prepare_only:
                download_snapshot(args.cache_dir.resolve(), args.previous_cache.resolve(), cutoff)
            manifest = prepare_snapshot(
                args.cache_dir.resolve(), output, previous, root / 'data/development',
                cutoff, args.previous_cache.resolve(),
            )
    print(json.dumps(manifest, indent=2))
    print(f'Prepared holdout: {output}')
