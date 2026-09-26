from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from .data import build_external_sequence_dataset, build_sequence_dataset


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def cache_metadata(cache: Path) -> tuple[int, int, pd.DataFrame]:
    paths = sorted(cache.glob('matches_c*_s*.parquet'))
    if len(paths) != 1:
        raise ValueError(f'{cache}: use one matches file per season cache')
    match = re.fullmatch(r'matches_c(\d+)_s(\d+)\.parquet', paths[0].name)
    if match is None:
        raise ValueError(f'Invalid matches filename: {paths[0].name}')
    return int(match[1]), int(match[2]), pd.read_parquet(paths[0])


def prepare_datasets(work_dir: Path, development_cache: Path, holdout_cache: Path) -> None:
    data_dir = work_dir / 'data'
    if data_dir.exists():
        raise FileExistsError(f'{data_dir} already exists; use a new work directory')
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=work_dir) as temporary:
        staged = Path(temporary) / 'data'
        datasets = {}
        for label, cache, builder in (
            ('development', development_cache, build_sequence_dataset),
            ('holdout', holdout_cache, build_external_sequence_dataset),
        ):
            competition, season, matches = cache_metadata(cache)
            if label == 'development':
                missing = [int(mid) for mid in matches['match_id']
                           if not (cache / f'events_m{int(mid)}.parquet').is_file()]
                if missing:
                    raise ValueError(f'Development cache lacks {len(missing)} event files')
            rows = builder(cache, staged / label, competition_id=competition, season_id=season)
            rows[['shot_seq_id', 'shot_event_id', 'match_id']].to_parquet(
                staged / label / 'sequence_ids.parquet', index=False,
            )
            rows.loc[rows['n_events'].le(1),
                     ['shot_seq_id', 'shot_event_id', 'match_id', 'match_date']].to_csv(
                staged / label / 'excluded_no_pre_shot_sequences.csv', index=False,
            )
            retained = rows.loc[rows['n_events'].gt(1)].copy()
            if retained.empty or retained['shot_seq_id'].duplicated().any():
                raise ValueError(f'{label}: empty data or duplicate sequence IDs')
            if label == 'development' and set(retained['split']) != {'train', 'validation', 'test'}:
                raise ValueError('Each temporal development split must contain nonempty sequences')
            path = staged / label / 'sequences_raw.parquet'
            retained.to_parquet(path, index=False)
            summary = {
                'source_cache': str(cache.resolve()),
                'competition_id': competition,
                'season_id': season,
                'sequence_length_including_reference_shot': 20,
                'filter': 'n_events > 1',
                'source_sequences': len(rows),
                'excluded_sequences': len(rows) - len(retained),
                'retained_sequences': len(retained),
                'matches': int(retained['match_id'].nunique()),
                'sequences_by_split': retained.groupby('split').size().to_dict(),
                'sha256': file_hash(path),
            }
            (staged / label / 'preparation.json').write_text(json.dumps(summary, indent=2))
            datasets[label] = retained
        development, holdout = datasets['development'], datasets['holdout']
        if set(development['match_id']) & set(holdout['match_id']):
            raise ValueError('Development and holdout matches overlap')
        if pd.to_datetime(development['match_date']).max() >= pd.to_datetime(holdout['match_date']).min():
            raise ValueError('Holdout dates must follow the development period')
        staged.rename(data_dir)
    print(f'Prepared nonempty development and holdout datasets in {data_dir}')
