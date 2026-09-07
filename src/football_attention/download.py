from __future__ import annotations
import argparse
from datetime import UTC, datetime
import getpass
import json
import os
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--competition-id', type=int, default=38)
    parser.add_argument('--season-id', type=int, default=318)
    parser.add_argument(
        '--auto-season',
        action='store_true',
        help='Resolve season ID from competition ID and expected season name.',
    )
    parser.add_argument('--expected-season', default='2025/2026')
    parser.add_argument('--match-week-start', type=int, default=None)
    parser.add_argument('--match-week-end', type=int, default=None)
    parser.add_argument('--match-date-start', default=None)
    parser.add_argument('--match-date-end', default=None)
    parser.add_argument(
        '--available-only',
        action='store_true',
        help='Download only matches whose StatsBomb status is available.',
    )
    parser.add_argument('--min-matches', type=int, default=None)
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path.cwd() / 'local' / 'cache' / 'development',
    )
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--metadata-only',
        action='store_true',
        help=(
            'Refresh and report match metadata without writing the matches '
            'cache or downloading events and lineups.'
        ),
    )
    parser.add_argument(
        '--metadata-report',
        type=Path,
        default=None,
        help='Optional JSON output path used with --metadata-only.',
    )
    return parser.parse_args()


def read_credentials() -> dict[str, str]:
    user = os.getenv('SB_USERNAME') or input('StatsBomb username/email: ').strip()
    passwd = os.getenv('SB_PASSWORD') or getpass.getpass('StatsBomb password: ')
    if not user or not passwd:
        raise RuntimeError('StatsBomb username and password are required.')
    return {'user': user, 'passwd': passwd}


def as_dataframe(data: object, resource: str) -> pd.DataFrame:
    '''Normalize statsbombpy DataFrame and team-keyed lineup responses.'''
    if isinstance(data, dict):
        frames = []
        for team_name, team_data in data.items():
            if not isinstance(team_data, pd.DataFrame):
                continue
            frame = team_data.copy()
            if 'team_name' not in frame.columns:
                frame.insert(0, 'team_name', str(team_name))
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise RuntimeError(
            f'StatsBomb returned no {resource}. If the API printed 401, '
            'the credentials or licence are invalid.'
        )
    return data


def stringify_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''Serialize mixed nested object columns so Arrow can write them safely.'''
    result = df.copy()
    for column in result.select_dtypes(include=['object']).columns:
        values = result[column]
        has_nested = values.map(
            lambda value: isinstance(value, (dict, list, tuple))
        ).any()
        if not has_nested:
            continue

        def encode(value: object) -> str | None:
            if value is None or (
                isinstance(value, float) and pd.isna(value)
            ):
                return None
            return json.dumps(value, ensure_ascii=False, default=str)

        result[column] = values.map(encode)
    return result


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    stringify_nested: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    if stringify_nested:
        df = stringify_nested_columns(df)
    df.to_parquet(temporary, index=False)
    temporary.replace(path)


def should_download(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.exists()


def build_metadata_report(
    matches: pd.DataFrame,
    *,
    competition_id: int,
    competition_name: str,
    season_id: int,
    season_name: str,
) -> dict[str, object]:
    '''Build a JSON-safe availability report without downloading resources.'''
    status = matches.get(
        'match_status',
        pd.Series('missing', index=matches.index, dtype='object'),
    )
    normalized_status = status.fillna('missing').astype(str).str.casefold()
    status_counts = {
        str(key): int(value)
        for key, value in normalized_status.value_counts().sort_index().items()
    }
    columns = [
        column
        for column in [
            'match_id',
            'match_week',
            'match_date',
            'kick_off',
            'home_team',
            'away_team',
            'home_score',
            'away_score',
            'match_status',
        ]
        if column in matches
    ]
    records = json.loads(
        matches[columns].to_json(
            orient='records',
            date_format='iso',
        )
    )
    return {
        'checked_at_utc': datetime.now(UTC).isoformat(),
        'competition_id': int(competition_id),
        'competition_name': competition_name,
        'season_id': int(season_id),
        'season_name': season_name,
        'match_count': int(len(matches)),
        'available_count': int(normalized_status.eq('available').sum()),
        'status_counts': status_counts,
        'matches': records,
    }


def select_competition(
    competitions: pd.DataFrame,
    *,
    competition_id: int,
    season_id: int,
    expected_season: str,
    auto_season: bool,
) -> pd.DataFrame:
    '''Select one licensed competition season without silent fallback.'''
    selected = competitions[
        competitions['competition_id'].eq(competition_id)
    ]
    if auto_season:
        if not expected_season:
            raise ValueError('--auto-season requires --expected-season.')
        selected = selected[
            selected['season_name'].astype(str).eq(expected_season)
        ]
    else:
        selected = selected[selected['season_id'].eq(season_id)]
    selected = as_dataframe(selected, 'requested competition/season')
    if len(selected) != 1:
        columns = [
            column
            for column in [
                'competition_id',
                'competition_name',
                'season_id',
                'season_name',
            ]
            if column in selected
        ]
        raise RuntimeError(
            'Competition/season selection is not unique: '
            f'{selected[columns].to_dict(orient="records")}'
        )
    season_name = str(selected.iloc[0].get('season_name', ''))
    if expected_season and season_name != expected_season:
        raise RuntimeError(
            f'Expected season {expected_season}, received {season_name}.'
        )
    return selected


def select_matches(
    matches: pd.DataFrame,
    *,
    match_week_start: int | None,
    match_week_end: int | None,
    limit: int | None,
    min_matches: int | None,
    match_date_start: str | None = None,
    match_date_end: str | None = None,
    available_only: bool = False,
) -> pd.DataFrame:
    '''Select a chronological block of match weeks for downloading.'''
    selected = matches.copy()
    if match_week_start is not None or match_week_end is not None:
        if 'match_week' not in selected:
            raise RuntimeError('StatsBomb matches do not contain match_week.')
        if (
            match_week_start is not None
            and match_week_end is not None
            and match_week_start > match_week_end
        ):
            raise ValueError('match-week-start cannot exceed match-week-end.')
        weeks = pd.to_numeric(selected['match_week'], errors='coerce')
        if match_week_start is not None:
            selected = selected[weeks.ge(match_week_start)]
            weeks = weeks.loc[selected.index]
        if match_week_end is not None:
            selected = selected[weeks.le(match_week_end)]
    if match_date_start is not None or match_date_end is not None:
        if 'match_date' not in selected:
            raise RuntimeError('StatsBomb matches do not contain match_date.')
        dates = pd.to_datetime(selected['match_date'], errors='raise')
        if match_date_start is not None:
            selected = selected[dates.ge(pd.Timestamp(match_date_start))]
            dates = dates.loc[selected.index]
        if match_date_end is not None:
            selected = selected[dates.le(pd.Timestamp(match_date_end))]
    if available_only:
        if 'match_status' not in selected:
            raise RuntimeError('StatsBomb matches do not contain match_status.')
        selected = selected[
            selected['match_status'].astype(str).str.casefold().eq('available')
        ]
    sort_columns = [
        column for column in ['match_date', 'match_week', 'match_id']
        if column in selected
    ]
    if sort_columns:
        selected = selected.sort_values(sort_columns)
    if limit is not None:
        selected = selected.head(limit)
    selected = as_dataframe(selected, 'matches in requested weeks')
    if min_matches is not None and len(selected) < min_matches:
        raise RuntimeError(
            f'Only {len(selected)} matches selected; expected at least '
            f'{min_matches}.'
        )
    return selected


def main() -> None:
    args = parse_args()
    try:
        from statsbombpy import sb
    except ImportError as error:
        raise RuntimeError('Install the download extra: pip install ".[download]"') from error
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    creds = read_credentials()

    competitions = as_dataframe(
        sb.competitions(creds=creds),
        'competitions',
    )
    selected = select_competition(
        competitions,
        competition_id=args.competition_id,
        season_id=args.season_id,
        expected_season=args.expected_season,
        auto_season=args.auto_season,
    )
    season_id = int(selected.iloc[0]['season_id'])
    season_name = str(selected.iloc[0]['season_name'])

    matches_path = cache_dir / (
        f'matches_c{args.competition_id}_s{season_id}.parquet'
    )
    if args.metadata_only or should_download(matches_path, args.overwrite):
        matches = as_dataframe(
            sb.matches(
                competition_id=args.competition_id,
                season_id=season_id,
                creds=creds,
            ),
            'matches',
        )
        if not args.metadata_only:
            save_parquet(matches, matches_path)
    else:
        matches = pd.read_parquet(matches_path)

    matches = select_matches(
        matches,
        match_week_start=args.match_week_start,
        match_week_end=args.match_week_end,
        limit=args.limit,
        min_matches=args.min_matches,
        match_date_start=args.match_date_start,
        match_date_end=args.match_date_end,
        available_only=args.available_only,
    )

    if args.metadata_only:
        report = build_metadata_report(
            matches,
            competition_id=args.competition_id,
            competition_name=str(
                selected.iloc[0].get('competition_name', '')
            ),
            season_id=season_id,
            season_name=season_name,
        )
        display_columns = [
            column
            for column in [
                'match_id',
                'match_week',
                'match_date',
                'home_team',
                'away_team',
                'match_status',
            ]
            if column in matches
        ]
        print('Authenticated. Match metadata refreshed; no resources downloaded.')
        print(matches[display_columns].to_string(index=False))
        print(
            'Availability: '
            f'{report["available_count"]}/{report["match_count"]}; '
            f'statuses={report["status_counts"]}'
        )
        if args.metadata_report is not None:
            report_path = args.metadata_report.resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            print(f'Report: {report_path}')
        return

    failures: list[dict[str, object]] = []
    match_ids = matches['match_id'].astype(int).tolist()
    print(f'Authenticated. Downloading {len(match_ids)} matches to {cache_dir}.')

    for index, match_id in enumerate(match_ids, start=1):
        print(f'[{index:03d}/{len(match_ids):03d}] match_id={match_id}')
        resources = {
            'events': (
                cache_dir / f'events_m{match_id}.parquet',
                lambda: sb.events(match_id=match_id, creds=creds),
            ),
            'lineups': (
                cache_dir / f'lineups_m{match_id}.parquet',
                lambda: sb.lineups(match_id=match_id, creds=creds),
            ),
        }

        for resource, (path, loader) in resources.items():
            if not should_download(path, args.overwrite):
                continue
            try:
                data = as_dataframe(loader(), f'{resource} for match {match_id}')
                save_parquet(
                    data,
                    path,
                    stringify_nested=resource == 'lineups',
                )
            except Exception as error:
                failures.append(
                    {
                        'match_id': match_id,
                        'resource': resource,
                        'error': f'{type(error).__name__}: {error}',
                    }
                )
                print(f'  WARNING: {resource} failed: {error}')

    failures_path = cache_dir / 'download_failures.json'
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    manifest = {
        'downloaded_at_utc': datetime.now(UTC).isoformat(),
        'competition_id': int(args.competition_id),
        'competition_name': str(selected.iloc[0].get('competition_name', '')),
        'season_id': season_id,
        'season_name': season_name,
        'match_week_start': args.match_week_start,
        'match_week_end': args.match_week_end,
        'match_date_start': args.match_date_start,
        'match_date_end': args.match_date_end,
        'available_only': bool(args.available_only),
        'match_count': len(match_ids),
        'match_ids': match_ids,
        'match_date_min': str(matches['match_date'].min()),
        'match_date_max': str(matches['match_date'].max()),
        'event_file_count': sum(
            (cache_dir / f'events_m{match_id}.parquet').exists()
            for match_id in match_ids
        ),
        'lineup_file_count': sum(
            (cache_dir / f'lineups_m{match_id}.parquet').exists()
            for match_id in match_ids
        ),
        'failure_count': len(failures),
    }
    (cache_dir / 'selection_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'Done. Failures: {len(failures)}. Report: {failures_path}')


if __name__ == '__main__':
    main()
