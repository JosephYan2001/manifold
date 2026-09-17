"""Losslessly merge completed suite logs, backing up originals per suite."""
import argparse
import json
from pathlib import Path
import zipfile
from ..training.logging import LOG_STREAMS, read_records


def merge_suite(suite, apply=False):
    suite = Path(suite).resolve()
    state = json.loads((suite/'status.json').read_text(encoding='utf-8'))
    runs = []
    for key, job in state['jobs'].items():
        if not key.startswith('training/') or job['status'] != 'complete':
            continue
        directory = (suite/key).resolve()
        directory.relative_to(suite)
        if json.loads((directory/'summary.json').read_text(encoding='utf-8')).get('stop_reason') == 'requested_pause':
            continue
        paths = [directory/f'{name}.jsonl' for name in LOG_STREAMS if (directory/f'{name}.jsonl').exists()]
        if not paths:
            continue
        if (directory/'events.jsonl').exists():
            raise ValueError(f'Mixed layouts require manual inspection: {directory}')
        for path in paths:
            path.resolve().relative_to(suite)
        originals = {path: path.read_bytes() for path in paths}
        records = {path: read_records(path) for path in paths}
        runs.append((directory, originals, records))
    if apply and runs:
        archive = suite/'legacy_logs.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as saved:
            for _, originals, _ in runs:
                for path, content in originals.items():
                    saved.writestr(path.relative_to(suite).as_posix(), content)
        with zipfile.ZipFile(archive) as saved:
            for _, originals, _ in runs:
                for path, content in originals.items():
                    if saved.read(path.relative_to(suite).as_posix()) != content:
                        raise ValueError(f'Archive verification failed: {path}')
        for directory, originals, records in runs:
            target = directory/'events.jsonl'
            with target.open('x', encoding='utf-8') as out:
                # Old streams have no global ordering; preserve order within each stream.
                for path, rows in records.items():
                    for row in rows:
                        out.write(json.dumps({'stream': path.stem, 'record': row}, ensure_ascii=False, allow_nan=False)+'\n')
            merged = {name: [] for name in LOG_STREAMS}
            for row in read_records(target):
                merged[row['stream']].append(row['record'])
            for path, rows in records.items():
                if merged[path.stem] != rows or path.read_bytes() != originals[path]:
                    raise ValueError(f'Log verification failed: {path}')
            for path in originals:
                path.unlink()
    return {'suite': suite.name, 'runs': len(runs),
            'old_files': sum(len(originals) for _, originals, _ in runs),
            'new_files': len(runs), 'applied': apply}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    for suite in sorted(args.results.glob('suite_*')):
        if (suite/'status.json').is_file():
            print(json.dumps(merge_suite(suite, args.apply)))
