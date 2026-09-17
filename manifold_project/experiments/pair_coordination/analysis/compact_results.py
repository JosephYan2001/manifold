"""Archive redundant files from completed suite runs, preserving raw logs/models.

Usage: python -m manifold_project.experiments.pair_coordination.analysis.compact_results RESULTS
Defaults to a preview; --apply creates and verifies one archive per suite before compacting.
"""
import argparse
import json
from pathlib import Path
import zipfile
from ..training.checkpoints import atomic_json


EXTRAS = ('runtime', 'requested_config', 'resolved_config', 'training_plan')
REDUNDANT = tuple(f'{name}.json' for name in EXTRAS) + (
    'actor.json', 'progress.json', 'sampling_progress.json')


def compact_suite(suite, apply=False):
    suite = Path(suite).resolve()
    status = json.loads((suite/'status.json').read_text(encoding='utf-8'))
    runs, originals = [], {}
    for key, state in status['jobs'].items():
        if not key.startswith('training/') or state['status'] != 'complete':
            continue
        directory = (suite/key).resolve()
        directory.relative_to(suite)
        names = [name for name in REDUNDANT if (directory/name).exists()]
        if not names:
            continue
        for name in (*names, 'config.json', 'summary.json', 'checkpoints/final.json', 'checkpoints/best.json'):
            (directory/name).resolve().relative_to(suite)
        config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        if summary.get('stop_reason') == 'requested_pause':
            continue
        final = directory/'checkpoints/final.json'
        if not final.is_file() or not (directory/'checkpoints/best.json').is_file():
            raise ValueError(f'Missing checkpoints: {directory}')
        if (directory/'actor.json').exists() and (directory/'actor.json').read_bytes() != final.read_bytes():
            raise ValueError(f'actor.json differs from final: {directory}')
        for name in (*names, 'config.json', 'summary.json'):
            originals[str((directory/name).relative_to(suite)).replace('\\', '/')] = (directory/name).read_bytes()
        for name in EXTRAS:
            if (directory/f'{name}.json').exists():
                config[name] = json.loads((directory/f'{name}.json').read_text(encoding='utf-8'))
        summary.pop('actor', None)
        runs.append((directory, names, config, summary))
    count = sum(len(names) for _, names, _, _ in runs)
    if apply and runs:
        archive = suite/'legacy_metadata.zip'
        # Exclusive creation prevents overwriting a previous backup.
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as out:
            for name, content in originals.items():
                out.writestr(name, content)
        with zipfile.ZipFile(archive) as saved:
            for name, content in originals.items():
                if saved.read(name) != content:
                    raise ValueError(f'Archive verification failed: {name}')
        for directory, names, config, summary in runs:
            atomic_json(directory/'config.json', config)
            atomic_json(directory/'summary.json', summary)
            for name in names:
                (directory/name).unlink()
    return {'suite': suite.name, 'runs': len(runs), 'redundant_files': count, 'applied': apply}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    for suite in sorted(args.results.glob('suite_*')):
        if (suite/'status.json').is_file():
            print(json.dumps(compact_suite(suite, args.apply)))
