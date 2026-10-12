"""Compatibility wrapper. Training recovery lives in the public experiments CLI."""
import argparse
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--plot-every', type=int)
    p.add_argument('--allow-duplicate-openmp', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    suite = next((parent for parent in args.checkpoint.resolve().parents if (parent/'manifest.json').is_file()), None)
    if suite is None:
        p.error('Use a checkpoint inside a public experiment suite with manifest.json.')
    command = ['--resume', str(args.output if args.resume else suite), '--budget', str(args.budget)]
    if not args.resume:
        command += ['--output', str(args.output)]
    for key in ('device', 'plot_every'):
        if getattr(args,key) is not None:
            command += ['--'+key.replace('_','-'), str(getattr(args,key))]
    for key in ('plot', 'allow_duplicate_openmp', 'dry_run'):
        if getattr(args,key):
            command += ['--'+key.replace('_','-')]
    print('Compatibility wrapper: delegating to python -m manifold_project.experiments', flush=True)
    from manifold_project.experiments.cli import main as public_main
    return public_main(command)


if __name__ == '__main__':
    raise SystemExit(main())
