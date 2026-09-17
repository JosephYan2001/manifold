"""Regenerate CSV summaries and panels without collecting trajectories."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from manifold_project.experiments.cooperative_navigation.evaluation.reporting import source_tables,write_csv,summaries
from manifold_project.experiments.cooperative_navigation.evaluation.transfer import cached_rows
from manifold_project.experiments.cooperative_navigation.evaluation.plotting import plot_suite,plot_transfer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args = p.parse_args()
    config = json.loads((args.directory/'manifest.json').read_text(encoding='utf-8'))['config']
    source_tables(args.directory,config)
    rows = list(cached_rows(args.directory/'transfer_results.csv',('condition','seed','n_agents')).values())
    if rows:
        write_csv(args.directory/'transfer_summary.csv',summaries(rows,config,transfer=True))
    plot_suite(args.directory)
    plot_transfer(args.directory)


if __name__ == '__main__':
    main()
