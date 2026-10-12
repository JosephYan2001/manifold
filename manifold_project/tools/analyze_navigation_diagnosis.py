"""Reproduce the three AN ablation summaries and scientific comparison figure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['an_base', 'an_direction64', 'an_batch64']
    labels = ['A0: base', 'A1: direction epochs 64', 'A2: batch episodes 64']
    data, report = {}, {}
    for name in names:
        run = args.root / name / 'N-C_source_learning/full/AN_s40'
        def csv_rows(path):
            with path.open(encoding='utf-8-sig', newline='') as f:
                return list(csv.DictReader(f))
        rows = csv_rows(run / 'training.csv')
        ev = csv_rows(run / 'final_evaluation/evaluation_episodes.csv')
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        direction = [e['record'] for e in events if e['stream'] == 'direction_fit']
        monitor = [e['record'] for e in events if e['stream'] == 'evaluation']
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        manifest = json.loads((args.root / name / 'manifest.json').read_text(encoding='utf-8'))
        tail = rows[-max(1, len(rows) // 5):]
        avg = lambda seq, key: float(np.mean([float(r[key]) for r in seq]))
        report[name] = dict(
            final_episodes=len(ev), final_successes=sum(float(r['success']) for r in ev),
            final_metrics={k: avg(ev, k) for k in ['discounted_return', 'coverage', 'mean_goal_distance', 'collisions_per_step']},
            rounds=len(rows), source_steps=summary['source_steps'], train_steps=summary['costs']['train'],
            monitoring_steps=summary['costs']['source_monitor'], train_hours=summary['train_seconds']/3600,
            monitor_hours=summary['evaluation_seconds']/3600,
            optimizer_steps=summary['optimizer_steps_by_module'], monitor_return_auc=summary['AUC'],
            train_successes=sum(int(r['train_success_count']) for r in rows),
            train_episodes=sum(int(r['train_episode_count']) for r in rows),
            tail_rounds=len(tail), tail_means={k: avg(tail, k) for k in ['train_J', 'policy_entropy', 'policy_step_kl', 'fit_kl_before', 'fit_kl_after', 'explained_variance']},
            fit_worse_rounds=sum(float(r['fit_kl_after']) > float(r['fit_kl_before']) for r in rows),
            q_max_over_079_rounds=sum(r['q_abs_max'] > .79 for r in direction),
            final_training_row=rows[-1], final_direction=direction[-1],
            experiment_code_sha256=manifest['metadata']['experiment_code_sha256'],
            final_checkpoint_sha256=hashlib.sha256((run / 'checkpoints/final.pt').read_bytes()).hexdigest())
        data[name] = dict(rows=rows, ev=ev, direction=direction, monitor=monitor)
    # Paired uncertainty describes these fixed trained policies over evaluation scenes.
    rng = np.random.default_rng(20261012)
    base = {int(r['scene_seed']): float(r['discounted_return']) for r in data[names[0]]['ev']}
    for name in names[1:]:
        diff = np.array([float(r['discounted_return'])-base[int(r['scene_seed'])] for r in data[name]['ev']])
        boot = rng.choice(diff, (10000, len(diff)), replace=True).mean(1)
        report[name]['paired_return_difference_vs_base'] = dict(mean=float(diff.mean()),
            percentile_95_interval=np.quantile(boot, [.025, .975]).tolist(),
            unit='evaluation scene; conditional on this single training seed')
    (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    for name, label in zip(names, labels):
        d = data[name]
        x = np.array([float(r['source_steps']) / 1e6 for r in d['rows']])
        # The final 8-scene monitor uses a different seed family; do not join it to the grid curve.
        mon = d['monitor'][:-1]
        axs[0, 0].plot([r['budget_checkpoint']/1e6 for r in mon], [r['J'] for r in mon], label=label)
        for ax, key in [(axs[0, 1], 'policy_entropy'), (axs[1, 0], 'policy_step_kl')]:
            ax.plot(x, [float(r[key]) for r in d['rows']], label=label, alpha=.85)
        axs[1, 1].plot(x, [float(r['fit_kl_after']) / max(1e-12, float(r['fit_kl_before'])) for r in d['rows']], label=label, alpha=.8)
        axs[2, 0].plot(x, [r['q_abs_mean'] for r in d['direction']], label=label, alpha=.8)
        axs[2, 1].plot(x, [r['q_abs_max'] for r in d['direction']], label=label, alpha=.8)
    titles = ['Source monitor return (same 8 grid scenes; higher is better)',
              'Policy entropy on each training batch', 'Actual old-to-new policy KL',
              'Actor fit: residual KL / initial target KL (>1 = worse)',
              'Direction mean absolute output', 'Direction maximum absolute output']
    for ax, title in zip(axs.flat, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Cumulative source steps (millions; includes monitoring)')
        ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=9)
    axs[0, 1].axhline(np.log(5), color='black', linestyle='--', linewidth=1)
    axs[1, 0].set_yscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].axhline(1, color='black', linestyle='--', linewidth=1)
    axs[2, 1].axhline(.8, color='black', linestyle='--', linewidth=1)
    fig.suptitle('Navigation AN diagnostics | seed 40 | 2M budget | all final policies: 0/20 successes')
    fig.savefig(args.output / 'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps({n: {k: v for k, v in report[n].items() if k not in ['final_training_row', 'optimizer_steps']} for n in names}, indent=2))


if __name__ == '__main__':
    main()
