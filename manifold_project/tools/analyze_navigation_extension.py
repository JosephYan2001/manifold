"""Compare saved 2M/4M navigation runs and render frozen policies on one scene."""
import csv
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'experiments/results'
OUT = ROOT / 'navigation_diagnosis_20261012_4m_s40_v1/analysis_20261014'
NAMES = ['an_base', 'an_direction64', 'an_batch64']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-gifs', action='store_true', help='Only refresh statistics and comparison plot')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for budget, folder in [(2, 'navigation_diagnosis_20261010_2m_s40_v1'),
                           (4, 'navigation_diagnosis_20261012_4m_s40_v1')]:
        for name in NAMES:
            data[f'{name}_{budget}m'] = ROOT / folder / name / 'N-C_source_learning/full/AN_s40'
    data['mappo_2m'] = ROOT / 'navigation_ep400_history8_2m_s40/mappo/N-C_source_learning/full/MAPPO-E_s40'
    report, evaluations, monitors, training = {}, {}, {}, {}
    for key, run in data.items():
        ev = read_csv(run / 'final_evaluation/evaluation_episodes.csv')
        rows = read_csv(run / 'training.csv')
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        mon = [e['record'] for e in events if e['stream'] == 'evaluation']
        directions = [e['record'] for e in events if e['stream'] == 'direction_fit']
        summary = json.loads((run / 'summary.json').read_text())
        avg = lambda seq, field: float(np.mean([float(r[field]) for r in seq]))
        tail = rows[-max(1, len(rows)//5):]
        report[key] = dict(final_episodes=len(ev), successes=sum(float(r['success']) for r in ev),
            final={field: avg(ev, field) for field in ['discounted_return', 'mean_goal_distance', 'coverage', 'collisions_per_step', 'steps']},
            rounds=len(rows), train_successes=sum(int(r['train_success_count']) for r in rows),
            train_episodes=sum(int(r['train_episode_count']) for r in rows),
            train_hours=summary['train_seconds']/3600, source_steps=summary['source_steps'],
            tail={field: avg(tail, field) for field in ['policy_entropy', 'policy_step_kl', 'explained_variance'] if tail[0].get(field)},
            monitor_success_nodes=[r for r in mon if r.get('success_rate', 0)>0])
        if directions:
            report[key]['last_direction'] = directions[-1]
            report[key]['tail'].update({field: avg(tail, field) for field in ['fit_kl_before', 'fit_kl_after']})
            report[key]['fit_worse_rounds'] = sum(float(r['fit_kl_after'])>float(r['fit_kl_before']) for r in rows)
        evaluations[key], monitors[key], training[key] = ev, mon, rows
    rng = np.random.default_rng(20261014)
    pairs = [(f'{n}_4m', f'{n}_2m') for n in NAMES] + [('an_direction64_4m', 'an_base_4m'), ('mappo_2m', 'an_direction64_4m')]
    comparisons = {}
    for new, old in pairs:
        a = {int(r['scene_seed']): float(r['discounted_return']) for r in evaluations[new]}
        b = {int(r['scene_seed']): float(r['discounted_return']) for r in evaluations[old]}
        assert a.keys() == b.keys()
        diff = np.array([a[s]-b[s] for s in sorted(a)])
        boot = rng.choice(diff, (20000, len(diff)), replace=True).mean(1)
        comparisons[f'{new}_minus_{old}'] = dict(mean=float(diff.mean()), scene_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(), improved_scenes=int((diff>0).sum()))
    (OUT/'metrics.json').write_text(json.dumps(dict(runs=report, paired_comparisons=comparisons,
        uncertainty_scope='Evaluation scenes conditional on one training seed; not across training seeds.'), indent=2), encoding='utf-8')
    table = [dict(run=key, **value['final'], successes=value['successes'],
                  final_episodes=value['final_episodes'], rounds=value['rounds'],
                  train_successes=value['train_successes'], train_episodes=value['train_episodes'],
                  train_hours=value['train_hours']) for key,value in report.items()]
    with (OUT/'comparison_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    labels = dict(an_base_4m='A0: base, 4M', an_direction64_4m='A1: direction64, 4M', an_batch64_4m='A2: batch64, 4M', mappo_2m='MAPPO-E, 2M')
    for key, label in labels.items():
        # Both inherited 2M final and new 4M final use a separate scene family.
        mon = [r for r in monitors[key] if r['budget_checkpoint'] not in (2000000, 4000000)]
        axes[0,0].plot([r['budget_checkpoint']/1e6 for r in mon], [r['J'] for r in mon], label=label)
        axes[0,1].plot([r['budget_checkpoint']/1e6 for r in mon], [r['mean_coverage'] for r in mon], label=label)
        rows = training[key]
        axes[1,0].plot([float(r['source_steps'])/1e6 for r in rows], [float(r['policy_entropy']) for r in rows], label=label, alpha=.8)
    xs = np.arange(4)
    vals = [report[k]['final']['discounted_return'] for k in labels]
    axes[1,1].bar(xs, vals, color=['C0','C1','C2','C3'])
    axes[1,1].set_xticks(xs, ['A0 4M','A1 4M','A2 4M','MAPPO 2M'])
    for i, key in enumerate(labels):
        axes[1,1].text(i, vals[i]-3, f"{vals[i]:.1f}\n{int(report[key]['successes'])}/20 success", ha='center', va='top', fontsize=9)
    axes[1,1].set_ylim(min(vals)-30, 0)
    for ax, title in zip(axes.flat, ['Fixed-grid monitor return', 'Time-averaged coverage (grid scenes)', 'Training policy entropy', 'Final evaluation: same 20 scenes']):
        ax.set_title(title); ax.grid(alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.axvline(2, color='gray', linestyle='--', alpha=.6)
        ax.set_xlabel('Source steps, millions (includes monitoring)')
    axes[0,0].legend(fontsize=8)
    axes[1,0].axhline(np.log(5), color='black', linestyle=':', alpha=.5)
    fig.suptitle('Navigation continuation | training seed 40 | stochastic evaluation')
    fig.savefig(OUT/'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(runs={k:{a:b for a,b in v.items() if a!='monitor_success_nodes'} for k,v in report.items()}, paired=comparisons), indent=2), flush=True)
    if args.skip_gifs:
        return

    from manifold_project.experiments.common.runtime import configure_runtime
    configure_runtime(True)
    import torch
    torch.set_num_threads(1)
    from manifold_project.experiments.visualize import render_checkpoint
    gifs = []
    metadata_path = OUT/'gif_metadata.json'
    existing = {r['output']:r for r in json.loads(metadata_path.read_text())} if metadata_path.exists() else {}
    for key in labels:
        output = OUT/f'{key}_seed1000000.gif'
        if not output.exists():
            result = render_checkpoint(data[key]/'checkpoints/final.pt', output, n_agents=4,
                seed=1000000, fps=20, frame_stride=5)
            gifs.append(result); print(json.dumps(result), flush=True)
        elif str(output) in existing:
            gifs.append(existing[str(output)])
    (OUT/'gif_metadata.json').write_text(json.dumps(gifs, indent=2), encoding='utf-8')
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (1200, 4*330), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, key in enumerate(labels):
        with Image.open(OUT/f'{key}_seed1000000.gif') as im:
            for col, frame in enumerate([0, im.n_frames//2, im.n_frames-1]):
                im.seek(frame)
                tile=im.convert('RGB');tile.thumbnail((300,300))
                sheet.paste(tile,(col*400,row*330+25))
        draw.text((8,row*330+5), labels[key]+' | start / middle / end', fill='black')
    sheet.save(OUT/'gif_preview.png')


if __name__ == '__main__':
    main()
