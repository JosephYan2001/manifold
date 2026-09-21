from pathlib import Path
import json
import time
import numpy as np
from ..models import Actor, load_actor
from ..configs import continuing_task, arrival_task
from ..training.storage import load_pt, seed_for
from ..training.collector import episode
from .evaluate import evaluate, ARRIVAL_METRICS
from .reporting import write_csv, read_csv, summaries, bootstrap


def cached_rows(path, dimensions):
    cached = {}
    for row in read_csv(path):
        if row['status'] != 'complete':
            continue
        for key in ('seed','n_agents','round','episodes','evaluation_steps','source_steps','count'):
            if key in row and row[key] != '':
                row[key] = int(row[key])
        for key in ('J','distance','coverage','all_covered','collision_pairs','collisions_per_agent',
                    'mean_reward','mean_coverage','tail_coverage','tail_all_covered',
                    'evaluation_seconds','mean','std','ci_low','ci_high', *ARRIVAL_METRICS):
            if key in row:
                row[key] = float(row[key]) if row[key] != '' else None
        for key in ('reused_source','candidate_accepted','supported_false_rejection'):
            if key in row:
                row[key] = None if row[key] == '' else row[key] == 'True'
        cached[tuple(row[key] for key in dimensions)] = row
    return cached


def transfer_suite(directory, manifest):
    directory = Path(directory)
    config = manifest['config']
    cache = cached_rows(directory/'transfer_results.csv',('condition','seed','n_agents'))
    rows = list(cache.values())
    for job in manifest['jobs']:
        if job['condition'] not in ('ours','sampled','mappo','ippo'):
            continue
        path = directory/job['path']
        if not (path/'summary.json').exists():
            for n in config['transfer_sizes']:
                rows.append(dict(condition=job['condition'],seed=job['seed'],n_agents=n,status='missing_source'))
            continue
        state = load_pt(path/'checkpoints/final.pt')
        actor = load_actor(state, config)
        summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
        for n in config['transfer_sizes']:
            if (job['condition'],job['seed'],n) in cache:
                continue
            start = time.perf_counter()
            if n == config['n_agents']:
                metrics = {k:summary[k] for k in ('J','distance','coverage','all_covered','collision_pairs','collisions_per_agent',
                           'mean_reward','mean_coverage','tail_coverage','tail_all_covered', *ARRIVAL_METRICS) if k in summary}
                if not arrival_task(config):
                    metrics['episode_steps'] = float(config['horizon'])
                count,steps = config['final_episodes'],0
            else:
                count = config['transfer_episodes']
                metrics,episodes = evaluate(actor,config,job['condition'],job['seed'],'transfer',count,n=n)
                steps = sum(e['episode_steps'] for e in episodes)
            rows.append(dict(condition=job['condition'],seed=job['seed'],n_agents=n,status='complete',
                             episodes=count,evaluation_steps=steps,reused_source=n==config['n_agents'],
                             evaluation_seconds=time.perf_counter()-start,**metrics))
            print(f'  transfer {job["condition"]} seed={job["seed"]} N={n}',flush=True)
            write_csv(directory/'transfer_results.csv',rows)
    write_csv(directory/'transfer_summary.csv',summaries(rows,config,transfer=True))
    return rows


def audit_suite(directory, manifest):
    directory = Path(directory)
    config = manifest['config']
    cache = cached_rows(directory/'audit_results.csv',('condition','seed','round'))
    rows = list(cache.values())
    for job in manifest['jobs']:
        if job['condition'] not in ('ours','no_return_check'):
            continue
        path = directory/job['path']
        if not (path/'checkpoints/final.pt').exists():
            continue
        state = load_pt(path/'checkpoints/final.pt')
        snapshots = {s['round']:s for s in state['snapshots']}
        for round_index in config['audit_rounds']:
            if (job['condition'],job['seed'],round_index) in cache:
                continue
            snap = snapshots.get(round_index)
            base = dict(condition=job['condition'],seed=job['seed'],round=round_index)
            if snap is None or snap['candidate'] is None:
                rows.append(dict(**base,status='missing_candidate',evaluation_steps=0,
                                 reason='round_not_reached' if snap is None else 'direction_rejected'))
                continue
            old, new = Actor(state['input_dim'],config),Actor(state['input_dim'],config)
            old.load_state_dict(snap['old'])
            new.load_state_dict(snap['candidate'])
            differences = []
            evaluation_steps = 0
            for i in range(config['audit_episodes']):
                reset = seed_for('audit-reset',job['seed'],round_index,i)
                episodes = [episode(actor,config,reset,seed_for('audit-action',job['condition'],job['seed'],round_index,i,side))
                          for side,actor in enumerate((old,new))]
                values = [e['J'] for e in episodes]
                evaluation_steps += sum(e['episode_steps'] for e in episodes)
                differences.append(values[1]-values[0])
            stats = bootstrap(differences,config['bootstrap_repeats'])
            outcome = 'uncertain'
            if stats['ci_high'] is not None and stats['ci_high'] < 0:
                outcome = 'supported_decline'
            elif stats['ci_low'] is not None and stats['ci_low'] > 0:
                outcome = 'supported_improvement'
            rows.append(dict(**base,status='complete',**stats,outcome=outcome,
                candidate_accepted=snap['decision']['accepted'],source_steps=snap['source_steps'],
                supported_false_rejection=(None if continuing_task(config) else
                    not snap['decision']['accepted'] and outcome=='supported_improvement'),
                score_kind='observed_first_arrival_return' if arrival_task(config) else 'observed_finite_window_return',
                evaluation_steps=evaluation_steps,
                interval_scope='paired_reset_within_candidate_uncorrected_not_training_seed_CI'))
            write_csv(directory/'audit_results.csv',rows)
    write_csv(directory/'audit_results.csv',rows)
    return rows
