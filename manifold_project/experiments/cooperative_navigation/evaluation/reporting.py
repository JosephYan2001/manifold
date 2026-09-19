import csv
import io
import json
from pathlib import Path
import numpy as np
from ..training.storage import atomic, seed_for

METRICS = ['J','AUC','coverage','distance','all_covered','collision_pairs','collisions_per_agent',
           'mean_reward','mean_coverage','tail_coverage','tail_all_covered',
           'source_steps','unused_budget','train_seconds','evaluation_seconds','optimizer_steps',
           'coverage_cost','fit_kl_after']


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    atomic(path, lambda p: Path(p).write_text(out.getvalue(), encoding='utf-8-sig'))


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def bootstrap(values, repeats=2000):
    x = np.asarray([float(v) for v in values if v is not None and v != ''], dtype=np.float64)
    if not len(x):
        return dict(count=0, mean=None, std=None, ci_low=None, ci_high=None)
    rng = np.random.default_rng(seed_for('bootstrap'))
    means = x[rng.integers(len(x), size=(repeats,len(x)))].mean(-1)
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x)>1 else None,
                ci_low=float(np.quantile(means,.025)) if len(x)>1 else None,
                ci_high=float(np.quantile(means,.975)) if len(x)>1 else None)


def summaries(rows, config, transfer=False):
    result = []
    groups = sorted({(r['condition'],r.get('n_agents',config['n_agents'])) for r in rows})
    for condition,n in groups:
        group = [r for r in rows if r['condition']==condition and r.get('n_agents',config['n_agents'])==n]
        ok = [r for r in group if r.get('status','complete')=='complete']
        for metric in METRICS + ([] if transfer else ['coverage_reached']):
            values = [(r.get('coverage_cost') is not None) if metric=='coverage_reached' else r.get(metric) for r in ok]
            if metric not in ('coverage_cost','coverage_reached') and all(v is None for v in values):
                continue
            result.append(dict(kind='condition',condition=condition,n_agents=n,metric=metric,
                               planned=len(group),failed=sum(r.get('status')=='failed' for r in group),
                               incomplete=sum(r.get('status','complete') not in ('complete','failed') for r in group),
                               **bootstrap(values,config['bootstrap_repeats'])))
    # Seed-block paired differences; no episode-level pseudo-replication.
    for n in sorted({g[1] for g in groups}):
        reference = {r['seed']:r for r in rows if r['condition']=='ours' and r.get('n_agents',config['n_agents'])==n and r.get('status','complete')=='complete'}
        for condition in sorted({g[0] for g in groups}-{'ours'}):
            peers = {r['seed']:r for r in rows if r['condition']==condition and r.get('n_agents',config['n_agents'])==n and r.get('status','complete')=='complete'}
            for metric in (['J','coverage','distance'] if transfer else ['J','AUC']):
                values = [reference[s][metric]-peers[s][metric] for s in sorted(reference.keys() & peers.keys())]
                if values:
                    result.append(dict(kind='paired_difference',condition='ours - '+condition,n_agents=n,
                                       metric=metric,**bootstrap(values,config['bootstrap_repeats'])))
    return result


def source_tables(directory, config):
    directory = Path(directory)
    runs, curves = [], []
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for job in manifest['jobs']:
        path = directory/job['path']
        if (path/'summary.json').exists():
            row = json.loads((path/'summary.json').read_text(encoding='utf-8'))
            costs = row.pop('costs')
            row.update({f'cost_{k}':v for k,v in costs.items()})
            runs.append(row)
            from ..training.storage import load_pt
            curves.extend(load_pt(path/'checkpoints/final.pt')['curves'])
        else:
            failure = json.loads((path/'failure.json').read_text(encoding='utf-8')) if (path/'failure.json').exists() else {}
            runs.append(dict(condition=job['condition'],seed=job['seed'],status='failed' if failure else 'pending',
                             source_steps=failure.get('used'),error=failure.get('message')))
    write_csv(directory/'training_results.csv',runs)
    write_csv(directory/'training_summary.csv',summaries(runs,config))
    write_csv(directory/'learning_curves.csv',curves)
    return runs
