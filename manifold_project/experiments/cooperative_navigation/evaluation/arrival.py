"""First simultaneous coverage diagnostics, including frozen legacy policies."""
import numpy as np
import torch
from ..envs.navigation import Navigation
from ..observations.history import History


def arrival_rollout(actor, config, reset_seed, action_seed, *, limit=500, n=None):
    """Keep the policy's observation convention; stop at its first success."""
    env = Navigation(dict(config, horizon=limit, task_horizon=limit), n)
    history = History(env.n, env.obs_dim, config['history'], limit, include_time=env.include_time)
    rng = np.random.default_rng(action_seed)
    device = next(actor.parameters()).device
    rewards, collisions, distance_travelled = [], [], 0.
    try:
        obs = env.reset(reset_seed)
        previous = None
        metrics = env.metrics()
        success = bool(metrics['all_covered'])
        while not success and env.t < limit:
            x = history.append(obs, previous, env.t)
            with torch.no_grad():
                p = actor(torch.as_tensor(x, device=device)).cpu().numpy().astype(np.float64)
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
                raise FloatingPointError('非法策略概率')
            cumulative = p.cumsum(-1)
            cumulative /= cumulative[:, -1:]
            actions = (rng.random(env.n)[:, None] >= cumulative).sum(-1)
            before = np.array([a.state.p_pos.copy() for a in env.world.agents])
            obs, reward, terminal, timeout = env.step(actions)
            after = np.array([a.state.p_pos.copy() for a in env.world.agents])
            distance_travelled += float(np.linalg.norm(after-before, axis=-1).sum())
            previous = actions
            metrics = env.metrics()
            rewards.append(reward)
            collisions.append(metrics['collision_pairs'])
            success = bool(metrics['all_covered'])
            if terminal or timeout:
                break
        return dict(success=int(success), success_steps=env.t if success else None,
                    episode_steps=env.t, restricted_steps=env.t if success else limit,
                    end_reason='success' if success else 'deadline',
                    J=float(np.dot(config['gamma']**np.arange(len(rewards)), rewards)),
                    final_coverage=metrics['coverage'], collision_pairs_total=float(sum(collisions)),
                    path_length=distance_travelled)
    finally:
        env.close()


def arrival_summary(rows, limit):
    hits = [r['success_steps'] for r in rows if r['success']]
    return dict(episodes=len(rows), successes=len(hits), success_rate=len(hits)/len(rows),
                success_steps_mean=float(np.mean(hits)) if hits else None,
                restricted_mean_steps=float(np.mean([r['restricted_steps'] for r in rows])),
                evaluation_steps=sum(r['episode_steps'] for r in rows), task_horizon=limit,
                **{f'success_by_{t}':float(np.mean([r['success'] and r['success_steps'] <= t for r in rows]))
                   for t in (100,200,500) if t <= limit},
                **{key:float(np.mean([r[key] for r in rows])) for key in
                   ('J','final_coverage','collision_pairs_total','path_length')})


def main(argv=None):
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from ..models import load_actor
    from ..training.storage import load_pt, seed_for
    from .reporting import write_csv
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, type=Path)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--conditions', nargs='+', default=['mappo','ippo'])
    p.add_argument('--output', type=Path)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args(argv)
    if args.steps < 1 or args.episodes < 2:
        p.error('steps须为正整数，episodes至少为2')
    manifest = json.loads((args.suite/'manifest.json').read_text(encoding='utf-8'))
    jobs = [j for j in manifest['jobs'] if j['condition'] in args.conditions]
    if {j['condition'] for j in jobs} != set(args.conditions):
        p.error('所选条件不在套件中')
    for job in jobs:
        if not (args.suite/job['path']/'summary.json').exists():
            p.error('所选训练尚未完成')
    output = args.output or args.suite/f'first_arrival_diagnostic_h{args.steps}'
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    all_rows, summaries = [], []
    for job in jobs:
        path = args.suite/job['path']/'checkpoints/final.pt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = load_pt(path)
        c = state['config']
        if c.get('task_mode','finite_horizon') == 'finite_horizon' and c['horizon'] != args.steps:
            p.error('旧有限时域策略的时钟含义不能改变')
        if c.get('task_mode') == 'first_arrival' and c['task_horizon'] != args.steps:
            p.error('首达策略须按训练时相同任务期限评价')
        actor = load_actor(state)
        rows = []
        print(f'首达诊断 {job["condition"]} seed={job["seed"]}: {args.episodes} scenes, limit={args.steps}', flush=True)
        for i in range(args.episodes):
            reset = seed_for('arrival-diagnostic-reset', i)
            action = seed_for('arrival-diagnostic-action', i)
            row = arrival_rollout(actor,c,reset,action,limit=args.steps)
            rows.append(row)
            all_rows.append(dict(condition=job['condition'],seed=job['seed'],scene=i,
                                 reset_seed=reset,action_seed=action,**row))
            if (i+1) % 25 == 0:
                print(f'  {i+1}/{args.episodes}, success={sum(r["success"] for r in rows)}',flush=True)
        summaries.append(dict(condition=job['condition'],seed=job['seed'],
                              training_task_mode=c.get('task_mode','finite_horizon'),
                              checkpoint_sha256=digest,**arrival_summary(rows,args.steps)))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        write_csv(output/'episodes.csv', all_rows)
        write_csv(output/'summary.csv', summaries)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        labels = [f'{s["condition"]}/s{s["seed"]}' for s in summaries]
        for label, summary in zip(labels,summaries):
            rows = [r for r in all_rows if r['condition']==summary['condition'] and r['seed']==summary['seed']]
            t = np.arange(args.steps+1)
            axes[0].plot(t,[sum(r['success'] and r['success_steps']<=k for r in rows)/len(rows) for k in t],label=label)
        axes[0].set(xlabel='Environment steps',ylabel='Cumulative success rate',ylim=(0,1))
        axes[0].legend()
        axes[1].bar(labels,[s['restricted_mean_steps'] for s in summaries])
        axes[1].set_ylabel('Mean capped completion steps (failure = limit)')
        axes[2].bar(labels,[s['collision_pairs_total'] for s in summaries])
        axes[2].set_ylabel('Collision pair-steps until stop')
        fig.savefig(output/'overview.png',dpi=150)
        plt.close(fig)
    print(f'已保存首达诊断: {output}；旧训练目标与新任务评价须分开标注。',flush=True)
    return 0
