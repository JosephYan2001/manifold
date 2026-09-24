"""Independent observed-reward evaluation, including long continuing rollouts."""
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = 'manifold_project.experiments.cooperative_navigation.evaluation'
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from ..training.collector import episode
from ..training.storage import seed_for
from ..configs import arrival_task

ARRIVAL_METRICS = ('success_rate', 'success_steps_mean', 'restricted_mean_steps',
                   'episode_steps', 'collision_pairs_total', 'path_length',
                   'success_by_100', 'success_by_200', 'success_by_500')


def evaluate(actor, config, condition, seed, purpose, count, n=None):
    n = n or config['n_agents']
    rows = [episode(actor, config,
                    seed_for('evaluation-reset', purpose, n, i),
                    seed_for('evaluation-action', condition, seed, purpose, n, i))
            for i in range(count)] if n == config['n_agents'] else [
                episode(actor, config, seed_for('evaluation-reset', purpose, n, i),
                        seed_for('evaluation-action', condition, seed, purpose, n, i), n=n)
                for i in range(count)]
    metrics = {key: float(np.mean([r[key] for r in rows], dtype=np.float64)) for key in rows[0]
               if key not in ('success_steps', 'end_reason')}
    if arrival_task(config) and config.get('environment', 'navigation') == 'navigation':
        hits = [r['success_steps'] for r in rows if r['success']]
        metrics.update(success_rate=metrics.pop('success'),
                       success_steps_mean=float(np.mean(hits)) if hits else None,
                       restricted_mean_steps=metrics.pop('restricted_steps'))
        for steps in (100, 200, 500):
            if steps <= config['task_horizon']:
                metrics[f'success_by_{steps}'] = float(np.mean([
                    r['success'] and r['success_steps'] <= steps for r in rows]))
    return metrics, rows


def main(argv=None):
    import argparse
    import hashlib
    import json
    import torch
    from ..configs import continuing_task
    from ..models import load_actor
    from ..training.storage import load_pt
    from .reporting import write_csv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--steps', type=int, default=500, help='持续模型为运行窗口；首达模型须等于训练task_horizon')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--conditions', nargs='+')
    p.add_argument('--output', type=Path, help='单个汇总 CSV；默认保存在套件中')
    args = p.parse_args(argv)
    if args.steps < 1 or args.episodes < 2:
        p.error('steps 必须为正整数，episodes 至少为 2')
    manifest = json.loads((args.suite/'manifest.json').read_text(encoding='utf-8'))
    available = {j['condition'] for j in manifest['jobs']}
    if args.conditions and not set(args.conditions) <= available:
        p.error('指定条件不在套件中')
    jobs = [j for j in manifest['jobs'] if not args.conditions or j['condition'] in args.conditions]
    if not jobs or any(not (args.suite/j['path']/'summary.json').exists() for j in jobs):
        p.error('所选训练必须已经完成，评价仅使用 final 模型')
    first_arrival = arrival_task(manifest['config'])
    if first_arrival and args.steps != manifest['config']['task_horizon']:
        p.error('首达模型评价上限必须等于训练 task_horizon，不能改变时钟含义')
    if not continuing_task(manifest['config']) and not first_arrival:
        p.error('旧有限时域模型含时间特征，不能作为持续任务的长窗口评价')
    output = args.output or args.suite/f'{"first_arrival" if first_arrival else "long_rollout"}_h{args.steps}.csv'
    if output.exists():
        p.error('输出已存在；请读取已有结果或指定新的 --output')
    if output.suffix.lower() != '.csv':
        p.error('--output 必须为 CSV 文件')
    torch.set_num_threads(1)
    rows = []
    for job in jobs:
        path = args.suite/job['path']/'checkpoints/final.pt'
        state = load_pt(path)
        if state['config'].get('task_mode') != manifest['config'].get('task_mode'):
            raise ValueError('checkpoint 的任务定义与持续任务套件不一致')
        config = dict(state['config'], horizon=args.steps, device='cpu')
        actor = load_actor(state, config)
        print(f'{"首达任务" if first_arrival else "长窗口"}评价 {job["condition"]} seed={job["seed"]}: '
              f'{args.episodes} 场景，步数上限 {args.steps}', flush=True)
        metrics, episodes = evaluate(actor, config, job['condition'], job['seed'],
                                     f'long-rollout-{args.steps}', args.episodes)
        for key in ('J', 'mean_reward', 'coverage', 'mean_coverage', 'tail_coverage', 'tail_all_covered'):
            metrics[key+'_se'] = float(np.std([r[key] for r in episodes], ddof=1)/np.sqrt(args.episodes))
        if first_arrival:
            for key, source in (('success_rate','success'),('restricted_mean_steps','restricted_steps')):
                metrics[key+'_se'] = float(np.std([r[source] for r in episodes],ddof=1)/np.sqrt(args.episodes))
        rows.append(dict(condition=job['condition'], seed=job['seed'], task_mode=config['task_mode'],
                         steps=args.steps, episodes=args.episodes,
                         evaluation_steps=sum(r['episode_steps'] for r in episodes),
                         checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **metrics))
        if first_arrival:
            print(f'  success={metrics["success_rate"]:.1%}, '
                  f'restricted_steps={metrics["restricted_mean_steps"]:.2f}, J={metrics["J"]:.4f}', flush=True)
        else:
            print(f'  J={metrics["J"]:.4f}, mean_coverage={metrics["mean_coverage"]:.1%}, '
                  f'tail_coverage={metrics["tail_coverage"]:.1%}', flush=True)
    write_csv(output, rows)
    print(f'已保存 {output}；仅实际奖励，无 Critic bootstrap；误差为评价回合标准误。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
