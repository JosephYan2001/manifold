"""Animate a frozen navigation policy without changing the training suite."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from manifold_project.experiments.cooperative_navigation.configs import load_config, continuing_task
from manifold_project.experiments.cooperative_navigation.envs.navigation import Navigation
from manifold_project.experiments.cooperative_navigation.models import load_actor
from manifold_project.experiments.cooperative_navigation.observations.history import History
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt


def rollout(actor, config, n, reset_seed, action_seed):
    env = Navigation(config, n)
    history = History(env.n, env.obs_dim, config['history'], env.horizon, include_time=env.include_time)
    rng = np.random.default_rng(action_seed)
    frames, rewards = [], []
    def snapshot():
        frames.append(dict(positions=np.array([a.state.p_pos.copy() for a in env.world.agents]),
                           landmarks=np.array([l.state.p_pos.copy() for l in env.world.landmarks]),
                           sizes=np.array([a.size for a in env.world.agents]),
                           metrics=env.metrics(), step=env.t))
    try:
        obs = env.reset(reset_seed)
        snapshot()
        previous = None
        for t in range(env.horizon):
            x = history.append(obs, previous, t)
            if actor is None:
                p = np.full((env.n, 5), .2)
            else:
                with torch.no_grad():
                    p = actor(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
            if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
                raise ValueError('Invalid action probabilities')
            actions = (rng.random(env.n)[:, None] > p.astype(np.float64).cumsum(-1)).sum(-1).clip(max=4)
            obs, reward, terminated, truncated = env.step(actions)
            rewards.append(reward)
            previous = actions
            snapshot()
        assert terminated or truncated
        metrics = dict(frames[-1]['metrics'])
        metrics['J'] = float(np.dot(np.power(config['gamma'], np.arange(env.horizon)), rewards))
        metrics['collision_pairs'] = float(np.mean([f['metrics']['collision_pairs'] for f in frames[1:]]))
        metrics['collisions_per_agent'] = float(np.mean([f['metrics']['collisions_per_agent'] for f in frames[1:]]))
        return frames, metrics
    finally:
        env.close()


def animate(frames, output, fps, title, show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    points = np.concatenate([np.concatenate((f['positions'], f['landmarks'])) for f in frames])
    lo, hi = points.min(axis=0)-.3, points.max(axis=0)+.3
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal', xlabel='x', ylabel='y')
    colors = plt.get_cmap('tab10')
    trails, circles, labels = [], [], []
    for i, (pos, size) in enumerate(zip(frames[0]['positions'], frames[0]['sizes'])):
        trails.append(ax.plot([], [], color=colors(i % 10), alpha=.45, lw=1)[0])
        circle = Circle(pos, size, color=colors(i % 10), alpha=.8)
        ax.add_patch(circle)
        circles.append(circle)
        labels.append(ax.text(*pos, str(i), ha='center', va='center', fontsize=9))
    landmarks = ax.scatter(*frames[0]['landmarks'].T, marker='x', s=90, color='black', label='Landmarks')
    ax.legend(loc='lower left')
    heading = ax.set_title('')
    def update(k):
        f = frames[k]
        for i, (circle, label, trail) in enumerate(zip(circles, labels, trails)):
            circle.center = f['positions'][i]
            label.set_position(f['positions'][i])
            path = np.array([v['positions'][i] for v in frames[:k+1]])
            trail.set_data(path[:, 0], path[:, 1])
        landmarks.set_offsets(f['landmarks'])
        m = f['metrics']
        heading.set_text(f"{title}\nstep {f['step']}/{len(frames)-1} | coverage {m['coverage']:.0%} | collision pairs {m['collision_pairs']:.0f}")
        return [*circles, *labels, *trails, landmarks, heading]
    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, repeat=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--suite', type=Path, help='Existing navigation suite with manifest.json')
    selection.add_argument('--random', action='store_true', help='Uniform random policy demo, not a trained result')
    p.add_argument('--condition', default='ours')
    p.add_argument('--seed', type=int, default=40, help='Training seed selecting the final checkpoint')
    p.add_argument('--agents', type=int, choices=[3, 4, 6, 8])
    p.add_argument('--reset-seed', type=int, default=20260918)
    p.add_argument('--action-seed', type=int, default=20260919)
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--steps', type=int, help='持续任务的回放长度，例如 500；期间不 reset')
    p.add_argument('--output', type=Path, default=Path('navigation_replay.gif'))
    p.add_argument('--show', action='store_true', help='Also open an animated window after saving')
    args = p.parse_args(argv)
    if args.fps < 1 or args.output.suffix.lower() != '.gif':
        p.error('fps must be positive and output must end in .gif')
    if args.output.exists():
        p.error('Output already exists; choose a new --output name')
    torch.set_num_threads(1)
    actor = None
    if args.random:
        config = load_config('pilot')
        title = 'UNIFORM RANDOM policy demo'
    else:
        manifest = json.loads((args.suite/'manifest.json').read_text(encoding='utf-8'))
        config = manifest['config']
        jobs = [j for j in manifest['jobs'] if j['condition'] == args.condition and j['seed'] == args.seed]
        if len(jobs) != 1:
            p.error('Requested condition/seed is not a unique job in this suite')
        run = args.suite/jobs[0]['path']
        if not (run/'summary.json').exists():
            p.error('This training job is not complete; wait for its final checkpoint')
        state = load_pt(run/'checkpoints/final.pt')
        config = state['config']
        actor = load_actor(state, config)
        actor.eval()
        title = f'{args.condition} | final | train seed {args.seed}'
    n = args.agents or config['n_agents']
    if args.steps is not None:
        if args.steps < 1 or not continuing_task(config):
            p.error('--steps 要求正整数且模型来自 continuing 任务，不能改变旧模型的时钟含义')
        config = dict(config, horizon=args.steps)
    print(f'Replaying {title}, N={n}; reset={args.reset_seed}, action={args.action_seed}', flush=True)
    frames, metrics = rollout(actor, config, n, args.reset_seed, args.action_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animate(frames, args.output, args.fps, f'{title} | N={n} | reset={args.reset_seed}', args.show)
    print(json.dumps(metrics, indent=2))
    print(f'Saved: {args.output.resolve()} (illustrative episode, not an aggregate evaluation)')


if __name__ == '__main__':
    main()
