from collections import Counter
import numpy as np
import torch
from ..envs.navigation import Navigation
from ..observations.history import History
from ..configs import arrival_task, episode_horizon
from .storage import seed_for


def episode(actor, config, reset_seed, action_seed, n=None, retain=False, on_step=None, max_steps=None):
    env = Navigation(config, n)
    history = History(env.n, env.obs_dim, config['history'], env.horizon, include_time=env.include_time)
    rng = np.random.default_rng(action_seed)
    device = next(actor.parameters()).device
    rows = {key: [] for key in ('x', 'state', 'actions', 'mu', 'reward', 'terminated', 'truncated')}
    rewards, collisions, robot_collisions = [], [], []
    coverages, full_coverages = [], []
    path_length = 0.
    try:
        obs = env.reset(reset_seed)
        previous = None
        metrics = env.metrics()
        limit = min(env.horizon, max_steps) if max_steps is not None else env.horizon
        for t in range(limit):
            if env.end_reason is not None:
                break
            x = history.append(obs, previous, t)
            with torch.no_grad():
                p = actor(torch.as_tensor(x, device=device)).cpu().numpy()
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
                raise FloatingPointError('非法 Actor 概率')
            cumulative = p.astype(np.float64).cumsum(-1)
            cumulative /= cumulative[:, -1:]
            actions = (rng.random(env.n)[:, None] >= cumulative).sum(-1)
            state = env.state() if retain else None
            positions = np.array([a.state.p_pos.copy() for a in env.world.agents])
            obs, reward, terminated, truncated = env.step(actions)
            path_length += float(np.linalg.norm(np.array([a.state.p_pos for a in env.world.agents])-positions, axis=-1).sum())
            # Cutting collection short does not end the task. Retain the actual
            # final observation and its task clock for value bootstrapping.
            if t+1 == limit and not (terminated or truncated):
                truncated = True
            if on_step:
                on_step()
            if not np.isfinite(reward):
                raise FloatingPointError('非有限环境奖励')
            metrics = env.metrics()
            collisions.append(metrics['collision_pairs'])
            robot_collisions.append(metrics['collisions_per_agent'])
            rewards.append(reward)
            coverages.append(metrics['coverage'])
            full_coverages.append(metrics['all_covered'])
            if retain:
                for key, value in zip(rows, (x, state, actions, p, reward, terminated, truncated)):
                    rows[key].append(value)
            previous = actions
            if terminated or truncated:
                break
        length = len(rewards)
        mean = lambda values: float(np.mean(values)) if len(values) else 0.
        metrics.update(J=float(np.dot(np.power(config['gamma'], np.arange(length)), rewards)),
                       collision_pairs=mean(collisions), collisions_per_agent=mean(robot_collisions),
                       mean_reward=mean(rewards), mean_coverage=mean(coverages),
                       tail_coverage=mean(coverages[length//2:]),
                       tail_all_covered=mean(full_coverages[length//2:]), episode_steps=length)
        if arrival_task(config):
            success = env.end_reason == 'success'
            metrics.update(success=float(success), success_steps=length if success else None,
                           restricted_steps=length if success else env.horizon,
                           end_reason=env.end_reason or 'collection_cutoff',
                           collision_pairs_total=float(sum(collisions)), path_length=path_length)
        if not retain:
            return metrics
        batch = {key: np.asarray(value) for key, value in rows.items()}
        # Construct s_T / h_T using the last action, before closing/resetting the env.
        batch.update(final_x=history.append(obs, previous, env.t), final_state=env.state())
        if not length:
            # An initially solved scene consumes zero transitions.
            shapes = dict(x=batch['final_x'].shape, state=batch['final_state'].shape,
                          actions=(env.n,), mu=(env.n, 5), reward=(), terminated=(), truncated=())
            for key, shape in shapes.items():
                batch[key] = np.empty((0, *shape))
        return batch
    finally:
        env.close()


class Collector:
    def __init__(self, config, condition, seed, log=None):
        self.config, self.condition, self.seed, self.log = config, condition, seed, log
        self.counts, self.costs = Counter(), Counter()

    @property
    def used(self):
        return sum(self.costs.values())

    def collect(self, actor, count, purpose, retain=True):
        arrival = arrival_task(self.config)
        fixed_steps = arrival and purpose == 'train'
        quota = count*self.config['horizon'] if fixed_steps else count*episode_horizon(self.config)
        if self.used+quota > self.config['budget']:
            raise RuntimeError('源预算不足，禁止超支采样')
        episodes = []
        initial = self.used
        attempts = 0
        while (self.used-initial < quota if fixed_steps else len(episodes) < count):
            attempts += 1
            if attempts > quota+1000:
                raise RuntimeError('Too many zero-step reset scenes; collection cannot progress')
            limit = min(episode_horizon(self.config), quota-(self.used-initial)) if fixed_steps else episode_horizon(self.config)
            index = self.counts[purpose]
            self.counts[purpose] += 1
            if self.log:
                self.log('sampling_start', purpose=purpose, episode=index,
                         reserved_through=self.used+limit)
            def debit():
                self.costs[purpose] += 1
            result = episode(actor, self.config,
                seed_for('train', self.condition, self.seed, purpose, index, 'reset'),
                seed_for('train', self.condition, self.seed, purpose, index, 'action'),
                retain=retain, on_step=debit, max_steps=limit)
            length = len(result['reward']) if retain else result['episode_steps']
            if not fixed_steps or length:
                episodes.append(result)
            if self.log:
                self.log('sampling', purpose=purpose, episode=index, used=self.used)
        if not retain:
            return episodes
        device = next(actor.parameters()).device
        if arrival:
            lengths = np.asarray([len(e['reward']) for e in episodes])
            width = max(1, int(lengths.max()))
            packed = {}
            for key in episodes[0]:
                if key.startswith('final_'):
                    packed[key] = np.stack([e[key] for e in episodes])
                else:
                    array = np.full((len(episodes), width, *episodes[0][key].shape[1:]),
                                    .2 if key == 'mu' else 0.)
                    for i, e in enumerate(episodes):
                        array[i, :lengths[i]] = e[key]
                    packed[key] = array
            packed['valid'] = np.arange(width)[None, :] < lengths[:, None]
            packed['lengths'] = lengths
            return {key: torch.as_tensor(value, device=device, dtype=(
                torch.long if key in ('actions', 'lengths') else
                torch.bool if key in ('terminated', 'truncated', 'valid') else torch.float32))
                for key, value in packed.items()}
        return {key: torch.as_tensor(np.stack([e[key] for e in episodes]), device=device,
                                     dtype=(torch.long if key == 'actions' else
                                            torch.bool if key in ('terminated', 'truncated') else torch.float32))
                for key in episodes[0]}
