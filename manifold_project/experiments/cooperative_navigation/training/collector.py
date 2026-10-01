from collections import Counter
import numpy as np
import torch
from ..envs.tasks import Navigation
from ..observations.history import History
from ..configs import arrival_task, episode_horizon, success_task
from .storage import seed_for


def episode(actor, config, reset_seed, action_seed, n=None, retain=False, on_step=None, max_steps=None, session=None):
    persistent = session is not None
    session = {} if session is None else session
    if not session or session['env'].end_reason is not None:
        if session:
            session['env'].close()
        env = Navigation(config, n)
        obs = env.reset(reset_seed)
        session.update(env=env, obs=obs, previous=None,
                       history=History(env.n, env.obs_dim, config['history'], env.horizon, include_time=env.include_time),
                       rng=np.random.default_rng(action_seed), reset_seed=reset_seed,
                       task_return=0., task_discounted_return=0.)
    env, history, rng = session['env'], session['history'], session['rng']
    device = next(actor.parameters()).device
    rows = {key: [] for key in ('x', 'state', 'actions', 'mu', 'reward', 'terminated', 'truncated')}
    rewards, collisions, robot_collisions = [], [], []
    coverages, full_coverages = [], []
    path_length = 0.
    try:
        obs, previous = session['obs'], session['previous']
        metrics = env.metrics()
        limit = (max_steps if max_steps is not None else env.horizon) if success_task(config) else (min(env.horizon, max_steps) if max_steps is not None else env.horizon)
        start_step = env.t
        for t in range(limit):
            if env.end_reason is not None:
                break
            x = history.append(obs, previous, env.t)
            with torch.no_grad():
                p = actor(torch.as_tensor(x, device=device)).cpu().numpy()
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
                raise FloatingPointError('非法 Actor 概率')
            cumulative = p.astype(np.float64).cumsum(-1)
            cumulative /= cumulative[:, -1:]
            actions = (rng.random(env.n)[:, None] >= cumulative).sum(-1)
            state = env.state() if retain else None
            positions = np.array([a.state.p_pos.copy() for a in env.world.agents]) if hasattr(env, 'world') else None
            obs, reward, terminated, truncated = env.step(actions)
            if positions is not None:
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
            collisions.append(metrics.get('collision_pairs', 0.))
            robot_collisions.append(metrics.get('collisions_per_agent', 0.))
            rewards.append(reward)
            session['task_return'] += reward
            session['task_discounted_return'] += config['gamma']**(env.t-1)*reward
            coverages.append(metrics.get('coverage', 0.))
            full_coverages.append(metrics.get('all_covered', 0.))
            if retain:
                for key, value in zip(rows, (x, state, actions, p, reward, terminated, truncated)):
                    rows[key].append(value)
            previous = actions
            if terminated or truncated:
                break
        session.update(obs=obs, previous=previous)
        length = len(rewards)
        mean = lambda values: float(np.mean(values)) if len(values) else 0.
        metrics.update(J=float(np.dot(np.power(config['gamma'], np.arange(length)), rewards)),
                       return_undiscounted=float(sum(rewards)),
                       collision_pairs=mean(collisions), collisions_per_agent=mean(robot_collisions),
                       mean_reward=mean(rewards), mean_coverage=mean(coverages),
                       tail_coverage=mean(coverages[length//2:]),
                       tail_all_covered=mean(full_coverages[length//2:]), episode_steps=length)
        if config.get('environment', 'navigation') != 'navigation':
            for key in ('collision_pairs', 'collisions_per_agent', 'mean_coverage', 'tail_coverage', 'tail_all_covered'):
                metrics.pop(key, None)
        if arrival_task(config) and config.get('environment', 'navigation') == 'navigation':
            success = env.end_reason == 'success'
            metrics.update(success=float(success), success_steps=length if success else None,
                           restricted_steps=length if success else limit,
                           end_reason=env.end_reason or 'collection_cutoff',
                           collision_pairs_total=float(sum(collisions)), path_length=path_length)
        if success_task(config):
            metrics.update(censored=float(env.end_reason != 'success'),
                           segment_start_step=start_step, task_elapsed_steps=env.t)
        session['last_metrics'] = metrics
        if not retain:
            return metrics
        batch = {key: np.asarray(value) for key, value in rows.items()}
        # Construct s_T / h_T using the last action, before closing/resetting the env.
        batch.update(final_x=history.append(obs, previous, env.t), final_state=env.state())
        if not length:
            # An initially solved scene consumes zero transitions.
            shapes = dict(x=batch['final_x'].shape, state=batch['final_state'].shape,
                          actions=(env.n,), mu=(env.n, config.get('action_dim', 5)), reward=(), terminated=(), truncated=())
            for key, shape in shapes.items():
                batch[key] = np.empty((0, *shape))
        return batch
    finally:
        if not persistent:
            env.close()


class Collector:
    def __init__(self, config, condition, seed, log=None):
        self.config, self.condition, self.seed, self.log = config, condition, seed, log
        self.counts, self.costs = Counter(), Counter()
        self.sessions = []

    def close(self):
        for session in self.sessions:
            if session:
                session['env'].close()
        self.sessions = []

    def state_dict(self):
        """Exact committed physical state; never restart a live task on resume."""
        return [self._session_state(s) if s else None for s in self.sessions]

    @staticmethod
    def _session_state(s):
        env = s['env']
        return dict(reset_seed=s['reset_seed'], t=env.t, end_reason=env.end_reason,
                    bodies=[dict(p_pos=b.state.p_pos.copy(), p_vel=b.state.p_vel.copy(),
                                 c=getattr(b.state, 'c', None)) for b in [*env.world.agents, *env.world.landmarks]],
                    obs=s['obs'].copy(), previous=s['previous'], frames=s['history'].frames.copy(),
                    rng=s['rng'].bit_generator.state, task_return=s['task_return'],
                    task_discounted_return=s['task_discounted_return'])

    def load_state_dict(self, saved):
        self.close()
        if saved is None:
            return
        self.sessions = [self._restore_session(s) if s is not None else {} for s in saved]

    def _restore_session(self, saved):
        env = Navigation(self.config)
        env.reset(saved['reset_seed'])
        for body, state in zip([*env.world.agents, *env.world.landmarks], saved['bodies']):
            body.state.p_pos, body.state.p_vel = state['p_pos'].copy(), state['p_vel'].copy()
            if state['c'] is not None:
                body.state.c = state['c'].copy()
        env.t, env.end_reason = saved['t'], saved['end_reason']
        env.env.unwrapped.steps = env.t
        history = History(env.n, env.obs_dim, self.config['history'], env.horizon, include_time=env.include_time)
        history.frames[:] = saved['frames']
        rng = np.random.default_rng()
        rng.bit_generator.state = saved['rng']
        return dict(env=env, history=history, rng=rng,
                    **{k:saved[k] for k in ('obs','previous','reset_seed','task_return','task_discounted_return')})

    @property
    def used(self):
        return sum(self.costs.values())

    def collect(self, actor, count, purpose, retain=True):
        arrival = arrival_task(self.config)
        streaming = success_task(self.config) and purpose == 'train'
        if streaming and not self.sessions:
            self.sessions = [{} for _ in range(count)]
        if streaming and len(self.sessions) != count:
            raise ValueError('Cannot change persistent training lane count within a run')
        remaining = [self.config['horizon']] * count
        fixed_steps = arrival and purpose == 'train' and not self.config.get('complete_episodes', False)
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
            lane = next((i for i, left in enumerate(remaining) if left), 0) if streaming else None
            if streaming:
                limit = remaining[lane]
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
                retain=retain, on_step=debit, max_steps=limit,
                session=self.sessions[lane] if streaming else None)
            length = len(result['reward']) if retain else result['episode_steps']
            if streaming:
                remaining[lane] -= length
            if not fixed_steps or length:
                episodes.append(result)
            if self.log:
                self.log('sampling', purpose=purpose, episode=index, used=self.used)
                if streaming:
                    session = self.sessions[lane]
                    m = session['last_metrics']
                    self.log('train_segment', lane=lane, segment_index=index, start_step=m['segment_start_step'],
                             end_step=m['task_elapsed_steps'], steps=length, discounted_return=m['J'],
                             return_undiscounted=m['return_undiscounted'], end_reason=m['end_reason'])
                    if session['env'].end_reason == 'success':
                        self.log('completed_task', lane=lane, steps=session['env'].t,
                                 return_undiscounted=session['task_return'],
                                 discounted_return=session['task_discounted_return'],
                                 policy_scope='training_policy_may_change_between_segments')
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
                                    1/self.config.get('action_dim', 5) if key == 'mu' else 0.)
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
