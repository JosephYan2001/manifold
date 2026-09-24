"""Complete-episode collection and frozen Actor evaluation.

Evaluation never constructs a target Critic, updates parameters, or selects a
checkpoint. Scene seeds and stochastic action seeds are fixed before a run.
"""
from pathlib import Path
import math
import time
import numpy as np
import torch

from .envs import make_env


def write_rows(path, rows):
    from ..common.reporting import write_csv
    write_csv(path, rows)


def tensor_obs(obs, device):
    return {key: torch.as_tensor(value, device=device,
            dtype=torch.bool if key == 'mask' else torch.float32) for key, value in obs.items()}


def load_actor(checkpoint, device='cpu'):
    """Load only our own trusted checkpoint format; returns (actor, payload)."""
    saved = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if 'current_config' in saved.get('config', {}):
        from ..cooperative_navigation.models import load_actor as legacy_load
        from ..cooperative_navigation.models.deployment import DeploymentActor
        legacy_config = saved['config']
        actor = DeploymentActor(legacy_load(saved).to(device), legacy_config).eval()
        normalized = dict(saved, config=legacy_config['current_config'],
            method=saved['current_method'], environment=saved['current_environment'],
            protocol=saved['current_protocol'], actor_config=actor.model_config,
            source_steps_total=sum(saved['costs'].values()), legacy_config=legacy_config)
        return actor, normalized
    raise ValueError('Expected a current-protocol Runner checkpoint. Retired framework checkpoints are not supported.')


@torch.no_grad()
def rollout_episode(env, actor, seed, gamma=.99, collect=False, deterministic=False):
    """A task deadline is terminal. Collection never cuts a task in this module."""
    device = next(actor.parameters()).device
    obs, info = env.reset(seed=int(seed))
    initial_success = bool(getattr(env, 'initial_done', False))
    rng = np.random.default_rng(int(seed) + 170000003)
    trajectory = []
    total, discounted, collisions, steps = 0., 0., 0., 0
    blocked, inference_seconds = 0, 0.
    entity_count = float(np.asarray(obs['mask']).sum(-1).mean())
    mode = getattr(actor, 'model_config', {}).get('mode', 'full')
    type_counts = np.stack([(np.asarray(obs['mask']) & (np.asarray(obs['entities'])[..., j] > .5)).sum(-1)
                            for j in range(2)], -1)
    truncated_entities = float(np.maximum(type_counts - 2, 0).sum(-1).mean()) if mode == 'nearest2' else 0.
    start = time.perf_counter()
    actor.eval()
    if not initial_success:
        for step in range(env.horizon):
            inference_start = time.perf_counter()
            p = actor(tensor_obs(obs, device)).detach().cpu().numpy()
            inference_seconds += time.perf_counter() - inference_start
            if not np.isfinite(p).all() or (p < 0).any():
                raise FloatingPointError('Actor produced an invalid action distribution.')
            actions = p.argmax(-1) if deterministic else np.array([rng.choice(len(row), p=(row / row.sum()).astype(np.float64) / (row / row.sum()).astype(np.float64).sum()) for row in p])
            state = env.state().copy() if collect else None
            next_obs, reward, terminated, truncated, info = env.step(actions)
            # Application adapters define both success and task deadline as true
            # terminals. A library time limit escaping this adapter is a bug;
            # treating it as zero-bootstrap would produce a false MC target.
            if truncated and not terminated:
                raise RuntimeError('Unexpected sampling truncation: full MC collection requires a true task terminal.')
            if collect:
                trajectory.append(dict(obs={key: value.copy() for key, value in obs.items()},
                    state=state, actions=actions.copy(), mu=p.copy(), reward=float(reward),
                    terminal=bool(terminated), time=step))
            steps += 1
            total += float(reward)
            discounted += gamma ** step * float(reward)
            collisions += float(info.get('collision_pairs', 0))
            blocked += int(info.get('blocked_forward_actions', 0))
            obs = next_obs
            if terminated:
                break
        if not terminated:
            raise RuntimeError('Environment did not terminate at the declared task horizon.')
    success = bool(info.get('success', initial_success))
    deliveries = float(info.get('deliveries_total', info.get('delivery_count', 0)))
    first_delivery = info.get('first_delivery_step', info.get('first_delivery_time', None))
    success_defined = 'success' in info or initial_success
    row = dict(scene_seed=int(seed), n_agents=env.n_agents, steps=steps,
        discounted_return=discounted, return_undiscounted=total,
        end_reason=info.get('end_reason', 'initial_success' if initial_success else 'task_terminal'),
        entity_count=entity_count,
        truncated_entity_count=truncated_entities, observation_mode=mode,
        inference_seconds=inference_seconds,
        inference_seconds_per_joint_step=inference_seconds / max(steps, 1),
        policy_mode='argmax' if deterministic else 'stochastic',
        wall_seconds=time.perf_counter() - start)
    if success_defined:
        row.update(success=float(success), initial_success=int(initial_success),
            completion_time=steps if success else env.horizon,
            success_completion_time=steps if success else '')
    for key in ('coverage', 'mean_goal_distance'):
        if key in info:
            row[key] = float(info[key])
    if 'collision_pairs' in info:
        row.update(collision_pairs=collisions, collisions_per_step=collisions / max(steps, 1),
            collision_participation_per_agent_step=2 * collisions / max(steps * env.n_agents, 1))
    if 'deliveries_total' in info or 'delivery_count' in info:
        row.update(deliveries=deliveries, throughput=deliveries / env.horizon,
            throughput_per_agent=deliveries / (env.horizon * env.n_agents),
            first_delivery_time=env.horizon if first_delivery is None else first_delivery,
            zero_delivery=int(deliveries == 0))
        if 'longest_no_delivery_streak' in info:
            row['longest_no_delivery_streak'] = info['longest_no_delivery_streak']
    if 'blocked_forward_actions' in info:
        row['blocked_forward_actions'] = blocked
    return row, trajectory


def summarize(rows):
    if not rows:
        return dict(episodes=0, environment_steps=0)
    out = dict(episodes=len(rows), environment_steps=sum(row['steps'] for row in rows))
    for key in ('discounted_return', 'return_undiscounted', 'success', 'completion_time',
                'coverage', 'mean_goal_distance', 'collisions_per_step',
                'collision_participation_per_agent_step', 'deliveries', 'throughput',
                'throughput_per_agent', 'first_delivery_time', 'zero_delivery',
                'longest_no_delivery_streak', 'blocked_forward_actions', 'entity_count', 'truncated_entity_count',
                'inference_seconds_per_joint_step', 'wall_seconds'):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, '')]
        values = [value for value in values if math.isfinite(value)]
        if values:
            out[key] = float(np.mean(values))
    if any('initial_success' in row for row in rows):
        out['initial_successes'] = sum(row.get('initial_success', 0) for row in rows)
    if 'success' in out:
        out['success_rate'] = out['success']
    if any('collision_pairs' in row for row in rows):
        out['collisions_per_step'] = sum(row.get('collision_pairs', 0) for row in rows) / max(out['environment_steps'], 1)
        out['collision_participation_per_agent_step'] = 2 * sum(row.get('collision_pairs', 0) for row in rows) / max(sum(row['steps'] * row['n_agents'] for row in rows), 1)
    out['inference_seconds_per_joint_step'] = sum(row['inference_seconds'] for row in rows) / max(out['environment_steps'], 1)
    return out


def evaluate_actor(actor, environment, config, seeds, n_agents=None, deterministic=False):
    env = make_env(environment, config, n_agents=n_agents)
    try:
        rows = [rollout_episode(env, actor, seed, config.get('gamma', .99),
                               deterministic=deterministic)[0] for seed in seeds]
        return rows, env.protocol()
    finally:
        env.close()


def evaluate_checkpoint(checkpoint, config, output, experiment='N-T'):
    actor, saved = load_actor(checkpoint, config.get('device', 'cpu'))
    # Frozen source protocol is authoritative. Only declared evaluation axes and
    # reporting settings can override it; target fitting hyperparameters cannot.
    c = dict(saved['config'])
    for key in ('evaluation_episodes', 'target_sizes', 'evaluation_seed', 'device',
                'deterministic_evaluation'):
        if key in config:
            c[key] = config[key]
    environment = saved['environment']
    base_seed = int(c.get('evaluation_seed', 1000000))
    count = int(c.get('evaluation_episodes', 500))
    sizes = list(dict.fromkeys([int(saved['config']['n_agents'])] + list(c.get('target_sizes', [2, 6, 8]))))
    state_before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    records, all_rows, protocols = [], [], []
    source_tags = dict(observation_mode=saved['actor_config'].get('mode', 'full'),
        source_experiment=saved['config'].get('experiment_id', 'unspecified'),
        relation_layers=saved['actor_config'].get('relation_layers', 2),
        source_profile=saved['config'].get('profile', 'unspecified'),
        source_horizon=saved['protocol'].get('horizon', saved['config'].get('horizon')),
        source_budget=saved['config'].get('budget'),
        source_smoke_only=bool(saved['config'].get('smoke_only', False)),
        smoke_only=bool(saved['config'].get('smoke_only', False) or config.get('smoke_only', False)),
        direction_check=bool(saved['config'].get('direction_check', True)) and saved['method'] in ('AN', 'SA'),
        return_check=bool(saved['config'].get('return_check', True)) and saved['method'] in ('AN', 'SA', 'DA'))
    for n_agents in sizes:
        target = dict(c)
        if experiment == 'N-D':
            target['initial_scale'] = float(saved['config'].get('initial_scale', 1.)) * math.sqrt(n_agents / saved['config']['n_agents'])
        if experiment == 'W-L':
            target['request_queue_size'] = n_agents
        rows, protocol = evaluate_actor(actor, environment, target,
            range(base_seed, base_seed + count), n_agents=n_agents,
            deterministic=bool(c.get('deterministic_evaluation', False)))
        for row in rows:
            row.update(experiment=experiment, method=saved['method'], seed=saved['config']['seed'],
                       source_n=saved['config']['n_agents'], target_n=n_agents,
                       theory_class='B_empirical', checkpoint=Path(checkpoint).name)
            row.update(source_tags)
        record = summarize(rows)
        record.update(experiment=experiment, method=saved['method'], seed=saved['config']['seed'],
                      source_n=saved['config']['n_agents'], target_n=n_agents,
                      evaluation_steps=record['environment_steps'],
                      theory_class='B_empirical', checkpoint=str(checkpoint))
        record.update(source_tags)
        records.append(record)
        all_rows.extend(rows)
        protocols.append(protocol)
    if any(not torch.equal(value, actor.state_dict()[key]) for key, value in state_before.items()):
        raise AssertionError('Frozen evaluation changed Actor parameters or buffers.')
    write_rows(Path(output) / 'evaluation_episodes.csv', all_rows)
    return dict(records=records, protocols=protocols, evaluation_steps=sum(row['steps'] for row in all_rows),
                frozen_actor_verified=True)
