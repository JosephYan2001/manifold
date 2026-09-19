"""Frozen-policy direction generalization probe; prints diagnostics, writes no results.

Uses fresh rollouts, so its interaction cost is additional to the original suite.
Direction networks and optional temporary candidate Actors are optimized;
the loaded source Actor/Critic and checkpoint files remain unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from manifold_project.experiments.cooperative_navigation.models import Actor, Critic, Direction
from manifold_project.experiments.cooperative_navigation.training.collector import Collector, episode
from manifold_project.experiments.cooperative_navigation.training.ppo import returns, gae, weighted, direction_terms
from manifold_project.experiments.cooperative_navigation.training.storage import seed_for


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def evaluate_candidates(actor, config, directions, fit_batch, args):
    """Fit final-round candidates on common data; evaluate on fresh paired scenes."""
    original = {key: value.clone() for key, value in actor.state_dict().items()}
    policies = [('unchanged', actor)]
    for name in args.candidate_variants:
        direction = directions[name]
        candidate = deepcopy(actor)
        with torch.no_grad():
            target = torch.softmax(fit_batch['mu'].log()+config['eta']*direction(fit_batch['x']), -1)
            target_kl = float(weighted((target*(target.log()-fit_batch['mu'].log())).sum(-1), config['gamma']))
        optimizer = torch.optim.Adam(candidate.parameters(), lr=config['actor_lr'])
        rng = np.random.default_rng(seed_for('diagnostic-candidate-fit', args.diagnostic_seed, args.repeats))
        updates = 0
        emit(stage='candidate_fit_start', variant=name, direction_round=args.repeats,
             fit_episodes=len(target), eta=config['eta'])
        for _ in range(config['actor_epochs']):
            order = rng.permutation(len(target))
            for start in range(0, len(order), config['minibatch_episodes']):
                idx = order[start:start+config['minibatch_episodes']]
                probability = candidate(fit_batch['x'][idx])
                loss = weighted((target[idx]*(target[idx].log()-probability.log())).sum(-1), config['gamma'])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), config['grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                updates += 1
        with torch.no_grad():
            probability = candidate(fit_batch['x'])
            residual = float(weighted((target*(target.log()-probability.log())).sum(-1), config['gamma']))
            step_kl = float(weighted((fit_batch['mu']*(fit_batch['mu'].log()-probability.log())).sum(-1), config['gamma']))
        emit(stage='candidate_fit_done', variant=name, target_kl=target_kl, fit_kl=residual,
             actual_step_kl=step_kl, actor_fit_updates=updates)
        candidate.eval()
        policies.append((name, candidate))
    evaluated, evaluation_steps = {}, 0

    def debit():
        nonlocal evaluation_steps
        evaluation_steps += 1

    for name, policy in policies:
        emit(stage='candidate_evaluation_start', variant=name, episodes=args.candidate_episodes,
             evaluation_seed=args.candidate_seed)
        rows = []
        for i in range(args.candidate_episodes):
            rows.append(episode(policy, config,
                seed_for('direction-candidate-reset', args.candidate_seed, i),
                seed_for('direction-candidate-action', args.candidate_seed, i), on_step=debit))
            if (i+1) % 128 == 0:
                emit(stage='candidate_evaluation_progress', variant=name, episodes_done=i+1)
        evaluated[name] = rows
        means, differences = {}, {}
        for key in ('J', 'coverage', 'distance', 'all_covered', 'collision_pairs'):
            values = np.asarray([r[key] for r in rows], dtype=np.float64)
            means[key] = float(values.mean())
            if name != 'unchanged':
                paired = values-np.asarray([r[key] for r in evaluated['unchanged']], dtype=np.float64)
                difference, se = float(paired.mean()), float(paired.std(ddof=1)/len(paired)**.5)
                differences[key] = dict(mean=difference, se=se,
                                         normal_95=[difference-1.96*se, difference+1.96*se])
        emit(candidate_result=name, episodes=args.candidate_episodes, means=means,
             paired_candidate_minus_unchanged=differences,
             note='Descriptive paired episode uncertainty, not corrected for multiple candidates; '
                  'one frozen training seed, final direction round only, no best-round selection.')
    assert evaluation_steps == len(policies)*args.candidate_episodes*config['horizon']
    assert all(torch.equal(value, original[key]) for key, value in actor.state_dict().items())
    return evaluation_steps


def compare_directions(actor, critic, state, config, collector, args):
    """Paired frozen-policy screening, never an online-training performance claim."""
    from time import perf_counter
    started = perf_counter()
    count = config['train_episodes']
    # Four times the data, one quarter the reuse: keep gradient-step counts equal
    # for the standard 16-episode / 4-minibatch / 16-epoch configuration.
    if count % config['minibatch_episodes'] or config['direction_epochs'] % 4:
        raise ValueError('Comparison requires whole minibatches and direction_epochs divisible by four')
    specs = [
        ('reset_mc', count, config['direction_epochs'], 'mc', 'reset'),
        ('warm_mc', count, config['direction_epochs'], 'mc', 'warm'),
        ('warm_ema_mc', count, config['direction_epochs'], 'mc', 'ema'),
        ('batch4x_mc', 4*count, config['direction_epochs']//4, 'mc', 'reset'),
        ('reset_gae', count, config['direction_epochs'], 'gae', 'reset'),
    ]

    def labels(batch):
        with torch.no_grad():
            prediction = critic(batch['state'])
            mc = returns(batch['reward'], config['gamma'])-prediction
            gae_label = gae(batch['reward'], prediction, config['gamma'], config['gae_lambda'])
            batch['mc'] = mc.unsqueeze(-1).expand_as(batch['actions'])
            batch['gae'] = gae_label.unsqueeze(-1).expand_as(batch['actions'])
        return batch

    def metrics(model, data, label):
        with torch.no_grad():
            q = model(data['x'])
            score, fisher = direction_terms(q, data['mu'], data['actions'])
            linear = 2*data[label]*score
            per_episode = torch.tensor([float(weighted((linear-fisher)[i:i+1], config['gamma']))
                                        for i in range(len(q))])
            per_linear = torch.tensor([float(weighted(linear[i:i+1], config['gamma']))
                                       for i in range(len(q))])
            first, curvature = float(per_linear.mean()), float(weighted(fisher, config['gamma']))
            target = torch.softmax(data['mu'].log()+config['eta']*q, -1)
            return dict(score=float(per_episode.mean()), score_se=float(per_episode.std()/len(q)**.5),
                        linear=first, linear_se=float(per_linear.std()/len(q)**.5), fisher=curvature,
                        normalized_linear=first/curvature**.5 if curvature > 1e-12 else None,
                        target_kl=float(weighted((target*(target.log()-data['mu'].log())).sum(-1), config['gamma'])))

    heldout = labels(collector.collect(actor, args.heldout_episodes, 'compare-heldout'))
    warm_models, records, final_directions = {}, {name: [] for name, *_ in specs}, {}
    emit(mode='compare', variants=[dict(name=name, train_episodes=n, epochs=e, label=label,
                                       initialization=mode, optimizer='fresh Adam each round')
                                   for name, n, e, label, mode in specs], ema_tau=args.ema_tau,
         note='Rounds share a frozen Actor/Critic and one heldout set. Warm models retain weights. '
              'All heldout scores use MC labels. Batch4x has four times the samples and equal gradient steps. '
              'This is screening, not independent seed replication or changing-policy validation.')
    for repetition in range(args.repeats):
        pool = labels(collector.collect(actor, 4*count, 'compare-fit'))
        for name, n, epochs, label, mode in specs:
            batch = {key: value[:n] for key, value in pool.items()}
            if mode != 'reset' and repetition:
                direction = warm_models[name]
            else:
                torch.manual_seed(seed_for('diagnostic-direction', args.diagnostic_seed, repetition))
                direction = Direction(state['input_dim'], config)
            # EMA interpolates along one continuing network's optimization path;
            # never average independently initialized hidden-unit representations.
            before = deepcopy(direction.state_dict()) if mode == 'ema' else None
            optimizer = torch.optim.Adam(direction.parameters(), lr=config['direction_lr'])
            rng = np.random.default_rng(seed_for('diagnostic-minibatches', repetition))
            updates = 0
            for _ in range(epochs):
                order = rng.permutation(n)
                for start in range(0, n, config['minibatch_episodes']):
                    idx = order[start:start+config['minibatch_episodes']]
                    score, fisher = direction_terms(direction(batch['x'][idx]), batch['mu'][idx], batch['actions'][idx])
                    loss = weighted(fisher-2*batch[label][idx]*score, config['gamma'])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(direction.parameters(), config['grad_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
            # First round establishes a common learned starting point for warm/EMA.
            if mode == 'ema' and repetition:
                direction.load_state_dict({key: (1-args.ema_tau)*before[key]+args.ema_tau*value
                                           for key, value in direction.state_dict().items()})
            if mode != 'reset':
                warm_models[name] = direction
            result = dict(variant=name, round=repetition+1, optimizer_steps=updates,
                          train=metrics(direction, batch, label), heldout_mc=metrics(direction, heldout, 'mc'))
            records[name].append(result)
            emit(**result)
            if args.candidate_episodes and repetition == args.repeats-1 and name in args.candidate_variants:
                final_directions[name] = deepcopy(direction)
    for name, n, epochs, label, mode in specs:
        tail = records[name][len(records[name])//2:]
        emit(summary=name, tail_rounds=len(tail),
             mean_heldout_mc={key: float(np.mean([row['heldout_mc'][key] for row in tail]))
                              for key in ('score', 'linear', 'fisher', 'target_kl')},
             nominal_training_steps=n*args.repeats*config['horizon'],
             optimizer_steps=sum(row['optimizer_steps'] for row in records[name]))
    evaluation_steps = 0
    if args.candidate_episodes:
        # Common final-round 64-episode pool for all Actor fits; label fields unused.
        # This isolates direction variants and is not a complete training update.
        evaluation_steps = evaluate_candidates(actor, config, final_directions, pool, args)
    emit(complete=True, additional_diagnostic_steps=collector.used+evaluation_steps,
         direction_probe_steps=collector.used, candidate_evaluation_steps=evaluation_steps,
         seconds=perf_counter()-started, actor_updated=False, critic_updated=False,
         candidate_actors_fitted=bool(args.candidate_episodes),
         note='Source Actor/Critic unchanged. Shared sampling charged once; nominal variant costs reported separately.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parent /
                        'cooperative_navigation/results/reference/nav_04_pilot/no_checks/seed_40/checkpoints/final.pt')
    parser.add_argument('--heldout-episodes', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--diagnostic-seed', type=int, default=90219)
    parser.add_argument('--compare', action='store_true',
                        help='Compare reset, warm start, warm+EMA, larger batch and GAE on a frozen policy')
    parser.add_argument('--ema-tau', type=float, default=.2,
                        help='EMA new-parameter weight per fitted round; only used by --compare')
    parser.add_argument('--candidate-episodes', type=int, default=0,
                        help='With --compare, fit final-round candidate Actors and evaluate on fresh paired episodes; 0 disables')
    parser.add_argument('--candidate-variants', nargs='+',
                        choices=['reset_mc', 'warm_mc', 'warm_ema_mc', 'batch4x_mc', 'reset_gae'],
                        default=['reset_mc', 'warm_mc', 'batch4x_mc'])
    parser.add_argument('--candidate-seed', type=int, default=90221,
                        help='Independent reset/action seed for paired candidate return evaluation')
    args = parser.parse_args(argv)
    if args.heldout_episodes < 2 or args.repeats < 1:
        parser.error('heldout-episodes must be >= 2 and repeats >= 1')
    if not 0 < args.ema_tau <= 1:
        parser.error('ema-tau must be in (0, 1]')
    if args.candidate_episodes < 0 or args.candidate_episodes == 1:
        parser.error('candidate-episodes must be 0 or >= 2')
    if args.candidate_episodes and not args.compare:
        parser.error('candidate-episodes requires --compare')
    if len(args.candidate_variants) != len(set(args.candidate_variants)):
        parser.error('candidate-variants must be unique')
    # Checkpoints must come from this project's trusted local training runs.
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['condition'] in ('mappo', 'ippo'):
        parser.error('This probe is for direction-based methods, not PPO')
    if args.compare and state['condition'] == 'sampled':
        parser.error('--compare currently screens the analytic Fisher objective only')
    config = dict(state['config'])
    config['device'] = 'cpu'
    total_steps = (args.heldout_episodes + args.repeats * config['train_episodes'] * (4 if args.compare else 1)) * config['horizon']
    planned_candidate_steps = (1+len(args.candidate_variants))*args.candidate_episodes*config['horizon']
    config['budget'] = total_steps  # Separate diagnostic ledger, not the completed training budget.
    torch.set_num_threads(1)
    actor = Actor(state['input_dim'], config)
    actor.load_state_dict(state['actor'])
    actor.eval()
    critic = Critic(state['critic']['net.0.weight'].shape[1], config)
    critic.load_state_dict(state['critic'])
    critic.eval()
    collector = Collector(config, 'frozen-plateau-diagnostic', args.diagnostic_seed)
    emit(checkpoint=str(args.checkpoint.resolve()), sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
         torch_version=torch.__version__, numpy_version=np.__version__, device='cpu',
         diagnostic_seed=args.diagnostic_seed, additional_steps=total_steps+planned_candidate_steps,
         direction_probe_steps=total_steps, candidate_evaluation_steps_planned=planned_candidate_steps,
         train_episodes=config['train_episodes'], direction_label=config['direction_label'],
         note='One shared heldout set; repetitions are not independent training seeds or independent test sets.')
    if args.compare:
        compare_directions(actor, critic, state, config, collector, args)
        return

    def labels(batch, name):
        with torch.no_grad():
            target = returns(batch['reward'], config['gamma'])
            prediction = critic(batch['state'])
            residual = target - prediction
            advantage = gae(batch['reward'], prediction, config['gamma'], config['gae_lambda']) \
                if config['direction_label'] == 'gae' else residual
            batch['advantage'] = advantage.unsqueeze(-1).expand_as(batch['actions'])
            variance = target.var(unbiased=False)
            emit(data=name, critic_fresh_mse=float(weighted(residual.square(), config['gamma'])),
                 critic_fresh_explained_variance=float(1-residual.var(unbiased=False)/variance) if variance > 0 else None,
                 MC_return_std=float(target.std()), advantage_std=float(advantage.std()))
        return batch

    heldout = labels(collector.collect(actor, args.heldout_episodes, 'heldout'), 'heldout')
    for repetition in range(args.repeats):
        batch = labels(collector.collect(actor, config['train_episodes'], 'direction-fit'), f'train_{repetition}')
        torch.manual_seed(seed_for('diagnostic-direction', args.diagnostic_seed, repetition))
        direction = Direction(state['input_dim'], config)
        optimizer = torch.optim.Adam(direction.parameters(), lr=config['direction_lr'])
        rng = np.random.default_rng(seed_for('diagnostic-minibatches', repetition))
        for _ in range(config['direction_epochs']):
            order = rng.permutation(config['train_episodes'])
            for start in range(0, len(order), config['minibatch_episodes']):
                idx = order[start:start+config['minibatch_episodes']]
                score, fisher = direction_terms(direction(batch['x'][idx]), batch['mu'][idx], batch['actions'][idx])
                quadratic = score.square() if state['condition'] == 'sampled' else fisher
                loss = weighted(quadratic-2*batch['advantage'][idx]*score, config['gamma'])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(direction.parameters(), config['grad_norm'], error_if_nonfinite=True)
                optimizer.step()
        for name, data in (('train', batch), ('heldout', heldout)):
            with torch.no_grad():
                q = direction(data['x'])
                score, fisher = direction_terms(q, data['mu'], data['actions'])
                linear = 2*data['advantage']*score
                quadratic = score.square() if state['condition'] == 'sampled' else fisher
                values = torch.tensor([float(weighted((linear-quadratic)[i:i+1], config['gamma']))
                                       for i in range(len(q))])
                target = torch.softmax(data['mu'].log()+config['eta']*q, -1)
                emit(repetition=repetition, data=name, episodes=len(q), direction_score=float(values.mean()),
                     score_se=float(values.std()/len(values)**.5), linear_2Aq=float(weighted(linear, config['gamma'])),
                     quadratic=float(weighted(quadratic, config['gamma'])),
                     target_kl=float(weighted((target*(target.log()-data['mu'].log())).sum(-1), config['gamma'])),
                     q_abs_mean=float(q.abs().mean()))
    emit(complete=True, additional_diagnostic_steps=collector.used, actor_updated=False, critic_updated=False)


if __name__ == '__main__':
    main()
