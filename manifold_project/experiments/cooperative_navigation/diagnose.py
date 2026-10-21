"""Frozen-checkpoint probes: direction/Actor/Critic fitting and teacher evaluation.

Checkpoint mode collects new samples. Fixed-target mode reuses a saved batch,
direction and target without sampling or direction training. teacher-eval executes
only the saved old policy and teacher, with no fitting. Sources are read-only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--fixed-target', type=Path, help='Replay one saved fixed target without collecting data or training a direction')
    source.add_argument('--critic-data', type=Path, help='Critic mode: replay a compact frozen value-data packet without sampling')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['direction', 'fit', 'teacher-eval', 'critic', 'critic-target'], required=True)
    p.add_argument('--critic-epochs', type=int, help='Critic-only fixed fitting budget; default 80, replay inherits')
    p.add_argument('--critic-starts', nargs='+', choices=['fresh', 'checkpoint'], help='Critic-only starts; default both, each compared with raw/fixed-normalized targets')
    p.add_argument('--seeds', nargs='+', type=int, help='Diagnostic data seeds; checkpoint default: 40 41 42; snapshot replay inherits its seed')
    p.add_argument('--batch-episodes', type=int, help='Defaults to the saved training batch size')
    p.add_argument('--validation-episodes', type=int)
    p.add_argument('--evaluation-episodes', type=int, help='Per policy; teacher-eval defaults to 100 complete pairs')
    p.add_argument('--evaluation-seed', type=int, help='Independent scene-set seed; teacher-eval requires a new explicit seed')
    direction_budget = p.add_mutually_exclusive_group()
    direction_budget.add_argument('--direction-epochs', nargs='+', type=int, help='Explicit direction budget, also supported in fit mode')
    p.add_argument('--sample-sizes', nargs='+', type=int, help='Expand an N x U matrix; same data seed gives nested batches')
    direction_budget.add_argument('--direction-updates', nargs='+', type=int, help='Exact direction updates in either mode; divisible by minibatches per epoch')
    p.add_argument('--init-seeds', nargs='+', type=int, help='Separate direction initialization seed axis')
    p.add_argument('--init-seed', type=int, help='One initialization seed, independent from --seeds data seeds')
    p.add_argument('--fit-trace-every-updates', type=int, help='Also recompute full train/validation KL every N optimizer steps; default 0 means epoch only')
    p.add_argument('--fit-epochs', type=int, help='Defaults to saved Actor epochs')
    p.add_argument('--prepare-only', action='store_true', help='Fit/critic mode: collect and save fixed data, then stop before fitting/evaluation')
    p.add_argument('--actor-lrs', nargs='+', type=float, help='Fit default: 0.0003 0.0001 0.00003')
    p.add_argument('--target-kl', type=float, help='Direction mode default: 0.0001')
    p.add_argument('--eta', type=float, help='Defaults to saved first step size')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--allow-duplicate-openmp', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def match_mixture(mu, proposed, weighted, requested):
    """Monotone KL(mu || (1-alpha)mu + alpha*proposed), on fixed states."""
    mu, proposed = normalized(mu), normalized(proposed)
    def kl(alpha):
        p = (1-alpha)*mu + alpha*proposed
        return max(0., float(weighted((mu*(mu.log()-p.log())).sum(-1))))
    raw = kl(1.)
    if requested > raw + 1e-9:
        raise ValueError('Mixture cannot increase a candidate KL')
    if requested == 0:
        return 0., 0.
    lo, hi = 0., 1.
    for _ in range(40):
        mid = (lo+hi)/2
        if kl(mid) < requested:
            lo = mid
        else:
            hi = mid
    alpha = (lo+hi)/2
    return alpha, kl(alpha)


def normalized(p):
    # Float32 probability sums differ from one enough to swamp tiny KLs.
    p = p.double()
    return p/p.sum(-1, keepdim=True)


def run(args):
    started = time.perf_counter()
    from ..common.runtime import configure_runtime
    configure_runtime(args.allow_duplicate_openmp)
    if args.mode == 'critic-target':
        from .training.critic_target_probe import run_critic_target_probe
        return run_critic_target_probe(args)
    if args.mode == 'critic':
        from .training.critic_probe import run_critic_probe
        return run_critic_probe(args)
    if args.critic_data is not None or args.critic_epochs is not None or args.critic_starts is not None:
        raise ValueError('Critic data/budget/starts require --mode critic')
    if args.mode == 'teacher-eval':
        from .evaluation.teacher_probe import run_teacher_evaluation
        return run_teacher_evaluation(args)
    # Preserve existing fit/direction defaults; None lets teacher-eval reject
    # explicitly supplied fitting arguments instead of silently ignoring them.
    args.actor_lrs = args.actor_lrs if args.actor_lrs is not None else [3e-4, 1e-4, 3e-5]
    args.target_kl = args.target_kl if args.target_kl is not None else 1e-4
    args.fit_trace_every_updates = args.fit_trace_every_updates if args.fit_trace_every_updates is not None else 0
    import torch
    from torch import nn
    from .training.runner import Runner
    from .training.collector import Collector
    from .training.ppo import direction_terms
    from .training.diagnostics import categorical_kl, direction_metrics, cpu_copy
    from .training.actor_fit import fit_actor, target_kl, state_changed
    from .training.fixed_target import Target, save_fixed_target, validate_packet, to_device, fingerprint
    from .training.storage import load_pt, save_pt, save_json, seed_for
    from .models import Direction
    from .evaluation.evaluate import evaluate
    from ..common.reporting import write_csv

    packet = None
    source = (args.fixed_target or args.checkpoint).resolve(strict=True)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    state = load_pt(source)
    if args.fixed_target:
        if args.mode != 'fit' or args.prepare_only:
            raise ValueError('--fixed-target requires fit mode and cannot be combined with --prepare-only')
        validate_packet(state)
        packet = state
        forbidden = ('seeds', 'batch_episodes', 'validation_episodes', 'direction_epochs',
                     'direction_updates', 'sample_sizes', 'init_seeds', 'init_seed', 'eta')
        if any(getattr(args, key) is not None for key in forbidden):
            raise ValueError('Fixed-target replay inherits data, direction, eta and seeds; do not override them')
        if args.fit_epochs is not None and args.fit_epochs != packet['fit_epochs']:
            raise ValueError('Replay must use the saved Actor optimization budget and minibatch schedule')
        args.seeds = [packet['data_seed']]
        args.validation_episodes = len(packet['validation']['reward'])
        args.fit_epochs = packet['fit_epochs']
        if args.evaluation_episodes is None:
            args.evaluation_episodes = packet['evaluation_episodes']
        if args.evaluation_seed is None:
            args.evaluation_seed = packet['evaluation_seed']
        args.init_seed = packet['provenance']['initialization_seed']
    elif (state.get('current_method') != 'AN' or state['config'].get('environment') != 'navigation'
          or state['config'].get('task_mode') != 'success_only' or not state['config'].get('complete_episodes')):
        raise ValueError('This probe requires an AN episodic success-only navigation checkpoint')
    c = deepcopy(state['config'])
    c.update(device=args.device, budget=10**12)
    args.seeds = args.seeds if args.seeds is not None else [40, 41, 42]
    args.validation_episodes = args.validation_episodes if args.validation_episodes is not None else 32
    args.evaluation_episodes = args.evaluation_episodes if args.evaluation_episodes is not None else 40
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be distinct and nonnegative')
    if args.fit_trace_every_updates < 0 or any(v is not None and v < 0 for v in (args.init_seed, args.evaluation_seed)):
        raise ValueError('Trace interval and seeds must be nonnegative')
    positive = [args.validation_episodes, args.evaluation_episodes, args.target_kl, *args.actor_lrs]
    positive += (args.direction_epochs or []) + (args.direction_updates or [])
    positive += [v for v in (args.batch_episodes, args.fit_epochs, args.eta) if v is not None]
    if any(not np.isfinite(v) or v <= 0 for v in positive) or len(set(args.actor_lrs)) != len(args.actor_lrs):
        raise ValueError('Counts/rates must be finite and positive; learning rates must be distinct')
    output = args.output.resolve()
    source_run = source.parent if packet else source.parent.parent
    if output == source_run or source_run in output.parents or output in source_run.parents:
        raise ValueError('Diagnostic output must be separate from its input directory')
    if output.exists():
        raise FileExistsError(f'Use a new diagnostic output directory: {output}')

    # Matrix expansion is separate from a single fixed-target fit.
    if args.sample_sizes or args.init_seeds or (args.mode == 'direction' and args.direction_updates):
        if args.mode != 'direction':
            raise ValueError('N x U expansion is supported in direction mode only')
        sizes = args.sample_sizes or [args.batch_episodes or c['train_episodes']]
        inits = args.init_seeds or [args.init_seed]
        if any(n <= 0 for n in sizes) or any(s is not None and s < 0 for s in inits):
            raise ValueError('Sample sizes must be positive and initialization seeds nonnegative')
        if len(set(sizes)) != len(sizes) or len(set(inits)) != len(inits):
            raise ValueError('Matrix axes must not contain duplicates')
        jobs = []
        for n in sizes:
            for initialization in inits:
                child = deepcopy(args)
                child.sample_sizes = child.direction_updates = child.init_seeds = None
                child.batch_episodes, child.init_seed = n, initialization
                child.output = output/f'n{n}_init{initialization if initialization is not None else "data"}'
                if args.direction_updates:
                    per_epoch = int(np.ceil(n/c['minibatch_episodes']))
                    if any(u % per_epoch for u in args.direction_updates):
                        raise ValueError('U must be divisible by ceil(N / saved minibatch_episodes)')
                    child.direction_epochs = [u//per_epoch for u in args.direction_updates]
                jobs.append(child)
        matrix = dict(checkpoint=str(source), sample_sizes=sizes, init_seeds=inits,
                      requested_updates=args.direction_updates, data_seeds=args.seeds)
        if not args.dry_run:
            output.mkdir(parents=True)
            save_json(output/'matrix_manifest.json', dict(matrix, status='running'))
        for child in jobs:
            run(child)
        if not args.dry_run:
            save_json(output/'matrix_manifest.json', dict(matrix, status='completed'))
        return matrix

    if args.prepare_only and args.mode != 'fit':
        raise ValueError('--prepare-only is supported in fit mode only')
    if args.batch_episodes is not None:
        c['train_episodes'] = args.batch_episodes
    per_epoch = int(np.ceil(c['train_episodes']/c['minibatch_episodes']))
    epochs_list = args.direction_epochs or ([16, 64] if args.mode == 'direction' else [c['direction_epochs']])
    if args.direction_updates:
        if any(u % per_epoch for u in args.direction_updates):
            raise ValueError('U must be divisible by ceil(N / saved minibatch_episodes)')
        epochs_list = [u//per_epoch for u in args.direction_updates]
    if packet:
        epochs_list = [packet['direction_epochs']]
    if len(set(epochs_list)) != len(epochs_list) or (args.mode == 'fit' and len(epochs_list) != 1):
        raise ValueError('Direction budgets must be distinct; fit mode requires exactly one fixed direction budget')
    fit_epochs = args.fit_epochs or c['actor_epochs']
    eta = packet['eta'] if packet else args.eta if args.eta is not None else c.get('step_sizes', [c['eta']])[0]
    rates = [c['actor_lr']] if args.mode == 'direction' else args.actor_lrs
    evaluation_purpose = ('frozen-checkpoint-probe' if args.evaluation_seed is None
                          else f'frozen-checkpoint-probe-scenes-{args.evaluation_seed}')
    provenance = packet['provenance'] if packet else dict(
        checkpoint=str(source), checkpoint_sha256=original_hash,
        source_round=state['round'], source_training_seed=state['seed'])
    plan = dict(**{k:v for k,v in provenance.items() if k != 'initialization_seed'},
        input_path=str(source), input_sha256=original_hash, input_kind='fixed_target' if packet else 'checkpoint',
        mode=args.mode, diagnostic_seeds=args.seeds, batch_episodes=c['train_episodes'],
        initialization_seed=args.init_seed, fit_trace_every_updates=args.fit_trace_every_updates,
        validation_episodes=args.validation_episodes, evaluation_episodes=args.evaluation_episodes,
        evaluation_scene_seed=args.evaluation_seed, evaluation_purpose=evaluation_purpose,
        direction_epochs=epochs_list, direction_updates=[e*per_epoch for e in epochs_list],
        fit_epochs=fit_epochs, actor_updates_per_trajectory=fit_epochs*per_epoch, actor_lrs=rates,
        prepare_only=args.prepare_only, eta=eta, requested_kl=args.target_kl, device=args.device,
        fit_semantics_version='actor-fit-selection-v1',
        fit_selection_atol=c.get('actor_fit_selection_atol', 1e-12),
        fit_selection_rtol=c.get('actor_fit_selection_rtol', 1e-6),
        kl_measure='discount-weighted mean KL(old || candidate) on independent old-policy validation states',
        match_rule=('common KL = min(requested, 0.95 * minimum raw candidate KL); probability mixture with old policy'
                    if args.mode == 'direction' else 'none; fit mode reports unmodified candidate policies'),
        best_fit_rule='minimum finite full-training-batch KL at epoch boundaries including zero; numerical ties favor old',
        scope='saved fixed target replay' if packet else 'new samples at frozen Actor/Critic; not historical batch replay',
        probe_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fit_code_sha256=hashlib.sha256((Path(__file__).parent/'training/actor_fit.py').read_bytes()).hexdigest())
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return plan
    output.mkdir(parents=True)
    save_json(output/'effective_runner_config.json', c)
    rows, traces, episode_rows, scores, target_records = [], [], [], [], []
    collection_steps = evaluation_steps = 0
    counters = []
    def persist(status='running', **extra):
        for filename, records in [('diagnostics.csv', rows), ('fit_trace.csv', traces),
                                  ('evaluation_episodes.csv', episode_rows), ('direction_scores.csv', scores)]:
            write_csv(output/filename, records)
        save_json(output/'manifest.json', dict(plan, status=status, fixed_targets=target_records,
            diagnostic_steps=collection_steps+evaluation_steps, collection_steps=collection_steps,
            evaluation_steps=evaluation_steps, direction_optimizer_updates=sum(v['direction'] for v in counters),
            actor_optimizer_updates=sum(v['actor_fit'] for v in counters), elapsed_seconds=time.perf_counter()-started, **extra))
    persist()

    class Mixture(nn.Module):
        def __init__(self, old, candidate, alpha):
            super().__init__(); self.old, self.candidate, self.alpha = old, candidate, alpha
        def forward(self, x):
            return (1-self.alpha)*normalized(self.old(x))+self.alpha*normalized(self.candidate(x))

    try:
        for seed in args.seeds:
            r = Runner(output/f'data_seed_{seed}', c, 'no_checks', seed)
            counters.append(r.optim_steps)
            r.actor.load_state_dict(state['actor']); r.critic.load_state_dict(state['critic'])
            r.round = state['round']
            old, critic = r.actor.eval(), r.critic.eval()
            validation_sampler = None
            try:
                if packet:
                    batch, validation = (to_device(packet[key], args.device) for key in ('batch', 'validation'))
                else:
                    validation_sampler = Collector(c, 'diagnostic', seed)
                    batch = r.sampler.collect(old, c['train_episodes'], 'train')
                    validation = validation_sampler.collect(old, args.validation_episodes, 'validation')
                    for b in (batch, validation):
                        b['advantage'], b['value_target'] = r.labels(b, critic)
                    collection_steps += r.sampler.used + validation_sampler.used
                def actual_kl(model):
                    with torch.no_grad():
                        return float(r.weighted(categorical_kl(validation['mu'], model(validation['x'])), validation))
                candidates = []
                for epochs in epochs_list:
                    print(f'data seed {seed}: direction {epochs} epochs ({epochs*per_epoch} updates)', flush=True)
                    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                        torch.manual_seed(seed_for('direction', seed if args.init_seed is None else args.init_seed, r.round+1))
                        direction = Direction(r.input_dim, c).to(args.device)
                    if packet:
                        direction.load_state_dict(packet['direction'])
                    else:
                        optimizer = torch.optim.Adam(direction.parameters(), lr=c['direction_lr'])
                        def objective(mb):
                            score, fisher = direction_terms(direction(mb['x']), mb['mu'], mb['actions'])
                            return r.weighted(fisher-2*mb['advantage']*score, mb), {}
                        r.optimize(direction, optimizer, batch, epochs, 'direction', objective)
                    target = Target(old, direction, eta).eval()
                    with torch.no_grad():
                        for name, b in (('train', batch), ('validation', validation)):
                            metrics = direction_metrics(direction, b, c)
                            scores.append(dict(data_seed=seed, direction_epochs=epochs, split=name,
                                init_seed=args.init_seed, optimizer_updates=epochs*per_epoch,
                                score=metrics['B_label'], **metrics,
                                interpretation='surrogate with frozen-critic GAE labels, not true advantage'))
                            if not packet:
                                b['q'] = direction(b['x']).detach()
                                b['target'] = target(b['x']).detach()
                    if packet:
                        fixed, fixed_path, fixed_hash = packet, source, original_hash
                    else:
                        fixed_path = output/'fixed_targets'/f'data_s{seed}_direction_u{epochs*per_epoch}.pt'
                        fixed = save_fixed_target(fixed_path, r, old, critic, direction, batch, validation,
                            eta=eta, direction_epochs=epochs, fit_epochs=fit_epochs,
                            provenance=dict(provenance, initialization_seed=seed if args.init_seed is None else args.init_seed),
                            evaluation_seed=args.evaluation_seed, evaluation_episodes=args.evaluation_episodes)
                        fixed_hash = hashlib.sha256(fixed_path.read_bytes()).hexdigest()
                    target_records.append(dict(path=str(fixed_path), sha256=fixed_hash,
                        frozen_sha256=fixed['frozen_sha256'], data_seed=seed,
                        direction_updates=epochs*per_epoch, newly_prepared=packet is None))
                    frozen_models = fingerprint(dict(actor=old.state_dict(), critic=critic.state_dict(), direction=direction.state_dict()))
                    frozen_data = fingerprint(dict(batch=batch, validation=validation))
                    persist()
                    if args.prepare_only:
                        continue
                    candidates.append((f'direction{epochs}_target', target, dict(
                        direction_epochs=epochs, direction_updates=epochs*per_epoch,
                        target_sha256=fixed_hash, candidate_kind='teacher')))
                    for lr in rates:
                        torch.set_rng_state(fixed['torch_rng'].cpu())
                        if args.device == 'cuda' and fixed['cuda_rng'] is not None:
                            torch.cuda.set_rng_state_all([value.cpu() for value in fixed['cuda_rng']])
                        initial_train_kl = target_kl(r, old, batch)
                        def trace(model, epoch, updates, scope):
                            traces.append(dict(data_seed=seed, init_seed=args.init_seed, direction_epochs=epochs,
                                actor_lr=lr, epoch=epoch, optimizer_updates=updates, scope=scope,
                                train_target_kl=target_kl(r, model, batch), validation_target_kl=target_kl(r, model, validation),
                                validation_actual_kl=actual_kl(model), initial_train_target_kl=initial_train_kl,
                                tiny_target_flag=initial_train_kl<1e-6, target_sha256=fixed_hash))
                            if scope != 'post_optimizer_step':
                                persist()
                        fitted = fit_actor(r, old, batch, fit_epochs, lr=lr, selection='last',
                            orders=fixed['fit_orders'], observer=trace, trace_every_updates=args.fit_trace_every_updates)
                        info = fitted['info']
                        candidate_path = output/'fit_candidates'/f'data_s{seed}_u{epochs*per_epoch}_lr{lr:g}.pt'
                        save_pt(candidate_path, cpu_copy(dict(format='actor-fit-candidates-v1',
                            target_path=str(fixed_path), target_sha256=fixed_hash, actor_lr=lr,
                            final=fitted['final'].state_dict(), best=fitted['best'].state_dict(),
                            history=fitted['history'], info=info, fit_orders=fitted['orders'])))
                        meta = dict(direction_epochs=epochs, direction_updates=epochs*per_epoch, init_seed=args.init_seed,
                            actor_lr=lr, fit_epochs=fit_epochs, actor_optimizer_updates=info['fit_optimizer_updates'],
                            initial_train_target_kl=info['fit_initial_train_kl'],
                            final_train_target_kl=info['fit_final_train_kl'], best_epoch=info['fit_best_epoch'],
                            best_train_target_kl=info['fit_best_train_kl'], fit_seconds=info['fit_seconds'],
                            target_sha256=fixed_hash, candidate_checkpoint=str(candidate_path), candidate_kind='actor')
                        for kind, rule, epoch, loss in (
                            ('final', 'last', fit_epochs, info['fit_final_train_kl']),
                            ('best', 'best_train_kl', info['fit_best_epoch'], info['fit_best_train_kl'])):
                            if args.mode == 'direction' and kind == 'best':
                                continue
                            model = fitted[kind]
                            candidates.append((f'direction{epochs}_lr{lr}_{kind}', model, dict(meta,
                                selection_rule=rule, selected_epoch=epoch, selected_train_target_kl=loss,
                                selected_validation_target_kl=target_kl(r, model, validation),
                                selected_zero_update=epoch == 0, actor_changed=state_changed(model, old))))
                        if fingerprint(dict(actor=old.state_dict(), critic=critic.state_dict(), direction=direction.state_dict())) != frozen_models:
                            raise AssertionError('Frozen model changed during Actor fit')
                        if fingerprint(dict(batch=batch, validation=validation)) != frozen_data:
                            raise AssertionError('Fixed labels/target/data changed during Actor fit')
                if not args.prepare_only:
                    raw_kls = [actual_kl(model) for _, model, _ in candidates]
                    common_kl = max(0., min(args.target_kl, .95*min(raw_kls))) if args.mode == 'direction' else None
                    evaluation_cache = {}
                    def evaluated(name, model):
                        nonlocal evaluation_steps
                        key = fingerprint(dict(state=model.state_dict(), kind=type(model).__name__,
                                               alpha=getattr(model, 'alpha', None), eta=getattr(model, 'eta', None)))
                        if key in evaluation_cache:
                            previous, result, episodes = evaluation_cache[key]
                            return result, episodes, previous
                        result, episodes = evaluate(model, c, 'diagnostic', seed, evaluation_purpose, args.evaluation_episodes)
                        evaluation_steps += sum(e['episode_steps'] for e in episodes)
                        evaluation_cache[key] = (name, result, episodes)
                        return result, episodes, None
                    baseline, baseline_eps, _ = evaluated('old', old)
                    for i, e in enumerate(baseline_eps):
                        episode_rows.append(dict(data_seed=seed, candidate='old', episode=i, **e))
                    for (name, model, meta), raw_kl in zip(candidates, raw_kls):
                        alpha, matched_kl = 1., raw_kl
                        if args.mode == 'direction':
                            with torch.no_grad():
                                alpha, matched_kl = match_mixture(validation['mu'], model(validation['x']),
                                    lambda value:r.weighted(value, validation), common_kl)
                            model = Mixture(old, model, alpha)
                        result, eps, reused = evaluated(name, model)
                        gains = np.array([e['J']-base['J'] for e, base in zip(eps, baseline_eps)])
                        boot_rng = np.random.default_rng(seed_for('paired-bootstrap', seed, name))
                        ci = np.quantile(boot_rng.choice(gains, (5000, len(gains)), replace=True).mean(1), [.025, .975])
                        rows.append(dict(data_seed=seed, candidate=name, **meta, raw_actual_kl=raw_kl,
                            common_kl=common_kl, matched_actual_kl=matched_kl, mixture_alpha=alpha,
                            low_common_kl=common_kl is not None and common_kl<.25*args.target_kl,
                            old_return=baseline['J'], candidate_return=result['J'], return_gain=float(gains.mean()),
                            gain_scene_ci_low=float(ci[0]), gain_scene_ci_high=float(ci[1]),
                            success_rate=result['success_rate'], old_success_rate=baseline['success_rate'],
                            evaluation_reused_from=reused, diagnostic_steps_total=collection_steps+evaluation_steps))
                        for i, e in enumerate(eps):
                            episode_rows.append(dict(data_seed=seed, candidate=name, episode=i, evaluation_reused_from=reused, **e))
                        persist()
                        print(f'{seed} {name}: gain={gains.mean():.5f}, KL={matched_kl:.6g}', flush=True)
            finally:
                r.sampler.close()
                if validation_sampler is not None:
                    validation_sampler.close()
            if fingerprint(old.state_dict()) != fingerprint(state['actor']) or fingerprint(critic.state_dict()) != fingerprint(state['critic']):
                raise AssertionError('Frozen source Actor/Critic changed')
        if hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
            raise AssertionError('Input checkpoint/snapshot changed')
        persist('completed', input_unchanged=True, source_checkpoint_unchanged=True if not packet else None,
                source_actor_critic_frozen=True)
    except Exception as exc:
        persist('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return rows


def main(argv=None):
    args = parser().parse_args(argv)
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
