"""Frozen-checkpoint probes: matched-KL directions and fixed-target Actor fitting.

New diagnostic samples are collected; this does not replay historical batches.
The source checkpoint and the training algorithm are never modified.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['direction', 'fit'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[40, 41, 42], help='Diagnostic data/initialization seeds, not training seeds')
    p.add_argument('--batch-episodes', type=int, help='Defaults to the saved training batch size')
    p.add_argument('--validation-episodes', type=int, default=32)
    p.add_argument('--evaluation-episodes', type=int, default=40)
    p.add_argument('--direction-epochs', nargs='+', type=int, default=[16, 64], help='Direction mode only; fit mode uses saved direction epochs')
    p.add_argument('--sample-sizes', nargs='+', type=int, help='Expand an N x U matrix; same data seed gives nested batches')
    p.add_argument('--direction-updates', nargs='+', type=int, help='Exact optimizer updates; must be divisible by minibatches per epoch')
    p.add_argument('--init-seeds', nargs='+', type=int, help='Separate direction initialization seed axis')
    p.add_argument('--init-seed', type=int, help='One initialization seed, independent from --seeds data seeds')
    p.add_argument('--fit-trace-every-updates', type=int, default=0, help='Also recompute full train/validation KL every N optimizer steps; 0 means epoch only')
    p.add_argument('--fit-epochs', type=int, help='Defaults to saved Actor epochs')
    p.add_argument('--actor-lrs', nargs='+', type=float, default=[3e-4, 1e-4, 3e-5])
    p.add_argument('--target-kl', type=float, default=1e-4)
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
    from ..common.runtime import configure_runtime
    configure_runtime(args.allow_duplicate_openmp)
    import torch
    from torch import nn
    from .training.runner import Runner
    from .training.collector import Collector
    from .training.ppo import direction_terms
    from .training.diagnostics import categorical_kl, direction_metrics
    from .training.storage import load_pt, save_json, seed_for
    from .models import Direction
    from .evaluation.evaluate import evaluate
    from ..common.reporting import write_csv

    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be distinct and nonnegative')
    if args.fit_trace_every_updates<0 or (args.init_seed is not None and args.init_seed<0):
        raise ValueError('Trace interval and initialization seed must be nonnegative')
    positive = [args.validation_episodes, args.evaluation_episodes, args.target_kl,
                *args.direction_epochs, *args.actor_lrs]
    positive += [v for v in [args.batch_episodes, args.fit_epochs, args.eta] if v is not None]
    if any(not np.isfinite(v) or v <= 0 for v in positive):
        raise ValueError('Counts, rates, eta and KL must be finite and positive')
    source = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    run_dir = source.parent.parent
    if output == run_dir or run_dir in output.parents or output in run_dir.parents:
        raise ValueError('Diagnostic output must be separate from the source run')
    if output.exists():
        raise FileExistsError(f'Use a new diagnostic output directory: {output}')
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    state = load_pt(source)
    c = deepcopy(state['config'])
    if (state.get('current_method') != 'AN' or c.get('environment') != 'navigation'
            or c.get('task_mode') != 'success_only' or not c.get('complete_episodes')):
        raise ValueError('This probe requires an AN episodic success-only navigation checkpoint')
    c.update(device=args.device, budget=10**12)
    # Matrix expansion keeps each N/init cell independently reviewable.
    if args.sample_sizes or args.direction_updates or args.init_seeds:
        if args.mode!='direction':
            raise ValueError('N x U expansion is supported in direction mode only')
        sizes=args.sample_sizes or [args.batch_episodes or c['train_episodes']]
        inits=args.init_seeds or [args.init_seed]
        if any(n<=0 for n in sizes) or any(s is not None and s<0 for s in inits):
            raise ValueError('Sample sizes must be positive and initialization seeds nonnegative')
        if len(set(sizes))!=len(sizes) or len(set(inits))!=len(inits):
            raise ValueError('Matrix axes must not contain duplicates')
        jobs=[]
        for n in sizes:
            for initialization in inits:
                child=deepcopy(args)
                child.sample_sizes=child.direction_updates=child.init_seeds=None
                child.batch_episodes=n;child.init_seed=initialization
                child.output=output/f'n{n}_init{initialization if initialization is not None else "data"}'
                if args.direction_updates:
                    per_epoch=int(np.ceil(n/c['minibatch_episodes']))
                    if any(u<1 or u%per_epoch for u in args.direction_updates):
                        raise ValueError('U must be positive and divisible by ceil(N / saved minibatch_episodes)')
                    child.direction_epochs=[u//per_epoch for u in args.direction_updates]
                jobs.append(child)
        matrix=dict(checkpoint=str(source),sample_sizes=sizes,init_seeds=inits,
                    requested_updates=args.direction_updates,data_seeds=args.seeds)
        if not args.dry_run:
            output.mkdir(parents=True)
            save_json(output/'matrix_manifest.json',dict(matrix,status='running'))
        for child in jobs:
            run(child)
        if not args.dry_run:
            save_json(output/'matrix_manifest.json',dict(matrix,status='completed'))
        return matrix
    if args.batch_episodes is not None:
        c['train_episodes'] = args.batch_episodes
    fit_epochs = args.fit_epochs or c['actor_epochs']
    eta = args.eta if args.eta is not None else c.get('step_sizes', [c['eta']])[0]
    plan = dict(checkpoint=str(source), checkpoint_sha256=original_hash,
        source_round=state['round'], source_training_seed=state['seed'],
        mode=args.mode, diagnostic_seeds=args.seeds, batch_episodes=c['train_episodes'],
        initialization_seed=args.init_seed,fit_trace_every_updates=args.fit_trace_every_updates,
        validation_episodes=args.validation_episodes, evaluation_episodes=args.evaluation_episodes,
        direction_epochs=args.direction_epochs if args.mode=='direction' else [c['direction_epochs']],
        fit_epochs=fit_epochs, actor_lrs=[c['actor_lr']] if args.mode=='direction' else args.actor_lrs,
        eta=eta, requested_kl=args.target_kl, device=args.device,
        kl_measure='discount-weighted mean KL(old || candidate) on independent old-policy validation states',
        match_rule=('common KL = min(requested, 0.95 * minimum raw candidate KL); probability mixture with old policy'
                    if args.mode=='direction' else 'none; fit mode reports unmodified candidate policies'),
        best_fit_rule='minimum training target KL, includes epoch 0; evaluation never selects candidates',
        scope='new samples at frozen saved Actor/Critic; not historical batch replay',
        probe_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return plan
    output.mkdir(parents=True)
    save_json(output/'manifest.json', dict(plan, status='running'))
    save_json(output/'effective_runner_config.json', c)
    rows, traces, episode_rows, scores = [], [], [], []
    total_steps = 0

    class Target(nn.Module):
        def __init__(self, old, direction):
            super().__init__(); self.old, self.direction = old, direction
        def forward(self, x):
            return torch.softmax(self.old(x).log()+eta*self.direction(x), -1)

    class Mixture(nn.Module):
        def __init__(self, old, candidate, alpha):
            super().__init__(); self.old, self.candidate, self.alpha = old, candidate, alpha
        def forward(self, x):
            return (1-self.alpha)*normalized(self.old(x))+self.alpha*normalized(self.candidate(x))

    def persist():
        for filename, records in [('diagnostics.csv', rows), ('fit_trace.csv', traces),
                                  ('evaluation_episodes.csv', episode_rows), ('direction_scores.csv', scores)]:
            write_csv(output/filename, records)

    for seed in args.seeds:
        r = Runner(output/f'data_seed_{seed}', c, 'no_checks', seed)
        r.actor.load_state_dict(state['actor']); r.critic.load_state_dict(state['critic'])
        r.round = state['round']
        old, critic = r.actor.eval(), r.critic.eval()
        validation_sampler = Collector(c, 'diagnostic', seed)
        try:
            batch = r.sampler.collect(old, c['train_episodes'], 'train')
            validation = validation_sampler.collect(old, args.validation_episodes, 'validation')
            for b in [batch, validation]:
                b['advantage'], b['value_target'] = r.labels(b, critic)
            total_steps += r.sampler.used + validation_sampler.used
            def fit_kl(model, b):
                with torch.no_grad():
                    p = model(b['x'])
                    return float(r.weighted(categorical_kl(b['target'],p), b))
            def actual_kl(model):
                with torch.no_grad():
                    p, mu = normalized(model(validation['x'])), normalized(validation['mu'])
                    return max(0.,float(r.weighted((mu*(mu.log()-p.log())).sum(-1), validation)))
            candidates = []
            epochs_list = args.direction_epochs if args.mode=='direction' else [c['direction_epochs']]
            for epochs in epochs_list:
                print(f'data seed {seed}: direction {epochs} epochs', flush=True)
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    torch.manual_seed(seed_for('direction',seed if args.init_seed is None else args.init_seed,r.round+1))
                    direction = Direction(r.input_dim, c).to(args.device)
                optimizer = torch.optim.Adam(direction.parameters(), lr=c['direction_lr'])
                def objective(mb):
                    score, fisher = direction_terms(direction(mb['x']), mb['mu'], mb['actions'])
                    return r.weighted(fisher-2*mb['advantage']*score, mb), {}
                r.optimize(direction, optimizer, batch, epochs, 'direction', objective)
                target = Target(old, direction).eval()
                with torch.no_grad():
                    for name, b in [('train', batch), ('validation', validation)]:
                        metrics=direction_metrics(direction,b,c)
                        scores.append(dict(data_seed=seed, direction_epochs=epochs, split=name,
                            init_seed=args.init_seed,optimizer_updates=epochs*int(np.ceil(len(batch['reward'])/c['minibatch_episodes'])),
                            score=metrics['B_label'],**metrics,
                            interpretation='surrogate with frozen-critic GAE labels, not true advantage'))
                        b['target'] = target(b['x']).detach()
                if args.mode=='direction':
                    candidates.append((f'direction{epochs}_target', target, dict(direction_epochs=epochs)))
                rates = [c['actor_lr']] if args.mode=='direction' else args.actor_lrs
                for lr in rates:
                    candidate = deepcopy(old)
                    optimizer = torch.optim.Adam(candidate.parameters(), lr=lr)
                    rng = np.random.default_rng(seed_for(seed, r.round+1, 'actor_fit', 'minibatches'))
                    best = deepcopy(candidate); best_epoch = 0; best_kl = fit_kl(candidate, batch)
                    initial_kl = best_kl
                    optimizer_updates=0
                    def trace(epoch,scope):
                        traces.append(dict(data_seed=seed,init_seed=args.init_seed,direction_epochs=epochs,
                            actor_lr=lr,epoch=epoch,optimizer_updates=optimizer_updates,scope=scope,
                            train_target_kl=fit_kl(candidate,batch),validation_target_kl=fit_kl(candidate,validation),
                            validation_actual_kl=actual_kl(candidate),initial_train_target_kl=initial_kl,
                            tiny_target_flag=initial_kl<1e-6))
                    for epoch in range(fit_epochs+1):
                        if epoch:
                            order = rng.permutation(len(batch['reward']))
                            for start in range(0,len(order),c['minibatch_episodes']):
                                idx = order[start:start+c['minibatch_episodes']]
                                mb = {k:v[idx] for k,v in batch.items()}
                                optimizer.zero_grad(set_to_none=True)
                                p = candidate(mb['x'])
                                loss = r.weighted((mb['target']*(mb['target'].log()-p.log())).sum(-1),mb)
                                if not torch.isfinite(loss):
                                    raise FloatingPointError('Nonfinite fit loss')
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(candidate.parameters(),c['grad_norm'],error_if_nonfinite=True)
                                optimizer.step()
                                optimizer_updates+=1
                                if args.fit_trace_every_updates and optimizer_updates%args.fit_trace_every_updates==0:
                                    trace(epoch,'post_optimizer_step')
                        train_kl = fit_kl(candidate,batch)
                        if train_kl < best_kl:
                            best, best_kl, best_epoch = deepcopy(candidate), train_kl, epoch
                        trace(epoch,'post_epoch' if epoch else 'initial')
                        persist()
                    meta = dict(direction_epochs=epochs,init_seed=args.init_seed,actor_lr=lr,fit_epochs=fit_epochs,
                                actor_optimizer_updates=optimizer_updates,
                                initial_train_target_kl=initial_kl,final_train_target_kl=train_kl,best_epoch=best_epoch)
                    candidates.append((f'direction{epochs}_lr{lr}_final',candidate,meta))
                    if args.mode=='fit':
                        candidates.append((f'direction{epochs}_lr{lr}_best',best,dict(meta,selected_train_target_kl=best_kl)))
            raw_kls = [actual_kl(model) for _,model,_ in candidates]
            common_kl = max(0., min(args.target_kl, .95*min(raw_kls))) if args.mode=='direction' else None
            baseline, baseline_eps = evaluate(old,c,'diagnostic',seed,'frozen-checkpoint-probe',args.evaluation_episodes)
            total_steps += sum(e['episode_steps'] for e in baseline_eps)
            for i,e in enumerate(baseline_eps):
                episode_rows.append(dict(data_seed=seed,candidate='old',episode=i,**e))
            for (name,model,meta), raw_kl in zip(candidates,raw_kls):
                alpha, matched_kl = 1., raw_kl
                if args.mode=='direction':
                    with torch.no_grad():
                        alpha, matched_kl = match_mixture(validation['mu'],model(validation['x']),
                            lambda value:r.weighted(value,validation),common_kl)
                    model = Mixture(old,model,alpha)
                result, eps = evaluate(model,c,'diagnostic',seed,'frozen-checkpoint-probe',args.evaluation_episodes)
                total_steps += sum(e['episode_steps'] for e in eps)
                gains = np.array([e['J']-base['J'] for e,base in zip(eps,baseline_eps)])
                boot_rng = np.random.default_rng(seed_for('paired-bootstrap',seed,name))
                ci = np.quantile(boot_rng.choice(gains,(5000,len(gains)),replace=True).mean(1),[.025,.975])
                rows.append(dict(data_seed=seed,candidate=name,**meta,raw_actual_kl=raw_kl,
                    common_kl=common_kl,matched_actual_kl=matched_kl,mixture_alpha=alpha,
                    low_common_kl=common_kl is not None and common_kl<.25*args.target_kl,
                    old_return=baseline['J'],candidate_return=result['J'],return_gain=float(gains.mean()),
                    gain_scene_ci_low=float(ci[0]),gain_scene_ci_high=float(ci[1]),
                    success_rate=result['success_rate'],old_success_rate=baseline['success_rate'],
                    diagnostic_steps_total=total_steps))
                for i,e in enumerate(eps):
                    episode_rows.append(dict(data_seed=seed,candidate=name,episode=i,**e))
                persist()
                print(f'{seed} {name}: gain={gains.mean():.5f}, KL={matched_kl:.6g}', flush=True)
        finally:
            r.sampler.close(); validation_sampler.close()
        if any(not torch.equal(v.cpu(),state['actor'][k].cpu()) for k,v in old.state_dict().items()):
            raise AssertionError('Frozen source Actor changed')
        if any(not torch.equal(v.cpu(),state['critic'][k].cpu()) for k,v in critic.state_dict().items()):
            raise AssertionError('Frozen source Critic changed')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=original_hash:
        raise AssertionError('Source checkpoint changed')
    save_json(output/'manifest.json',dict(plan,status='completed',diagnostic_steps=total_steps,
        source_checkpoint_unchanged=True,source_actor_critic_frozen=True))
    return rows


def main(argv=None):
    args = parser().parse_args(argv)
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
