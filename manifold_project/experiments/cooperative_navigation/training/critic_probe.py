"""Fixed-data Critic scale ablation, separate from policy training.

Raw and fixed-normalized fits share initial physical predictions, episode
orders, fresh Adam, clipping and update counts. A fresh/checkpoint start axis
separates relearning from rescuing an already saturated network. Targets always
come from the frozen source Critic; normalization never uses validation data.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ...common.reporting import write_csv
from ..configs import episode_horizon
from ..models import Actor, Critic
from .collector import Collector
from .diagnostics import cpu_copy, critic_metrics
from .fixed_target import fingerprint, model_fingerprint, to_device
from .ppo import batch_weighted, value_targets
from .storage import event, load_pt, save_json, save_pt, seed_for

FORMAT = 'navigation-fixed-critic-scale-v1'
NORMALIZATIONS = ('raw', 'fixed_normalized')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checksum(packet):
    return fingerprint({k: v for k, v in packet.items() if k != 'frozen_sha256'})


def probe_semantics_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def validate_packet(packet):
    if packet.get('format') != FORMAT or packet.get('frozen_sha256') != checksum(packet):
        raise ValueError('Invalid or modified fixed Critic data packet')
    if packet['model_code_sha256'] != model_fingerprint():
        raise ValueError('Model/label/weighting code changed since collection')
    if packet['probe_semantics_sha256'] != probe_semantics_hash():
        raise ValueError('Critic probe code changed; replay requires the original fitting semantics')
    count = len(packet['batch']['reward'])
    orders = packet['orders']
    if len(orders) != packet['epochs'] or any(sorted(o) != list(range(count)) for o in orders):
        raise ValueError('Invalid frozen Critic minibatch schedule')


def normalization(batch, config):
    y = batch['value_target'].double()
    weight = lambda v: batch_weighted(v, batch, config)
    mass = weight(torch.ones_like(y))
    if mass <= 0:
        raise ValueError('Critic training batch has no valid weight')
    mean = weight(y)/mass
    variance = weight((y-mean).square())/mass
    # The floor is fixed and recorded, not tuned using validation data.
    scale = variance.sqrt().clamp_min(1e-3)
    if not torch.isfinite(mean) or not torch.isfinite(scale):
        raise FloatingPointError('Nonfinite train-only value normalization statistics')
    return dict(mean=float(mean), scale=float(scale), variance=float(variance), std_floor=1e-3)


class ScaledCritic(nn.Module):
    """Optimize normalized outputs; expose physical values for all diagnostics."""
    def __init__(self, source, mean=0., scale=1.):
        super().__init__()
        self.base = deepcopy(source).requires_grad_(True)
        self.register_buffer('mean', torch.tensor(mean, dtype=next(source.parameters()).dtype,
                                                 device=next(source.parameters()).device))
        self.register_buffer('scale', torch.tensor(scale, dtype=self.mean.dtype, device=self.mean.device))
        with torch.no_grad():
            head = self.base.net[-1]
            head.weight.div_(self.scale)
            head.bias.sub_(self.mean).div_(self.scale)

    def forward(self, x):
        return self.base(x)*self.scale+self.mean


@torch.no_grad()
def metrics(model, batch, config, constant_mean):
    result = critic_metrics(model, batch, config)
    state = batch['state']
    h1 = model.base.net[1](model.base.net[0](state))
    h2 = model.base.net[3](model.base.net[2](h1))
    h2 = h2[batch['valid']].double()
    result.update(second_tanh_saturation=float((h2.abs() > .95).double().mean()),
                  second_tanh_derivative_lt_1e_minus_6=float((1-h2.square() < 1e-6).double().mean()),
                  constant_train_mean_mse=float(batch_weighted(
                      (batch['value_target'].double()-constant_mean).square(), batch, config)))
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f'Nonfinite Critic metric: {key}')
    return result


def fit_variant(start, batch, validation, config, orders, norm, variant, observer, on_update=None):
    normalized = variant == 'fixed_normalized'
    model = ScaledCritic(start, norm['mean'] if normalized else 0., norm['scale'] if normalized else 1.)
    # Normalization changes parameterization, not initial raw value predictions.
    with torch.no_grad():
        errors = [float((model(b['state'])-start(b['state'])).abs().max()) for b in (batch, validation)]
        for b in (batch, validation):
            if not torch.allclose(model(b['state']), start(b['state']), atol=5e-5, rtol=2e-6):
                raise AssertionError('Scale transform changed initial physical predictions')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['critic_lr'])
    updates = 0
    observer(model, 0, updates, dict(initial_raw_value_max_abs_error=max(errors)))
    for epoch, order in enumerate(orders, 1):
        norms, clipped = [], 0
        for offset in range(0, len(order), config['minibatch_episodes']):
            indices = order[offset:offset+config['minibatch_episodes']]
            mb = {k: v[indices] for k, v in batch.items()}
            prediction = model.base(mb['state'])
            target = (mb['value_target']-model.mean)/model.scale
            loss = config['value_coef']*batch_weighted((prediction-target).square(), mb, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Critic fitting loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'], error_if_nonfinite=True))
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite Critic parameters')
            norms.append(grad)
            clipped += grad > config['grad_norm']
            updates += 1
            if on_update is not None:
                on_update(updates)
        observer(model, epoch, updates, dict(grad_norm_before=float(np.mean(norms)),
                                            grad_clip_fraction=clipped/len(norms)))
    return model, updates


def compact_batch(batch, critic, config):
    values, target, advantage = value_targets(batch, critic, config)
    keys = ('state', 'reward', 'valid', 'lengths', 'terminated', 'truncated', 'final_state')
    return cpu_copy(dict({k: batch[k] for k in keys}, value_target=target,
                         frozen_old_values=values, frozen_gae=advantage))


def run_critic_probe(args):
    started = time.perf_counter()
    if args.fixed_target is not None:
        raise ValueError('Critic mode needs --checkpoint or --critic-data, not an Actor target')
    forbidden = ('evaluation_episodes', 'evaluation_seed', 'direction_epochs', 'direction_updates',
                 'sample_sizes', 'init_seeds', 'fit_trace_every_updates', 'fit_epochs', 'actor_lrs', 'target_kl', 'eta')
    if any(getattr(args, k) is not None for k in forbidden):
        raise ValueError('Critic mode does not accept direction, Actor fitting, or execution evaluation options')
    source = (args.critic_data or args.checkpoint).resolve(strict=True)
    output = args.output.resolve()
    source_root = source.parent if args.critic_data else source.parent.parent
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError('Use a separate Critic diagnostic output directory')
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    original_hash = file_hash(source)
    state = load_pt(source)
    packet = state if args.critic_data else None
    if packet:
        validate_packet(packet)
        if args.prepare_only or any(getattr(args, k) is not None for k in
                                   ('seeds', 'batch_episodes', 'validation_episodes', 'init_seed', 'critic_epochs', 'critic_starts')):
            raise ValueError('Critic replay inherits data, starts, normalization, and the entire optimization budget')
        c = deepcopy(packet['config'])
        data_seed, init_seed = packet['data_seed'], packet['init_seed']
        count, validation_count = len(packet['batch']['reward']), len(packet['validation']['reward'])
        epochs, starts = packet['epochs'], list(packet['starts'])
    else:
        c = deepcopy(state['config'])
        if (state.get('current_method') != 'AN' or c.get('environment') != 'navigation'
                or c.get('task_mode') != 'success_only' or not c.get('complete_episodes')):
            raise ValueError('Critic probe requires an AN episodic success-only navigation checkpoint')
        seeds = args.seeds if args.seeds is not None else [45]
        if len(seeds) != 1 or seeds[0] < 0:
            raise ValueError('Critic probe uses exactly one nonnegative data seed; no seed sweep')
        data_seed = seeds[0]
        init_seed = args.init_seed if args.init_seed is not None else state['seed']
        count = args.batch_episodes if args.batch_episodes is not None else 32
        validation_count = args.validation_episodes if args.validation_episodes is not None else 32
        epochs = args.critic_epochs if args.critic_epochs is not None else 80
        starts = args.critic_starts if args.critic_starts is not None else ['fresh', 'checkpoint']
    if min(count, validation_count, epochs) <= 0 or init_seed < 0 or len(set(starts)) != len(starts):
        raise ValueError('Positive counts/epochs, nonnegative init seed and distinct starts are required')
    cap = 0 if packet else (count+validation_count)*episode_horizon(c)
    c.update(device=args.device, budget=max(1, cap))
    updates_per_arm = epochs*math.ceil(count/c['minibatch_episodes'])
    plan = dict(mode='critic', semantics_version=FORMAT, input_path=str(source), input_sha256=original_hash,
                scope='fixed packet replay' if packet else 'new frozen-policy data; not historical replay',
                data_seed=data_seed, init_seed=init_seed, starts=starts, normalizations=list(NORMALIZATIONS),
                train_episodes=count, validation_episodes=validation_count, epochs=epochs,
                optimizer='fresh Adam in every arm; source moments not resumed',
                optimizer_updates_per_arm=updates_per_arm, critic_lr=c['critic_lr'],
                maximum_collection_steps=cap, device=args.device,
                actor_optimizer_updates=0, direction_optimizer_updates=0, evaluation_steps=0,
                target_rule='raw returns with frozen source Critic bootstrap; never recomputed during fitting',
                normalization_rule='fixed discounted train-target mean/std; no validation statistics; output-preserving head transform',
                primary_endpoint='final epoch raw-unit validation MSE/EV, paired by start; no best-epoch selection',
                mechanism_screen='validation MSE at least 10% lower and EV at least 0.05 higher than raw at the same start; descriptive, not significance',
                limitation='Frozen bootstrap targets are not true values; this does not establish return improvement',
                numpy_version=np.__version__, torch_version=str(torch.__version__),
                probe_code_sha256=file_hash(Path(__file__)), model_code_sha256=model_fingerprint())
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return plan
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    trace, results, episode_rows = [], [], []
    collector = None
    completed_updates = 0
    current_updates = 0
    packet_path = source if packet else output/'critic_data.pt'

    def persist(status='running', **extra):
        write_csv(output/'critic_trace.csv', trace)
        write_csv(output/'diagnostics.csv', results)
        write_csv(output/'episode_errors.csv', episode_rows)
        save_json(output/'manifest.json', dict(plan, status=status,
                  collection_steps=collector.used if collector else 0,
                  diagnostic_steps=collector.used if collector else 0,
                  critic_optimizer_updates=completed_updates+current_updates,
                  critic_data_path=str(packet_path), elapsed_seconds=time.perf_counter()-started, **extra))

    persist()
    try:
        if packet is None:
            with torch.random.fork_rng(devices=[]):
                old = Actor(state['input_dim'], c)
                label_critic = Critic(state['critic']['net.0.weight'].shape[1], c)
            old.load_state_dict(state['actor'])
            label_critic.load_state_dict(state['critic'])
            old = old.to(args.device).eval().requires_grad_(False)
            label_critic = label_critic.to(args.device).eval().requires_grad_(False)
            frozen_hash = fingerprint(dict(actor=old.state_dict(), critic=label_critic.state_dict()))
            def log(stream, **record):
                event(output/'collection_ledger.jsonl', stream, **record)
            collector = Collector(c, 'critic_scale_probe', data_seed, log)
            batch = compact_batch(collector.collect(old, count, 'train'), label_critic, c)
            persist()
            validation = compact_batch(collector.collect(old, validation_count, 'validation'), label_critic, c)
            if fingerprint(dict(actor=old.state_dict(), critic=label_critic.state_dict())) != frozen_hash:
                raise AssertionError('Source Actor/Critic changed during collection')
            if collector.used > cap:
                raise AssertionError('Collection exceeded its declared cap')
            initial = {}
            if 'checkpoint' in starts:
                initial['checkpoint'] = cpu_copy(label_critic.state_dict())
            if 'fresh' in starts:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed_for('critic-scale-fresh', init_seed))
                    fresh = Critic(batch['state'].shape[-1], c)
                initial['fresh'] = cpu_copy(fresh.state_dict())
            rng = np.random.default_rng(seed_for('critic-scale-orders', data_seed))
            packet = dict(format=FORMAT, config=deepcopy(c), data_seed=data_seed, init_seed=init_seed,
                          epochs=epochs, starts=starts, initial=initial, batch=batch, validation=validation,
                          orders=[rng.permutation(count).tolist() for _ in range(epochs)],
                          normalization=normalization(batch, c), model_code_sha256=model_fingerprint(),
                          probe_semantics_sha256=probe_semantics_hash(),
                          label_critic=cpu_copy(label_critic.state_dict()),
                          source=dict(path=str(source), sha256=original_hash, round=state['round'],
                                      actor_sha256=fingerprint(state['actor']), critic_sha256=fingerprint(state['critic'])),
                          original_collection_steps=collector.used, optimizer_policy=plan['optimizer'])
            packet['frozen_sha256'] = checksum(packet)
            save_pt(packet_path, packet)
            del old, label_critic, batch, validation
        validate_packet(packet)
        plan.update(frozen_data_sha256=packet['frozen_sha256'], critic_data_sha256=file_hash(packet_path),
                    normalization=packet['normalization'], source_provenance=packet['source'],
                    original_collection_steps=packet['original_collection_steps'])
        save_json(output/'effective_critic_config.json', c)
        if not args.prepare_only:
            batch, validation = (to_device(packet[k], args.device) for k in ('batch', 'validation'))
            data_hash = fingerprint(dict(batch=batch, validation=validation))
            for start_name in starts:
                with torch.random.fork_rng(devices=[]):
                    start = Critic(batch['state'].shape[-1], c)
                start.load_state_dict(packet['initial'][start_name])
                start = start.to(args.device).eval().requires_grad_(False)
                start_hash = fingerprint(start.state_dict())
                for variant in NORMALIZATIONS:
                    name = f'{start_name}_{variant}'
                    print(f'Critic {name}: {updates_per_arm} fixed updates', flush=True)
                    current_updates = 0
                    def observe(model, epoch, updates, optimization):
                        nonlocal current_updates
                        current_updates = updates
                        row = dict(candidate=name, start=start_name, normalization=variant,
                                   epoch=epoch, optimizer_updates=updates, **optimization)
                        for split, b in (('train', batch), ('validation', validation)):
                            row.update({f'{split}_{k}': v for k, v in metrics(model, b, c, packet['normalization']['mean']).items()})
                        trace.append(row)
                        persist()
                    def count_update(updates):
                        nonlocal current_updates
                        current_updates = updates
                    model, updates = fit_variant(start, batch, validation, c, packet['orders'],
                                                 packet['normalization'], variant, observe, count_update)
                    if updates != updates_per_arm or fingerprint(start.state_dict()) != start_hash:
                        raise AssertionError('Budget or frozen start changed')
                    results.append(dict(trace[-1], initial_state_sha256=start_hash))
                    with torch.no_grad():
                        for split, b in (('train', batch), ('validation', validation)):
                            for i in range(len(b['reward'])):
                                sub = {k: v[i:i+1] for k, v in b.items()}
                                mse = batch_weighted((model(sub['state']).double()-sub['value_target'].double()).square(), sub, c)
                                episode_rows.append(dict(candidate=name, split=split, episode=i, raw_mse=float(mse)))
                    save_pt(output/'candidates'/f'{name}.pt', cpu_copy(dict(
                        format=FORMAT, state_dict=model.state_dict(), config=c, start=start_name,
                        normalization=variant, frozen_data_sha256=packet['frozen_sha256'], final_epoch=epochs)))
                    completed_updates += updates
                    current_updates = 0
                    persist()
            if fingerprint(dict(batch=batch, validation=validation)) != data_hash:
                raise AssertionError('Frozen fitting tensors changed')
        if checksum(packet) != packet['frozen_sha256'] or file_hash(source) != original_hash:
            raise AssertionError('Input or frozen packet changed')
        comparisons = []
        for start_name in starts if results else []:
            raw, normalized = [next(r for r in results if r['candidate'] == f'{start_name}_{v}') for v in NORMALIZATIONS]
            raw_mse, norm_mse = raw['validation_mse'], normalized['validation_mse']
            ev0, ev1 = raw['validation_explained_variance'], normalized['validation_explained_variance']
            gain = 1-norm_mse/raw_mse if raw_mse > 1e-12 else None
            comparisons.append(dict(start=start_name, validation_mse_relative_reduction=gain,
                                    validation_ev_gain=ev1-ev0 if ev0 is not None and ev1 is not None else None,
                                    descriptive_screen_passed=(gain is not None and gain >= .1 and
                                                              ev0 is not None and ev1 is not None and ev1-ev0 >= .05)))
        save_json(output/'summary.json', dict(comparisons=comparisons, final_candidates=results,
                  interpretation='Fixed-target fitting only; no policy-return or true-value claim; do not pool fresh and checkpoint effects'))
        persist('prepared' if args.prepare_only else 'completed', input_unchanged=True,
                frozen_data_unchanged=True, source_actor_critic_frozen=True)
    except Exception as exc:
        persist('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if collector is not None:
            collector.close()
    return results
