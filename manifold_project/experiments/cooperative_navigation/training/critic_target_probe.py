"""One lambda-return fit against the completed long-return reference.

Reuse the source packet's fresh start, long-target scaling and complete orders.
Evaluate both fits on the SAME saved long-return targets; own lambda-target
errors are auxiliary only. No Actor, Collector, or new policy execution.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch

from ...common.reporting import write_csv
from ..models import Critic
from .critic_probe import ScaledCritic, file_hash, validate_packet, fit_variant, metrics
from .diagnostics import cpu_copy
from .fixed_target import fingerprint, to_device
from .ppo import batch_weighted
from .storage import load_pt, save_json, save_pt

SEMANTICS = 'fixed-critic-long-vs-lambda-v1'
REFERENCE = 'fresh_fixed_normalized'


def run_critic_target_probe(args):
    started = time.perf_counter()
    if args.critic_data is None or args.checkpoint is not None or args.fixed_target is not None:
        raise ValueError('critic-target requires --critic-data from the completed scale probe')
    forbidden = ('critic_epochs', 'critic_starts', 'seeds', 'batch_episodes', 'validation_episodes',
                 'init_seed', 'init_seeds', 'sample_sizes', 'evaluation_episodes', 'evaluation_seed',
                 'direction_epochs', 'direction_updates', 'fit_trace_every_updates', 'fit_epochs',
                 'actor_lrs', 'target_kl', 'eta')
    if args.prepare_only or any(getattr(args, k) is not None for k in forbidden):
        raise ValueError('critic-target inherits all data, initialization, scaling and fitting budgets; overrides are forbidden')
    source = args.critic_data.resolve(strict=True)
    output = args.output.resolve()
    reference_root = source.parent
    if output == reference_root or reference_root in output.parents or output in reference_root.parents:
        raise ValueError('Use an output directory separate from the source scale probe')
    if output.exists():
        raise FileExistsError(f'Use a new diagnostic output directory: {output}')
    packet = load_pt(source)
    validate_packet(packet)
    if 'fresh' not in packet['initial']:
        raise ValueError('Source packet must include the saved fresh Critic start')
    reference_path = reference_root/'candidates'/f'{REFERENCE}.pt'
    manifest_path = reference_root/'manifest.json'
    diagnostics_path = reference_root/'diagnostics.csv'
    import csv
    with diagnostics_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    reference_row = next(r for r in rows if r['candidate'] == REFERENCE)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    saved_reference = load_pt(reference_path)
    hashes = {str(p): file_hash(p) for p in (source, reference_path, manifest_path, diagnostics_path)}
    norm = packet['normalization']
    epochs = packet['epochs']
    config = deepcopy(packet['config'])
    config.update(device=args.device)
    expected_updates = epochs*math.ceil(len(packet['batch']['reward'])/config['minibatch_episodes'])
    if (manifest['status'] != 'completed' or manifest['critic_data_sha256'] != file_hash(source)
            or manifest['frozen_data_sha256'] != packet['frozen_sha256']
            or saved_reference['frozen_data_sha256'] != packet['frozen_sha256']
            or saved_reference['final_epoch'] != epochs or int(reference_row['optimizer_updates']) != expected_updates
            or reference_row['initial_state_sha256'] != fingerprint(packet['initial']['fresh'])
            or manifest['normalization'] != norm or manifest['critic_lr'] != config['critic_lr']):
        raise ValueError('Completed long-return reference does not match the saved data/start/scale/budget')
    # Identity of scale buffers is part of the intervention, not recomputed per target.
    for key, value in [('mean', norm['mean']), ('scale', norm['scale'])]:
        expected = torch.tensor(value, dtype=saved_reference['state_dict'][key].dtype)
        if not torch.equal(saved_reference['state_dict'][key].cpu(), expected):
            raise ValueError('Reference uses different normalization buffers')
    plan = dict(mode='critic-target', semantics_version=SEMANTICS, critic_data_path=str(source),
                source_files_sha256=hashes, frozen_data_sha256=packet['frozen_sha256'],
                reused_reference=REFERENCE, diagnostic_data_seed=packet['data_seed'], fresh_init_seed=packet['init_seed'],
                fitting_epochs=epochs, optimizer_updates=expected_updates, critic_lr=config['critic_lr'],
                gae_lambda=config['gae_lambda'], gamma=config['gamma'], normalization=norm,
                initial_state_sha256=fingerprint(packet['initial']['fresh']),
                scaling_rule='use original long-target train mean/std for BOTH targets',
                lambda_target_rule='saved frozen_old_values + saved frozen_gae; no new bootstrap/labels',
                primary_target='same saved long-return validation targets for reference and lambda candidate',
                endpoint='final saved epoch; no validation-based candidate selection',
                screen='lambda common-long-validation MSE <= 0.9 * train-mean-constant MSE and EV > 0.05',
                scope='exploratory mechanism diagnostic on previously inspected validation episodes',
                limitation='bootstrapped long returns are not true values; no policy-performance claim',
                original_collection_steps=packet['original_collection_steps'], collection_steps=0,
                evaluation_steps=0, actor_optimizer_updates=0, direction_optimizer_updates=0,
                probe_code_sha256=file_hash(Path(__file__)), device=args.device)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return plan
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    trace, endpoints, episode_rows = [], [], []
    updates_done = 0

    def persist(status='running', **extra):
        write_csv(output/'critic_trace.csv', trace)
        write_csv(output/'diagnostics.csv', endpoints)
        write_csv(output/'episode_errors.csv', episode_rows)
        save_json(output/'manifest.json', dict(plan, status=status, critic_optimizer_updates=updates_done,
                  elapsed_seconds=time.perf_counter()-started, **extra))

    persist()
    try:
        long_train, long_validation = (to_device(packet[k], args.device) for k in ('batch', 'validation'))
        lambda_train, lambda_validation = dict(long_train), dict(long_validation)
        for b in (lambda_train, lambda_validation):
            b['value_target'] = (b['frozen_old_values']+b['frozen_gae']).detach().clone()
        all_data = dict(long_train=long_train, long_validation=long_validation,
                        lambda_train=lambda_train, lambda_validation=lambda_validation)
        data_hash = fingerprint(all_data)
        with torch.random.fork_rng(devices=[]):
            start = Critic(long_train['state'].shape[-1], config)
            reference = ScaledCritic(start)
        start.load_state_dict(packet['initial']['fresh'])
        start = start.to(args.device).eval().requires_grad_(False)
        reference.load_state_dict(saved_reference['state_dict'])
        reference = reference.to(args.device).eval().requires_grad_(False)

        def measured(model, name, epoch, updates, **optimization):
            row = dict(candidate=name, epoch=epoch, optimizer_updates=updates, **optimization)
            for split, b in all_data.items():
                row.update({f'{split}_{k}': v for k, v in metrics(model, b, config, norm['mean']).items()})
            return row

        a = measured(reference, 'long_reference', epochs, 0, reused=True, original_optimizer_updates=expected_updates)
        for split, old_split in [('long_train', 'train'), ('long_validation', 'validation')]:
            for k in ('mse', 'explained_variance', 'value_variance'):
                value, recorded = a[f'{split}_{k}'], float(reference_row[f'{old_split}_{k}'])
                if not math.isclose(value, recorded, abs_tol=.05, rel_tol=5e-5):
                    raise AssertionError('Loaded reference does not reproduce the recorded common-target endpoint')
        endpoints.append(a)
        persist()

        def count_update(updates):
            nonlocal updates_done
            updates_done = updates

        def observe(model, epoch, updates, optimization):
            trace.append(measured(model, 'lambda_return', epoch, updates, **optimization))
            persist()

        candidate, updates = fit_variant(start, lambda_train, long_validation, config,
                                          packet['orders'], norm, 'fixed_normalized', observe, count_update)
        if updates != expected_updates or fingerprint(start.state_dict()) != plan['initial_state_sha256']:
            raise AssertionError('Frozen initialization or optimization budget changed')
        b = dict(trace[-1], reused=False)
        endpoints.append(b)
        with torch.no_grad():
            for name, model in [('long_reference', reference), ('lambda_return', candidate)]:
                for split, data in [('train', long_train), ('validation', long_validation)]:
                    for i in range(len(data['reward'])):
                        sub = {k: v[i:i+1] for k, v in data.items()}
                        error = model(sub['state']).double()-sub['value_target'].double()
                        episode_rows.append(dict(candidate=name, split=split, episode=i,
                                                  common_long_mse=float(batch_weighted(error.square(), sub, config))))
        constant = b['long_validation_constant_train_mean_mse']
        mse, ev = b['long_validation_mse'], b['long_validation_explained_variance']
        save_json(output/'summary.json', dict(primary_target=plan['primary_target'], final_candidates=endpoints,
                  lambda_common_validation_mse=mse, train_mean_constant_validation_mse=constant,
                  relative_mse_reduction_vs_constant=1-mse/constant if constant > 0 else None,
                  relative_mse_reduction_vs_long_reference=1-mse/a['long_validation_mse'],
                  descriptive_screen_passed=bool(mse <= .9*constant and ev is not None and ev > .05),
                  interpretation=plan['scope']+'; own lambda-target fit never determines success'))
        save_pt(output/'lambda_candidate.pt', cpu_copy(dict(format=SEMANTICS,
                state_dict=candidate.state_dict(), config=config, source_data_sha256=packet['frozen_sha256'],
                initialization_sha256=plan['initial_state_sha256'], epochs=epochs,
                normalization=norm, training_target='frozen old V + GAE')))
        if fingerprint(all_data) != data_hash:
            raise AssertionError('Frozen data or labels changed')
        for path, digest in hashes.items():
            if file_hash(path) != digest:
                raise AssertionError('Source packet/reference changed')
        persist('completed', input_unchanged=True, data_unchanged=True)
    except Exception as exc:
        persist('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return endpoints
