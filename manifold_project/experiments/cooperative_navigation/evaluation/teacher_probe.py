"""Bounded, paired execution of one saved teacher. No training objects are created."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ...common.reporting import write_csv
from ..configs import evaluation_horizon
from ..models import Actor, Direction
from ..training.collector import episode
from ..training.fixed_target import Target, fingerprint, validate_packet
from ..training.storage import load_pt, save_json, seed_for


SEMANTICS = 'fixed-teacher-evaluation-v1'
BOOTSTRAP_REPEATS = 50000
ZERO_TOLERANCE = 1e-12


def file_hash(path):
    # Full snapshots can be large; avoid allocating a second copy of the file.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def paired_summary(pairs, bootstrap_seed):
    """Predeclared percentile bootstrap of complete episode-pair differences."""
    gains = np.array([p['return_gain'] for p in pairs], dtype=np.float64)
    if len(gains) < 2 or not np.isfinite(gains).all():
        raise ValueError('At least two finite complete pairs are required')
    rng = np.random.default_rng(bootstrap_seed)
    means = np.empty(BOOTSTRAP_REPEATS)
    for start in range(0, BOOTSTRAP_REPEATS, 1000):
        end = min(start+1000, BOOTSTRAP_REPEATS)
        indices = rng.integers(0, len(gains), (end-start, len(gains)))
        means[start:end] = gains[indices].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return dict(evaluation_pairs=len(pairs),
        old_return=float(np.mean([p['old_return'] for p in pairs])),
        candidate_return=float(np.mean([p['teacher_return'] for p in pairs])),
        return_gain=float(gains.mean()), gain_scene_ci_low=float(low), gain_scene_ci_high=float(high),
        old_success_rate=float(np.mean([p['old_success'] for p in pairs])),
        success_rate=float(np.mean([p['teacher_success'] for p in pairs])),
        old_success_count=int(sum(p['old_success'] for p in pairs)),
        teacher_success_count=int(sum(p['teacher_success'] for p in pairs)),
        positive_pairs=int((gains > ZERO_TOLERANCE).sum()),
        negative_pairs=int((gains < -ZERO_TOLERANCE).sum()),
        zero_pairs=int((np.abs(gains) <= ZERO_TOLERANCE).sum()),
        median_gain=float(np.median(gains)), minimum_gain=float(gains.min()), maximum_gain=float(gains.max()),
        gain_p05=float(np.quantile(gains, .05)), gain_p95=float(np.quantile(gains, .95)))


def run_teacher_evaluation(args):
    started = time.perf_counter()
    if args.fixed_target is None or args.checkpoint is not None:
        raise ValueError('teacher-eval requires --fixed-target, not a source training checkpoint')
    forbidden = ('seeds', 'batch_episodes', 'validation_episodes', 'direction_epochs', 'direction_updates',
                 'sample_sizes', 'init_seeds', 'init_seed', 'eta', 'fit_epochs', 'actor_lrs',
                 'fit_trace_every_updates', 'target_kl')
    supplied = [key for key in forbidden if getattr(args, key) is not None]
    if supplied or args.prepare_only:
        raise ValueError('teacher-eval cannot change or fit the saved target: '+', '.join(supplied or ['prepare_only']))
    if args.evaluation_seed is None or args.evaluation_seed < 0:
        raise ValueError('teacher-eval requires an explicit nonnegative --evaluation-seed')
    count = args.evaluation_episodes if args.evaluation_episodes is not None else 100
    if count < 2:
        raise ValueError('teacher-eval requires at least two evaluation pairs')
    source = args.fixed_target.resolve(strict=True)
    output = args.output.resolve()
    if output == source.parent or source.parent in output.parents or output in source.parents:
        raise ValueError('Teacher evaluation output must be separate from its input directory')
    if output.exists():
        raise FileExistsError(f'Use a new teacher evaluation directory: {output}')
    original_hash = file_hash(source)
    packet = load_pt(source)
    validate_packet(packet)
    saved = packet['config']
    if saved.get('environment') != 'navigation' or saved.get('task_mode') != 'success_only' or not saved.get('complete_episodes'):
        raise ValueError('teacher-eval requires an episodic success-only navigation target')
    if args.evaluation_seed == packet['evaluation_seed']:
        raise ValueError('Use a new evaluation scene seed, different from the saved discovery evaluation')
    if not np.isfinite(packet['eta']) or packet['eta'] <= 0:
        raise ValueError('Saved teacher step size must be finite and positive')
    horizon = evaluation_horizon(saved)
    seed = packet['data_seed']
    purpose = f'frozen-checkpoint-probe-scenes-{args.evaluation_seed}'
    bootstrap_seed = seed_for('teacher-eval-bootstrap-v1', seed, args.evaluation_seed)
    expected_states = dict(actor=fingerprint(packet['actor']), direction=fingerprint(packet['direction']))
    teacher_hash = fingerprint(dict(actor=packet['actor'], direction=packet['direction'], eta=packet['eta']))
    code_root = Path(__file__).resolve().parents[1]
    paths = [code_root/'diagnose.py', Path(__file__), code_root/'training/collector.py',
             code_root/'training/fixed_target.py', code_root/'configs.py']
    paths += list((code_root/'envs').rglob('*.py')) + list((code_root/'observations').rglob('*.py'))
    execution_hashes = {p.relative_to(code_root).as_posix(): file_hash(p) for p in sorted(set(paths))}
    plan = dict(mode='teacher-eval', semantics_version=SEMANTICS,
        input_kind='fixed_target', input_path=str(source), input_sha256=original_hash,
        frozen_target_sha256=packet['frozen_sha256'], saved_config_sha256=packet['config_sha256'],
        source_provenance=packet['provenance'], source_round=packet['round'], data_seed=seed,
        saved_direction_updates=packet['direction_updates'], eta=packet['eta'],
        model_state_sha256=expected_states, teacher_policy_sha256=teacher_hash,
        model_code_sha256=packet['model_code_sha256'], execution_code_sha256=execution_hashes,
        evaluation_scene_seed=args.evaluation_seed, discovery_evaluation_scene_seed=packet['evaluation_seed'],
        evaluation_purpose=purpose, evaluation_episodes_per_policy=count, evaluation_pairs=count,
        evaluation_horizon=horizon, n_agents=saved['n_agents'], gamma=saved['gamma'],
        policies=['old', 'teacher'], device=args.device, numpy_version=np.__version__, torch_version=str(torch.__version__),
        maximum_evaluation_steps=2*count*horizon,
        pairing='same reset seed and inverse-CDF action uniform stream within each pair; actions follow each policy',
        reset_distribution='unchanged source task configuration',
        history_rule='reset per rollout, then update on that policy trajectory',
        primary_metric='teacher minus old realized discounted return J under the saved evaluation protocol',
        bootstrap=dict(method='paired_percentile', repeats=BOOTSTRAP_REPEATS, confidence_level=.95,
                       seed=bootstrap_seed, resampling_unit='complete episode pair', multiplicity_adjusted=False),
        gain_zero_tolerance=ZERO_TOLERANCE,
        stopping_rule='complete the predeclared pair count; no return-based early stopping or sample extension',
        collection_steps=0, training_collection_steps=0, validation_collection_steps=0,
        direction_optimizer_updates=0, actor_optimizer_updates=0, critic_optimizer_updates=0)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return plan

    # Match the diagnostic Runner's execution settings, without constructing a
    # Runner, Critic, Collector, optimizer, or any fitting batch on the GPU.
    torch.set_num_threads(saved['threads'])
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    config = deepcopy(saved)
    config.update(device=args.device, horizon=horizon)
    with torch.random.fork_rng(devices=[]):
        old = Actor(packet['input_dim'], config)
        direction = Direction(packet['input_dim'], config)
    old.load_state_dict(packet['actor']); direction.load_state_dict(packet['direction'])
    old = old.to(args.device).eval().requires_grad_(False)
    direction = direction.to(args.device).eval().requires_grad_(False)
    teacher = Target(old, direction, packet['eta']).eval().requires_grad_(False)
    if dict(actor=fingerprint(old.state_dict()), direction=fingerprint(direction.state_dict())) != expected_states:
        raise AssertionError('Loaded policy states differ from the saved teacher')
    # CPU-only packet validation is complete; release the potentially large data
    # tensors before execution. Evaluation never reads cached target probabilities.
    del packet
    output.mkdir(parents=True)
    save_json(output/'effective_evaluation_config.json', config)
    episodes, pairs = [], []
    evaluation_steps = 0

    def persist(status='running', **extra):
        write_csv(output/'evaluation_episodes.csv', episodes)
        write_csv(output/'paired_gains.csv', pairs)
        save_json(output/'manifest.json', dict(plan, status=status, completed_pairs=len(pairs),
            completed_policy_episodes=len(episodes), evaluation_steps=evaluation_steps,
            diagnostic_steps=evaluation_steps, elapsed_seconds=time.perf_counter()-started, **extra))

    def count_step():
        nonlocal evaluation_steps
        evaluation_steps += 1
        if evaluation_steps > plan['maximum_evaluation_steps']:
            raise AssertionError('Execution exceeded the predeclared evaluation budget')

    persist()
    try:
        for i in range(count):
            reset_seed = seed_for('evaluation-reset', purpose, config['n_agents'], i)
            action_seed = seed_for('evaluation-action', 'diagnostic', seed, purpose, config['n_agents'], i)
            metrics = {}
            for name, policy in [('old', old), ('teacher', teacher)]:
                before = evaluation_steps
                with torch.no_grad():
                    result = episode(policy, config, reset_seed, action_seed, on_step=count_step)
                if not np.isfinite(result['J']) or result['episode_steps'] != evaluation_steps-before:
                    raise AssertionError('Nonfinite return or inconsistent evaluation step accounting')
                if result['episode_steps'] > horizon:
                    raise AssertionError('Episode exceeded the saved evaluation horizon')
                metrics[name] = result
                episodes.append(dict(data_seed=seed, candidate=name, episode=i, pair_id=i,
                    evaluation_scene_seed=args.evaluation_seed, reset_seed=reset_seed, action_seed=action_seed, **result))
                persist()
            baseline, candidate = metrics['old'], metrics['teacher']
            pairs.append(dict(data_seed=seed, pair_id=i, reset_seed=reset_seed, action_seed=action_seed,
                old_return=baseline['J'], teacher_return=candidate['J'],
                return_gain=candidate['J']-baseline['J'], old_success=baseline['success'],
                teacher_success=candidate['success'], old_steps=baseline['episode_steps'], teacher_steps=candidate['episode_steps']))
            persist()
            if (i+1) % 10 == 0 or i+1 == count:
                print(f'data seed {seed}: {i+1}/{count} pairs, {evaluation_steps} evaluation steps', flush=True)
        after_states = dict(actor=fingerprint(old.state_dict()), direction=fingerprint(direction.state_dict()))
        if after_states != expected_states or teacher.eta != plan['eta']:
            raise AssertionError('Frozen teacher changed during evaluation')
        if file_hash(source) != original_hash:
            raise AssertionError('Input fixed target file changed during evaluation')
        if evaluation_steps != sum(row['episode_steps'] for row in episodes):
            raise AssertionError('Final evaluation cost does not match episode records')
        stats = paired_summary(pairs, bootstrap_seed)
        row = dict(data_seed=seed, candidate='teacher', candidate_kind='teacher',
                   target_sha256=original_hash, teacher_policy_sha256=teacher_hash, eta=plan['eta'],
                   evaluation_scene_seed=args.evaluation_seed, **stats, diagnostic_steps_total=evaluation_steps)
        write_csv(output/'diagnostics.csv', [row])
        worst = [p for p in sorted(pairs, key=lambda p: p['return_gain']) if p['return_gain'] < 0][:5]
        total_loss = -sum(min(0., p['return_gain']) for p in pairs)
        worst_loss = -sum(p['return_gain'] for p in worst)
        save_json(output/'summary.json', dict(row, worst_loss_pairs=worst,
            total_negative_gain_magnitude=total_loss, worst_five_loss_magnitude=worst_loss,
            worst_five_fraction_of_negative_gain=worst_loss/total_loss if total_loss else None,
            loss_contribution_definition='negative per-pair gains; all pairs remain in the primary mean and interval',
            inference_scope='one fixed teacher on new evaluation pairs; no across-training-seed or simultaneous guarantee'))
        persist('completed', input_unchanged=True, source_models_frozen=True, model_state_sha256_after=after_states,
                teacher_policy_sha256_after=fingerprint(dict(actor=old.state_dict(), direction=direction.state_dict(), eta=teacher.eta)))
        return row
    except BaseException as exc:
        try:
            unchanged = source.is_file() and file_hash(source) == original_hash
        except OSError:
            unchanged = None
        persist('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                error=f'{type(exc).__name__}: {exc}', input_unchanged=unchanged)
        raise
