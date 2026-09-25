"""Source-only KL matching and fixed-candidate gate diagnostics (pair)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..common.config import load_config
from ..common.reporting import read_csv, write_csv, write_json
from ..common.runtime import metadata
from .diagnostics import _fit, dataset_hash, settings
from .environments import pair_source, labels_for
from .numerics import episode_scores, fit_direct, kl, logit, tilted


def match(old, q, weights, requested):
    """Match source KL(old||new) on a fixed logit ray, without return selection."""
    high = 1.0
    while kl(old, tilted(old, q, high), weights) < requested and high < 2**20:
        high *= 2
    if kl(old, tilted(old, q, high), weights) < requested:
        return None, None
    low = 0.0
    for _ in range(70):
        mid = (low + high) / 2
        if kl(old, tilted(old, q, mid), weights) < requested:
            low = mid
        else:
            high = mid
    return tilted(old, q, high), high


def candidates(args, config):
    env = pair_source(config)
    old = env.initial_policy
    cfg = settings(config)
    rows = []
    for seed in args.seeds:
        batch = env.sample(old, args.samples, np.random.default_rng(seed))
        labels = labels_for(batch, "mc_raw", env, old)
        for method in ("AN", "SA", "DA"):
            if method == "DA":
                direct = fit_direct(old, batch, labels, cfg["actor_fit_steps"], cfg["actor_lr"])
                q = (logit(direct) - logit(old)) / 2
            else:
                q = _fit(batch, labels, env, method, cfg)[0]
            for requested in args.kl:
                candidate, eta = match(old, q, env.type_probabilities, requested)
                row = dict(seed=seed, method=method, sample_episodes=args.samples,
                           requested_kl=requested, dataset_sha256=dataset_hash(batch),
                           q_0=float(q[0]), q_1=float(q[1]), reachable=candidate is not None)
                if candidate is not None:
                    row.update(eta=eta, actual_kl=kl(old, candidate, env.type_probabilities),
                               candidate_p_0=float(candidate[0]), candidate_p_1=float(candidate[1]),
                               return_gain=env.oracle(candidate)["expected_return"]-env.oracle(old)["expected_return"],
                               normalized_derivative=env.oracle(old, q)["normalized_derivative"])
                rows.append(row)
        print(f"KL candidates: seed {seed}", flush=True)
    return rows


def gates(args, config):
    manifest_path = args.source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "completed" or manifest.get("experiment") != "pair_kl_v1":
        raise ValueError("Gate source must be a completed pair_kl_v1 run")
    if manifest["config"] != config:
        raise ValueError("Gate source configuration differs from this implementation")
    path = args.source / "diagnostics.csv"
    selected = [r for r in read_csv(path) if float(r["requested_kl"]) in args.kl]
    if not selected or any(not any(float(r['requested_kl']) == k for r in selected) for k in args.kl):
        raise ValueError("Requested KL absent from saved candidate source")
    env = pair_source(config)
    old = env.initial_policy
    rows = []
    for index, r in enumerate(selected):
        if r['reachable'] != 'True':
            rows.append(dict(method=r['method'], seed=int(r['seed']), requested_kl=float(r['requested_kl']), status='unreachable'))
            continue
        q0 = np.array([float(r['q_0']), float(r['q_1'])])
        eta = float(r['eta'])
        saved = np.array([float(r['candidate_p_0']), float(r['candidate_p_1'])])
        if not np.allclose(tilted(old, q0, eta), saved, rtol=0, atol=1e-12):
            raise ValueError("Saved candidate does not match its direction/path")
        # Prespecified original, reversed and near-zero rays; never cherry-pick by oracle.
        for variant, scale in [('original', 1.0), ('reversed', -1.0), ('near_zero', 1e-4)]:
            q = q0 * scale
            candidate = tilted(old, q, eta)
            gain = env.oracle(candidate)['expected_return'] - env.oracle(old)['expected_return']
            score = env.oracle(old, q)['score']
            for n in args.check_sizes:
                accepted_d, accepted_r, zero_r = [], [], []
                for repeat in range(args.repeats):
                    # Independent direction/return streams, shared across method/variant/size.
                    entropy = [args.check_seed, int(r['seed']), repeat]
                    direction_rng = np.random.default_rng(np.random.SeedSequence(entropy+[0]))
                    returns_seed = np.random.SeedSequence(entropy+[1])
                    check = env.sample(old, n, direction_rng)
                    d = float(episode_scores(q, check, labels_for(check, 'mc_raw', env, old)).mean())
                    previous = env.sample(old, n, np.random.default_rng(returns_seed))
                    proposed = env.sample(candidate, n, np.random.default_rng(returns_seed))
                    delta = proposed['returns'][:, 0] - previous['returns'][:, 0]
                    accepted_d.append(d > 0)
                    accepted_r.append(float(delta.mean()) >= 0)
                    zero_r.append(bool(np.all(delta == 0)))
                rate_d, rate_r = float(np.mean(accepted_d)), float(np.mean(accepted_r))
                applicable = r['method'] != 'DA'
                row = dict(method=r['method'], seed=int(r['seed']), requested_kl=float(r['requested_kl']),
                           variant=variant, check_episodes=n, repeats=args.repeats,
                           candidate_p_0=float(candidate[0]), candidate_p_1=float(candidate[1]),
                           actual_kl=kl(old,candidate,env.type_probabilities), exact_return_gain=gain,
                           return_class='positive' if gain>1e-8 else 'negative' if gain < -1e-8 else 'near_zero',
                           direction_gate_applicable=applicable, return_accept_rate=rate_r,
                           return_false_accept_rate=rate_r if gain < -1e-8 else None,
                           return_false_reject_rate=1-rate_r if gain > 1e-8 else None,
                           all_zero_pair_rate=float(np.mean(zero_r)), return_check_steps=2*n,
                           direction_check_steps=n if applicable else 0)
                if applicable:
                    row.update(exact_direction_score=score, direction_accept_rate=rate_d,
                               direction_false_accept_rate=rate_d if score < -1e-8 else None,
                               direction_false_reject_rate=1-rate_d if score > 1e-8 else None,
                               reject_positive_return_rate=1-rate_d if gain > 1e-8 else None)
                rows.append(row)
        print(f"Gate candidate {index+1}/{len(selected)}", flush=True)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=['kl','gates'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1000,1020)))
    p.add_argument('--samples', type=int, default=2048)
    p.add_argument('--kl', nargs='+', type=float, default=[1e-4,3e-4,1e-3])
    p.add_argument('--check-sizes', nargs='+', type=int, default=[8,32,128,512])
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--check-seed', type=int, default=20260925)
    args = p.parse_args(argv)
    if args.samples<1 or args.repeats<1 or any(n<1 for n in args.check_sizes) or any(not np.isfinite(k) or k<=0 for k in args.kl) or any(s<0 for s in args.seeds) or args.check_seed<0:
        p.error('Sample counts, KL and repeats must be positive; seeds nonnegative')
    if any(len(v)!=len(set(v)) for v in (args.kl,args.seeds,args.check_sizes)):
        p.error('Duplicate parameter values are not allowed')
    if args.experiment=='gates' and not args.source:
        p.error('gates requires --source KL_RUN')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output directory nonempty: {args.output}')
    config = load_config('pair','pilot')
    manifest = dict(experiment=f'pair_{args.experiment}_v1', status='running', config=config,
                    arguments=vars(args), metadata=metadata())
    if args.source:
        manifest['source_csv_sha256']=hashlib.sha256((args.source/'diagnostics.csv').read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'manifest.json',manifest)
    try:
        rows = candidates(args,config) if args.experiment=='kl' else gates(args,config)
        write_csv(args.output/'diagnostics.csv',rows)
        manifest.update(status='completed',records=len(rows))
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        raise
    finally:
        write_json(args.output/'manifest.json',manifest)
    print(f'Completed: {args.output}',flush=True)


if __name__=='__main__':
    main()
