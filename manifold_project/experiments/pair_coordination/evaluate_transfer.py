"""P-T: exact frozen-final evaluation of the 60-run pair confirmation suite."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig
from manifold_project.experiments.pair_coordination.evaluation.exact_pair import closed_form_expected_return
from manifold_project.experiments.pair_coordination.experiments.reporting import bootstrap, read_csv, write_csv
from manifold_project.experiments.pair_coordination.models.mlp import MLPActor

ROOT = Path(__file__).resolve().parent
METHODS = ('ours', 'sampled', 'pg', 'no_direction_check', 'no_return_check', 'fit_quarter')
SEEDS = tuple(range(100, 110))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def targets():
    source = PairConfig()
    return [('source', source)] + [
        (f'agents_{n}', replace(source, n_agents=n)) for n in (2, 6, 8)
    ] + [(f'composition_{p}', replace(source, type_probs=(p, 1-p))) for p in (.3, .9)] + [
        (f'strength_{r}', replace(source, interaction_strength=r)) for r in (0., .35, 1.4)
    ]


def load_models(suite):
    table = suite/'training_results.csv'
    rows = read_csv(table)
    expected = set(product(METHODS, SEEDS))
    keys = [(r['condition'], int(r['seed'])) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('P-T requires exactly six methods x seeds 100..109, without duplicates.')
    inputs = {str(table.relative_to(suite)): digest(table)}
    models = {}
    for row in rows:
        method, seed = row['condition'], int(row['seed'])
        run = suite/'training'/method/f'seed_{seed}'
        cfg = read_json(run/'config.json')
        if PairConfig(**cfg['source']) != PairConfig() or int(row['budget']) != 262144:
            raise ValueError(f'Unexpected source or budget: {method}/{seed}')
        actor = MLPActor.from_state(read_json(run/'checkpoints/final.json')['actor'])
        if actor.n_types != 2 or actor.beta != .02:
            raise ValueError(f'Unexpected policy constraints: {method}/{seed}')
        policy = actor.table()
        value = closed_form_expected_return(PairConfig(), policy)
        if not np.isclose(value, float(row['final_return']), rtol=0, atol=1e-10):
            raise ValueError(f'Final model does not match source result: {method}/{seed}')
        models[method, seed] = policy
        for file in (run/'config.json', run/'checkpoints/final.json'):
            inputs[str(file.relative_to(suite))] = digest(file)
    return models, inputs


def evaluate(models):
    rows = []
    corners = [np.column_stack((1-np.array(p), p)) for p in product((.01, .99), repeat=2)]
    oracle = np.array([[.99, .01], [.01, .99]])
    for name, cfg in targets():
        optimum = closed_form_expected_return(cfg, oracle)
        if not np.isclose(optimum, max(closed_form_expected_return(cfg, p) for p in corners), atol=1e-12, rtol=0):
            raise ValueError('Target no longer satisfies the shared-optimum protocol.')
        for (method, seed), policy in sorted(models.items()):
            value = closed_form_expected_return(cfg, policy)
            if value > optimum + 1e-12:
                raise ValueError('Policy exceeds the constrained oracle.')
            rows.append(dict(target=name, condition=method, seed=seed, n_agents=cfg.n_agents,
                             type0_probability=cfg.type_probs[0], interaction_strength=cfg.interaction_strength,
                             team_return=value, per_agent_return=value/cfg.n_agents,
                             optimal_team_return=optimum,
                             optimality_gap_per_agent=max(0., (optimum-value)/cfg.n_agents),
                             action1_type0=policy[0, 1], action1_type1=policy[1, 1]))
    return rows


def summarize(rows):
    result = []
    for name, _ in targets():
        groups = {m: {r['seed']: r for r in rows if r['target'] == name and r['condition'] == m} for m in METHODS}
        for metric in ('team_return', 'per_agent_return', 'optimality_gap_per_agent'):
            for method in METHODS:
                result.append(dict(target=name, kind='method', condition=method, metric=metric,
                                   ci_method='seed_bootstrap_95_pointwise',
                                   **bootstrap([groups[method][s][metric] for s in SEEDS])))
                if method != 'ours':
                    result.append(dict(target=name, kind='paired_ours_minus_comparator', condition=method,
                                       metric=metric, ci_method='paired_seed_bootstrap_95_pointwise',
                                       **bootstrap([groups['ours'][s][metric]-groups[method][s][metric] for s in SEEDS])))
    return result


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [('Population control', ['agents_2', 'source', 'agents_6', 'agents_8'], 'per_agent_return'),
              ('Composition shift', ['composition_0.3', 'source', 'composition_0.9'], 'optimality_gap_per_agent'),
              ('Interaction shift', ['strength_0.0', 'strength_0.35', 'source', 'strength_1.4'], 'optimality_gap_per_agent')]
    for ax, (title, names, metric) in zip(axes.flat, panels):
        for i, method in enumerate(METHODS):
            stats = [bootstrap([r[metric] for r in rows if r['condition'] == method and r['target'] == n]) for n in names]
            y = np.array([s['mean'] for s in stats])
            ax.errorbar(np.arange(len(names))+(i-2.5)*.025, y,
                        yerr=np.maximum(0., [y-np.array([s['ci_low'] for s in stats]), np.array([s['ci_high'] for s in stats])-y]),
                        label=method, marker='o', linestyle='-' if i < 3 else '--', capsize=2)
        ax.set_xticks(range(len(names)), names, rotation=15, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(metric.replace('_', ' '))
        ax.grid(alpha=.2)
        if metric.startswith('optimality'):
            ax.set_yscale('symlog', linthresh=1e-9)
            peak = max(r[metric] for r in rows if r['target'] in names)
            ax.set_ylim(0, max(1e-8, peak*3))
    ax = axes[1, 1]
    for method in METHODS:
        values = [r for r in rows if r['target'] == 'source' and r['condition'] == method]
        ax.scatter([r['action1_type0'] for r in values], [r['action1_type1'] for r in values], label=method, alpha=.7)
    ax.set(xlabel='P(action 1 | type 0)', ylabel='P(action 1 | type 1)', title='Frozen final policies (overlap expected)')
    ax.ticklabel_format(useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('P-T: exact frozen-policy evaluation; all targets share one constrained optimum\n'
                 '10 training seeds; pointwise 95% intervals; no target training')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite', type=Path, default=ROOT/'results/suite_09_论文比较与消融_10种子')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/suite_12_零样本迁移_10种子')
    parser.add_argument('--dry-run', action='store_true', help='Validate all 60 final models without target evaluation or output.')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args(argv)
    suite, output = args.source_suite.resolve(), args.output_dir.resolve()
    models, inputs = load_models(suite)
    print('Validated 60 final models; plan: 9 configurations, 540 exact evaluations.', flush=True)
    if args.dry_run:
        return 0
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new --output-dir: {output}')
    manifest = dict(experiment='P-T', status='planned', source_suite=str(suite),
                    created_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS, seeds=SEEDS,
                    budget=262144, checkpoint='final', beta=.02, evaluation='exact stochastic policy expectation',
                    targets={n: asdict(c) for n, c in targets()}, input_sha256=inputs,
                    bootstrap_repeats=10000, bootstrap_seed=20260916, intervals='pointwise 95%, not multiplicity corrected',
                    interpretation='Shared target optimum; not evidence of transfer-specific superiority.',
                    code_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                                 (Path(__file__), ROOT/'evaluation/exact_pair.py', ROOT/'models/mlp.py', ROOT/'experiments/reporting.py')})
    output.mkdir(parents=True)
    manifest_path = output/'manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save()
    try:
        rows = evaluate(models)
        write_csv(output/'records.csv', rows)
        write_csv(output/'summary.csv', summarize(rows))
        if args.plot:
            plot(rows, output/'overview.png')
        if any(digest(suite/p) != h for p, h in inputs.items()):
            raise RuntimeError('Source inputs changed during evaluation.')
        manifest.update(status='complete', records=len(rows))
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        save()
    print(f'Completed {len(rows)} exact evaluations: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
