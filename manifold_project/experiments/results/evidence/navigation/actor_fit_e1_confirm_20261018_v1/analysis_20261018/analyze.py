"""Audit the seed 43/44 fixed-target confirmation without sampling or training."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import torch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RUN = ROOT / 'a0_s43_44_u128'
NAV = ROOT.parent
PROJECT = next(p for p in OUT.parents if p.name == 'manifold_project')
CODE = PROJECT / 'experiments/cooperative_navigation'
SOURCE = NAV / 'navigation_diagnosis_20261012_4m_s40_v1/an_base/N-C_source_learning/full/AN_s40/checkpoints/final.pt'
N_BOOT = 50000
BOOT_SEED = 20261018


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def close(a, b):
    assert np.isclose(float(a), float(b), rtol=1e-10, atol=1e-14), (a, b)


def same_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def summarize_gains(gains, indices):
    n = len(gains)
    mean = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(n))
    half_width = float(t.ppf(.975, n-1)) * se
    worst = np.argsort(gains)[:2]
    leave_one_out = (gains.sum()-gains)/(n-1)
    return dict(mean=mean, bootstrap_ci95=np.quantile(gains[indices].mean(1), [.025, .975]).tolist(),
                t_ci95=[mean-half_width, mean+half_width], median=float(np.median(gains)),
                wins=int((gains > 1e-12).sum()), losses=int((gains < -1e-12).sum()),
                ties=int((np.abs(gains) <= 1e-12).sum()),
                minimum=float(gains.min()), maximum=float(gains.max()),
                leave_one_out_mean_range=[float(leave_one_out.min()), float(leave_one_out.max())],
                worst_two=[dict(episode=int(i), gain=float(gains[i])) for i in worst],
                worst_two_fraction_of_net_sum=float(gains[worst].sum()/gains.sum()) if gains.sum() else None)


def main():
    manifest = json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed' and manifest['diagnostic_seeds'] == [43, 44]
    assert manifest['direction_updates'] == [128] and manifest['actor_updates_per_trajectory'] == 512
    assert manifest['batch_episodes'] == 64 and manifest['validation_episodes'] == 32
    assert manifest['evaluation_episodes'] == 40 and manifest['evaluation_scene_seed'] == 20261018
    assert manifest['eta'] == .1 and manifest['mode'] == 'fit'
    assert sha(SOURCE) == manifest['checkpoint_sha256']
    old = torch.load(SOURCE, map_location='cpu', weights_only=True)['actor']
    source_code_checks = {}
    for filename, field in [('diagnose.py', 'probe_code_sha256'), ('training/actor_fit.py', 'fit_code_sha256')]:
        contents = (CODE/filename).read_bytes().replace(b'\r\n', b'\n')
        matches = [style for style, data in [('LF', contents), ('CRLF', contents.replace(b'\n', b'\r\n'))]
                   if hashlib.sha256(data).hexdigest() == manifest[field]]
        assert matches, filename
        source_code_checks[filename] = matches
    data = {name: read_csv(RUN/f'{name}.csv') for name in
            ('diagnostics', 'fit_trace', 'evaluation_episodes', 'direction_scores')}
    assert len(data['diagnostics']) == 10 and len(data['fit_trace']) == 2116
    report = dict(manifest=manifest, source_code_checks=source_code_checks,
                  artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()},
                  bootstrap=dict(draws=N_BOOT, seed=BOOT_SEED, paired=True, simultaneous_intervals=False),
                  limitations=['Frozen target packets absent: no KL recomputation or fit replay.',
                               'Both units use one source Actor and the same evaluation reset scenes.',
                               'Scene intervals do not measure variability across independently trained source models.',
                               't intervals and leave-one-out means are sensitivity descriptions; no observations excluded.'],
                  units={})
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), constrained_layout=True)
    colors = {3e-4: '#c65a30', 3e-5: '#1f77a5'}
    total_actor_updates = total_eval_steps = 0
    for ri, seed in enumerate([43, 44]):
        selected = {key: [r for r in records if int(r['data_seed']) == seed] for key, records in data.items()}
        d, trace, eps = selected['diagnostics'], selected['fit_trace'], selected['evaluation_episodes']
        target = next(r for r in manifest['fixed_targets'] if r['data_seed'] == seed)
        assert {r['target_sha256'] for r in d+trace} == {target['sha256']}
        unit = dict(target=target, fixed_target_present=(RUN/'fixed_targets'/f'data_s{seed}_direction_u128.pt').exists(),
                    direction_scores=selected['direction_scores'], fits={}, candidates=[], contrasts={})
        packets = []
        for lr in [3e-4, 3e-5]:
            trajectory = [r for r in trace if float(r['actor_lr']) == lr]
            epochs = [r for r in trajectory if r['scope'] != 'post_optimizer_step']
            steps = [r for r in trajectory if r['scope'] == 'post_optimizer_step']
            assert [int(r['epoch']) for r in epochs] == list(range(17))
            assert [int(r['optimizer_updates']) for r in steps] == list(range(1, 513))
            initial, final = epochs[0], epochs[-1]
            best = min(epochs, key=lambda r: float(r['train_target_kl']))
            packet = torch.load(RUN/'fit_candidates'/f'data_s{seed}_u128_lr{lr:g}.pt', map_location='cpu', weights_only=True)
            packets.append(packet)
            assert packet['target_sha256'] == target['sha256'] and packet['actor_lr'] == lr
            assert packet['info']['fit_best_epoch'] == int(best['epoch'])
            assert packet['info']['fit_optimizer_updates'] == 512 and len(packet['history']) == 17
            total_actor_updates += packet['info']['fit_optimizer_updates']
            for history, point in zip(packet['history'], epochs):
                assert history['epoch'] == int(point['epoch'])
                assert history['optimizer_updates'] == int(point['optimizer_updates'])
                close(history['train_target_kl'], point['train_target_kl'])
            for kind, point in [('final', final), ('best', best)]:
                record = next(r for r in d if r['actor_lr'] and float(r['actor_lr']) == lr and r['candidate'].endswith('_'+kind))
                close(record['selected_train_target_kl'], point['train_target_kl'])
                close(record['selected_validation_target_kl'], point['validation_target_kl'])
                assert int(record['selected_epoch']) == int(point['epoch'])
                unchanged = same_state(packet[kind], old)
                assert unchanged == (record['actor_changed'] == 'False')
                assert (int(point['epoch']) == 0) == (record['selected_zero_update'] == 'True')
                if int(point['epoch']) == 0:
                    assert unchanged
            stats = dict(best_epoch=int(best['epoch']), optimizer_updates=512,
                         initial_train_kl=float(initial['train_target_kl']), initial_validation_kl=float(initial['validation_target_kl']),
                         first_step_train_ratio=float(steps[0]['train_target_kl'])/float(initial['train_target_kl']),
                         steps_worse_than_initial=sum(float(r['train_target_kl']) > float(initial['train_target_kl']) for r in steps))
            for kind, point in [('best', best), ('final', final)]:
                for split in ['train', 'validation']:
                    stats[f'{kind}_{split}_kl'] = float(point[f'{split}_target_kl'])
                    stats[f'{kind}_{split}_ratio'] = float(point[f'{split}_target_kl'])/float(initial[f'{split}_target_kl'])
            stats['best_validation_reduction_vs_final'] = 1-stats['best_validation_kl']/stats['final_validation_kl']
            unit['fits'][str(lr)] = stats
            for ci, split in enumerate(['train', 'validation']):
                ax = axes[ri, ci]
                denominator = float(initial[f'{split}_target_kl'])
                ax.plot([int(r['optimizer_updates']) for r in steps], [float(r[f'{split}_target_kl'])/denominator for r in steps],
                        color=colors[lr], alpha=.25, lw=.65)
                ax.plot([int(r['optimizer_updates']) for r in epochs], [float(r[f'{split}_target_kl'])/denominator for r in epochs],
                        color=colors[lr], lw=1.5, label=f'LR {lr:g}')
                ax.scatter(int(best['optimizer_updates']), float(best[f'{split}_target_kl'])/denominator,
                           color=colors[lr], marker='*', s=110, zorder=5, edgecolors='white', linewidths=.4)
                if lr == 3e-4:
                    ax.axhline(1, color='#555555', ls='--', lw=.8)
                ax.set_yscale('log'); ax.grid(alpha=.15); ax.set_xlabel('Actor optimizer updates')
                ax.set_ylabel('Target-to-Actor KL / initial KL')
                ax.set_title(f'Data seed {seed}: {split} fit'); ax.legend(fontsize=9)
        assert packets[0]['fit_orders'] == packets[1]['fit_orders']
        assert len(packets[0]['fit_orders']) == 16
        assert all(sorted(order) == list(range(64)) for order in packets[0]['fit_orders'])
        unit['candidate_history_state_and_schedule_checks_passed'] = True
        groups = {name: sorted([r for r in eps if r['candidate'] == name], key=lambda r: int(r['episode']))
                  for name in {r['candidate'] for r in eps}}
        assert len(groups) == 6
        for records in groups.values():
            assert [int(r['episode']) for r in records] == list(range(40))
        returns = {name: np.array([float(r['J']) for r in records]) for name, records in groups.items()}
        indices = np.random.default_rng(BOOT_SEED).integers(0, 40, (N_BOOT, 40))
        unique_steps = sum(int(r['episode_steps']) for r in eps if not r['evaluation_reused_from'])
        total_eval_steps += unique_steps
        unit['evaluation_steps'] = unique_steps
        unit['success_counts'] = {name: sum(float(r['success']) for r in records) for name, records in groups.items()}
        for record in d:
            gains = returns[record['candidate']]-returns['old']
            close(returns[record['candidate']].mean(), record['candidate_return'])
            close(returns['old'].mean(), record['old_return']); close(gains.mean(), record['return_gain'])
            close(unit['success_counts'][record['candidate']]/40, record['success_rate'])
            original_ci = np.quantile(np.random.default_rng(seed_for('paired-bootstrap', seed, record['candidate']))
                                      .choice(gains, (5000, 40), replace=True).mean(1), [.025, .975])
            close(original_ci[0], record['gain_scene_ci_low']); close(original_ci[1], record['gain_scene_ci_high'])
            if record['evaluation_reused_from']:
                other = groups[record['evaluation_reused_from']]
                assert all(all(r[k] == s[k] for k in r if k not in ('candidate', 'evaluation_reused_from'))
                           for r, s in zip(groups[record['candidate']], other))
            unit['candidates'].append(dict(**record, recomputed=summarize_gains(gains, indices)))
        order = [d[0], d[1], d[3], d[2], d[4]]
        names = dict(zip(['Teacher', 'A', 'B', 'C', 'D'], [r['candidate'] for r in order]))
        ax = axes[ri, 2]
        for i, record in enumerate(order):
            gains = returns[record['candidate']]-returns['old']
            mean = gains.mean(); low, high = np.quantile(gains[indices].mean(1), [.025, .975])
            ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='o', color='#275366', capsize=3)
        ax.axvline(0, ls='--', color='#777777', lw=1)
        ax.set_yticks(range(5), ['Teacher', 'A: 3e-4 final', 'B: 3e-5 final', 'C: 3e-4 best', 'D: 3e-5 best'])
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        ax.set_title(f'Data seed {seed}: paired return gain')
        ax.set_xlabel('Mean gain and 95% scene bootstrap CI')
        for a, b in [('B', 'A'), ('C', 'A'), ('D', 'A'), ('D', 'B'), ('D', 'Teacher')]:
            unit['contrasts'][f'{a}-{b}'] = summarize_gains(returns[names[a]]-returns[names[b]], indices)
        report['units'][str(seed)] = unit
    assert total_eval_steps == manifest['evaluation_steps'] == 400000
    assert total_actor_updates == manifest['actor_optimizer_updates'] == 2048
    assert manifest['collection_steps'] == 76800 and manifest['diagnostic_steps'] == total_eval_steps + 76800
    assert manifest['direction_optimizer_updates'] == 256
    # Prior development target is contextual only, not a third held-out confirmation seed.
    old_path = NAV/'actor_fit_e1_20261017_v1/a0_s40_u128/diagnostics.csv'
    old_records = read_csv(old_path)
    old_best = next(r for r in old_records if r['candidate'] == 'direction4_lr3e-05_best')
    old_initial = next(r for r in read_csv(old_path.parent/'fit_trace.csv') if r['scope'] == 'initial')
    report['prior_development_seed40'] = dict(source=str(old_path), sha256=sha(old_path),
        validation_best_ratio=float(old_best['selected_validation_target_kl'])/float(old_initial['validation_target_kl']),
        best_epoch=int(old_best['selected_epoch']), return_gain=float(old_best['return_gain']))
    report['verified'] = True
    fig.suptitle('New data seeds: nonzero fit improvements repeat; return improvement remains unconfirmed', fontsize=14)
    fig.supxlabel('Stars: best training-KL epoch, including epoch 0. 40 paired scenes per target; unadjusted intervals.', fontsize=10)
    fig.savefig(OUT/'confirmation_fit_and_return.png', dpi=180)
    plt.close(fig)
    (OUT/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=True, fit_summary={seed:u['fits'] for seed,u in report['units'].items()},
                          evaluation_steps=total_eval_steps, actor_optimizer_updates=total_actor_updates), indent=2))


if __name__ == '__main__':
    main()
