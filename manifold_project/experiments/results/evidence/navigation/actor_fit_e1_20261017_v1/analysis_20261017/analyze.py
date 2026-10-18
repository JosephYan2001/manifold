"""Read-only E1 evidence checks and plots; no environment sampling or training."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
NAV = ROOT.parent
PROJECT = next(p for p in OUT.parents if p.name == 'manifold_project')
CODE = PROJECT / 'experiments/cooperative_navigation'


def rows(path):
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


def main():
    report = dict(scope='Two existing fixed-target diagnostic units, not multi-round training',
                  limitations=['Fixed-target packets are absent: cannot recompute KL or replay fits.',
                               'Scene bootstrap is conditional on each frozen training checkpoint and fitted target.',
                               'No multiplicity correction; repeated historical targets are not independent replications.'],
                  runs={})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), constrained_layout=True)
    plt.rcParams.update({'font.size': 10})
    colors = {0.0003: '#c65a30', 0.00003: '#1f77a5'}
    specs = [('a0_s40_u128', 'an_base', 'direction_nxu_s40/n64_init40'),
             ('a2_s41_u512', 'an_batch64', 'fit_s41_42')]
    for row_index, (name, method, historical) in enumerate(specs):
        directory = ROOT / name
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        diagnostics = rows(directory/'diagnostics.csv')
        trace = rows(directory/'fit_trace.csv')
        episodes = rows(directory/'evaluation_episodes.csv')
        direction = rows(directory/'direction_scores.csv')
        source = NAV/'navigation_diagnosis_20261012_4m_s40_v1'/method/'N-C_source_learning/full/AN_s40/checkpoints/final.pt'
        assert manifest['status'] == 'completed' and len(diagnostics) == 5
        assert sha(source) == manifest['checkpoint_sha256']
        # These are local project checkpoints; only tensors and simple data are needed.
        original = torch.load(source, map_location='cpu', weights_only=True)
        source_code_checks = {}
        for filename, field in [('diagnose.py', 'probe_code_sha256'), ('training/actor_fit.py', 'fit_code_sha256')]:
            contents = (CODE/filename).read_bytes().replace(b'\r\n', b'\n')
            matches = [style for style, data in [('LF', contents), ('CRLF', contents.replace(b'\n', b'\r\n'))]
                       if hashlib.sha256(data).hexdigest() == manifest[field]]
            assert matches, filename
            source_code_checks[filename] = matches
        target_sha = manifest['fixed_targets'][0]['sha256']
        assert {r['target_sha256'] for r in trace + diagnostics} == {target_sha}
        unit = dict(manifest=manifest, direction_scores=direction, source_code_checks=source_code_checks,
                    fixed_target_present=(directory/'fixed_targets'/Path(manifest['fixed_targets'][0]['path']).name).exists(),
                    artifact_sha256={str(f.relative_to(directory)): sha(f) for f in sorted(directory.rglob('*')) if f.is_file()},
                    fits={}, candidates=[], contrasts={})
        grouped = {name: sorted([r for r in episodes if r['candidate'] == name], key=lambda r: int(r['episode']))
                   for name in {r['candidate'] for r in episodes}}
        assert len(grouped) == 6
        for records in grouped.values():
            assert [int(r['episode']) for r in records] == list(range(40))
        baseline = np.array([float(r['J']) for r in grouped['old']])
        cost = sum(int(r['episode_steps']) for r in episodes if not r['evaluation_reused_from'])
        assert cost == manifest['evaluation_steps']
        assert manifest['diagnostic_steps'] == cost + manifest['collection_steps']
        assert all(float(r['success']) == 0 for r in episodes)
        unit['unique_evaluation_policies'] = len({r['candidate'] for r in episodes if not r['evaluation_reused_from']})
        unit['all_successes_zero'] = True
        # One shared resampling matrix gives identical intervals for identical policies.
        indices = np.random.default_rng(20261017).integers(0, 40, size=(50000, 40))
        returns = {'old': baseline}
        for r in diagnostics:
            values = np.array([float(e['J']) for e in grouped[r['candidate']]])
            returns[r['candidate']] = values
            gains = values-baseline
            close(values.mean(), r['candidate_return']); close(baseline.mean(), r['old_return'])
            close(gains.mean(), r['return_gain'])
            original_ci = np.quantile(np.random.default_rng(seed_for('paired-bootstrap', int(r['data_seed']), r['candidate']))
                                      .choice(gains, (5000, 40), replace=True).mean(1), [.025, .975])
            close(original_ci[0], r['gain_scene_ci_low']); close(original_ci[1], r['gain_scene_ci_high'])
            reused = r['evaluation_reused_from']
            if reused:
                assert all(all(e[k] == other[k] for k in e if k not in ('candidate', 'evaluation_reused_from'))
                           for e, other in zip(grouped[r['candidate']], grouped[reused]))
            ci = np.quantile(gains[indices].mean(1), [.025, .975])
            unit['candidates'].append(dict(**r, recomputed_paired_ci95=ci.tolist()))
        packets = []
        for lr in manifest['actor_lrs']:
            trajectory = [r for r in trace if float(r['actor_lr']) == lr]
            epochs = [r for r in trajectory if r['scope'] != 'post_optimizer_step']
            steps = [r for r in trajectory if r['scope'] == 'post_optimizer_step']
            assert len(epochs) == 17 and [int(r['optimizer_updates']) for r in steps] == list(range(1, 513))
            assert [int(r['epoch']) for r in epochs] == list(range(17))
            initial, final = epochs[0], epochs[-1]
            best = min(epochs, key=lambda r: float(r['train_target_kl']))
            packet_path = directory/'fit_candidates'/f'data_s{manifest["diagnostic_seeds"][0]}_u{manifest["direction_updates"][0]}_lr{lr:g}.pt'
            packet = torch.load(packet_path, map_location='cpu', weights_only=True)
            packets.append(packet)
            assert packet['target_sha256'] == target_sha
            assert len(packet['history']) == 17
            for h, r in zip(packet['history'], epochs):
                assert h['epoch'] == int(r['epoch']) and h['optimizer_updates'] == int(r['optimizer_updates'])
                close(h['train_target_kl'], r['train_target_kl'])
            assert packet['info']['fit_best_epoch'] == int(best['epoch'])
            for kind, reference in [('final', final), ('best', best)]:
                r = next(r for r in diagnostics if r['actor_lr'] and float(r['actor_lr']) == lr and r['candidate'].endswith('_'+kind))
                close(r['selected_train_target_kl'], reference['train_target_kl'])
                close(r['selected_validation_target_kl'], reference['validation_target_kl'])
                assert int(r['selected_epoch']) == int(reference['epoch'])
                changed = not same_state(packet[kind], original['actor'])
                assert changed == (r['actor_changed'] == 'True')
                if int(reference['epoch']) == 0:
                    assert not changed
            stats = dict(initial_train_kl=float(initial['train_target_kl']), initial_validation_kl=float(initial['validation_target_kl']),
                         best_epoch=int(best['epoch']), optimizer_updates=len(steps),
                         first_step_train_ratio=float(steps[0]['train_target_kl'])/float(initial['train_target_kl']),
                         steps_worse_than_initial=sum(float(r['train_target_kl']) > float(initial['train_target_kl']) for r in steps))
            for kind, point in [('best', best), ('final', final)]:
                for split in ['train', 'validation']:
                    stats[f'{kind}_{split}_kl'] = float(point[f'{split}_target_kl'])
                    stats[f'{kind}_{split}_ratio'] = float(point[f'{split}_target_kl'])/float(initial[f'{split}_target_kl'])
            unit['fits'][str(lr)] = stats
            for col, split in enumerate(['train', 'validation']):
                ax = axes[row_index, col]
                norm = float(initial[f'{split}_target_kl'])
                ax.plot([int(r['optimizer_updates']) for r in steps], [float(r[f'{split}_target_kl'])/norm for r in steps],
                        color=colors[lr], alpha=.25, lw=.65)
                ax.plot([int(r['optimizer_updates']) for r in epochs], [float(r[f'{split}_target_kl'])/norm for r in epochs],
                        color=colors[lr], lw=1.5, label=f'LR {lr:g}')
                ax.scatter(int(best['optimizer_updates']), float(best[f'{split}_target_kl'])/norm,
                           marker='*', s=110, color=colors[lr], zorder=5, edgecolors='white', linewidths=.4)
                ax.axhline(1, color='#555555', ls='--', lw=.8) if lr == manifest['actor_lrs'][0] else None
                ax.set_yscale('log'); ax.grid(alpha=.15); ax.set_xlabel('Actor optimizer updates')
                ax.set_ylabel('Target-to-Actor KL / initial KL')
                ax.set_title(f'{name.split("_")[0].upper()}: {split} fit')
                ax.legend(loc='upper right', fontsize=9)
        assert packets[0]['fit_orders'] == packets[1]['fit_orders']
        assert len(packets[0]['fit_orders']) == 16
        assert all(sorted(order) == list(range(64)) for order in packets[0]['fit_orders'])
        assert manifest['actor_optimizer_updates'] == sum(p['info']['fit_optimizer_updates'] for p in packets) == 1024
        unit['candidate_state_history_and_schedule_checks_passed'] = True
        old_trace = rows(NAV/'navigation_followup_20261015_v1'/historical/'fit_trace.csv')
        fields = ('data_seed', 'direction_epochs', 'actor_lr', 'scope', 'epoch', 'optimizer_updates')
        lookup = {tuple(r[k] for k in fields): r for r in old_trace}
        common = [(r, lookup[tuple(r[k] for k in fields)]) for r in trace if tuple(r[k] for k in fields) in lookup]
        unit['historical_overlap'] = dict(rows=len(common),
            target_kl_identical=all(float(a[k]) == float(b[k]) for a, b in common for k in ('train_target_kl', 'validation_target_kl')),
            max_actual_kl_difference=max(abs(float(a['validation_actual_kl'])-float(b['validation_actual_kl'])) for a, b in common))
        order = [diagnostics[0], diagnostics[1], diagnostics[3], diagnostics[2], diagnostics[4]]
        labels = ['Teacher', 'A: 3e-4 final', 'B: 3e-5 final', 'C: 3e-4 best', 'D: 3e-5 best']
        ax = axes[row_index, 2]
        for i, r in enumerate(order):
            gains = returns[r['candidate']]-baseline
            ci = np.quantile(gains[indices].mean(1), [.025, .975])
            mean = gains.mean()
            ax.errorbar(mean, i, xerr=[[mean-ci[0]], [ci[1]-mean]], fmt='o', capsize=3, color='#275366')
        ax.axvline(0, color='#777777', ls='--', lw=1)
        ax.set_yticks(range(5), labels); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        ax.set_title(f'{name.split("_")[0].upper()}: paired return gain')
        ax.set_xlabel('Mean gain and 95% scene bootstrap CI')
        for left, right in [('B', 'A'), ('C', 'A'), ('D', 'B'), ('D', 'Teacher')]:
            keys = dict(zip(['Teacher', 'A', 'B', 'C', 'D'], [r['candidate'] for r in order]))
            delta = returns[keys[left]]-returns[keys[right]]
            unit['contrasts'][f'{left}-{right}'] = dict(mean=float(delta.mean()), ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist())
        report['runs'][name] = unit
    report['totals'] = {key: sum(r['manifest'][key] for r in report['runs'].values())
                        for key in ('diagnostic_steps', 'collection_steps', 'evaluation_steps', 'direction_optimizer_updates', 'actor_optimizer_updates', 'elapsed_seconds')}
    report['verified'] = True
    fig.suptitle('Fixed-target Actor fitting: improvement in fit does not establish return improvement', fontsize=14)
    fig.supxlabel('Stars: best training-KL epoch (including epoch 0). Same target within each row. 40 paired scenes; unadjusted intervals.', fontsize=10)
    fig.savefig(OUT/'e1_fit_and_return.png', dpi=180)
    plt.close(fig)
    (OUT/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=True, totals=report['totals'], contrasts={k:v['contrasts'] for k,v in report['runs'].items()}), indent=2))


if __name__ == '__main__':
    main()
