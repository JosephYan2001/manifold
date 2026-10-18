"""Recompute the bounded teacher review from immutable episode records; no rollouts."""
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
RUN = OUT.parent
NAV = RUN.parent
PROJECT = next(p for p in OUT.parents if p.name == 'manifold_project')
CODE = PROJECT/'experiments/cooperative_navigation'
DISCOVERY = NAV/'actor_fit_e1_confirm_20261018_v1/a0_s43_44_u128'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def close(actual, expected):
    assert np.isclose(float(actual), float(expected), rtol=1e-11, atol=1e-12), (actual, expected)


def stats(gains, bootstrap_seed):
    # Independent implementation: resample the observed values, rather than
    # calling the runtime summary helper or trusting its saved interval.
    draws = np.random.default_rng(bootstrap_seed).choice(gains, size=(50000, len(gains)), replace=True)
    interval = np.quantile(draws.mean(axis=1), [.025, .975])
    return dict(mean=float(gains.mean()), bootstrap_ci95=interval.tolist(),
                standard_error=float(gains.std(ddof=1)/np.sqrt(len(gains))),
                median=float(np.median(gains)), minimum=float(gains.min()), maximum=float(gains.max()),
                positive=int((gains > 1e-12).sum()), negative=int((gains < -1e-12).sum()),
                zero=int((np.abs(gains) <= 1e-12).sum()),
                absolute_gain_le_01=int((np.abs(gains) <= .1).sum()),
                absolute_gain_le_1=int((np.abs(gains) <= 1).sum()))


def main():
    historical = read_json(DISCOVERY/'manifest.json')
    historical_config = read_json(DISCOVERY/'effective_runner_config.json')
    historical_rows = read_csv(DISCOVERY/'evaluation_episodes.csv')
    source = NAV/'navigation_diagnosis_20261012_4m_s40_v1/an_base/N-C_source_learning/full/AN_s40/checkpoints/final.pt'
    assert sha(source) == historical['checkpoint_sha256']
    old_actor_hash = tensor_fingerprint(torch.load(source, map_location='cpu', weights_only=True)['actor'])
    inputs = [p for seed in [43, 44] for p in sorted((RUN/f's{seed}').glob('*')) if p.is_file()]
    hashes = {p.relative_to(RUN).as_posix(): sha(p) for p in inputs}
    assert len(hashes) == 12
    report = dict(raw_artifact_sha256=hashes,
                  discovery_artifact_sha256={p.name: sha(p) for p in
                      [DISCOVERY/'manifest.json', DISCOVERY/'evaluation_episodes.csv', DISCOVERY/'effective_runner_config.json']},
                  primary_result='new 100 pairs per fixed teacher, separately from discovery 40 pairs',
                  units={}, total_evaluation_steps=0, total_elapsed_seconds=0.,
                  scope='record and code audit; original large fixed-target packets remain unavailable locally')
    gains_by_seed, scenes_by_seed, actions_by_seed = {}, {}, {}
    for seed in [43, 44]:
        folder = RUN/f's{seed}'
        manifest = read_json(folder/'manifest.json')
        summary = read_json(folder/'summary.json')
        config = read_json(folder/'effective_evaluation_config.json')
        rows, pairs, diagnostic = [read_csv(folder/f'{name}.csv') for name in
                                   ['evaluation_episodes', 'paired_gains', 'diagnostics']]
        target = next(t for t in historical['fixed_targets'] if t['data_seed'] == seed)
        assert manifest['status'] == 'completed' and manifest['mode'] == 'teacher-eval'
        assert manifest['data_seed'] == seed and manifest['eta'] == historical['eta'] == .1
        assert manifest['saved_direction_updates'] == 128
        assert manifest['evaluation_pairs'] == manifest['completed_pairs'] == 100
        assert manifest['completed_policy_episodes'] == 200 and len(rows) == 200 and len(pairs) == 100
        assert len(diagnostic) == 1
        assert manifest['evaluation_scene_seed'] == 20261019
        assert manifest['discovery_evaluation_scene_seed'] == historical['evaluation_scene_seed'] == 20261018
        assert manifest['input_sha256'] == summary['target_sha256'] == target['sha256']
        assert manifest['frozen_target_sha256'] == target['frozen_sha256']
        assert manifest['source_provenance']['checkpoint_sha256'] == historical['checkpoint_sha256']
        assert manifest['source_round'] == historical['source_round'] == 545
        assert manifest['model_state_sha256'] == manifest['model_state_sha256_after']
        assert manifest['model_state_sha256']['actor'] == old_actor_hash
        assert manifest['teacher_policy_sha256'] == manifest['teacher_policy_sha256_after'] == summary['teacher_policy_sha256']
        assert manifest['input_unchanged'] and manifest['source_models_frozen']
        zero_keys = ['collection_steps', 'training_collection_steps', 'validation_collection_steps',
                     'direction_optimizer_updates', 'actor_optimizer_updates', 'critic_optimizer_updates']
        assert all(manifest[k] == 0 for k in zero_keys)
        assert config == dict(historical_config, horizon=1000)
        assert manifest['evaluation_horizon'] == 1000 and manifest['gamma'] == config['gamma'] == .99
        assert manifest['maximum_evaluation_steps'] == 200000
        code_matches = {}
        for section in ['model_code_sha256', 'execution_code_sha256']:
            code_matches[section] = {}
            for name, digest in manifest[section].items():
                data = (CODE/name).read_bytes().replace(b'\r\n', b'\n')
                matches = [style for style, value in [('LF', data), ('CRLF', data.replace(b'\n', b'\r\n'))]
                           if hashlib.sha256(value).hexdigest() == digest]
                assert matches, name
                code_matches[section][name] = matches
        assert {r['candidate'] for r in rows} == {'old', 'teacher'}
        keyed = {(r['candidate'], int(r['pair_id'])): r for r in rows}
        assert len(keyed) == 200 and [int(p['pair_id']) for p in pairs] == list(range(100))
        gains = []
        scenes, actions = [], []
        success_pairs = []
        for i, p in enumerate(pairs):
            old, teacher = keyed['old', i], keyed['teacher', i]
            reset = seed_for('evaluation-reset', manifest['evaluation_purpose'], config['n_agents'], i)
            action = seed_for('evaluation-action', 'diagnostic', seed, manifest['evaluation_purpose'], config['n_agents'], i)
            for row in [p, old, teacher]:
                assert int(row['data_seed']) == seed and int(row['pair_id']) == i
                assert int(row['reset_seed']) == reset and int(row['action_seed']) == action
            for row in [old, teacher]:
                assert int(row['episode']) == i and int(row['evaluation_scene_seed']) == 20261019
                assert 0 <= int(row['episode_steps']) <= 1000 and np.isfinite(float(row['J']))
                assert row['end_reason'] == ('success' if float(row['success']) else 'collection_cutoff')
            gain = float(teacher['J'])-float(old['J'])
            close(gain, p['return_gain'])
            for name, row in [('old', old), ('teacher', teacher)]:
                close(p[f'{name}_return'], row['J'])
                close(p[f'{name}_success'], row['success'])
                assert int(p[f'{name}_steps']) == int(row['episode_steps'])
            if float(old['success']) or float(teacher['success']):
                success_pairs.append(dict(pair_id=i, old_steps=int(old['episode_steps']),
                    teacher_steps=int(teacher['episode_steps']), gain=gain,
                    old_success=float(old['success']), teacher_success=float(teacher['success'])))
            gains.append(gain); scenes.append(reset); actions.append(action)
        old_scenes = {seed_for('evaluation-reset', historical['evaluation_purpose'], config['n_agents'], i) for i in range(40)}
        assert len(set(scenes)) == 100 and not old_scenes.intersection(scenes)
        count_steps = sum(int(r['episode_steps']) for r in rows)
        assert count_steps == manifest['evaluation_steps'] == manifest['diagnostic_steps'] == summary['diagnostic_steps_total']
        boot = manifest['bootstrap']
        assert boot['method'] == 'paired_percentile' and boot['repeats'] == 50000 and boot['confidence_level'] == .95
        assert boot['resampling_unit'] == 'complete episode pair' and not boot['multiplicity_adjusted']
        assert boot['seed'] == seed_for('teacher-eval-bootstrap-v1', seed, 20261019)
        gains = np.asarray(gains)
        result = stats(gains, boot['seed'])
        for key, calculated in [('return_gain', result['mean']), ('gain_scene_ci_low', result['bootstrap_ci95'][0]),
                ('gain_scene_ci_high', result['bootstrap_ci95'][1]), ('median_gain', result['median']),
                ('minimum_gain', result['minimum']), ('maximum_gain', result['maximum']),
                ('positive_pairs', result['positive']), ('negative_pairs', result['negative']), ('zero_pairs', result['zero']),
                ('gain_p05', np.quantile(gains, .05)), ('gain_p95', np.quantile(gains, .95))]:
            close(summary[key], calculated); close(diagnostic[0][key], calculated)
        for name, field in [('old', 'old_return'), ('teacher', 'candidate_return')]:
            close(summary[field], np.mean([float(keyed[name, i]['J']) for i in range(100)]))
            successes = sum(float(keyed[name, i]['success']) for i in range(100))
            close(summary[f'{name}_success_count'], successes)
            close(summary['old_success_rate' if name == 'old' else 'success_rate'], successes/100)
        negative = -gains[gains < 0].sum()
        worst_indices = [int(i) for i in np.argsort(gains) if gains[i] < 0][:5]
        worst_loss = -gains[worst_indices].sum()
        assert [p['pair_id'] for p in summary['worst_loss_pairs']] == worst_indices
        close(summary['total_negative_gain_magnitude'], negative)
        close(summary['worst_five_loss_magnitude'], worst_loss)
        close(summary['worst_five_fraction_of_negative_gain'], worst_loss/negative)
        prior_old = {int(r['episode']): float(r['J']) for r in historical_rows if int(r['data_seed']) == seed and r['candidate'] == 'old'}
        prior_teacher = {int(r['episode']): float(r['J']) for r in historical_rows if int(r['data_seed']) == seed and r['candidate'] == 'direction4_target'}
        assert sorted(prior_old) == sorted(prior_teacher) == list(range(40))
        prior_gains = np.array([prior_teacher[i]-prior_old[i] for i in range(40)])
        prior_stats = stats(prior_gains, 20261018)
        result.update(old_return=summary['old_return'], teacher_return=summary['candidate_return'],
                      old_successes=summary['old_success_count'], teacher_successes=summary['teacher_success_count'],
                      successful_pairs=success_pairs, worst_five_negative_fraction=float(worst_loss/negative),
                      worst_loss_pairs=summary['worst_loss_pairs'], largest_gain_pair=int(gains.argmax()),
                      positive_gain_sum=float(gains[gains > 0].sum()), negative_gain_magnitude=float(negative))
        report['units'][seed] = dict(review=result, discovery=prior_stats, code_matches=code_matches,
            evaluation_steps=count_steps, elapsed_seconds=manifest['elapsed_seconds'],
            target_sha256=target['sha256'], actor_sha256=old_actor_hash,
            target_packet_present=(DISCOVERY/'fixed_targets'/f'data_s{seed}_direction_u128.pt').is_file(),
            new_scenes_disjoint_from_discovery=True, records_consistent=True)
        report['total_evaluation_steps'] += count_steps
        report['total_elapsed_seconds'] += manifest['elapsed_seconds']
        gains_by_seed[seed], scenes_by_seed[seed], actions_by_seed[seed] = gains, scenes, actions
    assert scenes_by_seed[43] == scenes_by_seed[44]
    assert all(a != b for a, b in zip(actions_by_seed[43], actions_by_seed[44]))
    report['shared_reset_scenes_across_teachers'] = True
    report['action_seeds_differ_across_teachers'] = True
    report['new_training_steps_and_optimizer_updates'] = 0
    report['raw_artifacts_unchanged'] = all(sha(RUN/p) == digest for p, digest in hashes.items())
    assert report['raw_artifacts_unchanged']
    report['verified'] = True
    (OUT/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    plot(report, gains_by_seed)
    print(json.dumps({k: report[k] for k in ['verified', 'total_evaluation_steps', 'total_elapsed_seconds', 'raw_artifacts_unchanged']}))
    for seed in [43, 44]:
        print(seed, json.dumps(report['units'][seed]['review']))


def plot(report, gains):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.1), constrained_layout=True)
    colors = {43: '#246B8E', 44: '#B85B38'}
    ax = axes[0, 0]
    labels = []
    for y, (seed, stage) in enumerate([(43, 'discovery'), (43, 'review'), (44, 'discovery'), (44, 'review')]):
        value = report['units'][seed][stage]
        mean = value['mean']; lo, hi = value['bootstrap_ci95']
        ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt='o' if stage == 'review' else 's',
                    color=colors[seed], alpha=1 if stage == 'review' else .5, capsize=4)
        labels.append(f"Seed {seed} | {'NEW 100 pairs' if stage == 'review' else 'discovery 40 pairs'}")
    ax.set_yticks(range(4), labels); ax.invert_yaxis(); ax.axvline(0, color='#666666', ls='--', lw=1)
    ax.set(xlabel='Mean teacher - old discounted return (95% paired bootstrap)', title='A. New intervals cross zero for both teachers')
    ax = axes[0, 1]
    for seed in [43, 44]:
        ax.plot(np.sort(gains[seed]), np.arange(1, 101)/100, color=colors[seed], label=f'Seed {seed}')
    ax.axvline(0, color='#666666', ls='--', lw=1)
    ax.set(xlabel='Per-pair return difference', ylabel='Fraction at or below difference', title='B. Complete gain distribution, including both tails')
    ax.legend(frameon=False)
    ax = axes[1, 0]
    for seed in [43, 44]:
        ax.scatter(np.arange(100), gains[seed], s=19, alpha=.75, color=colors[seed], label=f'Seed {seed}')
    ax.axhline(0, color='#666666', lw=1)
    ax.set(xlabel='New scene pair ID', ylabel='Teacher - old discounted return', title='C. All 100 pairs retained for each teacher')
    ax.legend(frameon=False)
    ax = axes[1, 1]
    bottom = np.zeros(2)
    for field, color, label in [('positive', '#49816F', 'Positive'), ('negative', '#B95F50', 'Negative'), ('zero', '#B6BBC1', 'Equal')]:
        counts = [report['units'][seed]['review'][field] for seed in [43, 44]]
        ax.bar([0, 1], counts, bottom=bottom, width=.55, color=color, label=label)
        for x, number, start in zip([0, 1], counts, bottom):
            ax.text(x, start+number/2, str(number), ha='center', va='center', color='white' if field != 'zero' else '#222222')
        bottom += counts
    ax.set_xticks([0, 1], ['Seed 43', 'Seed 44']); ax.set_ylim(0, 112)
    ax.set(ylabel='Complete pairs', title='D. Sign counts do not establish a mean benefit')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.08), ncol=3, frameon=False)
    fig.suptitle('Fixed teacher review | new scene seed 20261019 | no training updates', fontsize=15)
    fig.savefig(OUT/'teacher_review.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
