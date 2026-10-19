"""Audit the four short runs without sampling, training, or changing raw artifacts."""
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
GROUPS = dict(A='a_lr3e4_last', B='b_lr3e5_last', C='c_lr3e4_best', D='d_lr3e5_best')
SETTINGS = dict(A=(3e-4, 'last'), B=(3e-5, 'last'), C=(3e-4, 'best_train_kl'), D=(3e-5, 'best_train_kl'))


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b):
    assert np.isclose(float(a), float(b), rtol=1e-10, atol=1e-12), (a, b)


def truth(value):
    return value is True or value == 'True'


def contrast(values, indices):
    boot = values[indices].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                median=float(np.median(values)), positive=int((values > 1e-12).sum()),
                negative=int((values < -1e-12).sum()), zero=int((np.abs(values) <= 1e-12).sum()))


def main():
    raw = {p.relative_to(ROOT).as_posix(): sha(p) for name in GROUPS.values()
           for p in sorted((ROOT/name).rglob('*')) if p.is_file()}
    indices = np.random.default_rng(20261019).integers(0, 100, size=(50000, 100))
    report = dict(units={}, contrasts={}, raw_artifact_sha256=raw,
        bootstrap=dict(seed=20261019, repeats=50000, method='paired percentile', unit='complete final-evaluation scene',
                       confidence=.95, multiple_comparisons_adjusted=False,
                       scope='exploratory conditional comparison of four fixed final policies; one common training seed'),
        total_source_steps=0, total_final_evaluation_steps=0, total_experiment_seconds=0.)
    configs, all_events, evaluations, checkpoints = {}, {}, {}, {}
    for letter, name in GROUPS.items():
        folder = ROOT/name
        run = folder/'N-C_source_learning/full/AN_s40'
        manifest = read_json(folder/'manifest.json')
        c = manifest['experiments']['N-C']['config']
        rc = read_json(run/'config.json')
        summary = read_json(run/'summary.json')
        rows = read_csv(run/'training.csv')
        eps = sorted(read_csv(run/'final_evaluation/evaluation_episodes.csv'), key=lambda r: int(r['scene_seed']))
        events = [json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        event = lambda name: [e['record'] for e in events if e['stream'] == name]
        fits, directions, monitors = event('actor_fit_selection'), event('direction_fit'), event('evaluation')
        critics = [e for e in event('critic_full_batch') if e['stage'] == 'postfit']
        checkpoint = torch.load(run/'checkpoints/final.pt', map_location='cpu', weights_only=True)
        assert manifest['status'] == manifest['experiments']['N-C']['status'] == 'completed'
        assert summary['status'] == 'complete' and checkpoint['complete']
        assert (c['actor_lr'], c['actor_fit_selection']) == SETTINGS[letter]
        assert c['budget'] == 500000 and c['seeds'] == [40] and rc['seed'] == 40
        assert c['batch_episodes'] == 64 and c['direction_steps'] == 4 and c['actor_fit_steps'] == 16
        assert rc['config']['minibatch_episodes'] == 2
        assert not c['direction_check'] and not c['return_check'] and not c['save_fit_diagnostics']
        assert not manifest['command_arguments'].get('resume') and not manifest['command_arguments'].get('source')
        assert checkpoint['round'] == summary['rounds'] == len(rows) == len(fits) == len(directions) == len(critics) == 16
        assert checkpoint['config'] == rc['config']
        assert [int(r['round']) for r in rows] == list(range(1, 17))
        assert [int(e['scene_seed']) for e in eps] == list(range(20261020, 20261120))
        assert {e['checkpoint'] for e in eps} == {'final.pt'} and {int(e['target_n']) for e in eps} == {4}
        assert {e['policy_mode'] for e in eps} == {'stochastic'}
        assert len(monitors) == len(checkpoint['curves']) == 9
        expected_nodes = [0, 65536, 131072, 196608, 262144, 327680, 393216, 458752, 500000]
        assert [e['budget_checkpoint'] for e in monitors] == expected_nodes
        for e, saved in zip(monitors, checkpoint['curves']):
            for key in saved:
                assert e[key] == saved[key], key
        auc = sum((b['budget_checkpoint']-a['budget_checkpoint'])*a['J'] for a, b in zip(monitors, monitors[1:]))/c['budget']
        close(auc, summary['AUC'])
        round_metrics = []
        for i, (row, fit, direction, critic, stored) in enumerate(zip(rows, fits, directions, critics, checkpoint['rounds']), 1):
            assert all(r['round'] == i for r in [fit, direction, critic, stored])
            history = fit['history']
            assert [h['epoch'] for h in history] == list(range(17))
            assert [h['optimizer_updates'] for h in history] == list(range(0, 513, 32))
            assert all(h['finite'] for h in history)
            minimum = min(history, key=lambda h: h['train_target_kl'])
            initial, final = history[0]['train_target_kl'], history[-1]['train_target_kl']
            tolerance = fit['actor_fit_selection_atol'] + fit['actor_fit_selection_rtol']*abs(initial)
            best_epoch = 0 if initial-minimum['train_target_kl'] <= tolerance else minimum['epoch']
            selected_epoch = 16 if c['actor_fit_selection'] == 'last' else best_epoch
            assert fit['fit_minimum_epoch'] == minimum['epoch']
            assert fit['fit_best_epoch'] == best_epoch and fit['fit_selected_epoch'] == selected_epoch
            assert int(row['fit_optimizer_updates']) == fit['fit_optimizer_updates'] == 512
            close(row['fit_initial_train_kl'], initial)
            close(row['fit_final_train_kl'], final)
            close(row['fit_selected_train_kl'], history[selected_epoch]['train_target_kl'])
            close(row['fit_best_train_kl'], history[best_epoch]['train_target_kl'])
            close(stored['fit_selected_train_kl'], row['fit_selected_train_kl'])
            assert truth(row['fit_selected_zero_update']) == (selected_epoch == 0)
            assert truth(row['actor_changed']) == truth(stored['actor_changed'])
            assert int(row['train_batch_steps']) == 25600 and int(row['train_episode_count']) == 64
            if selected_epoch == 0:
                assert not truth(row['actor_changed']) and float(row['old_to_actor_kl_fp64']) < 1e-12
            round_metrics.append(dict(round=i, initial_target_kl=initial, final_target_kl=final,
                best_target_kl=history[best_epoch]['train_target_kl'], selected_target_kl=history[selected_epoch]['train_target_kl'],
                selected_epoch=selected_epoch, best_epoch=best_epoch,
                selected_ratio=history[selected_epoch]['train_target_kl']/initial,
                actor_changed=truth(row['actor_changed']), old_to_teacher_kl=float(row['old_to_target_kl_fp64']),
                old_to_actor_kl=float(row['old_to_actor_kl_fp64']), entropy=float(row['policy_entropy']),
                critic_postfit=critic, direction_train=direction))
        training_steps = sum(int(e['steps']) for e in event('train_episode'))
        monitor_steps = sum(int(e['episodes']*e['episode_steps']) for e in monitors)
        final_steps = sum(int(e['steps']) for e in eps)
        assert training_steps == summary['costs']['train'] == 409600
        assert monitor_steps == summary['costs']['source_monitor'] == summary['evaluation_steps'] == 72000
        assert training_steps+monitor_steps == summary['source_steps'] == 481600
        assert final_steps == 100000 and summary['unused_budget'] == 18400 and summary['diagnostic_steps'] == 0
        assert checkpoint['costs'] == summary['costs']
        for module, count in dict(critic=5120, direction=2048, actor_fit=8192).items():
            assert count == summary['optimizer_steps_by_module'][module] == checkpoint['optim_steps'][module]
            assert sum(e['updates'] for e in event('optimization') if e['module'] == module) == count
        assert summary['actor_changed_rounds'] == sum(r['actor_changed'] for r in round_metrics)
        assert summary['selected_zero_update_rounds'] == sum(r['selected_epoch'] == 0 for r in round_metrics)
        values = np.array([float(e['discounted_return']) for e in eps])
        success = sum(float(e['success']) for e in eps)
        table = {r['metric']: r for r in read_csv(folder/'summary.csv')}
        close(table['discounted_return']['mean'], values.mean())
        close(table['source_report_only_steps']['mean'], final_steps)
        assert table['discounted_return']['independent_units'] == '1'
        best_packet = torch.load(run/'checkpoints/best.pt', map_location='cpu', weights_only=True)
        result = dict(final_return=float(values.mean()), final_return_ci95=contrast(values, indices)['ci95'],
            successes=int(success), final_evaluation_episodes=len(eps), actual_source_steps=summary['source_steps'],
            training_steps=training_steps, monitor_steps=monitor_steps, final_evaluation_steps=final_steps,
            actor_changed_rounds=summary['actor_changed_rounds'], zero_update_rounds=summary['selected_zero_update_rounds'],
            monitor_AUC=auc, monitor_initial_return=monitors[0]['J'], monitor_last_grid_return=monitors[-2]['J'],
            monitor_final_other_scenes_return=monitors[-1]['J'],
            source_monitors=monitors, rounds=round_metrics,
            selected_ratio_median=float(np.median([r['selected_ratio'] for r in round_metrics])),
            selected_mid_epoch_rounds=[r['round'] for r in round_metrics if 0 < r['selected_epoch'] < 16],
            best_not_final_rounds=[r['round'] for r in round_metrics if r['best_epoch'] != 16],
            selected_worse_than_initial_rounds=[r['round'] for r in round_metrics if r['selected_ratio'] > 1+1e-6],
            old_to_actor_kl_median=float(np.median([r['old_to_actor_kl'] for r in round_metrics])),
            critic_postfit_ev=[r['explained_variance'] for r in critics],
            negative_train_B_rounds=[r['round'] for r in directions if r['B_label'] < 0],
            negative_train_D_rounds=[r['round'] for r in directions if r['D_label'] < 0],
            saturation_events=[r for r in directions if r['tanh_saturation'] > .01],
            final_entropy=round_metrics[-1]['entropy'], uniform_action_entropy=float(np.log(5)),
            train_seconds=summary['train_seconds'], experiment_seconds=manifest['experiments']['N-C']['elapsed_seconds'],
            recorded_code_sha256=manifest['metadata']['experiment_code_sha256'],
            final_checkpoint_sha256=sha(run/'checkpoints/final.pt'),
            best_checkpoint_node=best_packet['budget_checkpoint'])
        report['units'][letter] = result
        report['total_source_steps'] += training_steps+monitor_steps
        report['total_final_evaluation_steps'] += final_steps
        report['total_experiment_seconds'] += result['experiment_seconds']
        configs[letter], all_events[letter], evaluations[letter], checkpoints[letter] = c, fits, values, checkpoint
    assert len({u['recorded_code_sha256'] for u in report['units'].values()}) == 1
    for c in list(configs.values())[1:]:
        assert {k for k in c if c[k] != configs['A'][k]} <= {'actor_lr', 'actor_fit_selection'}
    # The same starting data/teacher are corroborated by equal first-round
    # direction metrics and targets. Later rounds need not share their targets.
    first = report['units']['A']['rounds'][0]
    for u in report['units'].values():
        assert u['rounds'][0]['direction_train'] == first['direction_train']
        close(u['rounds'][0]['initial_target_kl'], first['initial_target_kl'])
        close(u['monitor_initial_return'], report['units']['A']['monitor_initial_return'])
    for left, right, same_rounds in [('A', 'C', 2), ('B', 'D', 4)]:
        for i in range(same_rounds):
            assert all_events[left][i]['history'] == all_events[right][i]['history']
    for label, left, right in [('B-A','B','A'), ('C-A','C','A'), ('D-B','D','B'), ('D-C','D','C'), ('D-A','D','A')]:
        report['contrasts'][label] = contrast(evaluations[left]-evaluations[right], indices)
    report['verified'] = True
    report['raw_artifacts_unchanged'] = all(sha(ROOT/p) == h for p, h in raw.items())
    assert report['raw_artifacts_unchanged']
    report['limits'] = [
        'One common training seed; paired scene intervals do not establish across-training-seed superiority.',
        'No independent direction validation, per-round fixed batches, or saved per-round teachers in these runs.',
        'The endpoint source-final monitor uses a different scene set from source-grid; final 100 scenes are a third set.',
        'Equal recorded code fingerprints across runs; current whole-tree fingerprint was not independently matched.',
        'No new environment steps, training, or checkpoint selection performed by this analysis.']
    (OUT/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    draw(report)
    print(json.dumps(dict(verified=True, raw_files=len(raw), source_steps=report['total_source_steps'],
        final_steps=report['total_final_evaluation_steps'], elapsed_hours=report['total_experiment_seconds']/3600)))
    for label, result in report['contrasts'].items():
        print(label, json.dumps(result))


def draw(report):
    colors = dict(A='#246B8E', B='#D68B27', C='#328468', D='#AD5267')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11.3), constrained_layout=True)
    ax = axes[0, 0]
    for label, u in report['units'].items():
        nodes=u['source_monitors'][:-1]
        ax.plot([r['budget_checkpoint']/1000 for r in nodes], [r['J'] for r in nodes], '-o', ms=3, color=colors[label], label=label)
    ax.set(title='A. Fixed 8-scene source-grid monitor', xlabel='Declared source budget (thousands)', ylabel='Discounted return')
    ax.legend(ncol=4, frameon=False)
    ax = axes[0, 1]
    for i, (label, u) in enumerate(report['units'].items()):
        mean=u['final_return']; lo, hi=u['final_return_ci95']
        ax.errorbar(i, mean, yerr=[[mean-lo],[hi-mean]], fmt='o', capsize=4, color=colors[label])
        ax.text(i, mean-7.5, f'{mean:.2f}', ha='center', color=colors[label])
    ax.set_xticks(range(4), ['A\n3e-4 last','B\n3e-5 last','C\n3e-4 best','D\n3e-5 best'])
    ax.set(title='B. Final 100 scenes; successes 0/100 in every group', ylabel='Mean discounted return with scene interval')
    ax = axes[1, 0]
    for i,(label,r) in enumerate(report['contrasts'].items()):
        mean=r['mean'];lo,hi=r['ci95']
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color='#4B5664')
    ax.axvline(0,color='#888888',ls='--',lw=1);ax.set_yticks(range(5),list(report['contrasts']))
    ax.set(title='C. Paired differences; positive favors first group',xlabel='Mean difference, exploratory 95% interval')
    for label,u in report['units'].items():
        rounds=u['rounds'];x=[r['round'] for r in rounds]
        axes[1,1].plot(x,[r['selected_ratio'] for r in rounds],'-o',ms=3,label=label,color=colors[label])
        axes[2,0].semilogy(x,[max(r['old_to_actor_kl'],1e-16) for r in rounds],'-o',ms=3,label=label,color=colors[label])
        axes[2,1].plot(x,u['critic_postfit_ev'],'-o',ms=3,label=label,color=colors[label])
    axes[1,1].axhline(1,color='#888888',ls='--',lw=1)
    axes[1,1].set(title='D. Selected / initial training target KL',xlabel='Training round',ylabel='Residual ratio (different targets after branching)')
    axes[2,0].set(title='E. Nonzero parameter updates can still be very small',xlabel='Training round',ylabel='Mean KL(old || submitted Actor)')
    axes[2,0].annotate('C: zero update in round 16',xy=(16,report['units']['C']['rounds'][-1]['old_to_actor_kl']),xytext=(6,1e-12),arrowprops={'arrowstyle':'->'},fontsize=9)
    axes[2,1].axhline(0,color='#888888',lw=1)
    axes[2,1].set(title='F. Full-batch Critic explained variance after fitting',xlabel='Training round',ylabel='Explained variance')
    for ax in [axes[1,1],axes[2,0],axes[2,1]]:ax.legend(ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Actor short training | N64 / U128 | common seed 40 | 16 rounds per group',fontsize=16)
    fig.savefig(OUT/'short_training_comparison.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
