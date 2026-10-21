"""Read-only analysis of uploaded runs; no environment/model execution."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
NAV = SUITE.parent
NAMES = [f'n{n}_{s}' for n in (16, 32, 64) for s in ('fixed', 'linear')]
PATHS = {'old_base': NAV/'navigation_diagnosis_20261010_2m_s40_v1/an_base',
         **{name: SUITE/name for name in NAMES}}
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 20261022


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def values(rows, key):
    return np.array([float(r[key]) for r in rows if r.get(key) not in ('', None)])


def describe(x):
    if not len(x):
        return None
    x = np.asarray(x, dtype=float)
    assert np.isfinite(x).all()
    n = max(1, len(x)//4)
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()), last=float(x[-1]),
                first_quarter_mean=float(x[:n].mean()), last_quarter_mean=float(x[-n:].mean()))


def bootstrap_mean(x, rng):
    x = np.asarray(x, dtype=float)
    estimates = x[rng.integers(0, len(x), (BOOTSTRAPS, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), ci95_percentile=np.quantile(estimates, [.025, .975]).tolist())


def main():
    # Track all original run artifacts, including checkpoints and historical plots.
    source_files = sorted({p for root in PATHS.values() for p in root.rglob('*') if p.is_file()})
    raw_hashes = {str(p.relative_to(NAV)): sha(p) for p in source_files}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = dict(date='2026-10-22', bootstrap_repeats=BOOTSTRAPS,
                  bootstrap_seed=BOOTSTRAP_SEED,
                  interval_scope='conditional on these fixed policies; not across training seeds',
                  runs={}, paired_lr_contrasts={}, raw_files_sha256=raw_hashes)
    data = {}
    fields = ['fit_initial_train_kl', 'fit_selected_train_kl', 'fit_final_train_kl',
              'fit_ratio', 'fit_selected_epoch', 'old_to_actor_kl_fp64', 'policy_step_kl',
              'fit_kl_before', 'fit_kl_after', 'policy_entropy', 'actor_lr_used',
              'critic_postfit_explained_variance', 'critic_postfit_mse',
              'critic_postfit_target_mean', 'critic_postfit_value_mean',
              'critic_postfit_target_variance', 'critic_postfit_value_variance',
              'label_mean_unweighted', 'label_std_unweighted',
              'direction_train_B_label', 'direction_train_q_action_span',
              'direction_train_q_abs_mean', 'direction_train_tanh_saturation',
              'direction_train_effective_bound_fraction']
    for name, root in PATHS.items():
        d = root/'N-C_source_learning/full/AN_s40'
        manifest = read_json(root/'manifest.json')
        config = read_json(d/'config.json')['config']
        summary = read_json(d/'summary.json')
        rows = read_csv(d/'training.csv')
        episodes = read_csv(d/'final_evaluation/evaluation_episodes.csv')
        events = [json.loads(line) for line in (d/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        curves = [e['record'] for e in events if e['stream'] == 'evaluation']
        assert manifest['status'] == 'completed' and summary['status'] == 'complete'
        assert len(rows) == summary['rounds'] and len(curves) == 32
        assert len({int(e['scene_seed']) for e in episodes}) == len(episodes)
        assert all(e['checkpoint'] == 'final.pt' and e['policy_mode'] == 'stochastic' and
                   int(e['n_agents']) == 4 and int(e['evaluation_horizon']) == 1000 for e in episodes)
        assert summary['source_steps'] == sum(summary['costs'].values())
        assert summary['source_steps'] <= 2000000 and summary['costs']['source_monitor'] == 256000
        y = np.array([r['J'] for r in curves])
        x = np.array([r['budget_checkpoint'] for r in curves])
        # The runner integrates a left-continuous, sample-and-hold grid curve.
        # The separate source-final score is not a left endpoint and contributes no area.
        auc = float(np.dot(np.diff(x), y[:-1])/2000000)
        assert abs(auc-summary['AUC']) < 1e-8
        ev_return = values(episodes, 'discounted_return')
        info = dict(rounds=len(rows), source_steps=summary['source_steps'],
                    train_steps=summary['costs']['train'], monitoring_steps=256000,
                    evaluation_steps=int(values(episodes, 'steps').sum()),
                    monitor_final_return=summary['J'], monitor_auc=auc,
                    grid_last_return=curves[-2]['J'],
                    final_return=bootstrap_mean(ev_return, rng),
                    final_episodes=len(episodes), successes=int(values(episodes, 'success').sum()),
                    completion_steps=float(values(episodes, 'completion_time').mean()),
                    final_distance=float(values(episodes, 'mean_goal_distance').mean()),
                    final_collisions=float(values(episodes, 'collision_pairs').mean()),
                    scene_seeds=[int(e['scene_seed']) for e in episodes],
                    direction_epochs=config['direction_epochs'], train_episodes=config['train_episodes'],
                    optimizer_steps_by_module=summary['optimizer_steps_by_module'],
                    train_seconds=summary['train_seconds'],
                    metadata=manifest['metadata'],
                    diagnostics={k:describe(values(rows, k)) for k in fields},
                    first_round={k:rows[0].get(k) for k in ['train_J','fit_kl_before','policy_step_kl']},
                    source_grid=[{k:r[k] for k in ['budget_checkpoint','actor_source_steps','J','success_rate']} for r in curves])
        if name != 'old_base':
            assert info['scene_seeds'] == list(range(20262000, 20262100))
            audit = read_json(d/'final_evaluation/evaluation_audit.json')
            assert audit['status'] == 'completed' and audit['completed_episodes'] == 100
            assert audit['training_environment_steps'] == 0 and audit['optimizer_updates'] == 0
            assert audit['evaluation_steps'] == info['evaluation_steps']
            assert audit['checkpoint_sha256'] == sha(d/'checkpoints/final.pt')
            info['evaluation_audit'] = audit
            zero = sum(r['fit_selected_zero_update'] == 'True' for r in rows)
            assert zero == summary['selected_zero_update_rounds']
            info['zero_update_rounds'] = zero
            info['actor_changed_rounds'] = summary['actor_changed_rounds']
            info['negative_direction_score_rounds'] = int((values(rows,'direction_train_B_label')<0).sum())
            info['near_constant_critic_rounds'] = int((values(rows,'critic_postfit_value_variance')<1e-6).sum())
            info['selected_no_worse_than_initial'] = all(float(r['fit_selected_train_kl']) <=
                float(r['fit_initial_train_kl']) + 1e-12 + 1e-6*float(r['fit_initial_train_kl']) for r in rows)
            assert info['selected_no_worse_than_initial']
            info['intermediate_selection_rounds'] = sum(0<int(r['fit_selected_epoch'])<16 for r in rows)
            for r in rows:
                s, s0 = float(r['actor_lr_source_steps']), float(r['actor_lr_start_source_steps'])
                expected = .0003 if name.endswith('fixed') else .0003+(.00003-.0003)*np.clip((s-s0)/(2000000-s0),0,1)
                assert np.isclose(float(r['actor_lr_used']), expected, rtol=1e-12, atol=1e-15)
                assert float(r['fit_actor_lr']) == float(r['actor_lr_used'])
            info['lr_formula_and_fit_lr_verified'] = True
        result['runs'][name] = info
        data[name] = dict(rows=rows, episodes=episodes, curves=curves, config=config)
    for n in (16,32,64):
        a,b = (data[f'n{n}_{s}']['episodes'] for s in ('fixed','linear'))
        amap = {int(e['scene_seed']):float(e['discounted_return']) for e in a}
        bmap = {int(e['scene_seed']):float(e['discounted_return']) for e in b}
        assert amap.keys() == bmap.keys()
        diff = np.array([bmap[k]-amap[k] for k in sorted(amap)])
        result['paired_lr_contrasts'][f'n{n}_linear_minus_fixed'] = bootstrap_mean(diff, rng)
    old, new = data['old_base']['config'], data['n16_fixed']['config']
    result['old_vs_n16_fixed_config_diff'] = {k:[old.get(k),new.get(k)] for k in sorted(set(old)|set(new))
        if k!='current_config' and old.get(k)!=new.get(k)}
    result['new_total_source_steps'] = sum(result['runs'][n]['source_steps'] for n in NAMES)
    result['new_total_final_evaluation_steps'] = sum(result['runs'][n]['evaluation_steps'] for n in NAMES)
    result['mappo_reference_present'] = (SUITE/'mappo_reference_evaluation').exists()
    # Extract saved metadata only. Do not construct actors or execute environments.
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    import torch
    for name in NAMES:
        checkpoint = PATHS[name]/'N-C_source_learning/full/AN_s40/checkpoints/final.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        result['runs'][name]['initial_model_sha256'] = state['initial_model_sha256']
        result['runs'][name]['schedule_state'] = state['actor_lr_schedule_state']
        del state
    hashes = [json.dumps(result['runs'][n]['initial_model_sha256'],sort_keys=True) for n in NAMES]
    assert len(set(hashes)) == 1
    result['six_initial_models_identical'] = True
    for n in (16,32,64):
        a,b=(data[f'n{n}_{s}']['rows'][0] for s in ('fixed','linear'))
        for k in ['train_J','fit_initial_train_kl','fit_selected_train_kl','old_to_actor_kl_fp64']:
            assert a[k] == b[k], (n,k)
    result['first_round_fixed_linear_metrics_identical'] = True
    draw(data, result)
    assert all(sha(NAV/p)==digest for p,digest in raw_hashes.items())
    result['raw_files_unchanged_count'] = len(raw_hashes)
    (HERE/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name, info in result['runs'].items():
        print(name, {k:info[k] for k in ['rounds','monitor_final_return','monitor_auc','final_return','successes']})
    print('paired',result['paired_lr_contrasts'])
    print('verified',result['raw_files_unchanged_count'],'raw files; total source',result['new_total_source_steps'])


def draw(data, result):
    colors = dict(zip(NAMES, ['#1772ad','#71b6dc','#d36b16','#edb376','#258449','#8ac1a0']))
    colors['old_base']='#242424'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(3,2,figsize=(15,13),layout='constrained')
    ax=axes[0,0]
    for name in PATHS:
        grid=data[name]['curves'][:-1]  # source-final uses a different scene set
        ax.plot([r['budget_checkpoint']/1e6 for r in grid],[r['J'] for r in grid],
                color=colors[name],ls='--' if name.endswith('linear') else '-',
                lw=2 if name=='old_base' else 1.6,label=name)
    ax.set(title='Source monitoring: same 8 grid scenes',xlabel='Source joint steps (millions)',ylabel='Discounted return (higher is better)')
    ax.legend(ncol=2,fontsize=8)
    ax=axes[0,1]
    for i,name in enumerate(PATHS):
        stat=result['runs'][name]['final_return']; m=stat['mean'];lo,hi=stat['ci95_percentile']
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=colors[name],capsize=3)
        ax.text(hi+1,i,f'{m:.1f}',va='center',fontsize=9)
    ax.axhline(.5,color='#aaaaaa',ls=':')
    ax.set(yticks=range(7),yticklabels=['old_base (20 old scenes)']+NAMES,
           title='Final policy: 0 successes in every group',xlabel='Return with episode-bootstrap 95% interval')
    ax.invert_yaxis();ax.set_xlim(-162,-87)
    for name in NAMES:
        rows=data[name]['rows'];x=values(rows,'source_steps')/1e6
        axes[1,0].plot(x,values(rows,'critic_postfit_explained_variance'),color=colors[name],
                       lw=1.2,ls='--' if name.endswith('linear') else '-')
    axes[1,0].set(title='Critic: full-batch explained variance after fitting',xlabel='Source joint steps (millions)',ylabel='Explained variance')
    axes[1,0].axhline(0,color='#777777',lw=.6)
    for name in PATHS:
        rows=data[name]['rows'];x=values(rows,'source_steps')/1e6
        axes[1,1].plot(x,values(rows,'policy_entropy'),color=colors[name],
                       lw=1.3,ls='--' if name.endswith('linear') else '-')
    axes[1,1].axhline(np.log(5),color='#999999',ls=':',label='uniform log(5)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].set(title='Actor remains closer to uniform than old baseline',xlabel='Source joint steps (millions)',ylabel='Training-batch policy entropy')
    target=[result['runs'][n]['diagnostics']['fit_kl_before' if n=='old_base' else 'fit_initial_train_kl']['median'] for n in PATHS]
    axes[2,0].bar(range(7),target,color=[colors[n] for n in PATHS])
    axes[2,0].set(xticks=range(7),xticklabels=list(PATHS),yscale='log',
                  title='Median target-to-old KL per round',ylabel='KL (old: FP32; new: normalized FP64)')
    axes[2,0].tick_params(axis='x',labelrotation=25)
    zero=[result['runs'][n]['zero_update_rounds']/result['runs'][n]['rounds']*100 for n in NAMES]
    axes[2,1].bar(range(6),zero,color=[colors[n] for n in NAMES])
    for i,n in enumerate(NAMES):
        info=result['runs'][n]
        axes[2,1].text(i,zero[i]+.5,f"{info['zero_update_rounds']}/{info['rounds']}",ha='center',fontsize=9)
    axes[2,1].set(xticks=range(6),xticklabels=NAMES,ylim=(0,23),
                  title='Zero-update selection does not explain all weak runs',ylabel='Rounds selecting old Actor (%)')
    axes[2,1].tick_params(axis='x',labelrotation=25)
    for ax in axes.flat:ax.grid(axis='y',alpha=.16)
    fig.suptitle('AN 2M runs: batch size and Actor learning-rate schedule\nOne training seed (40); old baseline has 16 direction epochs, new runs have 4',fontsize=15)
    fig.savefig(HERE/'comparison.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
