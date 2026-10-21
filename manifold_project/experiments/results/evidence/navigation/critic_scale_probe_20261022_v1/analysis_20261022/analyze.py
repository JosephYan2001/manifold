"""Audit saved Critic data/candidates and summarize fixed-budget generalization.

No environment construction, optimizer steps, model selection or data collection.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = next(p for p in HERE.parents if (p/'manifold_project/experiments').is_dir())
sys.path.insert(0, str(REPO))
from manifold_project.experiments.cooperative_navigation.models import Critic
from manifold_project.experiments.cooperative_navigation.training.critic_probe import (
    ScaledCritic, file_hash, validate_packet, normalization)
from manifold_project.experiments.cooperative_navigation.training.fixed_target import fingerprint
from manifold_project.experiments.cooperative_navigation.training.ppo import value_targets

NAMES = ['fresh_raw', 'fresh_fixed_normalized', 'checkpoint_raw', 'checkpoint_fixed_normalized']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def field(rows, key):
    return np.array([float(r[key]) for r in rows])


def stats(y, prediction, w):
    mean = float((y*w).sum(1).mean())
    error = y-prediction
    bias = float((error*w).sum(1).mean())
    variance = float(((y-mean)**2*w).sum(1).mean())
    return dict(mse=float((error**2*w).sum(1).mean()),
                explained_variance=1-float(((error-bias)**2*w).sum(1).mean())/variance,
                target_mean=mean, target_variance=variance,
                value_mean=float((prediction*w).sum(1).mean()),
                value_variance=float(((prediction-(prediction*w).sum(1).mean())**2*w).sum(1).mean()))


def boot(delta, rng):
    samples=delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(1)
    return dict(mean=float(delta.mean()), ci95_percentile=np.quantile(samples,[.025,.975]).tolist(),
                positive_episodes=int((delta>0).sum()), negative_episodes=int((delta<0).sum()))


def main():
    torch.set_num_threads(1)
    originals = [p for p in ROOT.rglob('*') if p.is_file() and HERE not in p.parents]
    hashes={str(p.relative_to(ROOT)):file_hash(p) for p in originals}
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    trace=read_csv(ROOT/'critic_trace.csv')
    final=read_csv(ROOT/'diagnostics.csv')
    episode_rows=read_csv(ROOT/'episode_errors.csv')
    packet=torch.load(ROOT/'critic_data.pt',map_location='cpu',weights_only=False)
    validate_packet(packet)
    assert manifest['critic_data_sha256']==file_hash(ROOT/'critic_data.pt')
    assert packet['frozen_sha256']==manifest['frozen_data_sha256']
    assert normalization(packet['batch'],packet['config'])==packet['normalization']
    assert len(trace)==324 and len(final)==4 and len(episode_rows)==256
    assert manifest['critic_optimizer_updates']==5120 and manifest['collection_steps']==25600
    assert all(manifest[k]==0 for k in ('actor_optimizer_updates','direction_optimizer_updates','evaluation_steps'))
    ledger=[json.loads(l) for l in (ROOT/'collection_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    completed=[d['record'] for d in ledger if d['stream']=='sampling']
    assert len(completed)==64 and completed[-1]['used']==25600
    config=dict(packet['config'],device='cpu')
    source=ROOT.parent/'navigation_batch_lr_2m_s40_20261019_v1/n16_fixed/N-C_source_learning/full/AN_s40/checkpoints/final.pt'
    assert file_hash(source)==manifest['input_sha256']
    source_state=torch.load(source,map_location='cpu',weights_only=False)
    assert fingerprint(source_state['actor'])==packet['source']['actor_sha256']
    assert fingerprint(source_state['critic'])==packet['source']['critic_sha256']
    result=dict(date='2026-10-22',primary_endpoint_epoch=80,source_files_sha256=hashes,
                source_checkpoint_sha256=file_hash(source),model_and_packet_validation=True,
                maximum_collection_steps=25600,actual_collection_steps=25600,critic_updates=5120,
                added_analysis_environment_steps=0,added_analysis_optimizer_updates=0,
                elapsed_run_seconds=manifest['elapsed_seconds'],normalization=packet['normalization'],
                splits={},candidates={},paired_validation_differences={},
                interval_scope='Post-hoc descriptive bootstrap of 32 paired validation episodes, conditional on these fixed fits; no training-seed uncertainty or multiplicity correction',
                bootstrap_repeats=20000,bootstrap_seed=20261022)
    predictions={}
    for name in NAMES:
        rs=[r for r in trace if r['candidate']==name]
        assert [int(r['epoch']) for r in rs]==list(range(81))
        assert [int(r['optimizer_updates']) for r in rs]==[16*i for i in range(81)]
        candidate=torch.load(ROOT/'candidates'/f'{name}.pt',map_location='cpu',weights_only=False)
        assert candidate['frozen_data_sha256']==packet['frozen_sha256'] and candidate['final_epoch']==80
        model=ScaledCritic(Critic(packet['batch']['state'].shape[-1],config))
        model.load_state_dict(candidate['state_dict']);model.eval().requires_grad_(False)
        row=next(r for r in final if r['candidate']==name)
        result['candidates'][name]=dict(recomputed={},max_absolute_logged_mse_difference=0.,
            best_validation_mse_for_description_only=float(field(rs,'validation_mse').min()),
            descriptive_best_validation_epoch=int(rs[int(field(rs,'validation_mse').argmin())]['epoch']))
        predictions[name]={}
        for split,key in [('train','batch'),('validation','validation')]:
            b=packet[key]
            assert b['lengths'].tolist()==[400]*32 and int(b['truncated'].sum())==32 and int(b['terminated'].sum())==0
            w=np.power(config['gamma'],np.arange(400));w/=w.sum()
            with torch.no_grad():
                pred=model(b['state']).double().numpy()
            predictions[name][split]=pred
            y=b['value_target'].double().numpy()
            measured=stats(y,pred,w)
            result['candidates'][name]['recomputed'][split]=measured
            for k,v in measured.items():
                expected=float(row[f'{split}_{k}'])
                assert np.isclose(v,expected,atol=.05,rtol=5e-5),(name,split,k,v,expected)
            diff=abs(measured['mse']-float(row[f'{split}_mse']))
            result['candidates'][name]['max_absolute_logged_mse_difference']=max(diff,result['candidates'][name]['max_absolute_logged_mse_difference'])
            stored=np.array([float(r['raw_mse']) for r in episode_rows if r['candidate']==name and r['split']==split])
            actual=((y-pred)**2*w).sum(1)
            assert np.allclose(stored,actual,atol=.05,rtol=5e-5)
            assert np.isclose(stored.mean(),float(row[f'{split}_mse']),atol=1e-8)
    old=Critic(packet['batch']['state'].shape[-1],config).eval().requires_grad_(False)
    old.load_state_dict(packet['label_critic'])
    for split,key in [('train','batch'),('validation','validation')]:
        b=packet[key]
        values,target,gae=value_targets(b,old,config)
        target_diff=float((target-b['value_target']).abs().max())
        gae_diff=float((gae-b['frozen_gae']).abs().max())
        assert target_diff<.002 and gae_diff<.002
        s=b['state'].double().numpy();y=b['value_target'].double().numpy()
        w=np.power(config['gamma'],np.arange(400));w/=w.sum()
        means=(y*w).sum(1);mean=means.mean();variance=((y-mean)**2*w).sum(1).mean()
        a=s[:,:,:8].reshape(32,400,4,2);l=s[:,:,16:].reshape(32,400,4,2)
        negative_distance=-np.linalg.norm(a[:,:,:,None,:]-l[:,:,None,:,:],axis=-1).min(axis=2).mean(axis=2)
        xm=(negative_distance*w).sum(1).mean();xvar=((negative_distance-xm)**2*w).sum(1).mean()
        target_stats={}
        for label,z in [('current_return',y),('old_value_plus_gae',(b['frozen_old_values']+b['frozen_gae']).double().numpy())]:
            zm=(z*w).sum(1).mean();zv=((z-zm)**2*w).sum(1).mean()
            correlation=((negative_distance-xm)*(z-zm)*w).sum(1).mean()/np.sqrt(xvar*zv)
            target_stats[label]=dict(mean=float(zm),variance=float(zv),correlation_with_negative_goal_distance=float(correlation))
        result['splits'][split]=dict(target_max_abs_recompute_difference=target_diff,gae_max_abs_recompute_difference=gae_diff,
            constant_train_mean_mse=float(((y-packet['normalization']['mean'])**2*w).sum(1).mean()),
            between_episode_target_variance_fraction=float(np.var(means)/variance),
            within_episode_target_variance_fraction=float(((y-means[:,None])**2*w).sum(1).mean()/variance),
            target_definition_descriptive_statistics=target_stats)
    rng=np.random.default_rng(20261022)
    ep={n:np.array([float(r['raw_mse']) for r in episode_rows if r['candidate']==n and r['split']=='validation']) for n in NAMES}
    y=packet['validation']['value_target'].double().numpy()
    constant=((y-packet['normalization']['mean'])**2*w).sum(1)
    for lhs,rhs in [('fresh_fixed_normalized','fresh_raw'),('checkpoint_fixed_normalized','checkpoint_raw')]:
        result['paired_validation_differences'][lhs+'_minus_'+rhs]=boot(ep[lhs]-ep[rhs],rng)
    for name in ['fresh_raw','fresh_fixed_normalized']:
        result['paired_validation_differences'][name+'_minus_train_mean_constant']=boot(ep[name]-constant,rng)
    for c in summary['comparisons']:
        raw=next(r for r in final if r['candidate']==c['start']+'_raw')
        norm=next(r for r in final if r['candidate']==c['start']+'_fixed_normalized')
        assert np.isclose(1-float(norm['validation_mse'])/float(raw['validation_mse']),c['validation_mse_relative_reduction'])
        assert not c['descriptive_screen_passed']
    plot(trace,result)
    for rel,h in hashes.items():
        assert file_hash(ROOT/rel)==h,rel
    (HERE/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(original_files_verified=len(hashes),source_and_packet_verified=True,
          paired_validation_differences=result['paired_validation_differences'],splits=result['splits']),ensure_ascii=False,indent=2))


def plot(trace,result):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'fresh_raw':'#2166ac','fresh_fixed_normalized':'#d6604d'}
    labels={'fresh_raw':'Fresh / raw','fresh_fixed_normalized':'Fresh / fixed normalized'}
    for name in colors:
        rows=[r for r in trace if r['candidate']==name];x=field(rows,'epoch')
        for ax,split in [(axes[0,0],'train'),(axes[0,1],'validation')]:
            ax.plot(x,field(rows,split+'_mse'),label=labels[name],color=colors[name])
            ax.set_yscale('log');ax.set_ylabel('MSE in original return units')
        axes[1,0].plot(x,field(rows,'train_explained_variance'),color=colors[name],linestyle='--',label=labels[name]+' / train')
        axes[1,0].plot(x,field(rows,'validation_explained_variance'),color=colors[name],label=labels[name]+' / validation')
        axes[1,1].plot(x,field(rows,'validation_second_tanh_saturation')*100,color=colors[name],label=labels[name])
    for ax,split in [(axes[0,0],'train'),(axes[0,1],'validation')]:
        ax.axhline(result['splits'][split]['constant_train_mean_mse'],color='#444444',linestyle=':',label='Train-mean constant')
    axes[1,0].axhline(0,color='#444444',linestyle=':');axes[1,0].set_ylabel('Explained variance')
    axes[1,1].axhline(100,color='#444444',linestyle=':',label='Checkpoint starts (both)')
    axes[1,1].set_ylabel('Second-layer |tanh| > 0.95 (%)')
    titles=['Training fit: normalization learns much faster','Independent episodes: fit does not generalize',
            'Training and validation separate','Saturation improves without validation benefit']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=11);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Critic scale probe: one fixed dataset, final epoch 80 is the endpoint',fontsize=13)
    fig.savefig(HERE/'critic_fit_and_generalization.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
