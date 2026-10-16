"""Read-only audit of completed D1/D2 CSVs; writes derived JSON and a static figure."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent
WORKSPACE = next(p for p in OUT.parents if (p/'manifold_project').is_dir())


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def f(row,key):return float(row[key])


def paired(left,right,seed=20261015):
    delta=np.asarray(left)-np.asarray(right)
    rng=np.random.default_rng(seed)
    ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975])
    return dict(mean=float(delta.mean()),scene_bootstrap_95ci=ci.tolist(),n=len(delta))


def main():
    summary=dict(scope='Fixed-checkpoint diagnosis; one training seed and one diagnostic data seed',
                 bootstrap=dict(seed=20261015,repeats=20000,unit='paired evaluation scene'),runs={})
    raw={}
    code=(WORKSPACE/'manifold_project/experiments/cooperative_navigation/diagnose.py').read_bytes()
    lf=code.replace(b'\r\n',b'\n')
    hashes={hashlib.sha256(lf).hexdigest(),hashlib.sha256(lf.replace(b'\n',b'\r\n')).hexdigest()}
    for mode,expected_candidates in [('direction',4),('fit',6)]:
        folder=SOURCE/(mode+'_s40')
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        diag=read_csv(folder/'diagnostics.csv')
        trace=read_csv(folder/'fit_trace.csv')
        scores=read_csv(folder/'direction_scores.csv')
        eps=read_csv(folder/'evaluation_episodes.csv')
        config=json.loads((folder/'effective_runner_config.json').read_text(encoding='utf-8'))
        assert manifest['status']=='completed' and manifest['source_checkpoint_unchanged']
        assert manifest['source_actor_critic_frozen'] and manifest['probe_code_sha256'] in hashes
        tail=manifest['checkpoint'].replace('\\','/').split('manifold_project/',1)[1]
        checkpoint=WORKSPACE/'manifold_project'/tail
        if not checkpoint.exists():
            result_root=WORKSPACE/'manifold_project/experiments/results'
            recorded=checkpoint.relative_to(result_root).as_posix()
            moves=json.loads((result_root/'path_map.json').read_text(encoding='utf-8'))['moves']
            move=next(m for m in moves if recorded==m['from'] or recorded.startswith(m['from']+'/'))
            checkpoint=result_root/move['to']/Path(recorded).relative_to(move['from'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
        assert len(diag)==expected_candidates
        groups={name:sorted([e for e in eps if e['candidate']==name],key=lambda e:int(e['episode']))
                for name in {e['candidate'] for e in eps}}
        assert len(groups)==expected_candidates+1
        assert all([int(e['episode']) for e in group]==list(range(40)) for group in groups.values())
        values={name:[f(e,'J') for e in group] for name,group in groups.items()}
        base=np.array(values['old'])
        for row in diag:
            returns=np.array(values[row['candidate']])
            assert abs(returns.mean()-f(row,'candidate_return'))<1e-10
            assert abs((returns-base).mean()-f(row,'return_gain'))<1e-10
            assert abs(base.mean()-f(row,'old_return'))<1e-10
            assert abs(np.mean([f(e,'success') for e in groups[row['candidate']]])-f(row,'success_rate'))<1e-12
        for row in scores:
            assert abs(f(row,'B_label')-(2*f(row,'D_label')-f(row,'fisher_energy')))<1e-12
        fits=[]
        for epochs,lr in sorted({(r['direction_epochs'],r['actor_lr']) for r in trace}):
            series=sorted([r for r in trace if (r['direction_epochs'],r['actor_lr'])==(epochs,lr)],key=lambda r:int(r['epoch']))
            assert [int(r['epoch']) for r in series]==list(range(17))
            assert all(r['scope'] in ('initial','post_epoch') for r in series)
            first,last=series[0],series[-1]
            best=min(series,key=lambda r:f(r,'train_target_kl'))
            updates=16*int(np.ceil(manifest['batch_episodes']/config['minibatch_episodes']))
            assert int(last['optimizer_updates'])==updates
            selected=[r for r in diag if r['direction_epochs']==epochs and r['actor_lr']==lr]
            for row in selected:
                assert int(row['best_epoch'])==int(best['epoch'])
                ref=best if row['candidate'].endswith('_best') else last
                assert abs(f(row,'raw_actual_kl')-f(ref,'validation_actual_kl'))<1e-10
                assert abs(f(row,'final_train_target_kl')-f(last,'train_target_kl'))<1e-10
            fits.append(dict(direction_epochs=int(epochs),actor_lr=float(lr),updates=updates,
                initial_train=f(first,'train_target_kl'),initial_validation=f(first,'validation_target_kl'),
                final_train=f(last,'train_target_kl'),final_validation=f(last,'validation_target_kl'),
                best_epoch=int(best['epoch']),best_train=f(best,'train_target_kl'),best_validation=f(best,'validation_target_kl'),
                final_over_best_train=f(last,'train_target_kl')/f(best,'train_target_kl'),
                final_over_initial_train=f(last,'train_target_kl')/f(first,'train_target_kl'),
                final_over_initial_validation=f(last,'validation_target_kl')/f(first,'validation_target_kl'),
                first_epoch_over_initial=f(series[1],'train_target_kl')/f(first,'train_target_kl'),
                rising_epochs=sum(f(y,'train_target_kl')>f(x,'train_target_kl') for x,y in zip(series,series[1:]))))
        evaluation_steps=sum(int(e['episode_steps']) for e in eps)
        collection_steps=manifest['diagnostic_steps']-evaluation_steps
        assert collection_steps==(manifest['batch_episodes']+manifest['validation_episodes'])*400
        pairs={}
        if mode=='direction':
            common=f(diag[0],'common_kl')
            assert abs(common-min(manifest['requested_kl'],.95*min(f(r,'raw_actual_kl') for r in diag)))<1e-12
            match_error=max(abs(f(r,'matched_actual_kl')-common) for r in diag)
            assert match_error<1e-12
            pairs['target64_minus_target16']=paired(values['direction64_target'],values['direction16_target'])
            for epoch in (16,64):
                pairs[f'actor{epoch}_minus_target{epoch}']=paired(values[f'direction{epoch}_lr0.0003_final'],values[f'direction{epoch}_target'])
        else:
            common=match_error=None
            pairs['lr3e4_best_minus_final']=paired(values['direction16_lr0.0003_best'],values['direction16_lr0.0003_final'])
            for lr in ('0.0001','3e-05'):
                assert values[f'direction16_lr{lr}_best']==values[f'direction16_lr{lr}_final']
                pairs[f'lr{lr}_minus_lr3e4']=paired(values[f'direction16_lr{lr}_final'],values['direction16_lr0.0003_final'])
        run=dict(verified=True,manifest=manifest,record_counts=dict(candidates=len(diag),trace=len(trace),scores=len(scores),evaluation=len(eps)),
                 source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()},
                 code_matches_after_line_ending_normalization=True,common_kl=common,kl_match_max_abs_error=match_error,
                 collection_steps=collection_steps,evaluation_steps=evaluation_steps,fit_summary=fits,
                 pairwise_scene_comparisons=pairs)
        summary['runs'][mode]=run
        raw[mode]=dict(diag=diag,trace=trace,scores=scores)
    summary['total_diagnostic_steps']=sum(v['manifest']['diagnostic_steps'] for v in summary['runs'].values())
    (OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.5),constrained_layout=True)
    colors=['#2563eb','#d97706','#059669']
    ax=axes[0,0]
    for i,lr in enumerate(('0.0003','0.0001','3e-05')):
        rows=[r for r in raw['fit']['trace'] if r['actor_lr']==lr]
        x=[int(r['epoch']) for r in rows]
        for key,style in [('train_target_kl','-'),('validation_target_kl','--')]:
            ax.plot(x,[f(r,key)*1e6 for r in rows],style,color=colors[i],label=f'{lr} '+('train' if style=='-' else 'validation'))
    ax.set(title='D2: fixed-target Actor fitting',xlabel='Actor epoch',ylabel='KL(target || Actor), x 1e-6')
    ax.legend(fontsize=8,ncol=2)
    ax=axes[0,1]
    for i,split in enumerate(('train','validation')):
        rows=[r for r in raw['direction']['scores'] if r['split']==split]
        ax.plot([int(r['direction_epochs']) for r in rows],[f(r,'B_label') for r in rows],'o-',color=colors[i],label=split)
    ax.axhline(0,color='gray',lw=1)
    ax.set(title='D1: direction surrogate generalization',xlabel='Direction epoch',ylabel='B_label = 2D - E')
    ax.legend()
    ax=axes[1,0]
    for i,epoch in enumerate(('16','64')):
        rows=[r for r in raw['direction']['trace'] if r['direction_epochs']==epoch]
        for key,style in [('train_target_kl','-'),('validation_target_kl','--')]:
            scale=f(rows[0],key)
            ax.plot([int(r['epoch']) for r in rows],[f(r,key)/scale for r in rows],style,color=colors[i],label=f'direction {epoch} '+('train' if style=='-' else 'validation'))
    ax.axhline(1,color='gray',lw=1)
    ax.set(title='D1: Actor residual relative to initialization',xlabel='Actor epoch',ylabel='Target KL / initial target KL')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    rows=raw['direction']['diag']
    labels=['16 target','16 Actor','64 target','64 Actor']
    for i,row in enumerate(rows):
        mean=f(row,'return_gain');lo=f(row,'gain_scene_ci_low');hi=f(row,'gain_scene_ci_high')
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color=colors[i//2])
    ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='gray',lw=1)
    ax.set(title='D1: return gain at common KL = 3.703e-5',xlabel='Paired mean gain with 95% scene bootstrap interval')
    fig.suptitle('Frozen-checkpoint diagnosis: one data seed, 40 paired scenes; no performance claim',fontsize=12)
    fig.savefig(OUT/'diagnosis.png',dpi=180)
    plt.close(fig)
    print(json.dumps({mode:{k:v for k,v in run.items() if k in ('verified','record_counts','common_kl','kl_match_max_abs_error','fit_summary','pairwise_scene_comparisons')}
                      for mode,run in summary['runs'].items()},indent=2))


if __name__=='__main__':main()
