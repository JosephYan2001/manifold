"""Audit frozen-checkpoint follow-ups without sampling or modifying source CSVs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
RESULTS=ROOT.parent
WORKSPACE=next(p for p in OUT.parents if (p/'manifold_project').is_dir())


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def number(row,key):return float(row[key])


def main():
    folders={
        'fit_new':ROOT/'fit_s41_42',
        'n16':ROOT/'direction_nxu_s40/n16_init40',
        'n64':ROOT/'direction_nxu_s40/n64_init40',
        'fit_old':RESULTS/'navigation_fixed_diagnosis_20261014_v1/fit_s40',
        'direction_old':RESULTS/'navigation_fixed_diagnosis_20261014_v1/direction_s40'}
    summary=dict(scope='Fixed checkpoints; diagnostic data seeds are not independent training seeds',runs={})
    raw={}
    code=(WORKSPACE/'manifold_project/experiments/cooperative_navigation/diagnose.py').read_bytes().replace(b'\r\n',b'\n')
    code_hashes={hashlib.sha256(code).hexdigest(),hashlib.sha256(code.replace(b'\n',b'\r\n')).hexdigest()}
    for name,path in folders.items():
        manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8-sig'))
        cfg=json.loads((path/'effective_runner_config.json').read_text(encoding='utf-8-sig'))
        diag,trace,scores,eps=[csv_rows(path/f'{file}.csv') for file in ('diagnostics','fit_trace','direction_scores','evaluation_episodes')]
        assert manifest['status']=='completed' and manifest['source_checkpoint_unchanged'] and manifest['source_actor_critic_frozen']
        assert manifest['probe_code_sha256'] in code_hashes
        relative=manifest['checkpoint'].replace('\\','/').split('manifold_project/',1)[1]
        checkpoint=WORKSPACE/'manifold_project'/relative
        if not checkpoint.exists():
            result_root=WORKSPACE/'manifold_project/experiments/results'
            recorded=checkpoint.relative_to(result_root).as_posix()
            moves=json.loads((result_root/'path_map.json').read_text(encoding='utf-8'))['moves']
            move=next(m for m in moves if recorded==m['from'] or recorded.startswith(m['from']+'/'))
            checkpoint=result_root/move['to']/Path(recorded).relative_to(move['from'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
        count=manifest['evaluation_episodes']
        expected=6 if manifest['mode']=='fit' else 4
        assert len(diag)==expected*len(manifest['diagnostic_seeds'])
        assert len(eps)==(expected+1)*count*len(manifest['diagnostic_seeds'])
        groups={}
        for row in eps:groups.setdefault((row['data_seed'],row['candidate']),[]).append(row)
        for group in groups.values():
            group.sort(key=lambda r:int(r['episode']))
            assert [int(r['episode']) for r in group]==list(range(count))
        max_match_error=0.
        for row in diag:
            group=groups[(row['data_seed'],row['candidate'])]
            old=groups[(row['data_seed'],'old')]
            gain=np.array([number(x,'J')-number(y,'J') for x,y in zip(group,old)])
            assert abs(gain.mean()-number(row,'return_gain'))<1e-10
            assert abs(np.mean([number(x,'J') for x in group])-number(row,'candidate_return'))<1e-10
            assert abs(np.mean([number(x,'success') for x in group])-number(row,'success_rate'))<1e-12
            if manifest['mode']=='direction':
                common=min(manifest['requested_kl'],.95*min(number(r,'raw_actual_kl') for r in diag if r['data_seed']==row['data_seed']))
                assert abs(common-number(row,'common_kl'))<1e-12
                max_match_error=max(max_match_error,abs(common-number(row,'matched_actual_kl')))
        assert max_match_error<1e-12
        fit_summary=[]
        for key in sorted({(r['data_seed'],r['direction_epochs'],r['actor_lr']) for r in trace}):
            rows=[r for r in trace if (r['data_seed'],r['direction_epochs'],r['actor_lr'])==key]
            epochs=[r for r in rows if r['scope'] in ('initial','post_epoch')]
            steps=[r for r in rows if r['scope']=='post_optimizer_step']
            assert [int(r['epoch']) for r in epochs]==list(range(manifest['fit_epochs']+1))
            best=min(epochs,key=lambda r:number(r,'train_target_kl'))
            best_any=min(rows,key=lambda r:number(r,'train_target_kl'))
            first,last=epochs[0],epochs[-1]
            updates=manifest['fit_epochs']*math.ceil(manifest['batch_episodes']/cfg['minibatch_episodes'])
            assert int(last['optimizer_updates'])==updates
            if manifest['fit_trace_every_updates']==1:
                assert [int(r['optimizer_updates']) for r in steps]==list(range(1,updates+1))
                by_step={r['optimizer_updates']:r for r in steps}
                for epoch in epochs[1:]:
                    for metric in ('train_target_kl','validation_target_kl','validation_actual_kl'):
                        assert number(epoch,metric)==number(by_step[epoch['optimizer_updates']],metric)
            for row in diag:
                if (row['data_seed'],row['direction_epochs'],row['actor_lr'])==key:
                    assert int(row['best_epoch'])==int(best['epoch'])
                    chosen=best if row['candidate'].endswith('_best') else last
                    assert abs(number(row,'raw_actual_kl')-number(chosen,'validation_actual_kl'))<1e-10
                    assert abs(number(row,'final_train_target_kl')-number(last,'train_target_kl'))<1e-12
            initial=number(first,'train_target_kl')
            fit_summary.append(dict(data_seed=int(key[0]),direction_epochs=int(key[1]),actor_lr=float(key[2]),
                initial_train=initial,initial_validation=number(first,'validation_target_kl'),
                final_train=number(last,'train_target_kl'),final_validation=number(last,'validation_target_kl'),
                final_over_initial=number(last,'train_target_kl')/initial,
                best_epoch=int(best['epoch']),best_epoch_train=number(best,'train_target_kl'),
                best_observed_update=int(best_any['optimizer_updates']),best_observed_train=number(best_any,'train_target_kl'),
                first_step_over_initial=number(steps[0],'train_target_kl')/initial if steps else None,
                first_step_actual_kl=number(steps[0],'validation_actual_kl') if steps else None,
                steps_worse_than_initial=sum(number(r,'train_target_kl')>initial for r in steps) if steps else None,
                optimizer_updates=updates))
        evaluation_steps=sum(int(r['episode_steps']) for r in eps)
        collected=(manifest['batch_episodes']+manifest['validation_episodes'])*400*len(manifest['diagnostic_seeds'])
        assert evaluation_steps+collected==manifest['diagnostic_steps']
        for r in scores:
            assert abs(number(r,'B_label')-(2*number(r,'D_label')-number(r,'fisher_energy')))<1e-12
        summary['runs'][name]=dict(verified=True,manifest=manifest,
            record_counts=dict(diagnostics=len(diag),trace=len(trace),scores=len(scores),evaluation=len(eps)),
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()},
            fit_summary=fit_summary,evaluation_steps=evaluation_steps,collection_steps=collected,
            max_kl_match_error=max_match_error,
            successful_evaluations=[dict(data_seed=r['data_seed'],candidate=r['candidate'],episode=int(r['episode']),
                episode_steps=int(r['episode_steps'])) for r in eps if number(r,'success')>0])
        raw[name]=dict(diag=diag,trace=trace,scores=scores,episodes=eps)
    # The repeated N16 cell is a reproducibility check, not a new independent replicate.
    differences={}
    for table in ('diag','trace','scores','episodes'):
        old,new=raw['direction_old'][table],raw['n16'][table]
        assert len(old)==len(new)
        maximum=0.
        for a,b in zip(old,new):
            for key in a.keys() & b.keys():
                if key=='init_seed':continue
                try:
                    difference=abs(float(a[key])-float(b[key]))
                    assert difference<=1e-12,(table,key,difference)
                    maximum=max(maximum,difference)
                except ValueError:assert a[key]==b[key],(table,key)
        differences[table]=maximum
    summary['n16_repeat_max_abs_errors']=differences
    direction_matrix=[]
    for n in (16,64):
        for r in raw[f'n{n}']['scores']:
            if r['split']=='validation':
                d,e=number(r,'D_label'),number(r,'fisher_energy')
                direction_matrix.append(dict(N=n,U=int(r['optimizer_updates']),B=number(r,'B_label'),D=d,E=e,
                    normalized_label_alignment=d/math.sqrt(e),tanh_saturation=number(r,'tanh_saturation'),
                    effective_bound_fraction=number(r,'effective_bound_fraction')))
    summary['direction_validation_matrix']=direction_matrix
    summary['new_diagnostic_steps']=sum(summary['runs'][n]['manifest']['diagnostic_steps'] for n in ('fit_new','n16','n64'))
    summary['new_steps_with_previous_fixed_diagnosis']=summary['new_diagnostic_steps']+537600
    (OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    rates=[.0003,.0001,.00003];colors=['#dc2626','#2563eb','#059669']
    for ax,seed in zip(axes[0],(41,42)):
        for lr,color in zip(rates,colors):
            rows=[r for r in raw['fit_new']['trace'] if int(r['data_seed'])==seed and number(r,'actor_lr')==lr
                  and r['scope'] in ('initial','post_optimizer_step')]
            initial=number(rows[0],'train_target_kl')
            ax.plot([int(r['optimizer_updates']) for r in rows],[number(r,'train_target_kl')/initial for r in rows],color=color,lw=1,label=f'lr {lr:g}')
        ax.axhline(1,color='black',ls='--',lw=1,label='old Actor')
        ax.set_yscale('log');ax.set(title=f'D2 data seed {seed}: per-step fit residual',xlabel='Actor optimizer update',ylabel='KL(target || Actor) / initial KL')
        ax.legend(fontsize=9)
    ax=axes[1,0]
    grid=np.array([[next(r['B'] for r in direction_matrix if r['N']==n and r['U']==u) for u in (128,512)] for n in (16,64)])
    im=ax.imshow(grid,cmap='RdBu',vmin=-.07,vmax=.07,aspect='auto')
    ax.set_xticks([0,1],['U = 128','U = 512']);ax.set_yticks([0,1],['N = 16','N = 64'])
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{grid[i,j]:+.6f}',ha='center',va='center',fontsize=14,color='white' if abs(grid[i,j])>.04 else 'black')
    ax.set_title('A0: direction validation B = 2D - E\nOne data seed and initialization; no uncertainty estimate')
    fig.colorbar(im,ax=ax,shrink=.7,label='Frozen-Critic label surrogate')
    ax=axes[1,1]
    cells=[('fit_old',40,16),('fit_new',41,16),('fit_new',42,16),('n64',40,4),('n64',40,16)]
    for i,lr in enumerate(rates):
        x=[];y=[]
        for j,(name,seed,epochs) in enumerate(cells):
            found=[r for r in summary['runs'][name]['fit_summary'] if r['data_seed']==seed and r['direction_epochs']==epochs and r['actor_lr']==lr]
            if found:x.append(j+(i-1)*.23);y.append(found[0]['final_over_initial'])
        bars=ax.bar(x,y,width=.22,color=colors[i],label=f'lr {lr:g}')
        for bar,value in zip(bars,y):ax.text(bar.get_x()+bar.get_width()/2,value*1.08,f'{value:.2f}',ha='center',va='bottom',fontsize=8,rotation=90)
    ax.axhline(1,color='black',ls='--',lw=1);ax.set_yscale('log');ax.set_ylim(.1,28)
    ax.set_xticks(range(5),['A2\nseed 40','A2\nseed 41','A2\nseed 42','A0 N64\nU128','A0 N64\nU512'])
    ax.set(title='Final Actor fit residual relative to old Actor',ylabel='Final target KL / initial target KL')
    ax.legend(fontsize=8)
    fig.suptitle('Follow-up diagnosis: larger batches improve the surrogate; small targets expose Actor overshoot',fontsize=13)
    fig.savefig(OUT/'followup_diagnosis.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(new_steps=summary['new_diagnostic_steps'],repeat_errors=differences,
        direction_matrix=direction_matrix,fit=summary['runs']['fit_new']['fit_summary'],
        successes=summary['runs']['fit_new']['successful_evaluations']),indent=2))


if __name__=='__main__':main()
