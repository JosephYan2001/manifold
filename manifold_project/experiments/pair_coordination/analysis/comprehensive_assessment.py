"""Read-only audit of the 2026-09-18 evidence; emit one Markdown report and figure.

Run from repository root:
python -m manifold_project.experiments.pair_coordination.analysis.comprehensive_assessment
Original CSV, events, checkpoints and manifests are never rewritten.
"""
from collections import Counter, defaultdict
from itertools import product
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ..envs.pair_coordination import PairConfig
from ..evaluation.exact_pair import closed_form_expected_return
from ..experiments.reporting import bootstrap, binomial_interval, paired_direction_summary
from ..models.mlp import MLPActor

ROOT = Path(__file__).resolve().parents[1]/'results'
TRAIN = ROOT/'suite_09_论文比较与消融_10种子'
MECHANISM = ROOT/'suite_10_论文方向估计与运输_20数据种子'
CHECK = ROOT/'suite_11_论文检查规则_200数据种子'
LABELS = {'ours':'完整方法','sampled':'采样平方','pg':'PG','no_direction_check':'去方向检查',
          'no_return_check':'去回报检查','fit_quarter':'16 轮拟合'}
CONDITIONS = list(LABELS)


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mean(rows,key):
    return float(np.mean([float(r[key]) for r in rows]))


def true(x):
    return x is True or x == 'True'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def interval(stats,fmt='.6f'):
    return f"{stats['mean']:{fmt}} [{stats['ci_low']:{fmt}}, {stats['ci_high']:{fmt}}]"


def transfer_assessment():
    """Reproduce the saved frozen-policy evidence without writing its artifacts."""
    from ..evaluate_transfer import load_models, evaluate, summarize, digest, ROOT as PAIR_ROOT
    directory = ROOT/'pair_frozen_transfer_v1'
    manifest = js(directory/'manifest.json')
    models, hashes = load_models(TRAIN)
    assert manifest['status'] == 'complete' and manifest['records'] == 540
    assert manifest['input_sha256'] == hashes
    for filename, expected in manifest['code_sha256'].items():
        assert digest(PAIR_ROOT/filename) == expected, filename
    rows = evaluate(models)
    stats = summarize(rows)
    for expected_rows, filename in ((rows, 'records.csv'), (stats, 'summary.csv')):
        actual_rows = read(directory/filename)
        assert len(expected_rows) == len(actual_rows)
        for expected, actual in zip(expected_rows, actual_rows):
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    assert np.isclose(value, float(actual[key]), atol=1e-14, rtol=0), (filename, key)
                else:
                    assert value == actual[key], (filename, key)
    names = list(dict.fromkeys(r['target'] for r in rows))
    def stat(name, method, kind='method'):
        return next(s for s in stats if s['target'] == name and s['condition'] == method
                    and s['kind'] == kind and s['metric'] == 'per_agent_return')
    for (method, seed), _ in models.items():
        values = [r['per_agent_return'] for r in rows if r['condition'] == method and r['seed'] == seed
                  and r['target'] in ('source', 'agents_2', 'agents_6', 'agents_8')]
        assert np.allclose(values, values[0], atol=1e-14, rtol=0)
    differences = {c: max(abs(s['mean']) for s in stats if s['condition'] == c
                   and s['kind'] == 'paired_ours_minus_comparator' and s['metric'] == 'per_agent_return')
                   for c in CONDITIONS if c != 'ours'}
    gap = max(r['optimality_gap_per_agent'] for r in rows if r['condition'] == 'ours')
    print('Frozen-transfer audit passed: 540 records, 297 summary rows, input/code hashes and population invariance.')
    return [
        '## 7.1 新增 P-T：最终冻结策略的零样本评价',
        '已完成6方法×10原训练种子×9配置=540条精确评价，其中60条源参照、480条目标记录。复用suite_09的final，不新增训练种子或回合，不更新参数、不选best。逐条重算records.csv和297行summary.csv，输入文件与评价代码哈希均通过核对。此为事后增加的评价协议，不称原确认实验的预注册迁移检验。',
        table(['配置','完整方法人均回报','PG人均回报','ours−PG及95%配对区间'], [
            [name, f"{stat(name,'ours')['mean']:.9f}", f"{stat(name,'pg')['mean']:.9f}",
             interval(stat(name,'pg','paired_ours_minus_comparator'), '.9f')] for name in names]),
        '所有8个目标配置的ours−PG人均回报差均为正，逐目标95%配对区间均高于零。目标差值约0.0001054～0.0005434；原源差值为0.0003244。区间以10个训练seed重采样，未作多重比较校正；共享模型和目标结构使这些配置并非独立证据，不能把480条目标记录当独立训练重复。',
        f"完整方法在全部模型/配置中的最大目标及源人均最优差距为{gap:.3e}。跨配置最大的绝对平均人均回报差：ours与sampled为{differences['sampled']:.3e}，与去方向检查为{differences['no_direction_check']:.3e}，与去回报检查为{differences['no_return_check']:.3e}，与16轮拟合为{differences['fit_quarter']:.3e}。这些差异不构成有实际意义的迁移优势，也未进行预设等效界检验。",
        '仅改变人数时，所有方法、所有种子的人均回报均保持不变（核验容差1e-14），团队总回报随人数缩放。当前局部偏好和交互矩阵使全部目标共享动作1概率(0.01,0.99)的约束最优策略；ours、sampled和消融均逼近该策略，PG尚有源拟合差距。组成/强度变化会改变这一策略误差的回报代价，但本实验不能将目标领先与源终点精度分离。',
        '**新增结论：在当前固定源预算、PG配置和预定目标族下，完整方法的零样本目标回报高于PG，并保持近约束最优表现；未观察到相对sampled或消融的实际优势，亦未证明独立于源学习程度的迁移能力优势。** 这补齐了最终策略部署评价，不能替代导航/仓库对真实交互变化的比较。',
        '[迁移总览图](pair_frozen_transfer_v1/overview.png)；[逐模型记录](pair_frozen_transfer_v1/records.csv)；[均值与配对区间](pair_frozen_transfer_v1/summary.csv)；[协议与结构限制](../冻结策略迁移补充设计.md)。',
        table(['迁移输入','SHA256'], [[str((directory/name).relative_to(ROOT)), digest(directory/name)]
                                    for name in ('manifest.json', 'records.csv', 'summary.csv')])]


def audit_run(suite,row):
    directory=suite/'training'/row['condition']/f"seed_{row['seed']}"
    events=defaultdict(list)
    with (directory/'events.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line)
            if item['stream'] in ('policy','checks','batches','rounds'):
                events[item['stream']].append(item['record'])
    policies=events['policy']
    xs=np.asarray([p['source_episodes'] for p in policies]);ys=np.asarray([p['expected_return'] for p in policies])
    budget=int(row['budget'])
    assert np.all(np.diff(xs)>=0) and xs[0]==0 and xs[-1]<=budget
    auc=float(np.diff(np.append(xs,budget))@ys/budget)
    assert np.isclose(auc,float(row['budget_normalized_auc']),rtol=0,atol=1e-12)
    assert np.isclose(ys[-1],float(row['final_return']),rtol=0,atol=1e-12)
    costs=Counter()
    for batch in events['batches']:
        costs[batch['purpose']]+=batch['episodes']
    assert sum(costs.values())==int(row['source_episodes'])
    assert len(events['rounds'])==int(row['rounds'])
    cfg=js(directory/'config.json')
    checkpoint=js(directory/'checkpoints/final.json')
    actor=MLPActor.from_state(checkpoint['actor'])
    source=PairConfig(**{**cfg['source'],'type_probs':tuple(cfg['source']['type_probs']),
                        'local_bias':tuple(cfg['source']['local_bias']),
                        'pair_payoff':tuple(map(tuple,cfg['source']['pair_payoff']))})
    assert abs(closed_form_expected_return(source,actor.table())-ys[-1])<1e-10
    before={new['round']:old['expected_return'] for old,new in zip(policies,policies[1:])}
    counts=Counter()
    candidate_gains=[]
    for item in events['checks']:
        if item['kind']=='direction':
            counts['direction_checks']+=1;counts['direction_rejections']+=not item['accepted']
        else:
            skipped=item.get('check_skipped',False)
            counts['return_checks']+=not skipped
            counts['skipped_candidates']+=skipped
            gain=item['candidate_diagnostics']['expected_return']-before[item['round']]
            candidate_gains.append(gain)
            counts['negative_candidates']+=gain < -1e-10
            counts['return_rejections']+=not item['accepted']
            counts['false_rejections']+=(not item['accepted'] and gain > 1e-6)
    for old,new in zip(policies,policies[1:]):
        counts['accepted_updates']+=new['accepted']
        counts['declining_updates']+=new['accepted'] and new['expected_return']-old['expected_return'] < -1e-10
    assert counts['declining_updates']==int(row['declining_updates'])
    assert counts['return_rejections']==int(row['return_check_rejections'])
    for key,ckey in [('direction_check_rejections','direction_rejections'),('return_check_false_rejections','false_rejections')]:
        if key in row:
            assert counts[ckey]==int(row[key])
    thresholds={str(v):next((p['source_episodes'] for p in policies if p['expected_return']>=v),None) for v in (1.95,1.96)}
    for v,cost in thresholds.items():
        key='episodes_to_'+v.replace('.','_')
        if key in row:
            assert cost==(int(row[key]) if row[key] else None)
    return dict(counts=counts,costs=costs,thresholds=thresholds,policies=policies,
                tail5_gain=float(ys[-1]-ys[-6]) if len(ys)>=6 else None,
                candidate_min=min(candidate_gains) if candidate_gains else None)


def collect():
    suites={}
    for directory in sorted(ROOT.glob('suite*')):
        if not (directory/'manifest.json').exists():
            continue
        manifest,status=js(directory/'manifest.json'),js(directory/'status.json')
        assert all(job['status']=='complete' for job in status['jobs'].values()),directory
        rows=read(directory/'training_results.csv') if (directory/'training_results.csv').exists() else []
        assert len({(r['condition'],r['seed']) for r in rows})==len(rows)
        planned={(job['condition'],str(job['seed'])) for job in manifest['plan'].get('jobs',[])}
        assert {(r['condition'],r['seed']) for r in rows}==planned
        audits={}
        for row in rows:
            assert row['status']=='complete'
            audits[(row['condition'],row['seed'])]=audit_run(directory,row)
        suites[directory.name]=dict(directory=directory,manifest=manifest,rows=rows,audits=audits)
    assert len(suites[TRAIN.name]['rows'])==60
    assert {int(r['seed']) for r in suites[TRAIN.name]['rows']}==set(range(100,110))
    train=suites[TRAIN.name]
    groups={c:[r for r in train['rows'] if r['condition']==c] for c in CONDITIONS}
    assert all(len(g)==10 for g in groups.values())
    for c,rows in groups.items():
        for row in rows:
            directory=TRAIN/'training'/c/f"seed_{row['seed']}"
            cfg=js(directory/'config.json')['training']
            assert cfg['budget']==262144 and cfg['episodes']==1024 and cfg['acceptance']=='empirical'
            assert cfg['actor_epochs']==(16 if c=='fit_quarter' else 64)
            assert cfg['method']==('sampled' if c=='sampled' else 'analytic')
            assert cfg['algorithm']==('pg' if c=='pg' else 'direction')
    paired={}
    ref={r['seed']:r for r in groups['ours']}
    for c in CONDITIONS[1:]:
        paired[c]={}
        for metric in ('final_return','budget_normalized_auc','seconds','episodes_to_1_96'):
            paired[c][metric]=bootstrap([float(r[metric])-float(ref[r['seed']][metric]) for r in groups[c]],10000)
    m1=read(MECHANISM/'P-M1/records.csv');m3=read(MECHANISM/'P-M3/records.csv');ph=read(CHECK/'P-H/records.csv')
    assert len(m1)==640 and len(m3)==180 and len(ph)==9600
    assert len({(r['actor_id'],r['label'],r['method'],r['sample_size'],r['seed']) for r in m1})==640
    assert {int(r['seed']) for r in m1}==set(range(1000,1020))
    for row in paired_direction_summary(m1,10000):
        original=next(r for r in read(MECHANISM/'P-M1/paired_summary.csv') if all(r[k]==row[k] for k in ('actor_id','label','sample_size')))
        for k in ('mean','std','ci_low','ci_high'):
            assert np.isclose(row[k],float(original[k]),rtol=0,atol=1e-12)
    snapshots={r['seed']:r for r in m1 if r['actor_id']=='1' and r['label']=='return' and r['method']=='analytic' and r['sample_size']=='2048'}
    assert len({(r['target'],r['seed']) for r in m3})==180
    for row in m3:
        assert row['source_dataset_sha256']==snapshots[row['seed']]['dataset_sha256']
        assert true(row['bound_holds'])
        assert float(row['target_error'])<=float(row['transport_bound'])+1e-10
        assert float(row['target_error'])<=float(row['mechanism_transport_bound'])+1e-10
    pairs=defaultdict(dict)
    for row in ph:
        key=tuple(row[k] for k in ('kind','truth_class','sample_size','seed'))
        assert row['mode'] not in pairs[key]
        pairs[key][row['mode']]=row
    assert len(pairs)==4800 and {int(r['seed']) for r in ph}==set(range(2000,2200))
    for pair in pairs.values():
        assert pair['empirical']['evaluation_batch_id']==pair['hoeffding']['evaluation_batch_id']
        assert not true(pair['hoeffding']['accepted']) or true(pair['empirical']['accepted'])
    for row in read(CHECK/'P-H/summary.csv'):
        selected=[r for r in ph if all(r[k]==row[k] for k in ('kind','truth_class','sample_size','mode'))]
        assert len(selected)==200
        computed=binomial_interval([true(r[row['metric']]) for r in selected])
        for key in ('count','events','mean','ci_low','ci_high'):
            assert np.isclose(computed[key],float(row[key]),rtol=0,atol=1e-12)
    return suites,groups,paired,m1,m3,ph


def report(suites,groups,paired,m1,m3,ph):
    old=next(v for k,v in suites.items() if k.startswith('suite_01_'))
    development=[v for k,v in suites.items() if v['rows'] and k!=TRAIN.name]
    train=suites[TRAIN.name]
    means={c:{k:mean(rows,k) for k in ('final_return','budget_normalized_auc','seconds','rounds',
                    'optimizer_updates','episodes_to_1_95','episodes_to_1_96','source_episodes')} for c,rows in groups.items()}
    aggregate={c:sum((a['counts'] for (method,_),a in train['audits'].items() if method==c),Counter()) for c in CONDITIONS}
    scope=sum(len(v['rows']) for v in development)
    md=['# 成对协作：完整实验判断（2026-09-18）',
        '本报告综合环境 1 的现存开发与新增确认结果；不把其他环境结果纳入算法证据。数据目录及元数据路径已按内容整理，实验数值、模型与事件日志未改动。',
        '本报告为统一数值与复核依据；论文文字及跨环境验证安排见[论文结论与后续验证](论文结论与后续验证.md)。早期分批判断已清理，开发结果在第 7 节统一解释。',
        '**总体判断：该环境已经可以形成论文中的“可枚举机制核验＋受限任务上的性能与成本分析”结果节。现有证据支持有条件的方向估计收益、方向运输边界、Actor 表达限制及拟合计算冗余；不支持完整算法整体效率占优，也不支持 empirical 检查的普遍安全保证。**',
        '## 1. 数据完整性与统计口径',
        table(['层次','实际目录','已完成内容'],[
            ['早期机制',f'[{old["directory"].name}]({old["directory"].name}/)','P-M1/M2/M3/M4/P-H；其中 P-M2 24 条确定性记录、P-M4 3 条'],
            ['开发训练','suite_02～suite_08',f'{scope} 次训练；40–49 为开发种子，多个套件复用种子，不合并成独立大样本'],
            ['确认比较/消融',f'[{TRAIN.name}]({TRAIN.name}/)','6 条件×10 新种子（100–109）=60 次训练；预算 262144'],
            ['确认方向/运输',f'[{MECHANISM.name}]({MECHANISM.name}/)','P-M1 640 条；P-M3 180 条；20 个数据种子（1000–1019）'],
            ['确认检查规则',f'[{CHECK.name}]({CHECK.name}/)','P-H 9600 条规则记录；每条件200个数据种子（2000–2199）']]),
        f'共核对 {scope+60} 次非 smoke 完整训练，状态均 complete，未发现超预算、缺失种子或同套件重复条件/种子。逐运行从 events 重算 final、全轮次分段常数 AUC、成本与退化次数，并用 final checkpoint 重算精确回报；均与 CSV 一致。确认组另核对两项阈值成本、检查拒绝数与配置。P-M1 的配对数据哈希及配对区间、P-M3 源快照复用与上界、P-H 两规则共用数据及 Wilson 区间亦通过核对。',
        '训练统计单位是训练 seed；方向估计是独立数据 seed；P-H 是每个固定候选/样本量下的200个独立数据集。9600行不是9600个独立训练；P-H 的两条规则、不同候选/样本量之间存在数据种子复用，不池化为独立事件。',
        '新增组使用 Python 3.11.7、NumPy 1.26.4；训练为 PyTorch 2.7.1+cu128、RTX 4070 Ti SUPER、float64。当前分析解释器与其不同，只读取结果与重算统计，不重跑训练。墙钟仅作同套件硬件实测描述，未声称随机化运行顺序或控制机器负载。',
        '区间为训练/数据 seed 的10000次 percentile bootstrap，固定重采样种子20260916；P-H 为95% Wilson。所有区间逐比较，未作多重比较校正。确认训练的新增配对区间是本报告事后分析，不冒充预注册检验。无显著差异不等于等效；1e-10量级的浮点/平台差别不应包装成算法收益。',
        '## 2. 主比较：终点接近最优，整体效率并不占优',
        table(['条件','final J','AUC/B','到1.95回合数','到1.96回合数','耗时/秒','平均轮数'],[
            [LABELS[c],f'{means[c]["final_return"]:.9f}',f'{means[c]["budget_normalized_auc"]:.6f}',
             f'{means[c]["episodes_to_1_95"]:.1f}',f'{means[c]["episodes_to_1_96"]:.1f}',f'{means[c]["seconds"]:.2f}',f'{means[c]["rounds"]:.1f}'] for c in CONDITIONS]),
        '表中为10种子均值，各组均10/10达到两个阈值。回报由枚举任务的闭式公式精确评价，不含评价 Monte Carlo 噪声。AUC 使用全部轮次的已提交策略积分，不是对21个展示节点插值。预算包含训练与检查采样，未用剩余预算以最后策略延续至共同终点。',
        '本任务 beta=0.02，动作概率下限0.01；四个概率边界顶点核算的约束最优上确界为 **1.9626656**，无概率下限的最优值为 **2.024**。方向方法接近的是当前策略约束的上限，不能说已经达到无约束任务最优。该判断也得到回报关于动作均值的凸二次结构支持。',
        'PG 的 final 比完整方法低约0.001298，但 AUC 高约0.06865，且墙钟更少；到1.95更早、到1.96更晚，说明前期效率与后期逼近精度有取舍。PG最后5轮平均仍增加约3.98e-5，不能把本预算差距解释为PG最终可达上限。PG仅测试固定学习率0.001，不是充分调参后的最强基线。',
        table(['对照减完整方法','final 差及95%配对区间','AUC/B差及95%配对区间'],[
            [LABELS[c],interval(paired[c]['final_return'],'.3e'),interval(paired[c]['budget_normalized_auc'])] for c in CONDITIONS[1:]]),
        'sampled 的 AUC 比 ours 高0.001162，逐比较配对区间不跨零；差异幅度较小，不能说两者完全一致，更不能说解析法在完整训练中更优。final 差约1e-12，可视为实际性能上的同一平台。',
        '## 3. 三项消融：检查有成本，16轮拟合更经济',
        table(['条件','方向检查/拒绝','实际回报检查/拒绝','明确误拒绝>1e-6','已接受更新/真实退化'],[
            [LABELS[c],f'{aggregate[c]["direction_checks"]}/{aggregate[c]["direction_rejections"]}',
             f'{aggregate[c]["return_checks"]}/{aggregate[c]["return_rejections"]}',aggregate[c]['false_rejections'],
             f'{aggregate[c]["accepted_updates"]}/{aggregate[c]["declining_updates"]}'] for c in CONDITIONS]),
        '表为每组10种子合计；同一训练内更新相关，不能把更新数当独立训练重复来构造安全区间。去回报检查的1280个候选保留诊断但未采回报检查数据。所有已记录方向方法候选均未出现超过1e-10的真实退化；这说明当前任务没有充分产生保护场景，而非检查已阻止退化。',
        f'完整方法方向检查541次全部通过，实际源成本中检查占 **{100*sum(float(r[k]) for r in groups["ours"] for k in ("direction_check_episodes","return_old_episodes","return_candidate_episodes"))/sum(float(r["source_episodes"]) for r in groups["ours"]):.2f}%**。移除方向检查使AUC增加0.03969；移除回报检查使AUC增加0.08033。完整方法732次回报检查中拒绝277次，其中23次拒绝的候选真实改善>1e-6。余下拒绝不能统称有益拦截，绝大多数是极小正改善。',
        '去回报检查样本效率最好，但平均训练332.35秒，长于完整方法170.18秒：同预算允许128轮而不是54.1轮，优化次数由65177.6增到131072。节约交互与节约计算不是同一个结论。',
        f'64→16轮拟合：final平均差为{paired["fit_quarter"]["final_return"]["mean"]:.3e}，AUC差为{paired["fit_quarter"]["budget_normalized_auc"]["mean"]:.3e}；后者区间跨零。墙钟从170.18降至114.53秒，减少 **{100*(1-means["fit_quarter"]["seconds"]/means["ours"]["seconds"]):.2f}%**；优化次数减少 **{100*(1-means["fit_quarter"]["optimizer_updates"]/means["ours"]["optimizer_updates"]):.2f}%**。耗时配对差为{interval(paired["fit_quarter"]["seconds"],".2f")}秒。',
        '这与早期10开发种子中约32%的节时结果一致，支持“当前设置下64轮存在计算冗余、16轮是合理开发选择”。并未做预设等效界的等效检验，也没有测试所有环境或更少epochs。',
        '## 4. P-M1：解析二次项有条件地改善方向估计',
        '偏置冻结策略（两类型动作1概率0.2/0.8）、回报标签、20个配对数据种子的结果：']
    summary=read(MECHANISM/'P-M1/summary.csv');pairs=read(MECHANISM/'P-M1/paired_summary.csv')
    m1rows=[]
    for n in (128,512,2048,8192):
        base=[r for r in summary if r['actor_id']=='1' and r['label']=='return' and r['sample_size']==str(n) and r['metric']=='direction_error']
        a=next(float(r['mean']) for r in base if r['method']=='analytic');b=next(float(r['mean']) for r in base if r['method']=='sampled')
        pair=next(r for r in pairs if r['actor_id']=='1' and r['label']=='return' and r['sample_size']==str(n))
        m1rows.append([n,f'{a:.6g}',f'{b:.6g}',f'{100*(1-a/b):.1f}%',interval({k:float(pair[k]) for k in ('mean','ci_low','ci_high')},'.3g')])
    md += [table(['回合数','analytic误差','sampled误差','均值相对降低','analytic−sampled及95%配对区间'],m1rows),
        '四个回报标签条件均支持更小的平均方向误差；这是实际回报标签下的机制收益，不能直接替换成端到端训练收益。两方法的二次项不同，标签相同，不应称“解析标签优于采样标签”。',
        '限制同样清楚：均匀两动作策略下解析与采样平方在每个样本上相等，因此8个策略/标签/样本量配对行差均为0，是代数结构而非额外统计证据。偏置策略的精确优势标签下没有一致的解析优势；8192回合时 analytic−sampled 为+1.542e-5，逐比较95%区间[1.094e-6,3.137e-5]，反而略有利于sampled。该小差异未做多重比较校正，也不足以宣称sampled普遍更好。',
        '主训练与冻结估计的策略分布、函数类和迭代过程不同，且主训练接近平台；这些是收益未传递到主训练的可能解释，当前实验没有分别识别其因果贡献。',
        '## 5. P-M2 / P-M3 / P-M4：机制结论及边界',
        '**P-M2（沿用确定性核验）**：eta=0.2时目标真实增益0.0503703，table经过512步拟合实现相同增益且KL接近机器精度；tied_logit即使优化残差接近零，类内最佳KL仍约0.00119832，真实增益仅约1e-10。它说明参数化表达限制与优化未收敛必须分开；多拟合几轮不能消除表达误差。该构造不代表所有受限Actor都没有收益。',
        '**P-M3（20源方向×9目标=180条）**：所有记录均满足记录的精确方向运输界和机制上界；源方向来自P-M1指定快照，无目标训练。',
        table(['目标','目标方向误差均值','方向漂移平方范数','记录的运输界均值'],[
            [target,f'{mean([r for r in m3 if r["target"]==target],"target_error"):.6g}',
             f'{mean([r for r in m3 if r["target"]==target],"direction_shift"):.6g}',
             f'{mean([r for r in m3 if r["target"]==target],"transport_bound"):.6g}'] for target in dict.fromkeys(r['target'] for r in m3)]),
        '只改人数2/6/8时局部方向不变，是 iid 类型、完全图及1/(N−1)归一化的结构结果；不是复杂导航上的规模泛化证据。改类型比例或交互强度会产生方向漂移，即使源拟合误差小也可能有明显目标误差。P-M3评价的是冻结参考策略上的方向，不是训练final策略的跨任务性能竞赛；表里的方向漂移为确定性量，20次重复不能当作20次随机环境证据。',
        '**P-M4（沿用反例）**：一阶局部方向为零，共同扰动epsilon=0.01/0.05/0.1仍分别提高回报0.0004/0.01/0.04，符合4epsilon²。可用于说明一阶局部驻点不推出联合局部最优，不否定带明确条件的一阶改进结论。',
        '## 6. P-H：经验检查不是安全保证，置信检查也可能完全不接受',
        '固定候选真实回报差分别为−0.0233899、0、+0.0245989。下表是empirical回报检查接受数/200；负差接受即真实退化误接，零差接受是无改善误接，正差未接受是误拒绝。',
        table(['每侧回合数','退化候选接受','零改善候选接受','改善候选接受'],[
            [n,*[f'{sum(true(r["accepted"]) for r in ph if r["kind"]=="return" and r["mode"]=="empirical" and r["truth_class"]==str(sign) and r["sample_size"]==str(n))}/200' for sign in (-1,0,1)]] for n in (128,512,2048,8192)]),
        '真实退化误接率从34.5%降到0.5%，改善候选接受率从57.5%升到99.5%；增大样本显著改善本构造的区分能力，但8192回合仍有1/200误接，其Wilson区间约[0.088%,2.777%]。零改善接受率始终约43.5%–52.5%：两组独立、同分布的样本均值差仍可为正；它是缺乏显著性门槛的表现，不能说有一半概率造成严格退化。',
        'Hoeffding在所有候选、检查类型和四个样本量上均拒绝：零观察误接与100%改善误拒绝同时出现。alpha=6.4850843e-5，回报检查半径从128回合的2.01850下降到8192回合的0.252313，仍约为真实正改善0.024599的10.3倍。结论是当前界和预算下保守到失去接受能力，而非“更安全所以完整算法更好”。未观察误接也不能验证1e-5量级错误概率；0/200的逐条件Wilson上界仍约1.88%。',
        '方向检查的真实分数−0.18/0/+0.06下，empirical分别0/200、0/200、200/200接受，四个样本量一致；其中q=0分数恒等于0，严格大于0规则必然拒绝。候选只有这一组固定幅度，不能推广为接近零的任意方向判别能力。',
        '这与训练日志没有冲突：训练恰好没有产生显著退化候选，不能从训练未退化推出empirical检查能可靠拦截退化。P-H检验单次固定候选规则，不是自适应多轮训练安全定理的验证。',
        '## 7. 早期结果在综合判断中的位置',
        table(['开发套件','训练次数','均值AUC/B（按条件）'],[
            [v['directory'].name,len(v['rows']),'; '.join(f'{LABELS[c]} {mean([r for r in v["rows"] if r["condition"]==c],"budget_normalized_auc"):.6f}' for c in dict.fromkeys(r['condition'] for r in v['rows']))] for v in development]),
        '预算131072下的早期比较、四组小批量/大步长检查压力实验、10开发种子拟合比较，以及262144预算的3开发种子比较，方向上与新增确认结果一致：去检查提高交互效率；16轮拟合降低计算；PG增加预算后继续逼近；完整与sampled最终进入同一平台。旧机制P-H只有3数据种子，错误率判断应以新增200数据种子结果为主，不把两批简单池化。',
        '旧报告里“当前没有观察到退化”只针对已记录训练候选；如今应补充固定退化候选确实会被empirical回报检查误接。旧报告里“analytic/sampled几乎一致”应明确主要指最终回报，新增AUC配对分析发现轻微差异，方向并不有利于analytic。',
        '## 8. 可写入论文的判断与不能声称的内容',
        table(['研究主张','当前判断','适合的表述'],[
            ['解析方向训练目标有用','有条件支持','偏置冻结策略、回报标签下，四种样本量方向误差下降；不扩展到所有标签/策略'],
            ['完整算法整体更高效','不支持','固定预算下终点精度更高于当前PG，但AUC和耗时有明显取舍；去检查消融更省交互'],
            ['Actor表达限制重要','支持构造性证据','目标方向正确不保证受限参数化实现相同策略改进'],
            ['64轮拟合必要','不支持','当前任务16轮实际性能接近，确认种子平均节时约32.7%'],
            ['检查带来端到端保护收益','尚未展示','训练无退化候选，无法识别保护收益；检查消耗大量交互并存在误拒绝'],
            ['empirical检查安全','不支持','固定退化候选可被误接；样本增大仅在该候选幅度下降低误接'],
            ['方向可跨人数运输','在构造假设下支持','归一化iid完全图中局部方向对人数不敏感；组成/强度变化引起漂移'],
            ['最终策略零样本迁移优于其他方法','仅相对当前PG的目标回报支持','P-T已完成；ours高于当前PG但与sampled/消融实际接近，不能分离源终点差距与迁移能力'],
            ['一阶驻点意味着联合最优','反例否定','存在一阶方向为零但共同扰动产生二阶收益的任务'],
            ['已经验证复杂任务普遍优势','不支持','当前是小型可枚举任务；导航/仓库的实际学习与迁移尚需独立实验']]),
        '**可直接用于论文结果节的概括：** 在可枚举成对协作任务中，解析二次方向目标在偏置策略与回报标签条件下改善了方向估计精度，运输与表达受限构造揭示了局部方向改进的适用边界。然而，这一估计收益未转化为完整方法的整体训练效率优势。源交互预算包含检查开销时，去检查变体取得更高AUC，而缩短Actor拟合降低了计算成本。固定候选诊断进一步表明，经验回报检查存在有限样本误接，Hoeffding检查在当前界与预算下过于保守。',
        '## 9. 是否还需继续补成对实验',
        '若论文主张是“机制、适用边界和成本取舍”，环境1的预定补充清单已完成，可以封版整理，不必再次运行相同三组命令或盲目追加预算。下一项应按既定导航协议进行七条件源pilot，考察多步部分观测下是否出现不同的学习与检查行为。',
        'P-T最终冻结策略评价也已完成，无需默认重跑；它支持当前目标族中的近约束最优部署及相对当前PG的回报差，不支持算法普遍迁移领先。若要更强主张：整体优越需公平调优基线；逐次保护需预定退化场景和独立审计；复杂任务迁移优势需导航/仓库冻结final比较。不能用P-M3单轮方向运输或P-H固定候选替代这些证据。',
        '## 10. 复核与结果入口',
        '统计与图可重复生成：`python -m manifold_project.experiments.pair_coordination.analysis.comprehensive_assessment`。该脚本只读原始结果，重建本报告和一张综合图，不训练、不改原始统计。',
        '[综合图](完整实验判断_20260918.png)；确认训练原始表：[training_results.csv]('+TRAIN.name+'/training_results.csv)；方向配对表：[paired_summary.csv]('+MECHANISM.name+'/P-M1/paired_summary.csv)；检查区间表：[P-H/summary.csv]('+CHECK.name+'/P-H/summary.csv)。',
        '关键输入 SHA256（用于识别本报告读取的版本）：',
        table(['输入','SHA256'],[[str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()] for p in
              (TRAIN/'training_results.csv',MECHANISM/'P-M1/records.csv',MECHANISM/'P-M3/records.csv',CHECK/'P-H/records.csv')])]
    md[md.index('## 8. 可写入论文的判断与不能声称的内容'):md.index('## 8. 可写入论文的判断与不能声称的内容')] = transfer_assessment()
    (ROOT/'完整实验判断_20260918.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')


def plot(groups,m1,ph):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plot_training import chinese_fonts
    chinese_fonts(plt)
    fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    curves=read(TRAIN/'learning_curves.csv')
    for i,c in enumerate(CONDITIONS):
        group=[r for r in curves if r['condition']==c]
        xs=sorted({int(r['budget_checkpoint']) for r in group})
        stats=[bootstrap([float(r['expected_return']) for r in group if int(r['budget_checkpoint'])==x]) for x in xs]
        axes[0,0].step(xs,[s['mean'] for s in stats],where='post',color=colors[i],label=LABELS[c])
        axes[0,0].fill_between(xs,[s['ci_low'] for s in stats],[s['ci_high'] for s in stats],step='post',color=colors[i],alpha=.1)
    axes[0,0].set(title='确认训练：精确回报曲线',xlabel='累计源回合（含检查）',ylabel='团队期望回报')
    axes[0,0].legend(fontsize=8,ncol=2)
    for ax,key,title,scale in [(axes[0,1],'budget_normalized_auc','AUC/B：不等于最终性能',1),
                                (axes[0,2],'final_return','距离概率约束最优的差距',1e6),
                                (axes[1,0],'seconds','同预算训练墙钟：少检查不一定更快',1)]:
        for i,c in enumerate(CONDITIONS):
            vals=np.array([float(r[key]) for r in groups[c]])
            if key=='final_return': vals=(1.9626656-vals)*scale
            stats=bootstrap(vals)
            ax.bar(i,stats['mean'],color=colors[i],alpha=.7)
            ax.scatter(i+np.linspace(-.12,.12,len(vals)),vals,color='black',s=10)
            ax.errorbar(i,stats['mean'],yerr=[[max(0,stats['mean']-stats['ci_low'])],[max(0,stats['ci_high']-stats['mean'])]],color='black',capsize=3)
        ax.set(xticks=range(6),xticklabels=[LABELS[c] for c in CONDITIONS],title=title)
        ax.tick_params(axis='x',rotation=25,labelsize=9)
        if key=='final_return':
            ax.set_yscale('symlog',linthresh=.01)
            ax.set(ylabel='(1.9626656 − final J) × 10^6；对称对数轴')
        elif key=='seconds': ax.set(ylabel='秒')
        else: ax.set(ylabel='精确回报积分 / 262144',ylim=(1.76,1.91))
    for method,label in [('analytic','解析二次项'),('sampled','采样平方')]:
        selected=[r for r in m1 if r['actor_id']=='1' and r['label']=='return' and r['method']==method]
        xs=[128,512,2048,8192];stats=[bootstrap([float(r['direction_error']) for r in selected if int(r['sample_size'])==n]) for n in xs]
        line,=axes[1,1].plot(xs,[s['mean'] for s in stats],marker='o',label=label)
        axes[1,1].fill_between(xs,[s['ci_low'] for s in stats],[s['ci_high'] for s in stats],color=line.get_color(),alpha=.15)
    axes[1,1].set(title='P-M1：偏置策略＋回报标签',xscale='log',yscale='log',xlabel='独立回合数',ylabel='Fisher 加权方向误差')
    axes[1,1].legend(fontsize=9)
    for sign,label in [(-1,'退化候选'),(0,'零改善候选'),(1,'改善候选')]:
        xs=[128,512,2048,8192];stats=[]
        for n in xs:
            stats.append(binomial_interval([true(r['accepted']) for r in ph if r['kind']=='return' and r['mode']=='empirical' and int(r['truth_class'])==sign and int(r['sample_size'])==n]))
        line,=axes[1,2].plot(xs,[s['mean'] for s in stats],marker='o',label=label+'/empirical')
        axes[1,2].fill_between(xs,[s['ci_low'] for s in stats],[s['ci_high'] for s in stats],alpha=.12,color=line.get_color())
    axes[1,2].plot(xs,[0]*4,'k--',label='全部候选/Hoeffding')
    axes[1,2].set(title='P-H：独立回报检查接受率',xlabel='每侧独立回合数',ylabel='接受比例',xscale='log',ylim=(-.03,1.03))
    axes[1,2].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.15)
    fig.suptitle('成对协作综合证据：终点精度、过程效率与检查可靠性必须分开判断\n训练10种子、方向20数据种子：95% bootstrap；检查200数据集：Wilson；区间均未作多重比较校正',fontsize=14)
    fig.savefig(ROOT/'完整实验判断_20260918.png',dpi=170)
    plt.close(fig)


def main():
    suites,groups,paired,m1,m3,ph=collect()
    report(suites,groups,paired,m1,m3,ph)
    plot(groups,m1,ph)
    print('Audit passed: 143 training runs; 640 direction, 180 transport, 9600 check records.')
    print('Wrote comprehensive Markdown assessment and one overview PNG.')


if __name__=='__main__':
    main()
