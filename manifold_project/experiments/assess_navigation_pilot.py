"""Read-only log/CSV audit and Markdown assessment for the first navigation pilot."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import math

ROOT = Path(__file__).resolve().parent/'cooperative_navigation'
SUITE = ROOT/'results/archive/nav_01_pilot'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def js(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    m = js(SUITE/'manifest.json')
    cfg = m['config']
    rows, curves = read(SUITE/'training_results.csv'), read(SUITE/'learning_curves.csv')
    expected = {(j['condition'], j['seed']) for j in m['jobs']}
    assert len(rows) == len(expected) == 21
    assert {(r['condition'], int(r['seed'])) for r in rows} == expected
    all_summaries = []
    for job in m['jobs']:
        run = SUITE/job['path']
        summary = js(run/'summary.json')
        assert summary['status'] == 'complete' and not (run/'failure.json').exists()
        assert (run/'checkpoints/final.pt').is_file()
        all_summaries.append(summary)
        by_stream = defaultdict(list)
        for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            by_stream[item['stream']].append(item['record'])
        assert not by_stream['recovery'], 'Recovery requires a separate wasted-step audit'
        costs = Counter()
        for e in by_stream['sampling']:
            costs[e['purpose']] += cfg['horizon']
        assert dict(costs) == summary['costs']
        assert sum(costs.values()) == summary['source_steps'] <= cfg['budget']
        assert summary['source_steps']+summary['unused_budget'] == cfg['budget']
        commits = by_stream['commit']
        assert len(commits) == summary['rounds']
        assert sum(c['accepted'] for c in commits) == summary['accepted_rounds']
        assert len(by_stream['direction_check']) == summary['direction_checks']
        assert sum(not c['direction_pass'] for c in commits if c['direction_pass'] is not None) == summary['direction_rejections']
        assert len(by_stream['return_check']) == summary['return_checks']
        if summary['return_checks']:
            assert summary['return_checks']-summary['accepted_rounds'] == summary['return_rejections']
        cc = [r for r in curves if r['condition'] == job['condition'] and int(r['seed']) == job['seed']]
        assert len(cc) == 6 and [int(r['budget_checkpoint']) for r in cc] == [0,40000,80000,120000,160000,200000]
        assert len(by_stream['evaluation']) == len(cc)
        for a, b in zip(cc, by_stream['evaluation']):
            for key, value in a.items():
                assert str(b[key]) == value or (key != 'condition' and math.isclose(float(b[key]),float(value),abs_tol=1e-12))
        auc = sum((int(b['budget_checkpoint'])-int(a['budget_checkpoint']))*float(a['J']) for a,b in zip(cc,cc[1:]))/cfg['budget']
        assert math.isclose(auc,summary['AUC'],abs_tol=1e-12)
        for key in ('J','distance','coverage','all_covered','collision_pairs','collisions_per_agent'):
            assert math.isclose(float(cc[-1][key]),summary[key],abs_tol=1e-12)
        assert sum(int(c['episodes'])*cfg['horizon'] for c in cc) == summary['evaluation_steps']
        row = next(r for r in rows if r['condition']==job['condition'] and int(r['seed'])==job['seed'])
        for key,value in summary.items():
            if key=='costs':
                assert all(float(row['cost_'+k])==v for k,v in value.items())
            elif isinstance(value,dict):
                assert json.loads(row[key])==value
            elif value is None:
                assert row[key]==''
            elif isinstance(value,(float,int)):
                assert math.isclose(float(row[key]),value,abs_tol=1e-12)
            else:
                assert row[key]==value
    from manifold_project.experiments.cooperative_navigation.run_experiments import fingerprint
    from manifold_project.experiments.cooperative_navigation.evaluation.reporting import summaries
    recomputed = summaries(all_summaries,cfg)
    saved = read(SUITE/'training_summary.csv')
    assert len(recomputed)==len(saved)
    for a,b in zip(recomputed,saved):
        for key,v in a.items():
            if v is None: assert b[key]==''
            elif isinstance(v,(float,int)): assert math.isclose(v,float(b[key]),abs_tol=1e-12)
            else: assert v==b[key]
    names = list(dict.fromkeys(r['condition'] for r in rows))
    groups = {c:[s for s in all_summaries if s['condition']==c] for c in names}
    def mean(c,k): return st.mean(r[k] for r in groups[c])
    def check_fraction(c):
        g=groups[c]
        return 1-sum(r['costs']['train'] for r in g)/sum(r['source_steps'] for r in g)
    text = [
        '# 合作导航：第一批 pilot 实验判断',
        '依据 nav_01_pilot；7 条件×3 开发种子（40/41/42），每次源预算200000团队步。该批为源训练比较与消融，没有 N-T 迁移结果，也未启用 N-D 独立候选审计。',
        '**结论：已有学习信号，ours 相对 sampled 有一致的开发趋势，但当前源性能明显落后于 MAPPO/IPPO；所有方法覆盖水平偏低，尚不宜冻结正式实验或宣称零样本迁移优势。**',
        '## 1. 数据核验与范围',
        f"21/21训练完成，无failure或恢复事件；核对本批21份运行摘要、126个评价节点、采样分项成本、提交/检查计数、final指标及AUC，均与CSV一致；{len(saved)}行汇总与配对bootstrap亦重算一致。final.pt文件存在，但本报告不重跑环境、不重算checkpoint轨迹。",
        f"实际源采样合计{sum(s['source_steps'] for s in all_summaries):,}团队步，独立源评价{sum(s['evaluation_steps'] for s in all_summaries):,}步；预算单位不是回合，100团队步为一个回合。实际配置为CPU、1线程，尽管机器有RTX 4070 Ti SUPER，不能将这些耗时解释为GPU训练速度。",
        f"保存源码指纹：`{m['metadata']['code_sha256']}`；当前指纹：`{fingerprint()}`。两者{'相同' if fingerprint()==m['metadata']['code_sha256'] else '不同'}。本次验证是文件内部一致性，不证明当前代码与原训练实现完全相同；如需恢复或补N-T，使用原训练机器的同版代码/依赖，不能修改manifest强行绕过。manifest未记录KMP_DUPLICATE_LIB_OK或MKL_THREADING_LAYER，无法由本批结果确定实际运行时设置。",
        'AUC是6个评价节点上的左端分段常数积分再除以预算，不是每个训练轮次的精确曲线面积；final评价使用source-final随机流，前5节点使用source-grid，不能把最后两点差异直接解释为真实策略退化/收敛。每节点仅20回合。不同方法动作评价随机流不同，即使相同初始Actor也可有不同初始样本回报。三种子bootstrap只作开发描述，区间不跨零不等于充分正式证据。',
        '## 2. 源表现（越接近零的J/AUC越好）',
        table(['方法','final J','AUC','终点平均距离↓','终点覆盖率','每步碰撞对↓','训练秒'],[
            [c,f'{mean(c,"J"):.3f}',f'{mean(c,"AUC"):.3f}',f'{mean(c,"distance"):.3f}',f'{mean(c,"coverage"):.2%}',f'{mean(c,"collision_pairs"):.3f}',f'{mean(c,"train_seconds"):.1f}'] for c in names]),
        '覆盖率为第100步地标处于0.1距离阈值内的比例，碰撞为回合内每步平均碰撞对；二者时间口径不同。全部运行最终all_covered为0，所有coverage_cost为空，表示没有评价节点达到80%覆盖，不能写成零成本。低碰撞也可能伴随未到达目标，不是单独的成功证据。',
        table(['ours减对照','指标','均值差','逐比较95%区间'],[
            [r['condition'],r['metric'],f"{r['mean']:.3f}",f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"] for r in recomputed if r['kind']=='paired_difference']),
        'ours相对sampled的final差为+5.297、AUC差+4.428，三个种子方向一致；相对MAPPO/IPPO的final差分别为−19.324/−18.380。解析二次项出现值得继续验证的端到端开发趋势，但完整算法的总体性能优势未得到支持。PPO采用GAE与优势标准化，方向方法采用MC标签；与PPO的差异不能全归因于方向损失。',
        '## 3. 消融和瓶颈',
        table(['方法','平均轮数','平均接受轮数','方向拒绝/检查（3种子）','回报拒绝/检查（3种子）','检查交互占比'],[
            [c,f'{mean(c,"rounds"):.1f}',f'{mean(c,"accepted_rounds"):.1f}',f"{sum(r['direction_rejections'] for r in groups[c])}/{sum(r['direction_checks'] for r in groups[c])}",f"{sum(r['return_rejections'] for r in groups[c])}/{sum(r['return_checks'] for r in groups[c])}",f'{check_fraction(c):.2%}'] for c in names]),
        '完整方法方向检查拒绝71/161（44.1%），回报检查拒绝63/129（48.8%），有效接受66轮。检查占56.55%源交互。去方向检查使后续回报检查由129次增至222次，总检查占比反而升至59.68%，平均轮数由53.7降至50；因此不能机械套用pair中“去检查必然更省预算”的判断。没有候选独立真值，无法把这些拒绝判定为正确保护或误拒绝。',
        '去回报检查的final均值提高5.514，AUC提高5.179，但每步碰撞对从0.102升至0.140，训练秒从205.2升至226.7。它获得更多训练采样和提交机会，不能只用接受率或更高回报声称全面更好；也不能把碰撞增加等同于逐次策略退化。',
        f"16→4 Actor epochs：平均拟合KL从{mean('ours','fit_kl_after'):.3e}升至{mean('fit_quarter','fit_kl_after'):.3e}，约{mean('fit_quarter','fit_kl_after')/mean('ours','fit_kl_after'):.1f}倍；耗时仅减少{(1-mean('fit_quarter','train_seconds')/mean('ours','train_seconds')):.2%}，final均值下降5.126，但三种子配对区间跨零。暂保留16 epochs，不能继承pair中32.7%的节时结论。KL小本身也不保证目标有足够回报增益。",
        '## 4. 历史建议与当前入口',
        '本批之后已完成长预算开发；本页为历史记录，当前安排见[当前实验判断](../当前实验判断.md)及[实验运行说明](../../实验运行说明.md)。',
        '预算扩展保持原源配置、依赖和运行时设置，并保存原运行代码副本；如源码已变，先明确变更内容后作为新实现版本报告，不把差异全部归因于预算。同步可用动画检查覆盖/分工行为；不以挑选的好看回合替代统计。',
        '若仍需解释与PPO的差异，按既定N-L做标签对齐；公平超参数开发应对主算法提供相同候选预算，不能只调ours。正式冻结前决定是否启用N-D审计，当前未保存的拒绝候选不能事后补回。',
        'N-T已实现，但本批无迁移数据。若要先做探索性诊断，可在原训练代码与依赖环境中对本套件补N-T；目标结果不得用于选择源超参数或选模型。正式零样本主张仍以源参数冻结后的独立确认训练评价为依据。当前源码指纹不同，不能保证在本工作区直接resume成功。',
        '## 5. 论文状态',
        '本批可记录为开发证据：解析二次项相对采样平方的正向趋势、方向筛选与后续检查成本的相互作用、Actor拟合预算的环境依赖性。它不支持完整方法优于PPO、检查安全保证或零样本迁移优势。下一步是预算/源配置开发，不是宣布正式结论。',
        '[总览图](nav_01_pilot/overview.png)；[逐种子表](nav_01_pilot/training_results.csv)；[汇总表](nav_01_pilot/training_summary.csv)。复核命令：`python -m manifold_project.experiments.assess_navigation_pilot`。',
        table(['输入','SHA256'],[[f,hashlib.sha256((SUITE/f).read_bytes()).hexdigest()] for f in ('manifest.json','training_results.csv','training_summary.csv','learning_curves.csv')])]
    # One current assessment, no extra JSON/CSV logs or copied checkpoints.
    (ROOT/'results/archive/pilot_01_实验判断.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    print('Audit passed: 21 runs, 126 evaluation nodes and all summary rows. Assessment written.')


if __name__=='__main__':
    main()
