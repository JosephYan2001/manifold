"""Audit a completed single-seed ours suite; write one Markdown assessment, no resampling."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

if __package__:
    from .assess_navigation_stability import assess
else:
    from assess_navigation_stability import assess

ROOT = Path(__file__).resolve().parent/'cooperative_navigation'


def js(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def mean(rows, key):
    return st.mean(r[key] for r in rows)


def return_se(row):
    return math.sqrt(st.variance(row['old_returns'])/len(row['old_returns'])+
                     st.variance(row['candidate_returns'])/len(row['candidate_returns']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT/'results/reference/nav_03_pilot')
    args = parser.parse_args(argv)
    suite = args.suite.resolve()
    manifest = js(suite/'manifest.json')
    assert len(manifest['jobs']) == 1 and manifest['jobs'][0]['condition'] == 'ours', 'Expected one ours job'
    job = manifest['jobs'][0]
    c = manifest['config']
    run = suite/job['path']
    summary = js(run/'summary.json')
    assert summary['status'] == 'complete'
    assert js(suite/'status.json')['jobs'][job['path']] == 'complete'
    assert js(run/'config.json')['config'] == c
    streams = defaultdict(list)
    sampling = Counter()
    total_sampling_steps = 0
    with (run/'events.jsonl').open(encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            kind, record = item['stream'], item['record']
            if kind == 'sampling':
                sampling[record['purpose']] += 1
                total_sampling_steps = record['used']
            elif kind != 'sampling_start':
                streams[kind].append(record)
    commits = streams['commit']
    directions = streams['direction_check']
    checks = streams['return_check']
    evaluations = streams['evaluation']
    assert [r['completed_round'] for r in commits] == list(range(1, summary['rounds']+1))
    assert sum(r['accepted'] for r in commits) == summary['accepted_rounds']
    assert len(directions) == summary['direction_checks']
    assert sum(not r['direction_pass'] for r in directions) == summary['direction_rejections']
    assert len(checks) == len(streams['candidate']) == summary['return_checks']
    assert sum(not r['passed'] for r in checks) == summary['return_rejections']
    assert {k:v*c['horizon'] for k,v in sampling.items()} == summary['costs']
    assert total_sampling_steps == commits[-1]['source_steps'] == summary['source_steps'] <= c['budget']
    assert sum(r['episodes']*c['horizon'] for r in evaluations) == summary['evaluation_steps']
    previous = 0
    for row in commits:
        expected = c['horizon']*(c['train_episodes']+c['direction_check_episodes']+
                                2*c['return_check_episodes']*row['candidates'])
        assert row['source_steps']-previous == expected
        previous = row['source_steps']
    with (suite/'learning_curves.csv').open(encoding='utf-8-sig', newline='') as f:
        curves = list(csv.DictReader(f))
    assert len(curves) == len(evaluations) == len(c['eval_fractions'])
    for row, event in zip(curves, evaluations):
        assert int(row['budget_checkpoint']) == event['budget_checkpoint']
        assert int(row['actor_source_steps']) <= int(row['budget_checkpoint'])
        for key in ('J', 'coverage', 'distance', 'J_se', 'coverage_se', 'distance_se'):
            assert math.isclose(float(row[key]), event[key], rel_tol=1e-12, abs_tol=1e-12)
    auc = sum((b['budget_checkpoint']-a['budget_checkpoint'])*a['J']
              for a,b in zip(evaluations, evaluations[1:]))/c['budget']
    assert math.isclose(auc, summary['AUC'], rel_tol=1e-12)
    for key in ('J','coverage','distance','all_covered','collision_pairs'):
        assert math.isclose(summary[key], evaluations[-1][key], abs_tol=1e-12)
    with (suite/'training_results.csv').open(encoding='utf-8-sig', newline='') as f:
        results = list(csv.DictReader(f))
    assert len(results) == 1 and results[0]['status'] == 'complete'
    for key in ('J','AUC','coverage','source_steps'):
        assert math.isclose(float(results[0][key]), summary[key], rel_tol=1e-12)

    stability = assess(curves, c['budget'])
    last = evaluations[-1]
    accepted = summary['accepted_rounds']
    passed = len(directions)-summary['direction_rejections']
    cost_checks = summary['source_steps']-summary['costs']['train']
    phases = []
    for i in range(5):
        lo, hi = round(c['budget']*i/5), round(c['budget']*(i+1)/5)
        group = [r for r in commits if lo < r['source_steps'] <= hi]
        if not group:
            continue
        ids = {r['completed_round'] for r in group}
        gate = [r for r in directions if r['round'] in ids]
        ret = [r for r in checks if r['round'] in ids]
        critic = [r for r in streams['optimization'] if r['round'] in ids and r['module']=='critic']
        phases.append((lo, hi, group, gate, ret, critic))
    body = [f'# {suite.name}：完整方法长预算开发判断',
        '结论：本次参考组有持续学习信号，但尚未达到高覆盖表现，末段也未通过预定经验稳定性检查。下一步优先补齐同预算的检查消融，不把继续加预算作为唯一方案。',
        '本文仅核对已有文件，不重新训练、加载模型或采样；只新增本报告。单种子用于开发，不能给出跨训练种子的算法排名或机制因果确认。',
        '## 1. 配置与数据核对',
        f"实际仅ours/seed{job['seed']}一个任务；虽然manifest实验列表包含N-C/N-A，selected_conditions仅为ours。本次源预算{c['budget']:,}团队步（每回合{c['horizon']}步），Actor/方向/Critic学习率分别为{c['actor_lr']}/{c['direction_lr']}/{c['critic_lr']}，eta={c['eta']}，方向标签={c['direction_label']}。",
        f"设备为{c['device']}，GPU={manifest['metadata']['gpu']}，Torch={manifest['metadata']['versions']['torch']}。记录训练耗时{summary['train_seconds']/3600:.2f}小时、独立评价{summary['evaluation_seconds']/60:.2f}分钟；不含全部checkpoint/汇总I/O，不能称完整作业墙钟。",
        f"核对通过：{len(commits)}个连续提交轮次，{sum(sampling.values()):,}个源采样回合，{len(evaluations)}个评价节点，{len(checks)}次候选回报检查。用途成本、候选计数、JSONL与CSV曲线、final摘要和AUC重算一致；没有预算超支或未来Actor节点评价。独立评价共{summary['evaluation_steps']:,}团队步，另于源预算。",
        f"原训练源码SHA256：`{manifest['metadata']['code_sha256']}`。manifest记录KMP_DUPLICATE_LIB_OK={manifest['metadata'].get('runtime_environment',{}).get('KMP_DUPLICATE_LIB_OK')}，未记录CUBLAS_WORKSPACE_CONFIG，不能从本结果文件断言当时具体设置。当前工作区已增加监控及N-A5，不能改写原manifest或承诺直接resume；后续比较需记录版本和实现差异。",
        '## 2. 性能与末段趋势',
        table(['源步','独立J','覆盖率','终点距离','全覆盖率'],
              [(f"{r['budget_checkpoint']:,}", f"{r['J']:.3f}", f"{r['coverage']:.2%}",
                f"{r['distance']:.4f}", f"{r['all_covered']:.1%}") for r in evaluations
               if r['budget_checkpoint'] in (0, 2000000, 5000000, 7500000, 8000000, 9000000, 9500000, c['budget'])]),
        f"final J={last['J']:.3f}，AUC/B={auc:.3f}，覆盖率={last['coverage']:.2%}，全覆盖率={last['all_covered']:.1%}；后者只是final这{last['episodes']}个回合的观测比例。覆盖率指地标覆盖比例，不是任务成功率。回报和距离明显改善，但尚不足以支持任务已学好。",
        f"预定稳定性结果：**{stability['status']}**（{stability['code']}）。最后5个非final节点的J极差={stability['J_range']:.3f}、覆盖极差={stability['coverage_range']:.2%}、距离极差={stability['distance_range']:.4f}，对应阈值为2、2个百分点、0.02；窗口末两点与首两点平均J差为{stability['tail_return_change']:.3f}。",
        f"final近似95%回合均值误差半径：J ±{1.96*last['J_se']:.3f}，覆盖 ±{196*last['coverage_se']:.2f}个百分点，距离 ±{1.96*last['distance_se']:.4f}。末段评价精度未满足全部阈值，不能认定稳定；这也不是跨训练种子区间。",
        f"最后source-grid节点J={evaluations[-2]['J']:.3f}，final J={last['J']:.3f}，final采用另一套评价随机流，不能将这一下降直接解释为最后更新造成退化。网格节点复用场景且相关，也不能将节点数当作独立训练重复。",
        '## 3. 更新被挡在哪一层',
        table(['流程','数量 / 比例'], [
            ('完成训练轮次',len(commits)),
            ('方向检查拒绝',f"{summary['direction_rejections']} / {len(commits)} = {summary['direction_rejections']/len(commits):.2%}"),
            ('进入候选阶段',passed),
            ('通过方向后仍未接受',f'{passed-accepted} / {passed} = {(passed-accepted)/passed:.2%}'),
            ('最终接受更新',f'{accepted} / {len(commits)} = {accepted/len(commits):.2%}'),
            ('候选回报检查拒绝',f"{summary['return_rejections']} / {len(checks)} = {summary['return_rejections']/len(checks):.2%}")]),
        '方向检查是当前更新机会的主要限制。回报拒绝按候选计数，回退会导致一轮两次检查；不能把候选拒绝次数当成拒绝轮数，更不能当成已确认的误拒绝次数。',
        table(['预算段（万步，按轮结束归组）','轮数','方向拒绝率','最终接受率','回报候选检查数','Critic末小批MSE均值'],
              [(f'{lo/10000:g}–{hi/10000:g}',len(group),f"{sum(r['direction_pass'] is False for r in group)/len(group):.1%}",
                f"{sum(r['accepted'] for r in group)/len(group):.1%}",len(ret),f"{mean(critic,'value_mse'):.2f}")
               for lo,hi,group,gate,ret,critic in phases]),
        table(['源采样用途','团队步','占源预算'],[(k,f'{v:,}',f"{v/summary['source_steps']:.2%}") for k,v in summary['costs'].items()]),
        f"两项检查共消耗{cost_checks:,}步，占源采样{cost_checks/summary['source_steps']:.2%}。被拒绝轮次仍会训练Critic，不能称其全部计算都无用；但这些轮次没有提交新的Actor。",
        '## 4. 噪声、步幅与拟合诊断']
    attempt_rows = []
    for attempt in (1,2):
        group = [r for r in checks if r['attempt']==attempt]
        attempt_rows.append((attempt,len(group),sum(r['passed'] for r in group),f"{sum(r['passed'] for r in group)/len(group):.2%}"))
    body.append(table(['候选序号','检查数','接受数','接受比例'],attempt_rows))
    ses = [return_se(r) for r in checks]
    body += [f"各回报检查均值差的估计标准误平均为{st.mean(ses):.3f}；{st.mean(abs(r['difference'])<se for r,se in zip(checks,ses)):.2%}的观测差值绝对值小于自身1个标准误，{st.mean(abs(r['difference'])<1.96*se for r,se in zip(checks,ses)):.2%}小于1.96个标准误。规则只看正负号，说明需认真考虑噪声分辨率；这些不是误判率，也不是每次候选的真实收益。",
        '半步候选是在首个候选失败后才构造，并重新采样检查回合。因此两个候选的接受比例不能作为“半步更好”的随机对照证据。',
        table(['预算段（万步）','方向检查分数均值','逐批方向SE均值','逐批回报差SE均值'],
              [(f'{lo/10000:g}–{hi/10000:g}',f"{mean(gate,'direction_score'):.4f}",
                f"{st.mean(st.stdev(r['episode_scores'])/math.sqrt(len(r['episode_scores'])) for r in gate):.4f}",
                f"{st.mean(return_se(r) for r in ret):.3f}") for lo,hi,group,gate,ret,critic in phases]),
        '方向检查后期分数均值仍为负，但批次标准误同样不可忽略。方向网络训练目标与独立检查并非同一批数据；尚不能区分方向泛化不足、标签误差或检查噪声，也不能仅凭负分数断言真实候选会降低回报。']
    last_ids = {r['completed_round'] for r in commits[-500:]}
    for attempt in (1,2):
        candidates = [r for r in streams['candidate'] if r['round'] in last_ids and r['attempt']==attempt]
        body.append(f"末500轮，第{attempt}候选共{len(candidates)}次：平均目标拟合前KL={mean(candidates,'fit_kl_before'):.3g}，拟合后KL={mean(candidates,'fit_kl_after'):.3g}，候选相对旧策略KL={mean(candidates,'policy_step_kl'):.3g}。后两者分别是拟合残差与实际变化，不能混为同一个更新幅度。")
    critic = [r for r in streams['optimization'] if r['round'] in last_ids and r['module']=='critic']
    direction = [r for r in streams['direction_fit'] if r['round'] in last_ids]
    all_critic = [r for r in streams['optimization'] if r['module']=='critic']
    body.append(f"Critic误差并未归零：第一轮MSE={all_critic[0]['value_mse']:.2f}，最后一轮={all_critic[-1]['value_mse']:.2f}，全程最小={min(r['value_mse'] for r in all_critic):.2f}，精确为0的记录={sum(r['value_mse']==0 for r in all_critic)}。原线性纵轴被初期大误差撑高，导致后期误差看似贴零；监控现改为对数刻度（0附近线性）。这里的MSE是最后训练minibatch上的加权误差，优化loss另乘value_coef，不是独立验证误差。")
    body += [f"末500轮Critic训练MSE均值={mean(critic,'value_mse'):.2f}，方向输出最大绝对值的逐轮均值={mean(direction,'q_abs_max'):.3f}，接近q_max比例均值={mean(direction,'near_bound_fraction'):.2%}。当前没有方向输出贴满上界或拟合残差明显失控的证据；不优先机械增加q_max或拟合epochs。Critic训练误差下降不等于优势标签已准确或泛化无误。",
        '## 5. 与旧批次的比较边界']
    previous_suite = ROOT/'results/archive/nav_02_pilot'
    if (previous_suite/'ours/seed_40/summary.json').exists():
        old = js(previous_suite/'ours/seed_40/summary.json')
        same_node = next(r for r in evaluations if r['budget_checkpoint']==2000000)
        body.append(f"旧200万步ours final J={old['J']:.3f}、覆盖={old['coverage']:.2%}；本批200万节点J={same_node['J']:.3f}、覆盖={same_node['coverage']:.2%}，终点进一步改善至J={last['J']:.3f}。新旧同时改变Critic学习率、CPU/CUDA设备、评价数量/网格与源码版本；旧final和本批中间节点的评价随机流也不同，不能把差异全部归因于延长预算或某一个参数。不同预算/网格的AUC不直接排名。")
    body += ['本批只有ours，不能据此宣布优于PPO或消融。旧批次PPO的表现可以作为任务仍有提升空间的开发提示，但不是当前1000万步共同配置下的公平对比。也没有新种子确认、冻结迁移或预存候选审计数据。',
        '## 6. 下一步与可写结论',
        '1. 该参考的后续双检查消融已完成。本页保留参考分析；当前安排统一见合作导航目录下的实验运行说明和results/当前实验判断.md。',
        '2. 对比J/AUC、覆盖、距离、碰撞、接受轮数和源成本。去检查后更新更多是结构性结果，不单独作为成功证据；联合消融用于判断整套模块交互，不等于证明误拒绝。',
        '3. 随后按预定计划检查eta=0.3、batch64、GAE三个单因素候选，不同时改门槛和步长。选择依据只用源开发结果，再用41/42复核；PPO获得匹配的开发资源。',
        '4. 本批末段仍有改善，延长共同预算有研究价值，但当前先补检查对照更能区分瓶颈。暂不直接启动所有组2000万步或冻结正式预算；不能仅给ours更长正式预算。',
        '> 当前可写为开发结论：完整方法在该单种子长预算下持续改善源回报与终点距离，但覆盖仍低；两项检查消耗较大源交互预算，方向检查限制了大部分更新机会，且回报检查的观测差值常小于其估计噪声尺度。该结果支持进一步检验检查成本与更新控制，不足以证明其误拒绝、保护收益、算法优越性或收敛。',
        f"[训练监控图]({suite.name}/live_monitor.png)；[源总览]({suite.name}/overview.png)；[逐节点原始曲线]({suite.name}/learning_curves.csv)。",
        '## 7. 输入指纹与复核',
        '复核命令：`python manifold_project/experiments/assess_navigation_reference.py`。该命令仅重新生成本判断文档，不修改源结果。',
        table(['输入','SHA256'],[(p.relative_to(suite).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()) for p in
            (suite/'manifest.json',suite/'learning_curves.csv',suite/'training_results.csv',run/'summary.json',run/'events.jsonl')])]
    output = suite.parent/f'{suite.name}_实验判断.md'
    output.write_text('\n\n'.join(body)+'\n', encoding='utf-8')
    print(f'Audit passed; {len(commits)} commits; {len(checks)} candidate return checks; stability={stability["code"]}')
    print(output)


if __name__ == '__main__':
    main()
