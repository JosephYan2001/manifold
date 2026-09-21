"""Descriptive source-curve stability checks, never a convergence guarantee."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st


def assess_arrival(rows, budget, horizon, window=5):
    grid = sorted((r for r in rows if int(r['budget_checkpoint']) < budget), key=lambda r:int(r['budget_checkpoint']))
    final = [r for r in rows if int(r['budget_checkpoint']) == budget]
    if len(grid) < window or len(final) != 1 or int(grid[-window]['budget_checkpoint']) < budget*.7:
        return dict(code='insufficient_nodes', status='节点不足；需末30%预算内至少5个非final节点')
    tail, last = grid[-window:], final[0]
    tolerances = {'success_rate':.02, 'restricted_mean_steps':.02*horizon}
    ranges = {k:max(float(r[k]) for r in tail)-min(float(r[k]) for r in tail) for k in tolerances}
    precise = all(r.get(k+'_se') not in (None,'') and 1.96*float(r[k+'_se']) <= tol
                  for r in tail+[last] for k,tol in tolerances.items())
    changes = {k:st.mean(float(r[k]) for r in tail[-2:])-st.mean(float(r[k]) for r in tail[:2]) for k in tolerances}
    stable = all(ranges[k] <= tol and abs(float(last[k])-st.mean(float(r[k]) for r in tail[-2:])) <= tol
                 for k,tol in tolerances.items())
    if not precise:
        code,status = 'insufficient_precision','评价精度不足'
    elif changes['success_rate'] > .02 or changes['restricted_mean_steps'] < -.02*horizon:
        code,status = 'improving','首达表现仍改善，尚不判定平台'
    elif changes['success_rate'] < -.02 or changes['restricted_mean_steps'] > .02*horizon:
        code,status = 'declining','首达表现退化，检查优化稳定性'
    elif stable:
        high = float(last['success_rate']) >= .9
        code,status = ('stable_high','高成功率且经验稳定') if high else ('stable_low','低成功率平台，优先诊断')
    else:
        code,status = 'fluctuating','仍有波动'
    return dict(code=code,status=status,final_success_rate=float(last['success_rate']),
                final_restricted_mean_steps=float(last['restricted_mean_steps']),
                success_rate_change=changes['success_rate'], restricted_steps_change=changes['restricted_mean_steps'],
                ranges=ranges, evaluation_resolution_ok=precise)


def assess(rows, budget, window=5, j_tol=2., coverage_tol=.02, distance_tol=.02, success=.8):
    # Final uses a different random stream: report it separately, do not merge it into the window.
    grid = sorted((r for r in rows if int(r['budget_checkpoint']) < budget), key=lambda r:int(r['budget_checkpoint']))
    final = [r for r in rows if int(r['budget_checkpoint']) == budget]
    if len(grid) < window or len(final) != 1 or int(grid[-window]['budget_checkpoint']) < budget*.7:
        return dict(code='insufficient_nodes', status='节点不足或不够密集，不能判断后期稳定', detail='需末30%预算内至少5个非final节点')
    tail = grid[-window:]
    ranges = {k:max(float(r[k]) for r in tail)-min(float(r[k]) for r in tail) for k in ('J','coverage','distance')}
    delta = st.mean(float(r['J']) for r in tail[-2:])-st.mean(float(r['J']) for r in tail[:2])
    tolerances = {'J':j_tol,'coverage':coverage_tol,'distance':distance_tol}
    precise = all(r.get(k+'_se') not in (None,'') and 1.96*float(r[k+'_se']) <= tol
                  for r in tail+[final[0]] for k,tol in tolerances.items())
    stationary = all(ranges[k] <= tol for k,tol in tolerances.items())
    final_consistent = all(abs(float(final[0][k])-st.mean(float(r[k]) for r in tail[-2:])) <= tol for k,tol in tolerances.items())
    if not precise:
        code='insufficient_precision'
        status='评价精度不足，稳定性待确认'
    elif delta > j_tol:
        code='improving'
        status='末段回报仍改善，值得扩展预算'
    elif delta < -j_tol:
        code='declining'
        status='末段回报下降，先排查稳定性'
    elif stationary and final_consistent:
        code='stable_high' if float(final[0]['coverage']) >= success else 'stable_low'
        status=('高覆盖且经验稳定' if float(final[0]['coverage']) >= success else '低覆盖平台，优先调参/诊断')
    else:
        code='fluctuating'
        status='仍有波动，未达到经验稳定标准'
    return dict(code=code, status=status, tail_return_change=delta, J_range=ranges['J'], coverage_range=ranges['coverage'],
                distance_range=ranges['distance'], evaluation_resolution_ok=precise,
                final_J=float(final[0]['J']), final_coverage=float(final[0]['coverage']))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    args=p.parse_args(argv)
    manifest=json.loads((args.suite/'manifest.json').read_text(encoding='utf-8'))
    with (args.suite/'learning_curves.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if manifest['config'].get('task_mode') == 'first_arrival':
        lines = ['# 首达任务经验稳定性检查',
                 '末30%预算内最后5个非final节点：成功率极差≤0.02，截尾完成步数极差≤任务上限的2%；'
                 '对应近似95%误差半径及final与末段差值也须满足阈值。成功率≥0.9才称高平台。'
                 '这些是开发诊断约定，非收敛证明、统计等效检验或自动停止规则。',
                 '| 条件/种子 | 判断 | final成功率 | final截尾步数 |', '|---|---|---|---|']
        for job in manifest['jobs']:
            subset = [r for r in rows if r['condition']==job['condition'] and int(r['seed'])==job['seed']]
            result = assess_arrival(subset,manifest['config']['budget'],manifest['config']['task_horizon'])
            lines.append(f'| {job["condition"]}/{job["seed"]} | {result["status"]} | '
                         f'{result.get("final_success_rate", "—")} | {result.get("final_restricted_mean_steps", "—")} |')
            print(job['condition'],job['seed'],result)
        (args.suite/'稳定性检查.md').write_text('\n\n'.join(lines[:2])+'\n\n'+'\n'.join(lines[2:])+'\n',encoding='utf-8')
        return
    lines=['# 导航源策略经验稳定性检查',
           '这是预定实用阈值下的诊断，不是数学收敛证明、总体置信保证或自动停止规则。仍需多种子验证。',
           '末30%预算内取最后5个source-grid节点；回报极差≤2、覆盖极差≤0.02、距离极差≤0.02；各节点与final的近似95%误差半径也须低于相应阈值。final采用独立随机流，另核对其与末段均值一致。覆盖≥0.8才称高覆盖平台。阈值为开发约定，不随结果自动调整。',
           '各节点复用评价场景，节点不是独立训练重复。标准误来自同一冻结策略的独立回合；正态近似只是分辨率诊断，不据此声称等效或安全。缺失标准误的旧结果不补造不确定性。',
           '| 条件/种子 | 判断 | final J | final覆盖 | 末段J变化 |', '|---|---|---|---|---|']
    for job in manifest['jobs']:
        subset=[r for r in rows if r['condition']==job['condition'] and int(r['seed'])==job['seed']]
        result=assess(subset,manifest['config']['budget'])
        lines.append(f"| {job['condition']}/{job['seed']} | {result['status']} | {result.get('final_J','—')} | {result.get('final_coverage','—')} | {result.get('tail_return_change','—')} |")
        print(job['condition'],job['seed'],result)
    (args.suite/'稳定性检查.md').write_text('\n\n'.join(lines[:4])+'\n\n'+'\n'.join(lines[4:])+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
