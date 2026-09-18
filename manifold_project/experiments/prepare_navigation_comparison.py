"""Derive matched comparison and bounded PPO search configs from a completed ours suite."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from manifold_project.experiments.cooperative_navigation.configs import condition_config, validate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected-suite',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    source=args.selected_suite/'manifest.json'
    m=json.loads(source.read_text(encoding='utf-8'))
    jobs=[j for j in m['jobs'] if j['condition']=='ours']
    if not jobs or any(not (args.selected_suite/j['path']/'summary.json').exists() or
                       json.loads((args.selected_suite/j['path']/'summary.json').read_text(encoding='utf-8'))['status']!='complete' for j in jobs):
        p.error('Choose a completed source-development ours suite')
    if m['profile']!='pilot':
        p.error('Select a pilot suite; do not tune on formal results')
    c=condition_config(m['config'],'ours')
    c['method_overrides']={}
    c['protocol_status']='candidate_not_frozen'
    validate(c)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('matched.json',c)
    # Four PPO candidates per algorithm, using the same shared settings and source budget.
    for clip in (.1,.2):
        for epochs in (5,10):
            v=deepcopy(c)
            v['method_overrides']={a:{'ppo_clip':clip,'ppo_epochs':epochs} for a in ('mappo','ippo')}
            validate(v)
            save(f'ppo_clip{clip}_epochs{epochs}.json',v)
    (args.output_dir/'来源与使用.md').write_text(
        '# 比较配置来源\n\n'+f'选择依据来自源开发套件：{args.selected_suite.resolve()}\n\n'
        +f'manifest SHA256：{hashlib.sha256(source.read_bytes()).hexdigest()}\n\n'
        +'matched.json继承所选ours的公共参数；四个ppo候选只改变PPO专属clip/epochs。'
        +'这些配置未自动宣布最优或冻结formal。候选选择必须只用源开发结果；'
        +'预算、种子和设备由命令统一指定。sampled与检查消融必须继承matched配置，不单独调参。'
        +'若选择了不同的MAPPO/IPPO候选，分别合并其method_overrides后再冻结配置；不要更改公共环境/奖励/评价设置。\n',encoding='utf-8')
    print(f'Prepared matched and four PPO candidate configs: {args.output_dir}')


if __name__=='__main__':
    main()
