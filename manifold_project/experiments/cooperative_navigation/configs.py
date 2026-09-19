import json
from copy import deepcopy
from pathlib import Path

STANDARD_CONDITIONS = ['ours', 'sampled', 'mappo', 'ippo', 'no_direction_check', 'no_return_check', 'fit_quarter']
CONDITIONS = STANDARD_CONDITIONS + ['no_checks']
EXPERIMENTS = {'N-C': STANDARD_CONDITIONS[:4], 'N-A': STANDARD_CONDITIONS[:2]+STANDARD_CONDITIONS[4:],
               'N-A1': ['ours','sampled'], 'N-A2': ['ours','no_direction_check'],
               'N-A3': ['ours','no_return_check'], 'N-A4': ['ours','fit_quarter'],
               'N-A5': ['ours','no_checks'],
               'N-P': STANDARD_CONDITIONS, 'N-S1': STANDARD_CONDITIONS, 'N-D': ['ours','no_return_check'],
               'N-L': CONDITIONS[:4], 'N-T': []}


def continuing_task(config):
    # Saved configurations predating this field describe the finite-horizon task.
    return config.get('task_mode', 'finite_horizon') == 'continuing'


def load_config(profile, overrides=None):
    root = Path(__file__).parent/'configs'
    config = json.loads((root/'source.json').read_text(encoding='utf-8'))
    config.update(json.loads((root/f'{profile}.json').read_text(encoding='utf-8')))
    if overrides:
        values = json.loads(Path(overrides).read_text(encoding='utf-8-sig'))
        unknown = set(values)-set(config)
        if unknown:
            raise ValueError(f'未知配置键: {sorted(unknown)}')
        config.update(values)
    validate(config)
    return config


def validate(c):
    if c.get('task_mode', 'finite_horizon') not in ('continuing', 'finite_horizon'):
        raise ValueError('task_mode 只支持 continuing / finite_horizon')
    if continuing_task(c) and not 0 < c['gamma'] < 1:
        raise ValueError('持续任务要求 0 < gamma < 1')
    if not isinstance(c['method_overrides'],dict) or set(c['method_overrides'])-{'ours','mappo','ippo'}:
        raise ValueError('method_overrides 只允许 ours/mappo/ippo；sampled 与消融自动继承 ours')
    for method,values in c['method_overrides'].items():
        allowed = {'actor_lr','critic_lr','ppo_epochs','ppo_clip'} if method in ('mappo','ippo') else {'actor_lr','critic_lr','direction_lr','eta'}
        if not isinstance(values,dict) or set(values)-allowed:
            raise ValueError(f'{method} 分方法覆盖只允许 {sorted(allowed)}')
        merged = deepcopy(c)
        merged.update(values)
        merged['method_overrides'] = {}
        validate(merged)
    for key in ('budget','horizon','history','hidden','train_episodes','direction_check_episodes',
                'return_check_episodes','minibatch_episodes','direction_epochs','actor_epochs',
                'critic_epochs','ppo_epochs','attempts','eval_episodes','final_episodes',
                'transfer_episodes','audit_episodes','bootstrap_repeats','threads'):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] <= 0:
            raise ValueError(f'{key} 必须是正整数')
    if not 0 < c['beta'] < 1 or not 0 < c['gamma'] <= 1 or not 0 <= c['gae_lambda'] <= 1:
        raise ValueError('非法概率/折扣参数')
    for key in ('actor_lr','direction_lr','critic_lr','q_max','grad_norm','eta','value_coef','ppo_clip'):
        if not 0 < c[key] < float('inf'):
            raise ValueError(f'{key} 必须是有限正数')
    fractions = c['eval_fractions']
    if fractions != sorted(set(fractions)) or fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError('评价节点必须严格递增且包含 0 与 1')
    if len(set(round(x*c['budget']) for x in fractions)) != len(fractions):
        raise ValueError('预算过小，评价节点重合')
    if c['direction_label'] not in ('mc','gae') or c['ppo_label'] not in ('mc','gae'):
        raise ValueError('标签只支持 mc/gae')
    if not c['seeds'] or len(c['seeds']) != len(set(c['seeds'])):
        raise ValueError('种子不能为空或重复')
    if any(not isinstance(s,int) or isinstance(s,bool) or s < 0 for s in c['seeds']):
        raise ValueError('种子必须是非负整数')
    if c['protocol_status'] not in ('candidate_not_frozen','frozen'):
        raise ValueError('protocol_status 必须为 candidate_not_frozen 或 frozen')
    if any(not isinstance(n,int) or n < 2 for n in c['transfer_sizes']):
        raise ValueError('迁移人数必须为不小于 2 的整数')
    if c['n_agents'] != 4 or c['agent_neighbors'] != 2 or c['landmark_neighbors'] != 2 or c['local_ratio'] != .5:
        raise ValueError('当前协议固定 N=4、最近邻 2/2、local_ratio=0.5')


def condition_config(config, condition):
    result = deepcopy(config)
    group = condition if condition in ('mappo','ippo') else 'ours'
    result.update(config['method_overrides'].get(group,{}))
    return result
