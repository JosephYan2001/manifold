import json
from copy import deepcopy
from pathlib import Path

STANDARD_CONDITIONS = ['ours', 'sampled', 'mappo', 'ippo', 'no_direction_check', 'no_return_check', 'fit_quarter']
CONDITIONS = STANDARD_CONDITIONS + ['no_checks', 'direct']
EXPERIMENTS = {'N-C': ['ours','sampled','direct','mappo','ippo'], 'N-A': STANDARD_CONDITIONS[:2]+STANDARD_CONDITIONS[4:],
               'N-A1': ['ours','sampled'], 'N-A2': ['ours','no_direction_check'],
               'N-A3': ['ours','no_return_check'], 'N-A4': ['ours','fit_quarter'],
               'N-A5': ['ours','no_checks'],
               'N-P': STANDARD_CONDITIONS, 'N-S1': STANDARD_CONDITIONS, 'N-D': ['ours','no_return_check'],
               'N-L': CONDITIONS[:4], 'N-T': []}


def continuing_task(config):
    # Saved configurations predating this field describe the finite-horizon task.
    return config.get('task_mode', 'finite_horizon') == 'continuing'


def arrival_task(config):
    return config.get('task_mode') == 'first_arrival'


def episode_horizon(config):
    return config['task_horizon'] if arrival_task(config) else config['horizon']


def load_config(profile, overrides=None):
    root = Path(__file__).parent/'configs'
    config = json.loads((root/'source.json').read_text(encoding='utf-8'))
    config.update(json.loads((root/f'{profile}.json').read_text(encoding='utf-8')))
    if overrides:
        values = json.loads(Path(overrides).read_text(encoding='utf-8-sig'))
        # Canonical shared PPO names; preserve old MAPPO-only override files.
        for suffix in ('value_normalization', 'value_clipping', 'huber_loss'):
            old, new = f'mappo_{suffix}', f'ppo_{suffix}'
            if old in values:
                if new in values and values[new] != values[old]:
                    raise ValueError(f'{old} 与 {new} 设置冲突')
                values[new] = values.pop(old)
        unknown = set(values)-set(config)
        if unknown:
            raise ValueError(f'未知配置键: {sorted(unknown)}')
        config.update(values)
    validate(config)
    return config


def validate(c):
    if c.get('ppo_activation', 'relu') not in ('relu', 'tanh'):
        raise ValueError('ppo_activation 只支持 relu / tanh')
    for key in ('mappo_backend', 'ippo_backend'):
        if c.get(key, 'local') not in ('author', 'local'):
            raise ValueError(f'{key} 只支持 author / local')
    for key in ('ppo_value_normalization', 'ppo_value_clipping', 'ppo_huber_loss'):
        if key in c and not isinstance(c[key], bool):
            raise ValueError(f'{key} 必须是布尔值')
    if c.get('task_mode', 'finite_horizon') not in ('continuing', 'finite_horizon', 'first_arrival'):
        raise ValueError('task_mode 只支持 continuing / finite_horizon / first_arrival')
    if arrival_task(c) and (not isinstance(c.get('task_horizon'), int) or
                            isinstance(c['task_horizon'], bool) or c['task_horizon'] <= 0):
        raise ValueError('first_arrival 要求正整数 task_horizon')
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
    if c.get('observation_protocol') == 'entities_v1':
        if c['history'] != 1 or c['agent_neighbors'] is not None or c['landmark_neighbors'] is not None:
            raise ValueError('entities_v1 uses current-frame full native lists (neighbors=null, history=1)')
        if c['hidden'] % c['heads']:
            raise ValueError('hidden must be divisible by heads')
    elif c['agent_neighbors'] != 2 or c['landmark_neighbors'] != 2:
        raise ValueError('Legacy flat observation requires the original nearest 2/2 fields')
    if c['n_agents'] != 4 or c['local_ratio'] != .5:
        raise ValueError('当前协议固定 N=4、最近邻 2/2、local_ratio=0.5')


def condition_config(config, condition):
    result = deepcopy(config)
    group = condition if condition in ('mappo','ippo') else 'ours'
    result.update(config['method_overrides'].get(group,{}))
    if condition in ('mappo','ippo') and result.get(f'{condition}_backend', 'local') == 'author':
        if result['ppo_label'] != 'gae' or not result['ppo_normalize_advantage']:
            raise ValueError(f'作者 PPO 保留 GAE 和优势标准化；N-L 必须显式设置 {condition}_backend=local')
        from math import ceil
        batches = ceil(result['train_episodes']/result['minibatch_episodes'])
        if not result.get('complete_episodes', False) and result['train_episodes']*result['horizon']*result['n_agents'] % batches:
            raise ValueError('作者 PPO 总 agent 样本数必须可被小批次数整除')
    return result
