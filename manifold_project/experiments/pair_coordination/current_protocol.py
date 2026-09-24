"""Paper P-L configuration and reports around the original pair training loop."""
from collections import Counter
from pathlib import Path
import json
import numpy as np


def train(config, output):
    from .envs.pair_coordination import PairConfig
    from .models.mlp import load_actor
    from .training.config import TrainConfig
    from .training.runner import run_training
    from .training.logging import read_records
    from ..finite.diagnostics import settings, configured_labels
    from ..finite.environments import pair_source, PairGame
    from ..finite.training import _report_evaluation
    from ..common.reporting import write_csv
    cfg = settings(config)
    env = pair_source(config)
    source = PairConfig(n_agents=env.n, type_probs=(1-env.plus_type_probability, env.plus_type_probability),
                        local_bias=(-.1,.2), pair_payoff=((1.,1.),(1.,1.)),
                        interaction_strength=env.interaction_strength)
    records, training, evaluations = [], [], []
    for label in configured_labels(config, 'learning_labels', ('mc',), learning=True):
        if label not in ('mc_learned_baseline', 'mc_raw'):
            raise ValueError('P-L uses source MC labels; exact advantage remains a fixed-base diagnostic')
        for method in cfg['methods']:
            directory = Path(output)/label/f'{method}_s{cfg["seed"]}'
            args = TrainConfig(device=config.get('device','cpu'), algorithm='direct' if method == 'DA' else 'direction',
                mode='policy', method='sampled' if method == 'SA' else 'analytic',
                actor_model='table', direction_model='table', initial_probabilities=[.3,.7],
                baseline='table' if label == 'mc_learned_baseline' else 'zero', acceptance='empirical',
                seed=cfg['seed'], episodes=config.get('batch_episodes',128),
                critic_episodes=config.get('critic_episodes',128), check_episodes=config.get('check_episodes',8),
                direction_steps=cfg['direction_steps'], fit_steps=cfg['actor_fit_steps'],
                direction_lr=cfg['direction_lr'], fit_lr=cfg['actor_lr'], q_max=cfg['q_bound'], beta=1e-12,
                eta=cfg['step_sizes'][0], step_sizes=cfg['step_sizes'], attempts=len(cfg['step_sizes']), rounds=config['budget'],
                direction_threshold=cfg['direction_threshold'], return_tolerance=cfg['return_tolerance'],
                budget=config['budget'], direction_check_enabled=cfg['direction_check'],
                return_check_enabled=cfg['return_check'], checkpoint_selection='empirical',
                monitor_episodes=config.get('monitor_episodes',config.get('source_eval_episodes',64)),
                save_update_logs=False, log_every=max(1,cfg['direction_steps']))
            result = run_training(source, args, directory, compact_output=True)
            policy = load_actor(result['actor']).table()[:,1]
            costs = Counter(dict.fromkeys(('training','critic','direction_check','return_check','monitor'),0))
            for row in read_records(directory/'batches.jsonl'):
                purpose = {'direction_train':'training', 'pg_train':'training', 'return_old':'return_check',
                           'return_candidate':'return_check', 'source_monitor':'monitor'}.get(row['purpose'],row['purpose'])
                costs[purpose] += row['episodes']
            rows = read_records(directory/'rounds.jsonl')
            for row in rows:
                training.append(dict(row, experiment='P-L', method=method, label=label, seed=cfg['seed'],
                                     source_steps_total=row['source_episodes']))
            record = dict(experiment='P-L',method=method,label=label,seed=cfg['seed'],
                source_steps_total=result['source_episodes'], budget=config['budget'],
                unused_budget=config['budget']-result['source_episodes'], updates=result['rounds'],
                final_source_return_diagnostic=env.oracle(policy)['expected_return'],
                final_checkpoint=str(Path(result['checkpoint']).relative_to(output)),
                best_source_checkpoint=str(Path(result['best_checkpoint']).relative_to(output)),
                checkpoint_selection='final_primary_best_by_independent_source_monitor_only',
                training_backend='restored_pair_coordination.run_training',
                source_report_only_steps=cfg['evaluation_episodes'],
                target_report_only_steps=3*len(cfg['target_sizes'])*cfg['evaluation_episodes'])
            record.update({f'source_steps_{key}':value for key,value in costs.items()})
            records.append(record)
            evaluations.extend(_report_evaluation(env,policy,method,label,cfg['seed']+10_000_000,cfg['evaluation_episodes'],'source'))
            for variant in ('normalized','fixed-edge','composition'):
                for n in cfg['target_sizes']:
                    target=PairGame(n,.4 if variant=='composition' else env.plus_type_probability,variant,env.n,env.interaction_strength)
                    evaluations.extend(_report_evaluation(target,policy,method,label,cfg['seed']+10_000_000,cfg['evaluation_episodes'],variant))
                    records.append(dict(experiment='P-L-final-transfer',method=method,label=label,seed=cfg['seed'],
                        variant=variant,source_n=env.n,target_n=n,source_steps_total=result['source_episodes'],
                        frozen_actor_return_diagnostic=target.oracle(policy)['expected_return'],target_training_steps=0))
    write_csv(Path(output)/'training.csv',training)
    write_csv(Path(output)/'evaluation_episodes.csv',evaluations)
    return records, training, evaluations
