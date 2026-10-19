"""Translate the paper matrix into the restored Runner, without a second loop."""
from pathlib import Path
import math
import time


CONDITIONS = {'AN': 'ours', 'SA': 'sampled', 'DA': 'direct',
              'MAPPO-E': 'mappo', 'IPPO-E': 'ippo'}


def runner_config(config, environment='navigation', method=None):
    from .configs import load_config
    c = load_config('smoke' if config.get('smoke_only') else 'pilot')
    horizon = int(config['horizon'])
    c.update(environment=environment, current_config=dict(config),
        observation_protocol='entities_v1', observation_mode=config['observation_mode'],
        hidden=config['hidden'], heads=config['heads'], relation_layers=config['relation_layers'],
        network_chunk_size=config.get('network_chunk_size', 256),
        task_mode='first_arrival', task_horizon=horizon, horizon=horizon, history=config.get('history', 1),
        complete_episodes=True, device=config['device'], threads=config.get('torch_threads', 1),
        n_agents=config['n_agents'], budget=config['budget'], seeds=[config['seed']],
        beta=config['probability_floor'], q_max=config['q_bound'], gamma=config['gamma'],
        actor_lr=config['actor_lr'], direction_lr=config['direction_lr'], critic_lr=config['critic_lr'],
        direction_epochs=config['direction_steps'], actor_epochs=config['actor_fit_steps'],
        critic_epochs=config['critic_steps'], ppo_epochs=config['ppo_epochs'],
        train_episodes=config['batch_episodes'],
        minibatch_episodes=max(1,min(config['batch_episodes'],config['minibatch_size']//(horizon*config['n_agents']))),
        direction_check_episodes=config['check_episodes'], return_check_episodes=config['check_episodes'],
        direction_label=config['label_mode'], ppo_label='gae', gae_lambda=config['gae_lambda'],
        ppo_clip=config['ppo_clip'], entropy_coef=config['entropy_coef'], grad_norm=config['max_grad_norm'],
        eta=config['step_sizes'][0], step_sizes=config['step_sizes'], attempts=len(config['step_sizes']),
        direction_threshold=config['direction_threshold'], return_tolerance=config['return_tolerance'],
        ppo_value_normalization=config['value_normalization'],
        initial_scale=config.get('initial_scale', 1.), local_ratio=config.get('local_ratio', .5),
        coverage_radius=config.get('coverage_radius', .1),
        render_mode=config.get('render_mode'), budget_includes_monitoring=True,
        eval_episodes=config['source_eval_episodes'], final_episodes=config['source_eval_episodes'],
        eval_fractions=[0., 1.], transfer_sizes=config['target_sizes'],
        transfer_episodes=config['evaluation_episodes'], protocol_status='candidate_not_frozen')
    # Preserve requested monitor spacing, without making many duplicate nodes.
    interval = config['eval_every_steps']
    c['eval_fractions'] = sorted(set([0., 1.] + [i/c['budget'] for i in range(interval, c['budget'], interval)]))
    dims = {'navigation': (5, 6, 5), 'warehouse': (9, 11, 5), 'pair_entities': (3, 6, 2)}[environment]
    c.update(entity_self_dim=dims[0], entity_record_dim=dims[1], action_dim=dims[2])
    if environment == 'navigation' and config.get('navigation_task') == 'success_only':
        c.update(task_mode='success_only', complete_episodes=False, entity_self_dim=4,
                 evaluation_horizon=config['evaluation_horizon'],
                 train_reset_horizon=config.get('train_reset_horizon'))
        if config.get('collection_mode') == 'episodes':
            c.update(complete_episodes=True, train_reset_horizon=horizon)
    if method is not None:
        c.update(semantics_version='method-gates-v2', requested_method=method,
                 direction_objective='sampled_square' if method == 'SA' else 'analytic_fisher',
                 direction_gate=method in ('AN','SA') and config['direction_check'],
                 return_gate=method in ('AN','SA','DA') and config['return_check'])
    for key, default in [('direction_diagnostic_rounds', []), ('direction_diagnostic_episodes', 16),
                         ('actor_lr_schedule', 'fixed'), ('actor_lr_final', None), ('actor_lr_decay_steps', None),
                         ('save_direction_snapshots', False), ('actor_fit_selection', 'last'),
                         ('actor_fit_selection_atol', 1e-12), ('actor_fit_selection_rtol', 1e-6)]:
        c[key] = config.get(key, default)
    return c


def train(environment, method, config, output, resume=False, resume_from=None):
    from .training.runner import Runner, minimum_cost
    from .training.storage import load_pt, save_pt
    from ..applications.envs import make_env
    from ..common.reporting import write_csv, plot_overview
    from ..common.runtime import configure_torch
    configure_torch(config)
    c = runner_config(config, environment, method)
    saved = None
    if resume or resume_from:
        saved = load_pt(Path(output)/'checkpoints/final.pt' if resume else resume_from)
        from .training.semantics import validate_legacy_resume
        validate_legacy_resume(saved,method)
        from ..recovery import restored_config
        c = restored_config(saved, config)
    condition = CONDITIONS[method]
    if method in ('AN', 'SA'):
        if not config['direction_check'] and not config['return_check']:
            condition = 'no_checks'
        elif not config['direction_check']:
            condition = 'no_direction_check'
        elif not config['return_check']:
            condition = 'no_return_check'
    directory = Path(output)
    from .configs import evaluation_horizon
    monitor_cap = len(c['eval_fractions'])*c['eval_episodes']*evaluation_horizon(c)
    if saved is None and minimum_cost(c, condition)+monitor_cap > c['budget']:
        raise ValueError('Budget cannot reserve a complete source update and its declared source monitoring')
    env = make_env(environment, config)
    try:
        protocol = env.protocol()
    finally:
        env.close()
    def progress(runner):
        rows = [dict(row, method=method, seed=config['seed'], source_steps_total=row['source_steps'],
                     train_discounted_return=row['train_J']) for row in runner.rounds]
        for row in rows:
            row['protocol_version'] = config['protocol_version']
            eligible = [node for node in runner.curves if node['budget_checkpoint'] <= row['source_steps']]
            if eligible:
                node = eligible[-1]
                row.update(monitor_return=node['J'], monitor_budget_checkpoint=node['budget_checkpoint'])
                if 'success_rate' in node:
                    row.update(eval_success=node['success_rate'],
                               eval_restricted_mean_steps=node['restricted_mean_steps'])
            if c['task_mode'] == 'success_only':
                row['evaluation_horizon'] = c['evaluation_horizon']
                row['train_reset_horizon'] = c.get('train_reset_horizon')
        write_csv(directory/'training.csv', rows)
        if config.get('plot') and runner.round and runner.round % config.get('plot_every', 10) == 0:
            plot_overview(directory, rows)
    started = time.perf_counter()
    runner = Runner(directory, c, condition, config['seed'], progress=progress, resume=resume)
    if resume_from:
        runner.restore(saved,external=True)
        runner.complete = False
        runner.checkpoint()
        # Preserve historical monitoring/optimization events for the usual plots.
        source_events = Path(resume_from).parent.parent/'events.jsonl'
        if source_events.is_file():
            import shutil
            with source_events.open('rb') as src, (directory/'events.jsonl').open('xb') as dst:
                shutil.copyfileobj(src, dst)
    probe = None
    if method == 'AN' and config.get('save_fit_diagnostics', False):
        from .evaluation.fit_probe import FitProbe
        probe = FitProbe(runner, config, directory)
        runner.fit_probe = probe
    try:
        summary = runner.run()
    finally:
        runner.sampler.close()
    if probe:
        probe.finish()
    if not runner.round:
        raise RuntimeError('No training round completed')
    checkpoint = directory/'checkpoints/final.pt'
    # Enrich the original resumable checkpoint; do not save a parallel model.
    saved = load_pt(checkpoint)
    saved.update(current_method=method, current_environment=environment, current_protocol=protocol)
    save_pt(checkpoint, saved)
    record = dict(method=method, seed=config['seed'], source_steps_total=runner.sampler.used,
        budget=c['budget'], unused_budget=c['budget']-runner.sampler.used, rounds=runner.round,
        actor_updates=summary['actor_changed_rounds'], actor_parameters=summary['actor_parameters'],
        critic_parameters=summary['critic_parameters'], checkpoint=str(checkpoint),
        diagnostic_steps=runner.diagnostic_steps+(probe.steps if probe else 0),
        wall_seconds=time.perf_counter()-started, training_backend='restored_cooperative_navigation.Runner')
    record.update({f'source_{key}_steps': value for key,value in runner.sampler.costs.items()})
    return dict(records=[record], checkpoint=str(checkpoint), config=config, protocol=protocol,
                source_steps_total=runner.sampler.used, diagnostic_steps=runner.diagnostic_steps+(probe.steps if probe else 0))
