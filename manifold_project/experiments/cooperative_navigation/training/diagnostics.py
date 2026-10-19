"""Read-only numerical diagnostics; training objectives remain unchanged."""
from copy import deepcopy
import hashlib
import json
import torch
from .ppo import batch_weighted, valid_values, direction_terms
from .storage import seed_for, save_pt, event


def normalized(p):
    p = p.double()
    return p / p.sum(-1, keepdim=True)


def categorical_kl(p, q):
    p, q = normalized(p), normalized(q)
    # Per-state KL can be slightly negative only due to floating point rounding.
    return (p * (p.log()-q.log())).sum(-1).clamp_min(0.)


def saturation_metrics(u, q, q_max):
    k = q.shape[-1]
    bound = q_max*(k-1)/k
    return dict(tanh_saturation=float((u.abs()>.95).double().mean()),
                effective_bound_fraction=float((q.abs()>.95*bound).double().mean()),
                q_action_span=float((q.max(-1).values-q.min(-1).values).double().mean()),
                q_abs_mean=float(q.abs().double().mean()), q_abs_max=float(q.abs().max()),
                saturation_threshold=.95, effective_component_bound=bound,
                q_max=q_max, action_count=k)


@torch.no_grad()
def direction_metrics(direction, batch, config):
    u, q = direction.components(batch['x'])
    score, fisher = direction_terms(q.double(), normalized(batch['mu']), batch['actions'])
    d = float(batch_weighted(batch['advantage'].double()*score,batch,config))
    e = float(batch_weighted(fisher,batch,config))
    sampled_e = float(batch_weighted(score.square(),batch,config))
    return dict(B_label=2*d-e,D_label=d,fisher_energy=e,sampled_energy=sampled_e,
                analytic_risk=e-2*d,sampled_risk=sampled_e-2*d,
                **saturation_metrics(valid_values(u,batch),valid_values(q,batch),direction.q_max))


@torch.no_grad()
def critic_metrics(critic, batch, config, local=False):
    value = critic(batch['x'] if local else batch['state']).double()
    target = batch['value_target'].double()
    weight = lambda x:batch_weighted(x,batch,config)
    mass = weight(torch.ones_like(target)).clamp_min(1e-12)
    error = target-value
    variance = weight((target-weight(target)/mass).square())
    error_variance = weight((error-weight(error)/mass).square())
    return dict(mse=float(weight(error.square())),
                explained_variance=float(1-error_variance/variance) if variance>1e-12 else None,
                target_mean=float(weight(target)/mass),value_mean=float(weight(value)/mass),
                target_variance=float(variance/mass),
                value_variance=float(weight((value-weight(value)/mass).square())/mass),
                aggregation='discount_weighted_valid_transitions_fixed_episode_denominator')


def probe_steps(directory):
    """Incomplete probes are charged their reserved cap, including after recovery."""
    path = directory/'direction_probe_ledger.jsonl'
    entries = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record=json.loads(line)['record'];entries[record['id']]=record['steps']
            except (ValueError,KeyError):
                continue
    return sum(entries.values())


def observe_direction(runner, old, critic, direction, batch):
    if runner.round+1 not in runner.c.get('direction_diagnostic_rounds',[]):
        return None
    from .collector import Collector
    from ..configs import episode_horizon
    c=runner.c
    devices=list(range(torch.cuda.device_count()))
    ledger=runner.directory/'direction_probe_ledger.jsonl'
    probe_id=len(ledger.read_text(encoding='utf-8').splitlines()) if ledger.exists() else 0
    event(ledger,'start',id=probe_id,round=runner.round+1,
          steps=c['direction_diagnostic_episodes']*episode_horizon(c))
    with torch.random.fork_rng(devices=devices):
        collector=Collector(dict(c,budget=10**12),'read_only_direction',seed_for(runner.seed,runner.round+1,'probe'))
        try:
            validation=collector.collect(old,c['direction_diagnostic_episodes'],'validation')
            validation['advantage'],validation['value_target']=runner.labels(validation,critic)
            metrics=direction_metrics(direction,validation,c)
            runner.log('direction_probe',**metrics,data_split='independent_old_policy',
                       label_source=f'old_critic_round_{runner.round}',label_kind=c['direction_label'],
                       aggregation='discount_weighted_valid_transitions_agents',
                       probe_used_for_selection=False,probe_steps=collector.used,
                       checkpoint_id=f'round_{runner.round+1}')
            event(ledger,'complete',id=probe_id,round=runner.round+1,steps=collector.used)
        finally:
            collector.close()
    return validation


def cpu_copy(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    return deepcopy(value)


@torch.no_grad()
def save_direction_snapshot(runner, old, critic, direction, batch, validation, candidate, eta, info):
    if validation is None or not runner.c.get('save_direction_snapshots',False):
        return
    validation=dict(validation)
    if eta is not None:
        validation['target']=torch.softmax(validation['mu'].log()+eta*direction(validation['x']),-1)
    state=dict(format_version=1,config=runner.c,condition=runner.condition,seed=runner.seed,
               round=runner.round+1,input_dim=runner.input_dim,state_dim=runner.state_dim,
               actor=old.state_dict(),critic=critic.state_dict(),direction=direction.state_dict(),
               candidate=None if candidate is None else candidate.state_dict(),
               batch=batch,validation=validation,eta=eta,info=info,
               train_metrics=direction_metrics(direction,batch,runner.c),
               validation_metrics=direction_metrics(direction,validation,runner.c),
               optimizer_updates=dict(runner.optim_steps),
               config_sha256=hashlib.sha256(json.dumps(runner.c,sort_keys=True).encode()).hexdigest(),
               used_for_selection=False)
    path=runner.directory/'direction_snapshots'/f'round_{runner.round+1:06d}.pt'
    save_pt(path,cpu_copy(state))
    runner.log('direction_snapshot',snapshot_path=str(path),used_for_selection=False)
