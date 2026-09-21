"""One committed round is the recovery boundary; evaluations cannot select updates."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from ..envs.navigation import Navigation
from ..configs import continuing_task, arrival_task, episode_horizon
from ..models import Actor, Critic, Direction
from ..evaluation.evaluate import evaluate, ARRIVAL_METRICS
from .collector import Collector
from .ppo import returns, value_targets, batch_weighted, valid_values, direction_terms, ppo_loss
from .storage import seed_for, event, save_json, save_pt, load_pt
from .author_ppo import AuthorPPO, enabled as author_ppo_enabled


def direction_check_enabled(condition):
    return condition not in ('mappo', 'ippo', 'no_direction_check', 'no_checks')


def return_check_enabled(condition):
    return condition not in ('mappo', 'ippo', 'no_return_check', 'no_checks')


def minimum_cost(config, condition):
    cost = config['train_episodes']*config['horizon']
    episodes = 0
    if condition not in ('mappo','ippo'):
        episodes += config['direction_check_episodes'] if direction_check_enabled(condition) else 0
        episodes += 2*config['return_check_episodes'] if return_check_enabled(condition) else 0
    return cost+episodes*episode_horizon(config)


class Runner:
    def __init__(self, directory, config, condition, seed, audit=False, resume=False, progress=None):
        self.directory, self.c, self.condition, self.seed = Path(directory), config, condition, seed
        self.directory.mkdir(parents=True, exist_ok=True)
        self.audit = audit and condition in ('ours','no_return_check')
        self.progress = progress
        self.log_path = self.directory/'events.jsonl'
        self.sampler = Collector(config, condition, seed, self.log)
        self.round, self.curves, self.rounds, self.snapshots = 0, [], [], []
        self.optim_steps = Counter()
        self.train_seconds, self.eval_seconds, self.eval_steps = 0., 0., 0
        self.best = None
        torch.set_num_threads(config['threads'])
        torch.manual_seed(seed_for('initialization', seed))
        torch.use_deterministic_algorithms(True)
        probe = Navigation(config)
        self.input_dim = config['history']*(probe.obs_dim+6)+int(probe.include_time)
        self.state_dim = probe.state_dim
        probe.close()
        self.author_ppo = AuthorPPO(self.input_dim, self.state_dim, config, condition) if author_ppo_enabled(config, condition) else None
        self.actor_kind = self.author_ppo.actor_kind if self.author_ppo else 'local_mlp_v1'
        if self.author_ppo:
            self.actor, self.critic = self.author_ppo.actor, self.author_ppo.policy.critic
            self.actor_optimizer = self.author_ppo.policy.actor_optimizer
            self.critic_optimizer = self.author_ppo.policy.critic_optimizer
        else:
            self.actor = Actor(self.input_dim, config).to(config['device'])
            self.critic = Critic(self.input_dim if condition == 'ippo' else self.state_dim, config).to(config['device'])
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        self.grid = [round(f*config['budget']) for f in config['eval_fractions']]
        identity = {'config':config, 'condition':condition, 'seed':seed, 'audit':self.audit}
        self.complete = False
        if resume:
            if json.loads((self.directory/'config.json').read_text(encoding='utf-8')) != identity:
                raise ValueError('恢复配置与原运行不一致')
            # Let each module/optimizer place its state. Adam's non-capturable
            # step counters must stay on CPU even when parameters use CUDA.
            self.restore(load_pt(self.directory/'checkpoints/final.pt'))
            self.reconcile_interruption()
        else:
            if (self.directory/'config.json').exists():
                raise FileExistsError(f'拒绝覆盖运行: {self.directory}')
            save_json(self.directory/'config.json', identity)

    def log(self, stream, **record):
        event(self.log_path, stream, round=self.round+1, **record)

    def restore(self, state):
        if state['config'].get('task_mode', 'finite_horizon') != self.c.get('task_mode', 'finite_horizon'):
            raise ValueError('不能在不同任务定义之间直接恢复；请新建实验')
        if state.get('actor_kind', 'local_mlp_v1') != self.actor_kind:
            raise ValueError('PPO 实现或条件已更换；旧本地模型不能恢复为作者版，MAPPO/IPPO也不能互相续训')
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        if self.author_ppo:
            self.author_ppo.restore_normalizer(state['value_normalizer'])
        for name in ('round','curves','rounds','snapshots','train_seconds','eval_seconds','eval_steps','best','complete'):
            setattr(self, name, state[name])
        self.sampler.costs = Counter(state['costs'])
        self.sampler.counts = Counter(state['counts'])
        self.optim_steps = Counter(state['optim_steps'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])

    def reconcile_interruption(self):
        # Completed episodes in the append-only ledger are never erased on resume.
        maximum = self.sampler.used
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding='utf-8').splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # an interrupted final write is not a completed episode
                if item['stream'] == 'sampling':
                    maximum = max(maximum, item['record']['used'])
                if item['stream'] == 'sampling_start':
                    record = item['record']
                    maximum = max(maximum, record['reserved_through'])
                    self.sampler.counts[record['purpose']] = max(self.sampler.counts[record['purpose']],record['episode']+1)
        failure = self.directory/'failure.json'
        if failure.exists():
            maximum = max(maximum, json.loads(failure.read_text(encoding='utf-8')).get('used', maximum))
        wasted = maximum-self.sampler.used
        if wasted:
            self.sampler.costs['interrupted_uncommitted'] += wasted
            self.log('recovery', wasted_steps=wasted, boundary_round=self.round)
            self.checkpoint()

    def checkpoint(self):
        state = {name: getattr(self, name) for name in ('round','curves','rounds','snapshots',
                 'train_seconds','eval_seconds','eval_steps','best','complete')}
        state.update(format_version=2, config=self.c, condition=self.condition, seed=self.seed,
            actor_kind=self.actor_kind,
            input_dim=self.input_dim, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            costs=dict(self.sampler.costs), counts=dict(self.sampler.counts), optim_steps=dict(self.optim_steps),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        if self.author_ppo:
            state.update(value_normalizer=self.author_ppo.normalizer_state(), implementation=self.author_ppo.description())
        save_pt(self.directory/'checkpoints/final.pt', state)
        if self.best is not None:
            save_pt(self.directory/'checkpoints/best.pt', self.best)
        if self.audit and self.snapshots:
            save_pt(self.directory/'audit_snapshots.pt', self.snapshots)

    def evaluate_node(self, actor, budget_node, actor_cost):
        started = time.perf_counter()
        final = budget_node == self.c['budget']
        count = self.c['final_episodes'] if final else self.c['eval_episodes']
        print(f'  [eval] 开始 {self.condition} seed={self.seed} node={budget_node} '
              f'episodes={count}', flush=True)
        metrics, episodes = evaluate(actor, self.c, self.condition, self.seed,
                              'source-final' if final else 'source-grid', count)
        # Descriptive episode uncertainty; never used to accept training updates.
        for key in ('J', 'coverage', 'distance', 'mean_reward', 'mean_coverage', 'tail_coverage', 'tail_all_covered'):
            values = np.asarray([e[key] for e in episodes], dtype=np.float64)
            metrics[key+'_se'] = float(values.std(ddof=1)/np.sqrt(count)) if count > 1 else None
        self.eval_seconds += time.perf_counter()-started
        if arrival_task(self.c):
            for key, episode_key in (('success_rate', 'success'), ('restricted_mean_steps', 'restricted_steps')):
                values = np.asarray([e[episode_key] for e in episodes], dtype=np.float64)
                metrics[key+'_se'] = float(values.std(ddof=1)/np.sqrt(count)) if count > 1 else None
        self.eval_steps += sum(e['episode_steps'] for e in episodes)
        row = dict(condition=self.condition, seed=self.seed, budget_checkpoint=budget_node,
                   actor_source_steps=actor_cost, episodes=count, **metrics)
        self.curves.append(row)
        self.log('evaluation', **row)
        print(f'  [eval] {self.condition} seed={self.seed} node={budget_node} '
              f'actor_steps={actor_cost} J={metrics["J"]:.4f} '
              f'coverage={metrics["coverage"]:.1%} distance={metrics["distance"]:.4f} '
              f'episodes={count}', flush=True)
        if arrival_task(self.c):
            print(f'  [arrival] success={metrics["success_rate"]:.1%} '
                  f'restricted_steps={metrics["restricted_mean_steps"]:.2f} '
                  f'success_steps={metrics["success_steps_mean"]}', flush=True)
        score = ((metrics['success_rate'], -metrics['restricted_mean_steps'], -metrics['collision_pairs_total'])
                 if arrival_task(self.c) else metrics['J'])
        if self.best is None or score > self.best['score']:
            self.best = {'actor':deepcopy(actor.state_dict()), 'config':self.c, 'input_dim':self.input_dim,
                         'actor_kind':self.actor_kind, 'condition':self.condition, 'seed':self.seed,
                         'score':score, 'selection':('success_rate_then_restricted_steps_then_collisions_diagnostic_only'
                             if arrival_task(self.c) else 'independent_source_evaluation_diagnostic_only'),
                         'budget_checkpoint':budget_node, 'actor_source_steps':actor_cost}

    def labels(self, batch, critic):
        with torch.no_grad():
            local = self.condition == 'ippo'
            values, target, gae_advantage = value_targets(batch, critic, self.c, local=local)
            label = self.c['ppo_label'] if self.condition in ('mappo','ippo') else self.c['direction_label']
            advantage = gae_advantage if label == 'gae' else target-values
            if not local:
                advantage = advantage.unsqueeze(-1).expand_as(batch['actions'])
            # Continuing: n-step target with frozen V(s_T), also for GAE PPO.
            return advantage.detach(), target.detach()

    def weighted(self, value, batch):
        return batch_weighted(value, batch, self.c)

    def check_returns(self, actor, purpose, critic):
        if not continuing_task(self.c):
            rows = self.sampler.collect(actor, self.c['return_check_episodes'], purpose, retain=False)
            scores = [r['J'] for r in rows]
            return scores, scores
        batch = self.sampler.collect(actor, self.c['return_check_episodes'], purpose)
        _, target, _ = value_targets(batch, critic, self.c)
        observed = returns(batch['reward'], self.c['gamma'])[:, 0]
        return target[:, 0].double().tolist(), observed.double().tolist()

    def optimize(self, model, optimizer, batch, epochs, module, objective):
        rng = np.random.default_rng(seed_for(self.seed, self.round+1, module, 'minibatches'))
        last, grad_before, grad_after = {}, [], []
        for _ in range(epochs):
            order = rng.permutation(len(batch['reward']))
            for start in range(0, len(order), self.c['minibatch_episodes']):
                indices = order[start:start+self.c['minibatch_episodes']]
                mb = {k:v[indices] for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, last = objective(mb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{module}: nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.c['grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError(f'{module}: nonfinite parameter')
                grad_before.append(float(norm))
                grad_after.append(min(float(norm), self.c['grad_norm']))
                self.optim_steps[module] += 1
        last.update(loss=float(loss.detach()), grad_norm_before=float(np.mean(grad_before)),
                    grad_norm_after=float(np.mean(grad_after)), updates=len(grad_before))
        self.log('optimization', module=module, **last)
        return last

    def update(self):
        if self.author_ppo:
            return self.update_author_ppo()
        old = deepcopy(self.actor)
        old_critic = deepcopy(self.critic).eval()
        batch = self.sampler.collect(old, self.c['train_episodes'], 'train')
        episode_returns = returns(batch['reward'],self.c['gamma'])[:,0].double()
        self.log('train_batch',episodes=len(episode_returns),J_mean=float(episode_returns.mean()),
                 J_std=float(episode_returns.std(unbiased=False)))
        advantage, target = self.labels(batch, old_critic)
        batch.update(advantage=advantage, value_target=target)
        def critic_loss(mb):
            value = self.critic(mb['x'] if self.condition == 'ippo' else mb['state'])
            mse = self.weighted((value-mb['value_target']).square(), mb)
            with torch.no_grad():
                variance = valid_values(mb['value_target'], mb).var(unbiased=False)
                explained = float(1-valid_values(mb['value_target']-value, mb).var(unbiased=False)/variance) if variance > 1e-12 else None
            return self.c['value_coef']*mse, {'value_mse':float(mse.detach()), 'explained_variance':explained}
        critic_info = self.optimize(self.critic, self.critic_optimizer, batch, self.c['critic_epochs'], 'critic', critic_loss)
        info = dict(accepted=False, candidates=0, direction_pass=None, direction_score=None,
                    train_J=float(episode_returns.mean()), critic_mse=critic_info['value_mse'],
                    train_value_target=float(target[:, 0].mean()),
                    explained_variance=critic_info['explained_variance'],
                    return_difference=None, fit_kl_before=None, fit_kl_after=None,
                    fit_kl_p95=None, fit_kl_max=None, target_below_floor=None)
        if self.condition in ('mappo','ippo'):
            if self.c['ppo_normalize_advantage']:
                a = batch['advantage']
                real = valid_values(a, batch)
                batch['advantage'] = (a-real.mean())/(real.std(unbiased=False)+1e-8)
            info.update(self.optimize(self.actor, self.actor_optimizer, batch, self.c['ppo_epochs'], 'ppo',
                                      lambda mb: ppo_loss(self.actor, mb, mb['advantage'], self.c)))
            info['accepted'] = True
            return old, info
        # No global training random state is consumed by per-round initialization.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed_for('direction', self.seed, self.round+1))
            direction = Direction(self.input_dim, self.c).to(self.c['device'])
        optimizer = torch.optim.Adam(direction.parameters(), lr=self.c['direction_lr'])
        def direction_loss(mb):
            score, fisher = direction_terms(direction(mb['x']), mb['mu'], mb['actions'])
            quadratic = score.square() if self.condition == 'sampled' else fisher
            return self.weighted(quadratic-2*mb['advantage']*score, mb), {}
        self.optimize(direction, optimizer, batch, self.c['direction_epochs'], 'direction', direction_loss)
        with torch.no_grad():
            q = valid_values(direction(batch['x']), batch)
            self.log('direction_fit',q_abs_mean=float(q.abs().mean()),q_abs_max=float(q.abs().max()),
                     near_bound_fraction=float((q.abs()>.95*self.c['q_max']).float().mean()))
        if direction_check_enabled(self.condition):
            check = self.sampler.collect(old, self.c['direction_check_episodes'], 'direction_check')
            a, _ = self.labels(check, old_critic)
            with torch.no_grad():
                score, fisher = direction_terms(direction(check['x']), check['mu'], check['actions'])
                values = [float(self.weighted((2*a.double()*score.double()-fisher.double())[i:i+1],
                          {k:v[i:i+1] for k,v in check.items()})) for i in range(len(a))]
            info['direction_score'] = float(np.mean(values, dtype=np.float64))
            info['direction_pass'] = info['direction_score'] > 0
            self.log('direction_check', episode_scores=values, **info)
            if not info['direction_pass']:
                self.capture(None, old, info)
                return old, info
        for attempt in range(self.c['attempts']):
            if return_check_enabled(self.condition) and self.sampler.used+2*self.c['return_check_episodes']*episode_horizon(self.c) > self.c['budget']:
                break
            eta = self.c['eta']/(2**attempt)
            candidate = deepcopy(old)
            with torch.no_grad():
                batch['target'] = torch.softmax(batch['mu'].log()+eta*direction(batch['x']), -1).detach()
            def kl_values(model):
                with torch.no_grad():
                    return (batch['target']*(batch['target'].log()-model(batch['x']).log())).sum(-1)
            before = float(self.weighted(kl_values(candidate), batch))
            optimizer = torch.optim.Adam(candidate.parameters(), lr=self.c['actor_lr'])
            epochs = math.ceil(self.c['actor_epochs']/4) if self.condition == 'fit_quarter' else self.c['actor_epochs']
            def fit_loss(mb):
                kl = (mb['target']*(mb['target'].log()-candidate(mb['x']).log())).sum(-1)
                return self.weighted(kl, mb), {}
            self.optimize(candidate, optimizer, batch, epochs, 'actor_fit', fit_loss)
            kl = kl_values(candidate)
            with torch.no_grad():
                actual = candidate(batch['x'])
                step_kl = self.weighted((batch['mu']*(batch['mu'].log()-actual.log())).sum(-1), batch)
                entropy = self.weighted(-(actual*actual.log()).sum(-1), batch)
            info.update(candidates=attempt+1, fit_kl_before=before,
                        policy_step_kl=float(step_kl), policy_entropy=float(entropy),
                        fit_kl_after=float(self.weighted(kl, batch)),
                        fit_kl_p95=float(torch.quantile(valid_values(kl, batch).flatten(), .95)),
                        fit_kl_max=float(valid_values(kl, batch).max()),
                        target_below_floor=float(valid_values((batch['target'] < self.c['beta']/5).float(), batch).mean()))
            if not return_check_enabled(self.condition):
                info['accepted'] = True
            else:
                old_scores, old_observed = self.check_returns(old, 'return_old', old_critic)
                new_scores, new_observed = self.check_returns(candidate, 'return_candidate', old_critic)
                info['return_difference'] = float(np.mean(new_scores, dtype=np.float64)-np.mean(old_scores, dtype=np.float64))
                info['accepted'] = info['return_difference'] > 0
                self.log('return_check',attempt=attempt+1,old_returns=old_scores,
                         candidate_returns=new_scores,old_observed_returns=old_observed,
                         candidate_observed_returns=new_observed,
                         score_kind='bootstrapped_old_critic' if continuing_task(self.c) else 'observed_finite_return',
                         difference=info['return_difference'],passed=info['accepted'])
            self.log('candidate', attempt=attempt+1, eta=eta, **info)
            if attempt == 0:
                self.capture(candidate, old, info)
            if info['accepted']:
                self.actor.load_state_dict(candidate.state_dict())
                break
        return old, info

    def update_author_ppo(self):
        old = deepcopy(self.actor).eval()
        batch = self.sampler.collect(old, self.c['train_episodes'], 'train')
        observed = returns(batch['reward'], self.c['gamma'])[:, 0].double()
        self.log('train_batch', episodes=len(observed), J_mean=float(observed.mean()),
                 J_std=float(observed.std(unbiased=False)))
        info = self.author_ppo.update(batch)
        steps = info['updates']
        self.optim_steps['ppo'] += steps
        self.optim_steps['critic'] += steps
        self.log('optimization', module=f'author_{self.condition}', **info)
        info.update(accepted=True, candidates=0, direction_pass=None, direction_score=None,
                    train_J=float(observed.mean()), return_difference=None,
                    fit_kl_before=None, fit_kl_after=None, fit_kl_p95=None,
                    fit_kl_max=None, target_below_floor=None)
        return old, info

    def capture(self, candidate, old, info):
        if self.audit and self.round+1 in self.c['audit_rounds']:
            self.snapshots.append({'round':self.round+1, 'source_steps':self.sampler.used,
                'decision':deepcopy(info), 'old':deepcopy(old.cpu().state_dict()),
                'candidate':None if candidate is None else deepcopy(candidate.cpu().state_dict())})
            # .cpu() above only touched the frozen copies; candidate may be adopted on CUDA.
            old.to(self.c['device'])
            if candidate is not None:
                candidate.to(self.c['device'])

    def run(self, max_rounds=None):
        from ..evaluation.monitoring import progress_line
        if self.complete:
            summary = self.summary()
            save_json(self.directory/'summary.json', summary)
            if self.progress:
                self.progress(self)
            return summary
        try:
            if not self.curves:
                self.evaluate_node(self.actor, 0, self.sampler.used)
                self.checkpoint()
            if self.progress:
                self.progress(self)
            while self.sampler.used+minimum_cost(self.c, self.condition) <= self.c['budget']:
                if max_rounds is not None and self.round >= max_rounds:
                    return None  # test/development pause at a committed boundary
                before_cost = self.rounds[-1]['source_steps'] if self.rounds else 0
                started = time.perf_counter()
                old, info = self.update()
                info['update_seconds'] = time.perf_counter()-started
                self.train_seconds += info['update_seconds']
                self.round += 1
                self.rounds.append(dict(round=self.round, source_steps=self.sampler.used, **info))
                # A crossed node receives the previous committed policy, never the future one.
                while len(self.curves) < len(self.grid)-1 and self.grid[len(self.curves)] <= self.sampler.used:
                    node = self.grid[len(self.curves)]
                    crossed = node < self.sampler.used
                    self.evaluate_node(old if crossed else self.actor, node,
                                       before_cost if crossed else self.sampler.used)
                self.log('commit', completed_round=self.round, source_steps=self.sampler.used, **info)
                self.checkpoint()
                print('  ' + progress_line(self.condition, self.seed, self.c['budget'],
                                          self.rounds[-1], self.curves[-1]), flush=True)
                if self.progress:
                    self.progress(self)
            while len(self.curves) < len(self.grid):
                self.evaluate_node(self.actor, self.grid[len(self.curves)],
                                   self.rounds[-1]['source_steps'] if self.rounds else 0)
            self.complete = True
            self.checkpoint()
            summary = self.summary()
            save_json(self.directory/'summary.json', summary)
            if self.progress:
                self.progress(self)
            return summary
        except BaseException as error:
            save_json(self.directory/'failure.json', {'type':type(error).__name__, 'message':str(error),
                      'completed_round':self.round, 'used':self.sampler.used, 'costs':dict(self.sampler.costs),
                      'note':'保留失败记录；恢复将未提交采样按账本最大预留计费，当前轨迹至多高估预留长度减1步，不超预算。'})
            raise

    def summary(self):
        auc = sum((b['budget_checkpoint']-a['budget_checkpoint'])*a['J'] for a,b in zip(self.curves, self.curves[1:]))/self.c['budget']
        gates = [r for r in self.rounds if r['direction_pass'] is not None]
        fits = [r for r in self.rounds if r['fit_kl_after'] is not None]
        arrival = {}
        if arrival_task(self.c):
            arrival = {k:self.curves[-1][k] for k in ARRIVAL_METRICS if k in self.curves[-1]}
            arrival.update(task_horizon=self.c['task_horizon'],
                success_AUC=sum((b['budget_checkpoint']-a['budget_checkpoint'])*a['success_rate']
                                for a,b in zip(self.curves, self.curves[1:]))/self.c['budget'],
                success_cost=next((r['budget_checkpoint'] for r in self.curves if r['success_rate'] >= .9), None))
        return dict(condition=self.condition, seed=self.seed, status='complete', rounds=self.round,
            **arrival,
            implementation=({'backend':f'author_{self.condition}', 'commit':self.author_ppo.description()['commit']}
                            if self.author_ppo else {'backend':'local'}),
            source_steps=self.sampler.used, unused_budget=self.c['budget']-self.sampler.used,
            evaluation_steps=self.eval_steps, task_mode=self.c.get('task_mode', 'finite_horizon'),
            AUC=auc, **{k:self.curves[-1][k] for k in
                ('J','distance','coverage','all_covered','collision_pairs','collisions_per_agent',
                 'mean_reward','mean_coverage','tail_coverage','tail_all_covered') if k in self.curves[-1]},
            coverage_cost=next((r['budget_checkpoint'] for r in self.curves if r['coverage'] >= .8), None),
            accepted_rounds=sum(r['accepted'] for r in self.rounds),
            direction_rejections=sum(not r['direction_pass'] for r in gates), direction_checks=len(gates),
            candidates=sum(r['candidates'] for r in self.rounds),
            return_checks=sum(r['candidates'] for r in self.rounds) if return_check_enabled(self.condition) else 0,
            return_rejections=sum(r['candidates']-int(r['accepted']) for r in self.rounds) if return_check_enabled(self.condition) else 0,
            fit_kl_after=float(np.mean([r['fit_kl_after'] for r in fits])) if fits else None,
            train_seconds=self.train_seconds, evaluation_seconds=self.eval_seconds,
            optimizer_steps=sum(self.optim_steps.values()), optimizer_steps_by_module=dict(self.optim_steps),
            costs=dict(self.sampler.costs), actor_parameters=sum(p.numel() for p in self.actor.parameters()),
            critic_parameters=sum(p.numel() for p in self.critic.parameters()),
            direction_parameters=sum(p.numel() for p in self.actor.parameters()) if self.condition not in ('mappo','ippo') else 0)
