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
from ..models import Actor, Critic, Direction
from ..evaluation.evaluate import evaluate
from .collector import Collector
from .ppo import returns, gae, weighted, direction_terms, ppo_loss
from .storage import seed_for, event, save_json, save_pt, load_pt


def minimum_cost(config, condition):
    episodes = config['train_episodes']
    if condition not in ('mappo','ippo'):
        episodes += 0 if condition == 'no_direction_check' else config['direction_check_episodes']
        episodes += 0 if condition == 'no_return_check' else 2*config['return_check_episodes']
    return episodes*config['horizon']


class Runner:
    def __init__(self, directory, config, condition, seed, audit=False, resume=False):
        self.directory, self.c, self.condition, self.seed = Path(directory), config, condition, seed
        self.directory.mkdir(parents=True, exist_ok=True)
        self.audit = audit and condition in ('ours','no_return_check')
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
        self.input_dim = config['history']*(probe.obs_dim+6)+1
        self.state_dim = probe.state_dim
        probe.close()
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
            self.restore(load_pt(self.directory/'checkpoints/final.pt', config['device']))
            self.reconcile_interruption()
        else:
            if (self.directory/'config.json').exists():
                raise FileExistsError(f'拒绝覆盖运行: {self.directory}')
            save_json(self.directory/'config.json', identity)

    def log(self, stream, **record):
        event(self.log_path, stream, round=self.round+1, **record)

    def restore(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
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
        state.update(format_version=1, config=self.c, condition=self.condition, seed=self.seed,
            input_dim=self.input_dim, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            costs=dict(self.sampler.costs), counts=dict(self.sampler.counts), optim_steps=dict(self.optim_steps),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        save_pt(self.directory/'checkpoints/final.pt', state)
        if self.best is not None:
            save_pt(self.directory/'checkpoints/best.pt', self.best)
        if self.audit and self.snapshots:
            save_pt(self.directory/'audit_snapshots.pt', self.snapshots)

    def evaluate_node(self, actor, budget_node, actor_cost):
        started = time.perf_counter()
        final = budget_node == self.c['budget']
        count = self.c['final_episodes'] if final else self.c['eval_episodes']
        metrics, _ = evaluate(actor, self.c, self.condition, self.seed,
                              'source-final' if final else 'source-grid', count)
        self.eval_seconds += time.perf_counter()-started
        self.eval_steps += count*self.c['horizon']
        row = dict(condition=self.condition, seed=self.seed, budget_checkpoint=budget_node,
                   actor_source_steps=actor_cost, episodes=count, **metrics)
        self.curves.append(row)
        self.log('evaluation', **row)
        if self.best is None or metrics['J'] > self.best['score']:
            self.best = {'actor':deepcopy(actor.state_dict()), 'config':self.c, 'input_dim':self.input_dim,
                         'score':metrics['J'], 'selection':'independent_source_evaluation_diagnostic_only',
                         'budget_checkpoint':budget_node, 'actor_source_steps':actor_cost}

    def labels(self, batch, critic):
        with torch.no_grad():
            values = critic(batch['x'] if self.condition == 'ippo' else batch['state'])
            mc = returns(batch['reward'], self.c['gamma'])
            local = self.condition == 'ippo'
            target = mc.unsqueeze(-1).expand_as(values) if local else mc
            label = self.c['ppo_label'] if self.condition in ('mappo','ippo') else self.c['direction_label']
            advantage = gae(batch['reward'], values, self.c['gamma'], self.c['gae_lambda']) if label == 'gae' else target-values
            if not local:
                advantage = advantage.unsqueeze(-1).expand_as(batch['actions'])
            # Critic always regresses MC team return, documented also for GAE PPO.
            return advantage.detach(), target.detach()

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
            mse = weighted((value-mb['value_target']).square(), self.c['gamma'])
            return self.c['value_coef']*mse, {'value_mse':float(mse.detach())}
        self.optimize(self.critic, self.critic_optimizer, batch, self.c['critic_epochs'], 'critic', critic_loss)
        info = dict(accepted=False, candidates=0, direction_pass=None, direction_score=None,
                    return_difference=None, fit_kl_before=None, fit_kl_after=None,
                    fit_kl_p95=None, fit_kl_max=None, target_below_floor=None)
        if self.condition in ('mappo','ippo'):
            if self.c['ppo_normalize_advantage']:
                a = batch['advantage']
                batch['advantage'] = (a-a.mean())/(a.std(unbiased=False)+1e-8)
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
            return weighted(quadratic-2*mb['advantage']*score, self.c['gamma']), {}
        self.optimize(direction, optimizer, batch, self.c['direction_epochs'], 'direction', direction_loss)
        with torch.no_grad():
            q = direction(batch['x'])
            self.log('direction_fit',q_abs_mean=float(q.abs().mean()),q_abs_max=float(q.abs().max()),
                     near_bound_fraction=float((q.abs()>.95*self.c['q_max']).float().mean()))
        if self.condition != 'no_direction_check':
            check = self.sampler.collect(old, self.c['direction_check_episodes'], 'direction_check')
            a, _ = self.labels(check, old_critic)
            with torch.no_grad():
                score, fisher = direction_terms(direction(check['x']), check['mu'], check['actions'])
                values = [float(weighted((2*a.double()*score.double()-fisher.double())[i:i+1], self.c['gamma'])) for i in range(len(a))]
            info['direction_score'] = float(np.mean(values, dtype=np.float64))
            info['direction_pass'] = info['direction_score'] > 0
            self.log('direction_check', episode_scores=values, **info)
            if not info['direction_pass']:
                self.capture(None, old, info)
                return old, info
        for attempt in range(self.c['attempts']):
            if self.condition != 'no_return_check' and self.sampler.used+2*self.c['return_check_episodes']*self.c['horizon'] > self.c['budget']:
                break
            eta = self.c['eta']/(2**attempt)
            candidate = deepcopy(old)
            with torch.no_grad():
                batch['target'] = torch.softmax(batch['mu'].log()+eta*direction(batch['x']), -1).detach()
            def kl_values(model):
                with torch.no_grad():
                    return (batch['target']*(batch['target'].log()-model(batch['x']).log())).sum(-1)
            before = float(weighted(kl_values(candidate), self.c['gamma']))
            optimizer = torch.optim.Adam(candidate.parameters(), lr=self.c['actor_lr'])
            epochs = math.ceil(self.c['actor_epochs']/4) if self.condition == 'fit_quarter' else self.c['actor_epochs']
            def fit_loss(mb):
                kl = (mb['target']*(mb['target'].log()-candidate(mb['x']).log())).sum(-1)
                return weighted(kl, self.c['gamma']), {}
            self.optimize(candidate, optimizer, batch, epochs, 'actor_fit', fit_loss)
            kl = kl_values(candidate)
            info.update(candidates=attempt+1, fit_kl_before=before,
                        fit_kl_after=float(weighted(kl, self.c['gamma'])),
                        fit_kl_p95=float(torch.quantile(kl.flatten(), .95)), fit_kl_max=float(kl.max()),
                        target_below_floor=float((batch['target'] < self.c['beta']/5).float().mean()))
            if self.condition == 'no_return_check':
                info['accepted'] = True
            else:
                old_eval = self.sampler.collect(old, self.c['return_check_episodes'], 'return_old', retain=False)
                new_eval = self.sampler.collect(candidate, self.c['return_check_episodes'], 'return_candidate', retain=False)
                info['return_difference'] = float(np.mean([r['J'] for r in new_eval], dtype=np.float64)-np.mean([r['J'] for r in old_eval], dtype=np.float64))
                info['accepted'] = info['return_difference'] > 0
                self.log('return_check',attempt=attempt+1,old_returns=[r['J'] for r in old_eval],
                         candidate_returns=[r['J'] for r in new_eval],difference=info['return_difference'],passed=info['accepted'])
            self.log('candidate', attempt=attempt+1, eta=eta, **info)
            if attempt == 0:
                self.capture(candidate, old, info)
            if info['accepted']:
                self.actor.load_state_dict(candidate.state_dict())
                break
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
        if self.complete:
            summary = self.summary()
            save_json(self.directory/'summary.json', summary)
            return summary
        try:
            if not self.curves:
                self.evaluate_node(self.actor, 0, self.sampler.used)
                self.checkpoint()
            while self.sampler.used+minimum_cost(self.c, self.condition) <= self.c['budget']:
                if max_rounds is not None and self.round >= max_rounds:
                    return None  # test/development pause at a committed boundary
                before_cost = self.rounds[-1]['source_steps'] if self.rounds else 0
                started = time.perf_counter()
                old, info = self.update()
                self.train_seconds += time.perf_counter()-started
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
                print(f'  {self.condition} seed={self.seed} round={self.round} source={self.sampler.used}/{self.c["budget"]} accepted={info["accepted"]}', flush=True)
            while len(self.curves) < len(self.grid):
                self.evaluate_node(self.actor, self.grid[len(self.curves)],
                                   self.rounds[-1]['source_steps'] if self.rounds else 0)
            self.complete = True
            self.checkpoint()
            summary = self.summary()
            save_json(self.directory/'summary.json', summary)
            return summary
        except BaseException as error:
            save_json(self.directory/'failure.json', {'type':type(error).__name__, 'message':str(error),
                      'completed_round':self.round, 'used':self.sampler.used, 'costs':dict(self.sampler.costs),
                      'note':'保留失败记录；恢复将最后未完成回合按已预留整回合计费，最多高估 horizon-1 步，不超预算。'})
            raise

    def summary(self):
        auc = sum((b['budget_checkpoint']-a['budget_checkpoint'])*a['J'] for a,b in zip(self.curves, self.curves[1:]))/self.c['budget']
        gates = [r for r in self.rounds if r['direction_pass'] is not None]
        fits = [r for r in self.rounds if r['fit_kl_after'] is not None]
        return dict(condition=self.condition, seed=self.seed, status='complete', rounds=self.round,
            source_steps=self.sampler.used, unused_budget=self.c['budget']-self.sampler.used,
            evaluation_steps=self.eval_steps, AUC=auc, **{k:self.curves[-1][k] for k in
                ('J','distance','coverage','all_covered','collision_pairs','collisions_per_agent')},
            coverage_cost=next((r['budget_checkpoint'] for r in self.curves if r['coverage'] >= .8), None),
            accepted_rounds=sum(r['accepted'] for r in self.rounds),
            direction_rejections=sum(not r['direction_pass'] for r in gates), direction_checks=len(gates),
            candidates=sum(r['candidates'] for r in self.rounds),
            return_checks=sum(r['candidates'] for r in self.rounds) if self.condition not in ('mappo','ippo','no_return_check') else 0,
            return_rejections=sum(r['candidates']-int(r['accepted']) for r in self.rounds) if self.condition not in ('mappo','ippo','no_return_check') else 0,
            fit_kl_after=float(np.mean([r['fit_kl_after'] for r in fits])) if fits else None,
            train_seconds=self.train_seconds, evaluation_seconds=self.eval_seconds,
            optimizer_steps=sum(self.optim_steps.values()), optimizer_steps_by_module=dict(self.optim_steps),
            costs=dict(self.sampler.costs), actor_parameters=sum(p.numel() for p in self.actor.parameters()),
            critic_parameters=sum(p.numel() for p in self.critic.parameters()),
            direction_parameters=sum(p.numel() for p in self.actor.parameters()) if self.condition not in ('mappo','ippo') else 0)
