"""Fixed-q diagnostic hook; never replaces the original Runner's Actor."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from ..training.collector import Collector
from ..training.ppo import direction_terms
from .evaluate import evaluate
from ...common.reporting import write_csv


class FitProbe:
    def __init__(self, runner, config, output):
        self.runner, self.config, self.output = runner, config, Path(output)
        self.pending = list(config.get('fit_snapshot_fractions', [.25,.5,1.]))
        self.rows, self.last = [], None
        self.steps = 0

    def __call__(self, old, direction, batch, critic):
        self.last = (old, direction, batch, critic)
        fraction = self.runner.sampler.used/self.runner.c['budget']
        for mark in list(self.pending):
            if mark <= fraction:
                self.measure(mark)

    def finish(self):
        # Use the last predeclared update for an unspent budget tail; expose its
        # actual source cost, rather than claiming the budget was exhausted.
        if self.last:
            for mark in list(self.pending):
                self.measure(mark)

    def measure(self, mark):
        old, direction, batch, critic = self.last
        r, c = self.runner, self.runner.c
        device = next(old.parameters()).device
        devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            sampler = Collector(dict(c,budget=10**12), 'fit_probe', r.seed+int(mark*10000))
            validation = sampler.collect(old, c['return_check_episodes'], 'validation')
            labels, _ = r.labels(validation, critic)
            eta = c.get('step_sizes',[c['eta']])[0]
            with torch.no_grad():
                target = torch.softmax(batch['mu'].log()+eta*direction(batch['x']),-1)
                independent_target = torch.softmax(validation['mu'].log()+eta*direction(validation['x']),-1)
                score,fisher = direction_terms(direction(validation['x']),validation['mu'],validation['actions'])
                fixed_score = float(r.weighted(2*labels*score-fisher, validation))
            count = self.config['evaluation_episodes']
            purpose = f'fixed-q-{mark}'
            before, episodes = evaluate(old,c,r.condition,r.seed,purpose,count)
            self.steps += sampler.used+sum(e['episode_steps'] for e in episodes)
            for steps in self.config['fit_steps']:
                candidate = deepcopy(old)
                optimizer = torch.optim.Adam(candidate.parameters(),lr=c['actor_lr'])
                start = time.perf_counter()
                for _ in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = r.weighted((target*(target.log()-candidate(batch['x']).log())).sum(-1),batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(candidate.parameters(),c['grad_norm'])
                    optimizer.step()
                elapsed=time.perf_counter()-start
                with torch.no_grad():
                    kl=float(r.weighted((independent_target*(independent_target.log()-candidate(validation['x']).log())).sum(-1),validation))
                after, episodes=evaluate(candidate,c,r.condition,r.seed,purpose,count)
                self.steps+=sum(e['episode_steps'] for e in episodes)
                self.rows.append(dict(method='AN',seed=r.seed,budget_fraction=mark,
                    source_steps_total=r.sampler.used,fit_steps=steps,independent_fit_kl=kl,
                    direction_score=fixed_score,step_size=eta,source_return_gain=after['J']-before['J'],
                    fit_seconds=elapsed,diagnostic_steps=self.steps,used_for_selection=False))
                write_csv(self.output/'diagnostics.csv',self.rows)
        self.pending.remove(mark)
