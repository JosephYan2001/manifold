"""Small CPU engineering checks, never a training-performance experiment."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from manifold_project.experiments.tests.test_direction_remediation import settings
from manifold_project.experiments.tests.test_recovery import same
import torch
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.ppo import batch_weighted, direction_terms
from manifold_project.experiments.cooperative_navigation.training.diagnostics import (
    direction_metrics, critic_metrics, categorical_kl, normalized)


EVIDENCE = {}


def reference_kl(p, q):
    """Independent scalar math implementation; no production KL helper."""
    shape = p.shape[:-1]
    rows = []
    for pp, qq in zip(p.detach().double().reshape(-1, p.shape[-1]).tolist(),
                      q.detach().double().reshape(-1, q.shape[-1]).tolist()):
        ps, qs = math.fsum(pp), math.fsum(qq)
        rows.append(math.fsum((a/ps)*math.log((a/ps)/(b/qs)) for a,b in zip(pp,qq)))
    return torch.tensor(rows, dtype=torch.float64).reshape(shape)


def reference_weight(value, batch, config, episode_block=1, time_block=1):
    """Absolute time weights and one global denominator, including masked padding."""
    v = value.detach().double()
    e,t = v.shape[:2]
    n = v.shape[2] if v.ndim == 3 else 1
    horizon = config['task_horizon']
    denominator = e*n*math.fsum(config['gamma']**j for j in range(horizon))
    parts = []
    for es in range(0,e,episode_block):
        for ts in range(0,t,time_block):
            terms = []
            for i in range(es,min(es+episode_block,e)):
                for j in range(ts,min(ts+time_block,t)):
                    if bool(batch['valid'][i,j]):
                        terms.extend(x*config['gamma']**j for x in v[i,j].reshape(-1).tolist())
            parts.append(math.fsum(terms))
    return math.fsum(parts)/denominator


class RepairAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='acceptance_test_',dir=Path(__file__).resolve().parents[3])
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_frozen_full_batch_block_invariance(self):
        c = settings()
        c.update(train_episodes=5)
        r = Runner(self.root/'blocks',c,'no_checks',40)
        evidence = {}

        def inspect(old,direction,batch,critic):
            b = deepcopy(batch)
            # Unequal valid lengths deliberately exercise padding and unequal chunks.
            b['valid'] = torch.arange(8)[None,:] < torch.tensor([8,5,3,7,1])[:,None]
            before = deepcopy((old.state_dict(),direction.state_dict(),critic.state_dict(),b))
            full = direction_metrics(direction,b,c)
            cm = critic_metrics(critic,b,c)
            with torch.no_grad():
                q = direction(b['x']).double()
                score,fisher = direction_terms(q,normalized(b['mu']),b['actions'])
                value = critic(b['state']).double()
            target = b['value_target'].double()
            fields = dict(D_label=b['advantage'].double()*score,fisher_energy=fisher,
                          sampled_energy=score.square(),mse=(target-value).square())
            max_error = 0.
            for key,values in fields.items():
                # Poison padding: masking must exclude NaNs rather than multiply them by zero.
                values = values.clone()
                values[~b['valid']] = float('nan')
                expected = full[key] if key in full else cm[key]
                for eb,tb in ((1,1),(2,3),(3,5),(5,8)):
                    actual = reference_weight(values,b,c,eb,tb)
                    max_error = max(max_error,abs(expected-actual))
                    self.assertAlmostEqual(expected,actual,delta=1e-10)
                self.assertAlmostEqual(float(batch_weighted(values,b,c)),expected,delta=1e-10)
            weight = lambda x:reference_weight(x,b,c,2,3)
            mass = weight(torch.ones_like(target))
            variance = weight((target-weight(target)/mass).square())
            error = target-value
            ev = 1-weight((error-weight(error)/mass).square())/variance
            self.assertAlmostEqual(cm['explained_variance'],ev,delta=1e-10)
            # Re-evaluate the frozen networks on uneven episode chunks and reversed order.
            net_error = 0.
            for order in (torch.arange(5),torch.arange(4,-1,-1)):
                for block in (1,2,3,5):
                    aggregate = dict.fromkeys(('D_label','fisher_energy','sampled_energy'),0.)
                    mse = 0.
                    moments = dict.fromkeys(('mass','target','target2','error','error2'),0.)
                    for start in range(0,5,block):
                        idx = order[start:start+block]
                        mb = {k:v[idx] for k,v in b.items()}
                        dm = direction_metrics(direction,mb,c)
                        for key in aggregate:aggregate[key] += len(idx)/5*dm[key]
                        mse += len(idx)/5*critic_metrics(critic,mb,c)['mse']
                        with torch.no_grad():
                            mt = mb['value_target'].double()
                            me = mt-critic(mb['state']).double()
                        for key,values in dict(mass=torch.ones_like(mt),target=mt,target2=mt.square(),
                                               error=me,error2=me.square()).items():
                            moments[key] += len(idx)/5*float(batch_weighted(values,mb,c))
                    aggregate['B_label'] = 2*aggregate['D_label']-aggregate['fisher_energy']
                    aggregate['analytic_risk'] = -aggregate['B_label']
                    aggregate['sampled_risk'] = aggregate['sampled_energy']-2*aggregate['D_label']
                    for key,actual in aggregate.items():
                        net_error = max(net_error,abs(actual-full[key]))
                        self.assertAlmostEqual(actual,full[key],delta=2e-6+2e-5*abs(full[key]))
                    self.assertAlmostEqual(mse,cm['mse'],delta=2e-6+2e-5*abs(cm['mse']))
                    ev_chunk = 1-(moments['error2']-moments['error']**2/moments['mass'])/(
                        moments['target2']-moments['target']**2/moments['mass'])
                    self.assertAlmostEqual(ev_chunk,cm['explained_variance'],delta=2e-6)
            self.assertTrue(same(before,(old.state_dict(),direction.state_dict(),critic.state_dict(),b)))
            evidence.update(cached_reduction_max_abs_error=max_error,
                            network_rechunk_direction_max_abs_error=net_error,
                            cached_atol=1e-10,network_atol=2e-6,network_rtol=2e-5,
                            valid_lengths=[8,5,3,7,1],time_blocks=[1,3,5,8],episode_blocks=[1,2,3,5])
        r.fit_probe = inspect
        try:r.update()
        finally:r.sampler.close()
        self.assertTrue(evidence)
        EVIDENCE['full_batch'] = evidence

    def test_frozen_snapshots_fit_origin_and_three_kls(self):
        c = settings()
        c.update(direction_epochs=2,actor_epochs=3)
        r = Runner(self.root/'freeze',c,'no_checks',40)
        captured = {}
        initial_actor = deepcopy(r.actor.state_dict())
        original_optimize = r.optimize
        original_collect = r.sampler.collect

        def grads(model):return [None if p.grad is None else p.grad.clone() for p in model.parameters()]

        def collect(actor,episodes,purpose,*args,**kwargs):
            if purpose == 'train':
                captured['sampling_old'] = actor
                captured['sampling_old_grads'] = grads(actor)
            return original_collect(actor,episodes,purpose,*args,**kwargs)
        r.sampler.collect = collect

        def capture(old,direction,batch,critic):
            captured.update(old=old,direction=direction,critic=critic,
                            frozen=deepcopy((old.state_dict(),direction.state_dict(),critic.state_dict())),
                            gradients=grads(old)+grads(direction)+grads(critic),
                            data=deepcopy(batch))
            self.assertTrue(same(initial_actor,old.state_dict()))

        def optimize(model,optimizer,batch,epochs,module,objective):
            original_step = optimizer.step
            steps = 0
            if module == 'direction':direction_data = deepcopy(batch)
            if module == 'actor_fit':
                old,direction,critic = (captured[k] for k in ('old','direction','critic'))
                self.assertTrue(same(model.state_dict(),old.state_dict()))
                with torch.no_grad():
                    torch.testing.assert_close(model(batch['x']),old(batch['x']),rtol=0,atol=0)
                    captured['zero_loss'] = float(objective(batch)[0])
                target = batch['target'].clone()
                self.assertFalse(batch['target'].requires_grad)
                self.assertIsNone(batch['target'].grad_fn)
                optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
                self.assertEqual(optimized,{id(p) for p in model.parameters()})
                self.assertFalse(optimized & {id(p) for m in (old,direction,critic) for p in m.parameters()})
                def check_frozen():
                    self.assertTrue(same(captured['frozen'],(old.state_dict(),direction.state_dict(),critic.state_dict())))
                    self.assertTrue(same(captured['gradients'],grads(old)+grads(direction)+grads(critic)))
                    self.assertTrue(same(captured['data'],{k:batch[k] for k in captured['data']}))
                    torch.testing.assert_close(batch['target'],target,rtol=0,atol=0)
            def step(*args,**kwargs):
                nonlocal steps
                if module == 'actor_fit':check_frozen()
                result = original_step(*args,**kwargs)
                steps += 1
                if module == 'direction':
                    self.assertTrue(same(initial_actor,r.actor.state_dict()))
                    self.assertTrue(same(initial_actor,captured['sampling_old'].state_dict()))
                    self.assertTrue(same(captured['sampling_old_grads'],grads(captured['sampling_old'])))
                    self.assertTrue(same(direction_data,batch))
                if module == 'actor_fit':check_frozen()
                return result
            with patch.object(optimizer,'step',side_effect=step):
                result = original_optimize(model,optimizer,batch,epochs,module,objective)
            if module == 'direction':captured['direction_steps_checked'] = steps
            if module == 'actor_fit':
                self.assertFalse(same(model.state_dict(),old.state_dict()))
                with torch.no_grad():
                    p = model(batch['x'])
                    mu = old(batch['x'])
                captured['fit_steps_checked'] = steps
                captured['kls'] = {
                    'fit_kl_before_fp64':float(r.weighted(reference_kl(target,mu),batch)),
                    'fit_kl_after_fp64':float(r.weighted(reference_kl(target,p),batch)),
                    'old_to_actor_kl_fp64':float(r.weighted(reference_kl(mu,p),batch)),
                    'old_to_target_kl_fp64':float(r.weighted(reference_kl(mu,target),batch))}
            return result
        r.fit_probe = capture
        r.optimize = optimize
        try:_,info = r.update()
        finally:r.sampler.close()
        self.assertEqual(captured['zero_loss'],info['fit_kl_before'])
        errors = {key:abs(value-info[key]) for key,value in captured['kls'].items()}
        for key,error in errors.items():self.assertLessEqual(error,2e-14,key)
        self.assertGreater(captured['direction_steps_checked'],0)
        self.assertGreater(captured['fit_steps_checked'],0)
        EVIDENCE['freeze_and_kl'] = dict(zero_loss=captured['zero_loss'],
            fit_kl_before=info['fit_kl_before'],reference_kls=captured['kls'],absolute_errors=errors,
            direction_steps_checked=captured['direction_steps_checked'],
            fit_steps_checked=captured['fit_steps_checked'],kl_atol=2e-14)

    def test_kl_orientation_independent_reference(self):
        old = torch.tensor([[.70,.20,.10]],dtype=torch.float64)
        target = torch.tensor([[.20,.50,.30]],dtype=torch.float64)
        candidate = torch.tensor([[.40,.15,.45]],dtype=torch.float64)
        actual = []
        for p,q in ((target,old),(target,candidate),(old,candidate)):
            expected = reference_kl(p,q)
            torch.testing.assert_close(categorical_kl(p,q),expected,rtol=1e-13,atol=1e-13)
            self.assertGreater(abs(float(expected-reference_kl(q,p))),1e-3)
            actual.append(float(expected))
        self.assertEqual(len({round(x,8) for x in actual}),3)
        EVIDENCE['asymmetric_kl_fixture'] = dict(zip(
            ('target_to_old','target_to_candidate','old_to_candidate'),actual))


if __name__ == '__main__':unittest.main()
