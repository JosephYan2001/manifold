"""Method/gate truth table, read-only probe invariance, and snapshot replay."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
import json

from manifold_project.experiments.common.runtime import configure_runtime
configure_runtime(True)
import torch
from manifold_project.experiments.common.config import load_config
from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
from manifold_project.experiments.cooperative_navigation.training.runner import Runner, minimum_cost
from manifold_project.experiments.cooperative_navigation.training.semantics import effective_settings,validate_legacy_resume
from manifold_project.experiments.cooperative_navigation.training.ppo import direction_loss_values
from manifold_project.experiments.cooperative_navigation.training.diagnostics import saturation_metrics,categorical_kl
from manifold_project.experiments.cooperative_navigation.training.storage import event,load_pt,save_pt
from manifold_project.experiments.cooperative_navigation.replay_direction import replay
from manifold_project.experiments.tests.test_recovery import same


def settings(method='AN',direction=False,returns=False):
    p=load_config('navigation','smoke')
    p.update(hidden=8,heads=1,relation_layers=0,history=1,horizon=8,evaluation_horizon=8,
             device='cpu',batch_episodes=2,minibatch_size=32,direction_steps=1,actor_fit_steps=1,
             critic_steps=1,check_episodes=1,seed=40,budget=4096,save_fit_diagnostics=False,
             direction_check=direction,return_check=returns)
    return runner_config(p,method=method)


class RemediationTests(unittest.TestCase):
    def setUp(self):
        workspace=Path(__file__).resolve().parents[3]
        self.tmp=tempfile.TemporaryDirectory(prefix='remediation_test_',dir=workspace)
        self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()

    def test_method_and_gate_truth_table(self):
        q=torch.tensor([[.6,-.3,-.3]],dtype=torch.float64)
        mu=torch.tensor([[.2,.3,.5]],dtype=torch.float64)
        a=torch.tensor([0]);adv=torch.tensor([.7])
        center=(mu*q).sum(-1);score=q[:,0]-center
        fisher=(mu*q*q).sum(-1)-center.square()
        for method in ('AN','SA','DA'):
            for dg in (False,True):
                for rg in (False,True):
                    c=settings(method,dg,rg)
                    condition='direct' if method=='DA' else ('sampled' if method=='SA' else 'ours')
                    if method!='DA':
                        condition=('no_checks' if not dg and not rg else 'no_direction_check' if not dg
                                   else 'no_return_check' if not rg else condition)
                    effective=effective_settings(c,condition)
                    self.assertEqual(effective['direction_gate'],dg and method!='DA')
                    self.assertEqual(effective['return_gate'],rg)
                    if method!='DA':
                        expected=(score.square() if method=='SA' else fisher)-2*adv*score
                        torch.testing.assert_close(direction_loss_values(q,mu,a,adv,c['direction_objective']),expected)
                    self.assertEqual(minimum_cost(c,condition),16+8*(int(dg and method!='DA')+2*int(rg)))
                    r=Runner(self.root/f'{method}_{dg}_{rg}',c,condition,40)
                    try:
                        _,info=r.update()
                        self.assertEqual(info['effective_objective'],effective['effective_objective'])
                        self.assertEqual(r.sampler.counts['direction_check']>0,dg and method!='DA')
                        # A direction rejection can legitimately prevent a return check.
                        if not rg:
                            self.assertEqual(r.sampler.counts['return_old'],0)
                            self.assertEqual(r.sampler.counts['return_candidate'],0)
                        elif method=='DA' or info['direction_pass'] is not False:
                            self.assertGreater(r.sampler.counts['return_old'],0)
                        if method=='DA' and not rg:
                            self.assertTrue(info['accepted'])
                            self.assertEqual(r.sampler.used,16)
                    finally:r.sampler.close()

    def test_probe_is_read_only_and_snapshot_replays(self):
        results=[]
        for enabled in (False,True):
            c=settings()
            c.update(direction_diagnostic_rounds=[1] if enabled else [],
                     direction_diagnostic_episodes=2,save_direction_snapshots=enabled)
            r=Runner(self.root/str(enabled),c,'no_checks',40)
            try:
                r.update()
                results.append(deepcopy(dict(actor=r.actor.state_dict(),critic=r.critic.state_dict(),
                    optimizer=r.critic_optimizer.state_dict(),counts=r.sampler.counts,costs=r.sampler.costs,
                    torch_rng=torch.get_rng_state(),optim_steps=r.optim_steps)))
                self.assertEqual(r.diagnostic_steps,16 if enabled else 0)
                if enabled:
                    snapshot=r.directory/'direction_snapshots/round_000001.pt'
                    self.assertTrue(replay(snapshot)['verified'])
                    padded=load_pt(snapshot)
                    for key in ('batch','validation'):
                        b=padded[key]
                        for field,value in list(b.items()):
                            if not field.startswith('final_') and field!='lengths':
                                tail=torch.zeros_like(value[:,:1])
                                if field=='mu':tail.fill_(1/value.shape[-1])
                                b[field]=torch.cat([value,tail],dim=1)
                    padded_path=self.root/'padded.pt';save_pt(padded_path,padded)
                    self.assertTrue(replay(padded_path)['verified'])
                    r.checkpoint()
                    restored=Runner(r.directory,c,'no_checks',40,resume=True)
                    self.assertEqual(restored.diagnostic_steps,16);restored.sampler.close()
                    extended=Runner(self.root/'extended',c,'no_checks',40)
                    extended.restore(load_pt(r.directory/'checkpoints/final.pt'),external=True)
                    self.assertEqual(extended.diagnostic_steps,16)
                    extended.checkpoint()
                    event(extended.directory/'direction_probe_ledger.jsonl','start',id=0,steps=16)
                    extended.sampler.close()
                    recovered=Runner(extended.directory,c,'no_checks',40,resume=True)
                    self.assertEqual(recovered.diagnostic_steps,32)
                    recovered.sampler.close()
            finally:r.sampler.close()
        self.assertTrue(same(results[0],results[1]))

    def test_saturation_and_small_kl(self):
        for k in (2,5):
            u=torch.full((1,k),-1.);u[0,0]=1.
            q=.5*(u-u.mean(-1,keepdim=True))
            m=saturation_metrics(u,q,1.)
            self.assertAlmostEqual(m['q_abs_max'],(k-1)/k,places=6)
            self.assertGreater(m['effective_bound_fraction'],0)
            self.assertEqual(m['tanh_saturation'],1)
            self.assertEqual(saturation_metrics(torch.zeros_like(u),torch.zeros_like(q),1.)['effective_bound_fraction'],0)
            self.assertEqual(saturation_metrics(torch.ones_like(u),torch.zeros_like(q),1.)['tanh_saturation'],1)
        p=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float32)
        self.assertEqual(float(categorical_kl(p,p)),0.)
        q=p.clone();q[0,0]+=1e-5;q[0,1]-=1e-5
        self.assertGreater(float(categorical_kl(p,q)),0.)

    def test_legacy_buggy_resumes_are_rejected(self):
        for method,condition in [('SA','no_checks'),('DA','direct')]:
            with self.assertRaises(ValueError):
                validate_legacy_resume(dict(current_method=method,condition=condition,
                    config=dict(current_config=dict(return_check=False))))
        validate_legacy_resume(dict(current_method='AN',condition='no_checks',config={}))


if __name__=='__main__':unittest.main()
