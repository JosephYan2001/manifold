"""Run: python -m unittest discover -s .../cooperative_navigation/tests -v"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from manifold_project.experiments.cooperative_navigation.configs import load_config, condition_config, validate, CONDITIONS, EXPERIMENTS
from manifold_project.experiments.cooperative_navigation.envs.navigation import Navigation
from manifold_project.experiments.cooperative_navigation.observations.history import History
from manifold_project.experiments.cooperative_navigation.models import Actor, Direction
from manifold_project.experiments.cooperative_navigation.training.collector import episode
from manifold_project.experiments.cooperative_navigation.training.ppo import returns, gae, weighted, direction_terms, ppo_loss
from manifold_project.experiments.cooperative_navigation.training.runner import Runner, minimum_cost
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt, event
from manifold_project.experiments.cooperative_navigation.evaluation.evaluate import evaluate
from manifold_project.experiments.cooperative_navigation.evaluation.transfer import transfer_suite, audit_suite
from manifold_project.experiments.cooperative_navigation.evaluation.reporting import summaries


def tiny():
    c = load_config('smoke')
    c['task_mode'] = 'finite_horizon'  # Preserve coverage of the legacy protocol.
    c.update(horizon=5,history=2,hidden=8,train_episodes=2,direction_check_episodes=1,
             return_check_episodes=1,minibatch_episodes=1,direction_epochs=1,actor_epochs=1,
             critic_epochs=1,ppo_epochs=1,eval_episodes=1,final_episodes=1,
             transfer_episodes=1,audit_episodes=2,budget=50,eval_fractions=[0,.2,.4,.6,.8,1],
             bootstrap_repeats=50,audit_rounds=[1,2,3])
    return c


class EnvironmentTests(unittest.TestCase):
    def test_sizes_native_reward_terminal_and_reproducibility(self):
        c = load_config('smoke')
        c['task_mode'] = 'finite_horizon'
        for n in (2,3,4,6,8):
            env = Navigation(c,n)
            try:
                first = env.reset(321)
                self.assertEqual(first.shape,(n,16))
                np.testing.assert_array_equal(first,env.reset(321))
                self.assertEqual(env.state().shape,(16*n+1,))
                rewards = []
                for t in range(100):
                    _,r,terminated,truncated = env.step(np.arange(n)%5)
                    m = env.metrics()
                    self.assertAlmostEqual(r,-.5*n*m['distance']-m['collision_pairs']/n,places=6)
                    self.assertFalse(terminated)
                    self.assertEqual(truncated,t==99)
                    rewards.append(r)
                env.reset(321)
                for r in rewards:
                    self.assertEqual(r,env.step(np.arange(n)%5)[1])
            finally:
                env.close()

    def test_nearest_slots_padding_and_state(self):
        c = load_config('smoke')
        c['task_mode'] = 'finite_horizon'
        for n in (2,4):
            env = Navigation(c,n)
            try:
                obs = env.reset(4)
                world = env.world
                a = world.agents[0]
                others = sorted(world.agents[1:],key=lambda x:np.linalg.norm(x.state.p_pos-a.state.p_pos))
                landmarks = sorted(world.landmarks,key=lambda x:np.linalg.norm(x.state.p_pos-a.state.p_pos))
                np.testing.assert_allclose(obs[0,4:8],np.concatenate([x.state.p_pos-a.state.p_pos for x in landmarks[:2]]),rtol=1e-6)
                np.testing.assert_allclose(obs[0,8:10],others[0].state.p_pos-a.state.p_pos,rtol=1e-6)
                if n==2:
                    np.testing.assert_array_equal(obs[0,10:12],0)
                np.testing.assert_array_equal(obs[0,12:16],0)
                np.testing.assert_array_equal(env.state()[:-1],obs.flatten())
            finally:
                env.close()

    def test_history_only_past_own_actions(self):
        h = History(2,16,8,100)
        x = h.append(np.ones((2,16)),None,0)
        self.assertEqual(x.shape,(2,177))
        self.assertTrue((x[:,:7*22]==0).all())
        self.assertTrue((x[:,7*22+16:7*22+21]==0).all())
        self.assertTrue((x[:,7*22+21]==1).all())
        y = h.append(2*np.ones((2,16)),np.array([1,3]),1)
        np.testing.assert_array_equal(y[0,7*22+16:7*22+21],[0,1,0,0,0])
        np.testing.assert_array_equal(y[1,7*22+16:7*22+21],[0,0,0,1,0])
        self.assertAlmostEqual(float(y[0,-1]),.01)


class ObjectiveTests(unittest.TestCase):
    def test_method_override_inheritance(self):
        c = tiny()
        c['method_overrides'] = {'ours':{'eta':.05},'mappo':{'ppo_epochs':5}}
        validate(c)
        for method in ('ours','sampled','fit_quarter','no_return_check','no_direction_check','no_checks'):
            self.assertEqual(condition_config(c,method)['eta'],.05)
        self.assertEqual(condition_config(c,'mappo')['ppo_epochs'],5)
        self.assertEqual(condition_config(c,'ippo')['ppo_epochs'],1)

    def test_ppo_ratio_initially_one_and_frozen_targets(self):
        c = tiny()
        actor = Actor(45,c)
        x = torch.randn(2,5,4,45)
        mu = actor(x).detach()
        batch = {'x':x,'mu':mu,'actions':torch.zeros(2,5,4,dtype=torch.long)}
        advantage = torch.randn(2,5,4)
        loss,info = ppo_loss(actor,batch,advantage,c)
        entropy = -(mu*mu.log()).sum(-1)
        torch.testing.assert_close(loss,-weighted(advantage+c['entropy_coef']*entropy,c['gamma']))
        self.assertEqual(info['ppo_kl'],0)
        self.assertEqual(info['clip_fraction'],0)
        loss.backward()
        self.assertIsNone(mu.grad)
        self.assertTrue(any(p.grad is not None for p in actor.parameters()))

    def test_terminal_returns_and_gae(self):
        r = torch.tensor([[1.,2.,3.]])
        values = torch.tensor([[.4,.5,.6]])
        expected = torch.tensor([[1+2*.9+3*.81,2+3*.9,3.]])
        torch.testing.assert_close(returns(r,.9),expected)
        torch.testing.assert_close(gae(r,values,.9,1),expected-values)
        self.assertAlmostEqual(float(gae(r,values,.9,.95)[0,-1]),2.4,places=5)

    def test_fisher_equals_expected_sampled_and_gradient(self):
        mu = torch.tensor([.1,.2,.3,.15,.25],dtype=torch.float64)
        q = torch.tensor([.3,-.5,.1,.4,-.3],dtype=torch.float64,requires_grad=True)
        fisher = (mu*q.square()).sum()-(mu*q).sum().square()
        sampled = (mu*(q-(mu*q).sum()).square()).sum()
        torch.testing.assert_close(fisher,sampled)
        torch.testing.assert_close(torch.autograd.grad(fisher,q,retain_graph=True)[0],torch.autograd.grad(sampled,q)[0])

    def test_probability_bounds_and_weight_normalization(self):
        c = tiny()
        actor, direction = Actor(45,c),Direction(45,c)
        x = 100*torch.randn(2,5,4,45)
        p,q = actor(x),direction(x)
        torch.testing.assert_close(p.sum(-1),torch.ones_like(p[...,0]))
        self.assertTrue((p>=c['beta']/5).all())
        self.assertTrue((q.abs()<=c['q_max']).all())
        torch.testing.assert_close(q.sum(-1),torch.zeros_like(q[...,0]),atol=1e-6,rtol=0)
        self.assertAlmostEqual(float(weighted(torch.ones(2,5,4),.99)),1)

    def test_evaluation_does_not_consume_training_rng(self):
        c = tiny()
        actor = Actor(45,c)
        before = torch.get_rng_state().clone()
        a,_ = evaluate(actor,c,'ours',0,'test',2)
        torch.testing.assert_close(before,torch.get_rng_state())
        b,_ = evaluate(actor,c,'ours',0,'test',2)
        self.assertEqual(a,b)


class RunnerTests(unittest.TestCase):
    def test_joint_ablation_only_trains_and_accepts_first_candidate(self):
        c = tiny()
        c['budget'] = 20
        self.assertEqual(minimum_cost(c, 'no_checks'), 10)
        self.assertEqual(EXPERIMENTS['N-A5'], ['ours', 'no_checks'])
        for experiment in ('N-A', 'N-P', 'N-S1'):
            self.assertNotIn('no_checks', EXPERIMENTS[experiment])
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(folder, c, 'no_checks', 40)
            original = runner.sampler.collect
            def collect(actor, count, purpose, retain=True):
                self.assertEqual(purpose, 'train')
                return original(actor, count, purpose, retain)
            before = {k:v.clone() for k,v in runner.actor.state_dict().items()}
            with patch.object(runner.sampler, 'collect', collect):
                result = runner.run()
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(result['accepted_rounds'], 2)
            self.assertEqual(result['costs'], {'train':20})
            self.assertEqual(result['direction_checks'], 0)
            self.assertEqual(result['return_checks'], 0)
            self.assertEqual(result['return_rejections'], 0)
            self.assertTrue(all(r['candidates']==1 and r['direction_pass'] is None for r in runner.rounds))
            for module in ('critic','direction','actor_fit'):
                self.assertGreater(result['optimizer_steps_by_module'][module], 0)
            self.assertTrue(any(not torch.equal(before[k], v) for k,v in runner.actor.state_dict().items()))

    def test_mc_aligned_ppo_both_critics(self):
        c = tiny()
        c.update(mappo_backend='local',ippo_backend='local',ppo_label='mc',ppo_normalize_advantage=False,budget=10)
        with tempfile.TemporaryDirectory() as folder:
            for condition in ('mappo','ippo'):
                runner = Runner(Path(folder)/condition,c,condition,0)
                runner.run()
                self.assertEqual(runner.round,1)
                self.assertTrue(runner.complete)

    def test_strict_return_gate_backtracks_from_zero(self):
        c = tiny()
        c['budget'] = 35
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(folder,c,'ours',2,audit=True)
            original_collect = runner.sampler.collect
            calls = {'candidate':0}
            def collect(actor,count,purpose,retain=True):
                batch = original_collect(actor,count,purpose,retain)
                if not retain:
                    if purpose=='return_candidate':
                        calls['candidate'] += 1
                    for row in batch:
                        row['J'] = (0 if calls['candidate']==1 else 1) if purpose=='return_candidate' else 0
                return batch
            def labels(batch,critic):
                return torch.ones_like(batch['actions'],dtype=torch.float32),returns(batch['reward'],c['gamma'])
            def terms(q,mu,actions):
                score = q[...,0]*0+1  # preserves an autograd path with a zero gradient
                return score,score*0
            with patch.object(runner.sampler,'collect',collect), patch.object(runner,'labels',labels), patch(
                    'manifold_project.experiments.cooperative_navigation.training.runner.direction_terms',terms):
                runner.run()
            self.assertEqual(runner.rounds[0]['candidates'],2)
            self.assertTrue(runner.rounds[0]['accepted'])
            self.assertEqual(runner.sampler.used,35)
            self.assertFalse(runner.snapshots[0]['decision']['accepted'])
            self.assertEqual(runner.summary()['return_rejections'],1)

    def test_all_conditions_and_no_future_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            for condition in CONDITIONS:
                c = tiny()
                runner = Runner(Path(folder)/condition,c,condition,0,audit=True)
                summary = runner.run()
                self.assertLessEqual(summary['source_steps'],c['budget'])
                self.assertGreater(summary['rounds'],0)
                for row in runner.curves:
                    self.assertLessEqual(row['actor_source_steps'],row['budget_checkpoint'])
                self.assertEqual(sum(summary['costs'].values()),summary['source_steps'])
                if condition in ('mappo','ippo','no_checks'):
                    self.assertEqual(set(summary['costs']),{'train'})
                elif condition=='no_direction_check':
                    self.assertNotIn('direction_check',summary['costs'])
                elif condition=='no_return_check':
                    self.assertNotIn('return_old',summary['costs'])
                    self.assertNotIn('return_candidate',summary['costs'])
                self.assertTrue((runner.directory/'checkpoints/best.pt').exists())
                state = load_pt(runner.directory/'checkpoints/final.pt')
                torch.testing.assert_close(state['actor'],runner.actor.state_dict())
                # AUC uses the left endpoint, not interpolation or final-only score.
                expected = sum((b['budget_checkpoint']-a['budget_checkpoint'])*a['J'] for a,b in zip(runner.curves,runner.curves[1:]))/c['budget']
                self.assertEqual(summary['AUC'],expected)

    def test_resume_matches_uninterrupted_all_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            for condition in CONDITIONS:
                c = tiny()
                continuous = Runner(Path(folder)/(condition+'_full'),c,condition,4)
                continuous.run()
                paused = Runner(Path(folder)/(condition+'_resume'),c,condition,4)
                paused.run(max_rounds=1)
                resumed = Runner(paused.directory,c,condition,4,resume=True)
                resumed.run()
                torch.testing.assert_close(continuous.actor.state_dict(),resumed.actor.state_dict(),rtol=0,atol=0)
                torch.testing.assert_close(continuous.critic.state_dict(),resumed.critic.state_dict(),rtol=0,atol=0)
                self.assertEqual(continuous.curves,resumed.curves)
                self.assertEqual(continuous.sampler.costs,resumed.sampler.costs)
                self.assertEqual(continuous.optim_steps,resumed.optim_steps)

    def test_budget_reservations_and_rejected_direction(self):
        c = tiny()
        self.assertEqual(minimum_cost(c,'ours'),25)
        self.assertEqual(minimum_cost(c,'mappo'),10)
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(folder,c,'ours',0,audit=True)
            # Exactly zero direction makes empirical score zero: strict >0 rejects.
            def zero(self,x):
                return self.net(x)*0
            with patch.object(Direction,'forward',zero):
                runner.run()
            self.assertTrue(all(r['direction_pass'] is False for r in runner.rounds))
            self.assertNotIn('return_old',runner.sampler.costs)
            self.assertTrue(all(s['candidate'] is None for s in runner.snapshots))

    def test_interrupted_episode_reserved_conservatively(self):
        c = tiny()
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(folder,c,'mappo',0)
            runner.run(max_rounds=1)
            used = runner.sampler.used
            event(runner.log_path,'sampling_start',purpose='train',episode=runner.sampler.counts['train'],reserved_through=used+c['horizon'])
            resumed = Runner(folder,c,'mappo',0,resume=True)
            self.assertEqual(resumed.sampler.used,used+c['horizon'])
            resumed.run()
            self.assertLessEqual(resumed.sampler.used,c['budget'])

    def test_label_alignment_transfer_and_audit_missing(self):
        c = tiny()
        c.update(direction_label='gae',ppo_label='mc',budget=25)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner = Runner(root/'run',c,'ours',0,audit=True)
            runner.run()
            manifest = {'config':c,'jobs':[dict(path='run',condition='ours',seed=0)]}
            rows = transfer_suite(root,manifest)
            source = next(r for r in rows if r['n_agents']==4)
            self.assertEqual(source['evaluation_steps'],0)
            self.assertEqual(source['J'],runner.summary()['J'])
            self.assertEqual({r['n_agents'] for r in rows},{3,4,6,8})
            # Re-running transfer must reuse complete results without new trajectories.
            with patch('manifold_project.experiments.cooperative_navigation.evaluation.transfer.evaluate',side_effect=AssertionError('unexpected resampling')):
                self.assertEqual(transfer_suite(root,manifest),rows)
            audit = audit_suite(root,manifest)
            self.assertEqual(len(audit),len(c['audit_rounds']))
            self.assertTrue(any(r['status']=='missing_candidate' for r in audit))


if __name__ == '__main__':
    unittest.main()
