"""Success terminals, external cutoffs and live-state recovery are different."""
import unittest
import numpy as np
import torch

from manifold_project.experiments.common.config import load_config
from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
from manifold_project.experiments.cooperative_navigation.models import Actor
from manifold_project.experiments.cooperative_navigation.training.collector import Collector
from manifold_project.experiments.cooperative_navigation.training.ppo import value_targets, gae
from manifold_project.experiments.cooperative_navigation.training.author_ppo import AuthorPPO
from manifold_project.experiments.applications.envs import make_env
from manifold_project.experiments.applications.evaluation import rollout_episode, summarize


class SuccessNavigationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.public = load_config('navigation', 'smoke')
        self.public.update(seed=40, horizon=3, evaluation_horizon=8, budget=1000,
                           batch_episodes=2, coverage_radius=1e-9)
        self.c = runner_config(self.public)
        self.actor = Actor(6*4+6, self.c)

    def test_no_clock_no_deadline_and_success_terminal(self):
        env = make_env('navigation', self.public)
        try:
            obs, _ = env.reset(seed=71)
            self.assertEqual(obs['self'].shape, (4,4))
            self.assertEqual(env.state().shape, (24,))
            self.assertEqual(env.protocol()['sampling_horizon'],3)
            self.assertEqual(env.protocol()['evaluation_horizon'],8)
            self.assertNotIn('elapsed_time_fraction',env.protocol()['critic_fields'])
            for _ in range(9):
                _, _, terminal, truncated, _ = env.step([0]*4)
                self.assertFalse(terminal or truncated)
            # Force a collision-free simultaneous cover; success is terminal
            # even after both collection and evaluation cutoffs have elapsed.
            for i, (a, goal) in enumerate(zip(env.world.agents, env.world.landmarks)):
                a.state.p_pos[:] = [2*i, 0]
                a.state.p_vel[:] = 0
                goal.state.p_pos[:] = a.state.p_pos
            _, _, terminal, truncated, info = env.step([0]*4)
            self.assertTrue(terminal)
            self.assertFalse(truncated)
            self.assertEqual(info['end_reason'], 'success')
        finally:
            env.close()

    def test_persistent_lanes_and_exact_collector_restore(self):
        original, restored = Collector(self.c,'no_checks',40), Collector(self.c,'no_checks',40)
        try:
            first = original.collect(self.actor,2,'train')
            self.assertEqual([s['env'].t for s in original.sessions], [3,3])
            self.assertTrue(first['truncated'][:,-1].all())
            self.assertFalse(first['terminated'].any())
            restored.costs.update(original.costs)
            restored.counts.update(original.counts)
            restored.load_state_dict(original.state_dict())
            second = original.collect(self.actor,2,'train')
            recovered = restored.collect(self.actor,2,'train')
            self.assertEqual([s['env'].t for s in original.sessions], [6,6])
            torch.testing.assert_close(first['final_state'],second['state'][:,0],rtol=0,atol=0)
            for key in second:
                torch.testing.assert_close(second[key],recovered[key],rtol=0,atol=0)
            # Independent checks must not advance training environments.
            original.collect(self.actor,1,'direction_check')
            self.assertEqual([s['env'].t for s in original.sessions], [6,6])
        finally:
            original.close(); restored.close()

    def test_cutoff_bootstraps_but_terminal_does_not(self):
        class Constant(torch.nn.Module):
            def forward(self,x): return x.new_full(x.shape[:-1], 5.)
        batch = dict(state=torch.zeros(2,2,1),final_state=torch.zeros(2,1),
                     reward=torch.tensor([[1.,2.],[1.,2.]]),
                     terminated=torch.tensor([[False,False],[False,True]]),
                     truncated=torch.tensor([[False,True],[False,False]]),
                     valid=torch.ones(2,2,dtype=torch.bool),lengths=torch.tensor([2,2]))
        c=dict(self.c,gamma=.9,gae_lambda=1.)
        values,target,advantage=value_targets(batch,Constant(),c)
        expected=torch.tensor([[1+.9*(2+.9*5),2+.9*5],[1+.9*2,2.]])
        torch.testing.assert_close(target,expected)
        torch.testing.assert_close(advantage,expected-values)

    def test_training_reset_keeps_pre_reset_bootstrap_and_restore(self):
        c = dict(self.c, train_reset_horizon=5)
        events = []
        original = Collector(c, 'no_checks', 40, lambda stream, **row: events.append((stream,row)))
        restored = Collector(c, 'no_checks', 40)
        try:
            original.collect(self.actor, 2, 'train')  # each lane has age 3
            batch = original.collect(self.actor, 2, 'train')
            self.assertEqual(batch['lengths'].tolist(), [2,1,2,1])
            self.assertEqual([s['env'].t for s in original.sessions], [1,1])
            self.assertFalse(batch['terminated'].any())
            self.assertTrue(batch['truncated'][torch.arange(4),batch['lengths']-1].all())
            self.assertEqual(original.costs['train'], 12)
            self.assertEqual(original.last_training_metrics['train_reset_count'], 2)
            self.assertEqual(original.last_training_metrics['train_task_age_max'], 5)
            self.assertEqual(len([row for stream,row in events if stream == 'training_reset']), 2)
            # The bootstrapped state belongs to the scene BEFORE reset.
            self.assertFalse(torch.equal(batch['final_state'][0],batch['state'][1,0]))
            class PositionValue(torch.nn.Module):
                def forward(self, x): return 5+x[...,0]
            _, target, _ = value_targets(batch, PositionValue(), c)
            for i, length in enumerate(batch['lengths'].tolist()):
                expected = batch['reward'][i,length-1]+c['gamma']*(5+batch['final_state'][i,0])
                torch.testing.assert_close(target[i,length-1],expected)
            # Reach a reset exactly at a round boundary and restore that flag.
            original.collect(self.actor,2,'train')
            original.collect(self.actor,2,'train')
            original.collect(self.actor,2,'train')
            self.assertEqual([s['env'].t for s in original.sessions], [5,5])
            self.assertTrue(all(s['reset_pending'] for s in original.sessions))
            restored.costs.update(original.costs)
            restored.counts.update(original.counts)
            restored.load_state_dict(original.state_dict())
            left, right = original.collect(self.actor,2,'train'), restored.collect(self.actor,2,'train')
            for key in left:
                torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
            self.assertEqual([s['env'].t for s in original.sessions], [3,3])
        finally:
            original.close(); restored.close()

    def test_author_gae_uses_each_segment_final_state(self):
        for normalize, reset_horizon in ((False,None),(True,None),(False,2),(True,2)):
            c=dict(self.c,ppo_value_normalization=normalize,train_reset_horizon=reset_horizon)
            ppo=AuthorPPO(30,24,c)
            if normalize:
                ppo.trainer.value_normalizer.update(torch.tensor([[-1000.],[-2000.],[-3000.]]))
            sampler=Collector(c,'mappo',40)
            try:
                batch=sampler.collect(ppo.actor,2,'train')
                for i, length in enumerate(batch['lengths'].tolist()):
                    batch['reward'][i,:length]=torch.arange(1,length+1)*(10.**i)
                boundary=int(batch['lengths'][0])-1
                batch['terminated'][0,boundary]=True
                batch['truncated'][0,boundary]=False
                inputs,final_inputs=ppo.critic_inputs(batch)
                # Compute an independent masked GAE reference before packing.
                with torch.no_grad():
                    v=ppo.values(inputs).squeeze(-1)
                    f=ppo.values(final_inputs).squeeze(-1)
                    norm=ppo.trainer.value_normalizer
                    if norm is not None:
                        v=torch.as_tensor(norm.denormalize(v.unsqueeze(-1))).squeeze(-1)
                        f=torch.as_tensor(norm.denormalize(f.unsqueeze(-1))).squeeze(-1)
                    nxt=torch.cat([v[:,1:],f.unsqueeze(1)],dim=1)
                    for i, length in enumerate(batch['lengths'].tolist()):
                        nxt[i,length-1]=f[i]
                    expected=gae(batch['reward'],v,c['gamma'],c['gae_lambda'],next_values=nxt,
                                 terminated=batch['terminated'],truncated=batch['truncated'])+v
                buffer=ppo.make_buffer(batch)
                actual=torch.as_tensor(buffer.returns[:-1,0,:,0])
                torch.testing.assert_close(actual,expected[batch['valid']].expand(-1,4),rtol=2e-5,atol=2e-5)
            finally:
                sampler.close()

    def test_evaluation_cutoff_is_censored_observed_return(self):
        from manifold_project.experiments.cooperative_navigation.models.deployment import DeploymentActor
        env=make_env('navigation',self.public)
        try:
            row,_=rollout_episode(env,DeploymentActor(self.actor,self.c),71)
            self.assertEqual(row['steps'],8)
            self.assertEqual(row['end_reason'],'evaluation_cutoff')
            self.assertEqual(row['censored'],1)
            self.assertFalse(row['terminated'])
            self.assertEqual(row['protocol_version'], 'navigation-success-v3')
            self.assertEqual(summarize([row])['censored_count'],1)
            self.assertIsNone(summarize([row])['success_completion_time'])
        finally:
            env.close()
