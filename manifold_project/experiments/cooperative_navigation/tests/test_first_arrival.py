"""First-arrival task boundaries, variable batches and frozen-policy evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manifold_project.experiments.cooperative_navigation.tests.test_protocol import tiny
from manifold_project.experiments.cooperative_navigation.configs import CONDITIONS, load_config
from manifold_project.experiments.cooperative_navigation.envs.navigation import Navigation
from manifold_project.experiments.cooperative_navigation.models import Actor, load_actor
from manifold_project.experiments.cooperative_navigation.training.collector import Collector, episode
from manifold_project.experiments.cooperative_navigation.training.ppo import value_targets, batch_weighted
from manifold_project.experiments.cooperative_navigation.training.author_ppo import AuthorPPO
from manifold_project.experiments.cooperative_navigation.training.runner import Runner, minimum_cost
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt
from manifold_project.experiments.cooperative_navigation.evaluation.evaluate import evaluate
from manifold_project.experiments.cooperative_navigation.evaluation.transfer import transfer_suite
from manifold_project.experiments.cooperative_navigation.evaluation.reporting import source_tables, read_csv
from manifold_project.experiments.cooperative_navigation.evaluation.plotting import plot_suite, plot_transfer
from manifold_project.experiments.cooperative_navigation.evaluation.monitoring import TrainingMonitor


def arrival_config(**kw):
    config = dict(tiny(), task_mode='first_arrival', task_horizon=7, budget=70,
                  eval_fractions=[0, .5, 1])
    config.update(kw)
    return config


class FirstArrivalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native_success_and_deadline_are_terminal_with_last_reward(self):
        c = arrival_config()
        for success, horizon in ((False,7),(True,7),(True,1)):
            c['task_horizon'] = horizon
            env = Navigation(c)
            try:
                env.reset(72)
                for i, (agent, landmark) in enumerate(zip(env.world.agents, env.world.landmarks)):
                    agent.state.p_pos = np.array([i*.5, 0.])
                    agent.state.p_vel = np.zeros(2)
                    landmark.state.p_pos = agent.state.p_pos + ([0., 0.] if success else [0., 2.])
                steps = 1 if success else c['task_horizon']
                for t in range(steps):
                    obs, reward, terminal, truncated = env.step(np.zeros(4, dtype=int))
                    self.assertEqual(terminal, t == steps-1)
                    self.assertFalse(truncated)
                    m = env.metrics()
                    self.assertAlmostEqual(reward, -.5*4*m['distance']-m['collision_pairs']/4, places=6)
                self.assertEqual(env.end_reason, 'success' if success else 'deadline')
                self.assertEqual(obs.shape, (4,16))
                self.assertAlmostEqual(float(env.state()[-1]), steps/c['task_horizon'], places=6)
                with self.assertRaises(RuntimeError):
                    env.step(np.zeros(4, dtype=int))
            finally:
                env.close()

    def test_fixed_real_step_quota_padding_and_pre_reset_clock(self):
        c = arrival_config()
        actor = Actor(45,c)
        sampler = Collector(c,'ours',1)
        batch = sampler.collect(actor,2,'train')
        self.assertEqual(sampler.used,10)
        self.assertEqual(batch['lengths'].tolist(),[7,3])
        self.assertEqual(int(batch['valid'].sum()),10)
        self.assertTrue(batch['terminated'][0,6])
        self.assertTrue(batch['truncated'][1,2])
        self.assertFalse(batch['terminated'][1,2])
        torch.testing.assert_close(batch['final_x'][:,:,-1], torch.tensor([[1.]*4,[3/7]*4]))
        torch.testing.assert_close(batch['final_x'][1,:,22:38].flatten(),batch['final_state'][1,:-1])
        self.assertEqual(minimum_cost(c,'ours'),10+3*7)
        self.assertEqual(minimum_cost(c,'mappo'),10)
        sampler.collect(actor,1,'direction_check')
        self.assertEqual(sampler.used,17)
        rows = Collector(c,'ours',1).collect(actor,1,'train',retain=False)
        self.assertEqual(sum(r['episode_steps'] for r in rows),5)
        self.assertEqual(rows[-1]['end_reason'],'collection_cutoff')

    def test_targets_never_bootstrap_task_terminal_or_train_on_padding(self):
        c = arrival_config()
        actor = Actor(45,c)
        batch = Collector(c,'ours',1).collect(actor,2,'train')
        class Constant(torch.nn.Module):
            def forward(self,x):
                return x[...,0]*0-10
        for local in (False, True):
            values,target,adv = value_targets(batch,Constant(),c,local=local)
            last = batch['reward'][0,6].expand_as(target[0,6])
            torch.testing.assert_close(target[0,6],last)
            last = (batch['reward'][1,2]-10*c['gamma']).expand_as(target[1,2])
            torch.testing.assert_close(target[1,2],last)
            self.assertTrue((target[~batch['valid']]==0).all())
            self.assertTrue((adv[~batch['valid']]==0).all())
        x = torch.ones_like(batch['reward'],requires_grad=True)
        loss = batch_weighted(x,batch,c)
        loss.backward()
        self.assertTrue((x.grad[~batch['valid']]==0).all())
        expected = (sum(c['gamma']**i for i in range(7))+sum(c['gamma']**i for i in range(3)))/(2*sum(c['gamma']**i for i in range(7)))
        self.assertAlmostEqual(float(loss),expected,places=6)

    def test_author_packing_matches_independent_gae_and_direct_upstream_update(self):
        for condition in ('mappo','ippo'):
            for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
                c = arrival_config(device=device)
                model = AuthorPPO(45,65,c,condition)
                batch = Collector(c,condition,1).collect(model.actor,2,'train')
                if condition=='ippo':
                    del batch['state'],batch['final_state']
                buffer = model.make_buffer(batch)
                self.assertEqual(buffer.rewards.shape,(10,1,4,1))
                self.assertEqual(float(buffer.masks[7].sum()),0.)
                self.assertEqual(float(buffer.masks[10].sum()),4.)
                # Oracle computes each episode independently on unnormalized values.
                inputs,final_inputs = model.critic_inputs(batch)
                norm = model.trainer.value_normalizer
                values = norm.denormalize(model.values(inputs).cpu().numpy())
                finals = norm.denormalize(model.values(final_inputs).cpu().numpy())
                expected = []
                for e,length in enumerate(batch['lengths'].tolist()):
                    targets, tail = [], 0.
                    for t in reversed(range(length)):
                        terminal = bool(batch['terminated'][e,t])
                        next_v = finals[e] if t==length-1 else values[e,t+1]
                        delta = float(batch['reward'][e,t])+c['gamma']*(0 if terminal else next_v)-values[e,t]
                        tail = delta+c['gamma']*c['gae_lambda']*(0 if terminal else tail)
                        targets.append(tail+values[e,t])
                    expected.extend(reversed(targets))
                expected = np.broadcast_to(np.stack(expected)[:,None],buffer.returns[:-1].shape)
                np.testing.assert_allclose(buffer.returns[:-1],expected,atol=1e-5)
                oracle = deepcopy(model)
                rng = torch.get_rng_state()
                oracle.trainer.train(oracle.make_buffer(batch))
                torch.set_rng_state(rng)
                stats = model.update(batch)
                self.assertTrue(np.isfinite(stats['critic_mse']))
                torch.testing.assert_close(model.actor.state_dict(),oracle.actor.state_dict(),rtol=0,atol=0)
                torch.testing.assert_close(model.normalizer_state(),oracle.normalizer_state(),rtol=0,atol=0)

    def test_success_metrics_count_early_stop_including_initial_success(self):
        c = arrival_config()
        actor = Actor(45,c)
        # Independently specified mixed outcomes ensure that failures are not
        # silently dropped from duration or success-rate statistics.
        base = episode(actor,c,72,73)
        rows = [dict(base,success=1.,success_steps=2,restricted_steps=2,episode_steps=2,end_reason='success'),
                dict(base,success=0.,success_steps=None,restricted_steps=7,episode_steps=7,end_reason='deadline')]
        with patch('manifold_project.experiments.cooperative_navigation.evaluation.evaluate.episode',side_effect=rows):
            metrics,_ = evaluate(actor,c,'ours',0,'test',2)
        self.assertEqual(metrics['success_rate'],.5)
        self.assertEqual(metrics['success_steps_mean'],2)
        self.assertEqual(metrics['restricted_mean_steps'],4.5)
        reset = Navigation.reset
        def solved(env,seed):
            obs = reset(env,seed)
            env.end_reason = 'success'
            return obs
        with patch.object(Navigation,'reset',solved),patch.object(Navigation,'step',side_effect=AssertionError('must stop at reset')):
            row = episode(actor,c,72,73)
        self.assertEqual((row['success'],row['episode_steps'],row['J']),(1.,0,0.))

    def test_new_configs_separate_deadline_from_train_quota(self):
        root = Path(__file__).resolve().parents[1]/'configs/first_arrival'
        for name in ('mc','gae','author_mpe'):
            c = load_config('pilot',root/f'{name}_20m.json')
            self.assertEqual(c['task_mode'],'first_arrival')
            self.assertEqual(c['task_horizon'],500)
            self.assertEqual(c['train_episodes']*c['horizon'],1600)

    def test_all_conditions_resume_and_frozen_transfer(self):
        c = arrival_config()
        with tempfile.TemporaryDirectory(dir=Path.cwd(),prefix='arrival_check_') as folder:
            root = Path(folder)
            for condition in CONDITIONS:
                full = Runner(root/(condition+'_full'),c,condition,4)
                full.run()
                paused = Runner(root/(condition+'_resume'),c,condition,4)
                paused.run(max_rounds=1)
                resumed = Runner(paused.directory,c,condition,4,resume=True)
                resumed.run()
                torch.testing.assert_close(full.actor.state_dict(),resumed.actor.state_dict(),rtol=0,atol=0)
                torch.testing.assert_close(full.critic.state_dict(),resumed.critic.state_dict(),rtol=0,atol=0)
                self.assertEqual(full.curves,resumed.curves)
                self.assertEqual(full.sampler.costs,resumed.sampler.costs)
                self.assertLessEqual(full.sampler.used,c['budget'])
                self.assertEqual(full.eval_steps,sum(r['episodes']*r['episode_steps'] for r in full.curves))
                self.assertEqual(full.best['score'],max((r['success_rate'],-r['restricted_mean_steps'],-r['collision_pairs_total']) for r in full.curves))
                state = load_pt(full.directory/'checkpoints/final.pt')
                actor = load_actor(state)
                before = deepcopy(actor.state_dict())
                for n in (3,4,6,8):
                    m = episode(actor,c,42,43,n=n)
                    self.assertLessEqual(m['episode_steps'],7)
                torch.testing.assert_close(actor.state_dict(),before,rtol=0,atol=0)
            manifest = dict(config=c,jobs=[dict(path='mappo_full',condition='mappo',seed=4),
                                          dict(path='ours_full',condition='ours',seed=4)])
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            source_tables(root,c)
            rows = transfer_suite(root,manifest)
            self.assertTrue(all('success_rate' in r for r in rows))
            with patch('manifold_project.experiments.cooperative_navigation.evaluation.transfer.evaluate',side_effect=AssertionError('must reuse')):
                self.assertEqual(transfer_suite(root,manifest),rows)
            plot_suite(root)
            plot_transfer(root)
            TrainingMonitor(root/'training_monitor.png',1)(full)
            for name in ('overview.png','transfer_overview.png','training_monitor.png'):
                self.assertGreater((root/name).stat().st_size,1000)
            self.assertTrue(any(r['metric']=='success_rate' for r in read_csv(root/'training_summary.csv')))
            from manifold_project.experiments.cooperative_navigation.evaluation.evaluate import main as evaluate_main
            self.assertEqual(evaluate_main(['--suite',str(root),'--steps','7','--episodes','2']),0)
            self.assertIn('success_rate_se',read_csv(root/'first_arrival_h7.csv')[0])

    def test_gif_rollout_stops_on_success(self):
        from manifold_project.experiments.visualize_navigation import rollout
        c = arrival_config()
        native_step = Navigation.step
        def success_step(env,actions):
            result = native_step(env,actions)
            env.end_reason = 'success'
            return result[0],result[1],True,False
        with patch.object(Navigation,'step',success_step):
            frames,m = rollout(Actor(45,c),c,4,72,73)
        self.assertEqual(len(frames),2)
        self.assertTrue(m['success'])
        self.assertEqual(frames[-1]['end_reason'],'success')

    def test_stability_uses_success_and_failure_inclusive_duration(self):
        from manifold_project.experiments.assess_navigation_stability import assess_arrival
        rows = [dict(budget_checkpoint=t,success_rate=.95,restricted_mean_steps=70.,
                     success_rate_se=.005,restricted_mean_steps_se=2.) for t in (70,75,80,85,90,100)]
        self.assertEqual(assess_arrival(rows,100,500)['code'],'stable_high')
        low = [dict(r,success_rate=.1) for r in rows]
        self.assertEqual(assess_arrival(low,100,500)['code'],'stable_low')
        self.assertEqual(assess_arrival(rows[:3],100,500)['code'],'insufficient_nodes')

    def test_first_arrival_continuation_preserves_protocol(self):
        from manifold_project.experiments.continue_navigation import make_plan, ContinuationRunner
        c = arrival_config(budget=10,eval_fractions=[0,1],eval_episodes=2)
        with tempfile.TemporaryDirectory(dir=Path.cwd(),prefix='arrival_branch_check_') as folder:
            root = Path(folder)
            parent = Runner(root/'parent',c,'no_checks',4)
            parent.run()
            path = parent.directory/'checkpoints/final.pt'
            state = load_pt(path)
            plan = make_plan(path,state,[1],20,eval_episodes=2)
            job = plan['jobs'][0]
            branch = ContinuationRunner(root/'branch',job['config'],plan['source'],job['variant'],
                                        plan['evaluation_seed'],parent=state)
            result = branch.run()
            self.assertIn('success_AUC',result)
            self.assertEqual(branch.eval_steps,sum(r['episode_steps']*r['episodes'] for r in branch.curves))
            self.assertIsInstance(branch.best['score'],tuple)


if __name__ == '__main__':
    unittest.main()
