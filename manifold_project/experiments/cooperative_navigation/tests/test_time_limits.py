"""Continuing-task regressions: timeouts bootstrap without crossing resets."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manifold_project.experiments.cooperative_navigation.configs import continuing_task, load_config, validate, CONDITIONS
from manifold_project.experiments.cooperative_navigation.envs.navigation import Navigation
from manifold_project.experiments.cooperative_navigation.models import Actor, Critic
from manifold_project.experiments.cooperative_navigation.observations.history import History
from manifold_project.experiments.cooperative_navigation.training.collector import episode, Collector
from manifold_project.experiments.cooperative_navigation.training.ppo import returns, gae, value_targets
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from test_protocol import tiny


class TimeLimitTests(unittest.TestCase):
    def test_native_timeout_preserves_final_observation_and_reward(self):
        c = dict(tiny(), task_mode='continuing', horizon=3)
        env = Navigation(c)
        try:
            initial = env.reset(72)
            self.assertEqual(env.state_dim, 64)
            for t in range(c['horizon']):
                obs, reward, terminal, timeout = env.step(np.ones(env.n, dtype=int))
                self.assertFalse(terminal)
                self.assertEqual(timeout, t == c['horizon']-1)
                self.assertEqual(obs.shape, initial.shape)
                np.testing.assert_array_equal(env.state(), obs.flatten())
                m = env.metrics()
                self.assertAlmostEqual(reward, -.5*env.n*m['distance']-m['collision_pairs']/env.n, places=6)
            self.assertFalse(np.array_equal(obs, initial))
        finally:
            env.close()

    def test_coverage_is_not_terminal_and_clock_does_not_enter_input(self):
        c = dict(tiny(), task_mode='continuing')
        env = Navigation(c)
        try:
            env.reset(72)
            for i, (a, landmark) in enumerate(zip(env.world.agents, env.world.landmarks)):
                a.state.p_pos = np.array([i*.5, 0.])
                landmark.state.p_pos = a.state.p_pos.copy()
                a.state.p_vel = np.zeros(2)
            _, _, terminal, timeout = env.step(np.zeros(env.n, dtype=int))
            self.assertEqual(env.metrics()['coverage'], 1.)
            self.assertFalse(terminal or timeout)
        finally:
            env.close()
        h1, h2 = (History(4, 16, 2, 100, include_time=False) for _ in range(2))
        x = h1.append(np.ones((4, 16)), None, 0)
        np.testing.assert_array_equal(x, h2.append(np.ones((4, 16)), None, 700))
        self.assertEqual(x.shape, (4, 44))

    def test_collector_records_pre_reset_final_history(self):
        c = dict(tiny(), task_mode='continuing')
        actor = Actor(44, c)
        batch = episode(actor, c, 123, 456, retain=True)
        self.assertFalse(batch['terminated'].any())
        self.assertEqual(batch['truncated'].tolist(), [False]*4+[True])
        np.testing.assert_array_equal(batch['final_x'][:, :22], batch['x'][-1, :, -22:])
        np.testing.assert_array_equal(batch['final_x'][:, -6:-1], np.eye(5)[batch['actions'][-1]])
        np.testing.assert_array_equal(batch['final_x'][:, -22:-6].flatten(), batch['final_state'])
        metrics = episode(actor, c, 123, 456)
        self.assertAlmostEqual(metrics['J'], float(np.dot(c['gamma']**np.arange(5), batch['reward'])))
        self.assertEqual(metrics['mean_reward'], float(np.mean(batch['reward'])))
        self.assertTrue(0 <= metrics['tail_coverage'] <= 1)

    def test_timeout_bootstrap_negative_value_and_true_terminal_precedence(self):
        r = torch.tensor([[1., 2., 3.]])
        v = torch.tensor([[.4, .5, .6]])
        nv = torch.tensor([[.5, .6, -10.]])
        end = torch.tensor([[False, False, True]])
        expected = returns(r, .9) + torch.tensor([[-10*.9**3, -10*.9**2, -10*.9]])
        target = returns(r, .9, next_values=nv, truncated=end)
        torch.testing.assert_close(target, expected)
        torch.testing.assert_close(gae(r, v, .9, 1., next_values=nv, truncated=end), target-v)
        torch.testing.assert_close(returns(r, .9, next_values=nv, terminated=end, truncated=end), returns(r, .9))
        torch.testing.assert_close(gae(r, v, .9, 1., next_values=nv, terminated=end, truncated=end), returns(r, .9)-v)

    def test_no_trace_or_rewards_leak_across_reset(self):
        # The third transition belongs to another episode, with a large reward.
        r = torch.tensor([[1., 2., 1000.]])
        v = torch.zeros_like(r)
        nv = torch.tensor([[0., -10., 0.]])
        timeout = torch.tensor([[False, True, False]])
        expected = torch.tensor([[1+.9*(2-9), 2-9, 1000.]])
        torch.testing.assert_close(returns(r, .9, next_values=nv, truncated=timeout), expected)
        torch.testing.assert_close(gae(r, v, .9, 1., next_values=nv, truncated=timeout), expected)

    def test_all_methods_bootstrap_using_their_own_frozen_critic(self):
        c = dict(tiny(), task_mode='continuing')
        actor = Actor(44, c)
        batch = Collector(c, 'ours', 0).collect(actor, 2, 'train')
        class FirstCoordinate(torch.nn.Module):
            def forward(self, x):
                return x[..., 0]
        for local in (False, True):
            # Make each agent's final local value distinct from the global value.
            batch['final_x'][..., 0] = torch.tensor([-1., -2., -3., -4.])
            batch['final_state'][..., 0] = -20.
            v, target, a = value_targets(batch, FirstCoordinate(), c, local=local)
            final = batch['final_x'][..., 0] if local else batch['final_state'][..., 0]
            r = batch['reward'][:, -1]
            expected = (r.unsqueeze(-1) if local else r) + c['gamma']*final
            torch.testing.assert_close(target[:, -1], expected)
            torch.testing.assert_close(a[:, -1], expected-v[:, -1])
            self.assertFalse(target.requires_grad or a.requires_grad)
            _, old_target, _ = value_targets(batch, FirstCoordinate(), dict(c, task_mode='finite_horizon'), local=local)
            old = returns(batch['reward'], c['gamma'])
            torch.testing.assert_close(old_target, old.unsqueeze(-1).expand_as(v) if local else old)

    def test_long_replay_and_old_configuration_compatibility(self):
        from manifold_project.experiments.visualize_navigation import rollout
        c = dict(tiny(), task_mode='continuing', horizon=120)
        frames, _ = rollout(Actor(44, c), c, 4, 72, 73)
        self.assertEqual(len(frames), 121)
        self.assertEqual([f['step'] for f in frames], list(range(121)))
        old = dict(tiny())
        old.pop('task_mode')
        self.assertFalse(continuing_task(old))
        self.assertEqual(episode(Actor(45, old), old, 72, 73, retain=True)['x'].shape[-1], 45)
        with self.assertRaises(ValueError):
            validate(dict(c, gamma=1.))

    def test_continuing_all_conditions_resume_exactly(self):
        c = dict(tiny(), task_mode='continuing', budget=50)
        with tempfile.TemporaryDirectory() as folder:
            for condition in CONDITIONS:
                full = Runner(Path(folder)/(condition+'_full'), c, condition, 4)
                full.run()
                paused = Runner(Path(folder)/(condition+'_resume'), c, condition, 4)
                paused.run(max_rounds=1)
                resumed = Runner(paused.directory, c, condition, 4, resume=True)
                resumed.run()
                torch.testing.assert_close(full.actor.state_dict(), resumed.actor.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(full.critic.state_dict(), resumed.critic.state_dict(), rtol=0, atol=0)
                self.assertEqual(full.curves, resumed.curves)
                self.assertEqual(full.sampler.costs, resumed.sampler.costs)
                self.assertLessEqual(full.sampler.used, c['budget'])
                self.assertIn('tail_coverage', full.summary())

    def test_return_gate_uses_same_frozen_tail_for_both_policies(self):
        c = dict(tiny(), task_mode='continuing')
        class ConstantCritic(torch.nn.Module):
            def forward(self, x):
                return x[..., 0]*0-10
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(folder, c, 'ours', 0)
            for purpose in ('return_old', 'return_candidate'):
                scores, observed = runner.check_returns(runner.actor, purpose, ConstantCritic())
                np.testing.assert_allclose(np.array(scores)-observed, -10*c['gamma']**c['horizon'], atol=2e-6)
            self.assertEqual(runner.sampler.used, 2*c['return_check_episodes']*c['horizon'])

    def test_long_evaluation_cli_uses_observed_rewards_and_does_not_change_checkpoint(self):
        from manifold_project.experiments.cooperative_navigation.evaluation.evaluate import main
        from manifold_project.experiments.cooperative_navigation.evaluation.reporting import read_csv
        c = dict(tiny(), task_mode='continuing', budget=10, eval_fractions=[0, 1])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner = Runner(root/'mappo/seed_4', c, 'mappo', 4)
            runner.run()
            (root/'manifest.json').write_text(json.dumps(dict(config=c, jobs=[
                dict(condition='mappo', seed=4, path='mappo/seed_4')])), encoding='utf-8')
            checkpoint = runner.directory/'checkpoints/final.pt'
            before = checkpoint.read_bytes()
            with patch.object(Critic, 'forward', side_effect=AssertionError('evaluation must not bootstrap')):
                self.assertEqual(main(['--suite', str(root), '--steps', '7', '--episodes', '2']), 0)
            rows = read_csv(root/'long_rollout_h7.csv')
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]['evaluation_steps']), 14)
            self.assertTrue(0 <= float(rows[0]['tail_coverage']) <= 1)
            self.assertEqual(before, checkpoint.read_bytes())
            with self.assertRaises(SystemExit):
                main(['--suite', str(root), '--steps', '7', '--episodes', '2'])

    def test_learning_configs_use_same_continuing_protocol(self):
        root = Path(__file__).resolve().parents[1]/'configs'
        mc = load_config('pilot', root/'learning_20m/mc.json')
        g = load_config('pilot', root/'learning_20m/gae.json')
        self.assertTrue(continuing_task(mc) and continuing_task(g))
        self.assertEqual({k for k in mc if mc[k] != g[k]}, {'direction_label'})
        self.assertFalse(continuing_task(load_config('pilot', root/'study_v3/reference.json')))


if __name__ == '__main__':
    unittest.main()
