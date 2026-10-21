"""Mechanism-probe checks: scale semantics, production loss, immutable replay."""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from manifold_project.experiments.common.runtime import configure_runtime
configure_runtime(True)
import torch
from manifold_project.experiments.common.config import load_config
from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
from manifold_project.experiments.cooperative_navigation.diagnose import main
from manifold_project.experiments.cooperative_navigation.models import Critic
from manifold_project.experiments.cooperative_navigation.training.critic_probe import (
    ScaledCritic, fit_variant, normalization, file_hash, validate_packet)
from manifold_project.experiments.cooperative_navigation.training.fixed_target import fingerprint
from manifold_project.experiments.cooperative_navigation.training.ppo import batch_weighted
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt, save_pt


class CriticProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)

    def test_scale_preserves_predictions_and_ignores_padding(self):
        c = dict(hidden=8, gamma=.9, task_mode='success_only', task_horizon=8)
        source = Critic(3, c).eval().requires_grad_(False)
        x = torch.randn(4, 7, 3)
        before = fingerprint(source.state_dict())
        scaled = ScaledCritic(source, -217., 35.)
        torch.testing.assert_close(scaled(x), source(x), atol=3e-5, rtol=2e-5)
        self.assertTrue(torch.equal(scaled.base.net[0].weight, source.net[0].weight))
        self.assertTrue(torch.equal(scaled.base.net[2].weight, source.net[2].weight))
        self.assertEqual(fingerprint(source.state_dict()), before)
        batch = dict(value_target=torch.tensor([[-10., -20., 1e10], [-30., -40., 1e10]]),
                     valid=torch.tensor([[True, True, False], [True, True, False]]))
        n = normalization(batch, c)
        expected = (-10-20*.9-30-40*.9)/(2*1.9)
        self.assertAlmostEqual(n['mean'], expected, places=12)
        batch['value_target'][:, 2] = -1e15
        self.assertEqual(normalization(batch, c), n)

    def test_raw_arm_reproduces_production_optimizer(self):
        c = dict(hidden=8, gamma=.99, task_mode='success_only', task_horizon=8,
                 minibatch_episodes=2, critic_lr=.001, value_coef=.5, grad_norm=.5)
        start = Critic(3, c).eval()
        batch = dict(state=torch.randn(4, 6, 3), value_target=-100+torch.randn(4, 6)*20,
                     reward=torch.zeros(4, 6), valid=torch.ones(4, 6, dtype=torch.bool))
        orders = [[2, 0, 3, 1], [3, 1, 0, 2]]
        reference = deepcopy(start)
        optimizer = torch.optim.Adam(reference.parameters(), lr=c['critic_lr'])
        harness = SimpleNamespace(c=c, seed=45, round=0, optim_steps=Counter(), log=lambda *a, **k: None)
        objective = lambda b: (c['value_coef']*batch_weighted((reference(b['state'])-b['value_target']).square(), b, c), {})
        Runner.optimize(harness, reference, optimizer, batch, 2, 'critic', objective, orders=orders)
        fitted, updates = fit_variant(start, batch, batch, c, orders, normalization(batch, c), 'raw', lambda *a: None)
        self.assertEqual(updates, 4)
        self.assertEqual(fingerprint(fitted.base.state_dict()), fingerprint(reference.state_dict()))

    def test_four_arms_frozen_compact_packet_and_zero_sampling_replay(self):
        workspace = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix='critic_probe_test_', dir=workspace) as tmp:
            root = Path(tmp)
            public = load_config('navigation', 'smoke')
            public.update(hidden=8, heads=1, relation_layers=0, history=1, horizon=8,
                          evaluation_horizon=8, device='cpu', batch_episodes=2,
                          direction_steps=1, actor_fit_steps=1, critic_steps=1,
                          minibatch_size=64, seed=40, budget=256, threads=1)
            config = runner_config(public)
            runner = Runner(root/'source', config, 'no_checks', 40)
            runner.checkpoint()
            runner.sampler.close()
            source = root/'source/checkpoints/final.pt'
            state = load_pt(source)
            state['current_method'] = 'AN'
            save_pt(source, state)
            before = file_hash(source)
            args = ['--mode', 'critic', '--checkpoint', str(source), '--device', 'cpu',
                    '--seeds', '45', '--init-seed', '40', '--batch-episodes', '2',
                    '--validation-episodes', '2', '--critic-epochs', '2', '--allow-duplicate-openmp']
            with redirect_stdout(StringIO()):
                main(args+['--output', str(root/'dry'), '--dry-run'])
            self.assertFalse((root/'dry').exists())
            # No direction fitting or execution evaluation is part of this probe.
            with patch('manifold_project.experiments.cooperative_navigation.models.Direction', side_effect=AssertionError), \
                 patch('manifold_project.experiments.cooperative_navigation.evaluation.evaluate.evaluate', side_effect=AssertionError), \
                 redirect_stdout(StringIO()):
                main(args+['--output', str(root/'run')])
            meta = json.loads((root/'run/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['status'], 'completed')
            self.assertEqual(meta['collection_steps'], 32)
            self.assertEqual(meta['critic_optimizer_updates'], 8)
            for key in ('actor_optimizer_updates', 'direction_optimizer_updates', 'evaluation_steps'):
                self.assertEqual(meta[key], 0)
            packet_path = root/'run/critic_data.pt'
            packet_hash = file_hash(packet_path)
            packet = load_pt(packet_path)
            validate_packet(packet)
            self.assertNotIn('x', packet['batch'])
            self.assertNotIn('mu', packet['batch'])
            self.assertEqual(packet['normalization'], normalization(packet['batch'], config))
            replay = ['--mode', 'critic', '--critic-data', str(packet_path), '--output', str(root/'replay'),
                      '--device', 'cpu', '--allow-duplicate-openmp']
            with patch('manifold_project.experiments.cooperative_navigation.training.critic_probe.Collector', side_effect=AssertionError), \
                 patch('manifold_project.experiments.cooperative_navigation.training.critic_probe.Actor', side_effect=AssertionError), \
                 redirect_stdout(StringIO()):
                main(replay)
            replay_meta = json.loads((root/'replay/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(replay_meta['collection_steps'], 0)
            self.assertEqual(replay_meta['original_collection_steps'], 32)
            self.assertEqual(replay_meta['critic_optimizer_updates'], 8)
            for p in (root/'run/candidates').glob('*.pt'):
                other = root/'replay/candidates'/p.name
                self.assertEqual(fingerprint(load_pt(p)['state_dict']), fingerprint(load_pt(other)['state_dict']))
            self.assertEqual(file_hash(source), before)
            self.assertEqual(file_hash(packet_path), packet_hash)
            with self.assertRaises(FileExistsError):
                main(args+['--output', str(root/'run')])
            with self.assertRaises(ValueError):
                main(['--mode', 'critic', '--critic-data', str(packet_path), '--output', str(root/'bad'),
                      '--device', 'cpu', '--critic-epochs', '3'])
            modified = deepcopy(packet)
            modified['batch']['value_target'][0, 0] += 1
            with self.assertRaises(ValueError):
                validate_packet(modified)


if __name__ == '__main__':
    unittest.main()
