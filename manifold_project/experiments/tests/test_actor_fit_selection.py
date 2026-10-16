"""Controlled optimizer trajectories, real commits, and self-contained target replay."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from manifold_project.experiments.tests.test_direction_remediation import settings
from manifold_project.experiments.tests.test_recovery import same
import torch
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.actor_fit import fit_actor
from manifold_project.experiments.cooperative_navigation.training.fixed_target import validate_packet
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt, save_pt
from manifold_project.experiments.cooperative_navigation.diagnose import main, run, parser

EVIDENCE = {}


class ScalarActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.))
    def forward(self, x):
        return torch.stack((self.logit, self.logit*0)).softmax(-1).expand(*x.shape[:-1], 2)


class ActorFitTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[3]
        self.tmp = tempfile.TemporaryDirectory(prefix='actor_selection_test_', dir=workspace)
        self.root = Path(self.tmp.name).resolve()
        assert self.root.is_relative_to(workspace)
    def tearDown(self):
        self.tmp.cleanup()

    def test_public_configuration_routes_selection_and_rejects_invalid_values(self):
        from manifold_project.experiments.common.config import load_config
        from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
        override = self.root/'selection.json'
        override.write_text(json.dumps(dict(actor_fit_selection='best_train_kl', actor_lr=3e-5)))
        public = load_config('navigation', 'smoke', override)
        c = runner_config(public, method='AN')
        self.assertEqual(c['actor_fit_selection'], 'best_train_kl')
        self.assertEqual(c['actor_lr'], 3e-5)
        self.assertEqual(c['actor_fit_selection_atol'], 1e-12)
        for values in (dict(actor_fit_selection='best_return'), dict(actor_fit_selection_atol=-1),
                       dict(actor_fit_selection_rtol=True)):
            override.write_text(json.dumps(values))
            with self.assertRaises(ValueError): load_config('navigation', 'smoke', override)

    def fixture(self, name, target_logit):
        c = settings()
        c.update(minibatch_episodes=2, actor_lr=.2)
        runner = Runner(self.root/name, c, 'no_checks', 40)
        self.addCleanup(runner.sampler.close)
        actor = ScalarActor()
        x = torch.zeros(2, 1, 1, 1)
        batch = dict(x=x, reward=torch.zeros(2, 1), valid=torch.ones(2, 1, dtype=torch.bool),
                     target=torch.tensor([target_logit, 0.]).softmax(-1).expand(2, 1, 1, 2).clone())
        return runner, actor, batch

    def test_intermediate_best_full_budget_and_legacy_trajectory(self):
        runner, old, batch = self.fixture('middle', .2)
        frozen = deepcopy((old.state_dict(), batch))
        result = fit_actor(runner, old, batch, 3, selection='best_train_kl')
        self.assertEqual(result['info']['fit_best_epoch'], 1)
        self.assertEqual(result['info']['fit_optimizer_updates'], 3)
        self.assertTrue(result['info']['fit_candidate_changed'])
        self.assertGreater(result['info']['fit_final_train_kl'], result['info']['fit_best_train_kl'])
        # Compare the final iterate to the previous optimizer loop, not a second selector.
        legacy = deepcopy(old)
        optimizer = torch.optim.Adam(legacy.parameters(), lr=.2)
        def loss(mb):
            p = legacy(mb['x'])
            return runner.weighted((mb['target']*(mb['target'].log()-p.log())).sum(-1), mb), {}
        runner.optimize(legacy, optimizer, batch, 3, 'actor_fit', loss)
        self.assertTrue(same(legacy.state_dict(), result['final'].state_dict()))
        self.assertTrue(same(frozen, (old.state_dict(), batch)))
        saved_best = deepcopy(result['best'].state_dict())
        with torch.no_grad(): result['final'].logit.add_(1.)
        self.assertTrue(same(saved_best, result['best'].state_dict()))
        EVIDENCE['actor_selection_middle'] = dict(selected_epoch=1, total_updates=3,
            final_iterate_matches_legacy=True, independent_candidate_storage=True)

    def test_zero_candidate_and_ties(self):
        runner, old, batch = self.fixture('zero', .001)
        result = fit_actor(runner, old, batch, 3, selection='best_train_kl')
        self.assertEqual(result['info']['fit_best_epoch'], 0)
        self.assertEqual(result['info']['fit_optimizer_updates'], 3)
        self.assertFalse(result['info']['fit_candidate_changed'])
        self.assertFalse(same(result['final'].state_dict(), old.state_dict()))
        self.assertTrue(same(result['selected'].state_dict(), old.state_dict()))
        batch['target'] = old(batch['x']).detach()
        tied = fit_actor(runner, old, batch, 3, selection='best_train_kl')
        self.assertEqual(tied['info']['fit_best_epoch'], 0)

    def test_production_commit_and_zero_update_cost(self):
        for selection in ('last', 'best_train_kl'):
            c = settings()
            c.update(actor_epochs=3, attempts=3, actor_fit_selection=selection)
            if selection == 'best_train_kl':
                # Force the tolerance branch to exercise zero-update commit semantics.
                c['actor_fit_selection_atol'] = 1.
            runner = Runner(self.root/selection, c, 'no_checks', 40)
            before = deepcopy(runner.actor.state_dict())
            captured = {}
            def fit(*args, **kwargs):
                result = fit_actor(*args, **kwargs)
                captured['selected'] = deepcopy(result['selected'].state_dict())
                return result
            try:
                with patch('manifold_project.experiments.cooperative_navigation.training.runner.fit_actor', side_effect=fit):
                    _, info = runner.update()
                self.assertTrue(same(runner.actor.state_dict(), captured['selected']))
                self.assertEqual(info['fit_optimizer_updates'], 6)
                self.assertEqual(runner.optim_steps['actor_fit'], 6)
                self.assertEqual(runner.sampler.used, 16)
                self.assertEqual(info['candidates'], 1)
                self.assertTrue(info['accepted'])
                self.assertEqual(info['actor_changed'], selection == 'last')
                if selection == 'best_train_kl':
                    self.assertEqual(info['zero_update_reason'], 'selected_epoch_zero')
                    self.assertTrue(same(before, runner.actor.state_dict()))
            finally:
                runner.sampler.close()
        EVIDENCE['actor_selection_commit'] = dict(selected_state_committed=True,
            zero_update_training_steps=16, zero_update_optimizer_steps=6, no_extra_eta_retry=True)

    def test_prepare_and_replay_without_source_sampling_or_direction_training(self):
        c = settings()
        runner = Runner(self.root/'source', c, 'no_checks', 40)
        runner.checkpoint(); runner.sampler.close()
        checkpoint = self.root/'source/checkpoints/final.pt'
        state = load_pt(checkpoint)
        state['current_method'] = 'AN'
        save_pt(checkpoint, state)
        base = ['--mode', 'fit', '--device', 'cpu', '--allow-duplicate-openmp']
        prepare = base+['--checkpoint', str(checkpoint), '--output', str(self.root/'prepared'),
            '--seeds', '41', '--init-seed', '40', '--batch-episodes', '2', '--validation-episodes', '2',
            '--evaluation-episodes', '2', '--evaluation-seed', '20261017', '--direction-updates', '4',
            '--fit-epochs', '2', '--prepare-only']
        with redirect_stdout(StringIO()): main(prepare)
        prepared = json.loads((self.root/'prepared/manifest.json').read_text())
        self.assertEqual(prepared['direction_epochs'], [2])
        self.assertEqual(prepared['direction_optimizer_updates'], 4)
        self.assertEqual(prepared['actor_optimizer_updates'], 0)
        self.assertEqual(prepared['evaluation_steps'], 0)
        snapshot = Path(prepared['fixed_targets'][0]['path'])
        snapshot_bytes = snapshot.read_bytes()
        packet = load_pt(snapshot)
        validate_packet(packet)
        checkpoint.rename(checkpoint.with_suffix('.unavailable'))
        modules = []
        original_optimize = Runner.optimize
        def optimize(runner, model, optimizer, batch, epochs, module, objective, **kwargs):
            modules.append(module)
            return original_optimize(runner, model, optimizer, batch, epochs, module, objective, **kwargs)
        with patch('manifold_project.experiments.cooperative_navigation.training.collector.Collector.collect',
                   side_effect=AssertionError('Replay recollected a training/validation batch')), \
             patch.object(Runner, 'labels', side_effect=AssertionError('Replay regenerated labels')), \
             patch.object(Runner, 'optimize', optimize), redirect_stdout(StringIO()):
            for name, rates in (('replay_a', ['0.0003', '0.00003']), ('replay_b', ['0.0003'])):
                main(base+['--fixed-target', str(snapshot), '--output', str(self.root/name),
                           '--actor-lrs', *rates, '--fit-trace-every-updates', '1'])
        self.assertEqual(modules, ['actor_fit']*3)
        outputs = []
        for name in ('replay_a', 'replay_b'):
            folder = self.root/name
            manifest = json.loads((folder/'manifest.json').read_text())
            self.assertEqual(manifest['collection_steps'], 0)
            self.assertEqual(manifest['direction_optimizer_updates'], 0)
            self.assertEqual(manifest['evaluation_scene_seed'], 20261017)
            outputs.append(load_pt(next((folder/'fit_candidates').glob('*lr0.0003.pt'))))
        for key in ('final', 'best', 'history', 'fit_orders'):
            self.assertTrue(same(outputs[0][key], outputs[1][key]), key)
        self.assertEqual(snapshot.read_bytes(), snapshot_bytes)
        with (self.root/'replay_a/diagnostics.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(row['candidate_kind']=='teacher' for row in rows), 1)
        with (self.root/'replay_a/evaluation_episodes.csv').open(encoding='utf-8-sig') as stream:
            episodes = list(csv.DictReader(stream))
        actual_cost = sum(int(row['episode_steps']) for row in episodes if not row.get('evaluation_reused_from'))
        manifest = json.loads((self.root/'replay_a/manifest.json').read_text())
        self.assertEqual(manifest['evaluation_steps'], actual_cost)
        for flag in (['--eta', '.2'], ['--direction-updates', '2'], ['--seeds', '42'], ['--fit-epochs', '3']):
            with self.assertRaises(ValueError), redirect_stdout(StringIO()):
                run(parser().parse_args(base+['--fixed-target', str(snapshot), '--output', str(self.root/'bad'), *flag]))
        self.assertFalse((self.root/'bad').exists())
        corrupt = deepcopy(packet)
        corrupt['batch']['target'][0,0,0,0] += .01
        with self.assertRaises(ValueError): validate_packet(corrupt)
        EVIDENCE['fixed_target_replay'] = dict(source_checkpoint_not_required=True,
            training_validation_sampling_calls=0, direction_updates=0, labels_regenerated=False,
            repeated_fit_states_identical=True, raw_teacher_evaluated=True, evaluation_cost_deduplicated=True)


if __name__ == '__main__': unittest.main()
