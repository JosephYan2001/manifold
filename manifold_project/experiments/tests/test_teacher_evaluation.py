"""Teacher-only execution, independent pairing, zero optimization, and bounded cost."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from manifold_project.experiments.tests.test_direction_remediation import settings
from manifold_project.experiments.cooperative_navigation.diagnose import main, parser, run
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.collector import Collector
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt, save_pt
from manifold_project.experiments.cooperative_navigation.training.fixed_target import Target, fingerprint
from manifold_project.experiments.cooperative_navigation.evaluation.evaluate import evaluate
from manifold_project.experiments.cooperative_navigation.evaluation import teacher_probe
import torch

EVIDENCE = {}


def records(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


class TeacherEvaluationTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[3]
        self.tmp = tempfile.TemporaryDirectory(prefix='teacher_eval_test_', dir=workspace)
        self.root = Path(self.tmp.name).resolve()
        self.assertTrue(self.root.is_relative_to(workspace))
        self.addCleanup(self.tmp.cleanup)
        c = settings()
        c['history'] = 2
        runner = Runner(self.root/'source', c, 'no_checks', 40)
        runner.checkpoint(); runner.sampler.close()
        checkpoint = self.root/'source/checkpoints/final.pt'
        state = load_pt(checkpoint); state['current_method'] = 'AN'; save_pt(checkpoint, state)
        with redirect_stdout(StringIO()):
            main(['--mode', 'fit', '--checkpoint', str(checkpoint), '--output', str(self.root/'prepared'),
                  '--seeds', '43', '--init-seed', '40', '--batch-episodes', '2', '--validation-episodes', '2',
                  '--direction-updates', '4', '--fit-epochs', '2', '--prepare-only', '--evaluation-episodes', '2',
                  '--evaluation-seed', '20261018', '--device', 'cpu', '--allow-duplicate-openmp'])
        self.source = self.root/'prepared/fixed_targets/data_s43_direction_u4.pt'
        self.packet = load_pt(self.source)
        self.original_bytes = self.source.read_bytes()
        checkpoint.rename(checkpoint.with_suffix('.unavailable'))
        self.base = ['--mode', 'teacher-eval', '--fixed-target', str(self.source),
                     '--evaluation-seed', '20261019', '--device', 'cpu', '--allow-duplicate-openmp']

    def test_real_execution_has_no_training_and_matches_existing_evaluation(self):
        calls, inputs, policies = [], {}, {}
        original_episode = teacher_probe.episode
        def spy(policy, config, reset_seed, action_seed, **kwargs):
            name = 'teacher' if isinstance(policy, Target) else 'old'
            policies[name] = policy
            self.assertFalse(policy.training)
            self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
            before = []
            hook = policy.register_forward_pre_hook(lambda module, args: before.append(args[0].detach().clone()))
            try:
                result = original_episode(policy, config, reset_seed, action_seed, **kwargs)
            finally:
                hook.remove()
            calls.append((name, reset_seed, action_seed, result['episode_steps']))
            inputs.setdefault(name, []).append(before)
            return result
        out = self.root/'actual'
        forbidden = AssertionError('Teacher evaluation entered a training path')
        with ExitStack() as stack:
            for method in ['__init__', 'optimize', 'labels']:
                stack.enter_context(patch.object(Runner, method, side_effect=forbidden))
            stack.enter_context(patch.object(Collector, '__init__', side_effect=forbidden))
            stack.enter_context(patch.object(Collector, 'collect', side_effect=forbidden))
            stack.enter_context(patch('torch.optim.Adam', side_effect=forbidden))
            stack.enter_context(patch('manifold_project.experiments.cooperative_navigation.training.actor_fit.fit_actor', side_effect=forbidden))
            stack.enter_context(patch.object(teacher_probe, 'episode', side_effect=spy))
            stack.enter_context(redirect_stdout(StringIO()))
            main(self.base+['--output', str(out), '--evaluation-episodes', '4'])
        meta = json.loads((out/'manifest.json').read_text())
        self.assertEqual(meta['status'], 'completed')
        self.assertEqual(meta['completed_pairs'], 4)
        for field in ['collection_steps', 'training_collection_steps', 'validation_collection_steps',
                      'direction_optimizer_updates', 'actor_optimizer_updates', 'critic_optimizer_updates']:
            self.assertEqual(meta[field], 0, field)
        self.assertEqual(meta['evaluation_steps'], sum(c[3] for c in calls))
        self.assertEqual(meta['model_state_sha256'], meta['model_state_sha256_after'])
        self.assertEqual(meta['teacher_policy_sha256'], meta['teacher_policy_sha256_after'])
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertEqual(len(calls), 8)
        for i in range(0, 8, 2):
            self.assertEqual([calls[i][0], calls[i+1][0]], ['old', 'teacher'])
            self.assertEqual(calls[i][1:3], calls[i+1][1:3])
        self.assertEqual(len({c[1] for c in calls}), 4)
        self.assertEqual(len({c[2] for c in calls}), 4)
        for rollouts in inputs.values():
            for frames in rollouts:
                self.assertGreaterEqual(len(frames), 2)
                width = frames[0].shape[-1]//2
                self.assertTrue(torch.equal(frames[0][:, :width], torch.zeros_like(frames[0][:, :width])))
                self.assertTrue(torch.all(frames[1][:, width-1] == 1))
                self.assertFalse(torch.equal(frames[0], frames[1]))
        teacher = policies['teacher']
        x = inputs['teacher'][0][1]
        with torch.no_grad():
            unnormalized = teacher.old(x) * torch.exp(teacher.eta * teacher.direction(x))
            torch.testing.assert_close(teacher(x), unnormalized/unnormalized.sum(-1, keepdim=True), rtol=1e-6, atol=1e-7)
        episodes = records(out/'evaluation_episodes.csv')
        for name, policy in policies.items():
            _, reference = evaluate(policy, self.packet['config'], 'diagnostic', 43, meta['evaluation_purpose'], 4)
            actual = [r for r in episodes if r['candidate'] == name]
            self.assertEqual([float(r['J']) for r in actual], [r['J'] for r in reference])
        self.assertEqual({r['candidate'] for r in episodes}, {'old', 'teacher'})
        self.assertFalse((out/'fit_candidates').exists())
        self.assertFalse((out/'fit_trace.csv').exists())
        EVIDENCE['teacher_evaluation_execution'] = dict(training_collection_calls=0, optimizer_constructions=0,
            direction_updates=0, actor_updates=0, old_and_teacher_only=True, existing_evaluation_returns_identical=True,
            complete_pairs=4, source_checkpoint_not_required=True, frozen_target_bytes_unchanged=True,
            model_hashes_unchanged=True, histories_reset_and_advance=True, teacher_recomputed_on_new_inputs=True)

    def test_full_count_primary_statistic_and_early_success_cost(self):
        calls = []
        differences = [0., -2., 4., -1.]
        def fake(policy, config, reset_seed, action_seed, on_step):
            pair = len(calls)//2
            is_teacher = isinstance(policy, Target)
            length = 1 if is_teacher else 3
            for _ in range(length): on_step()
            calls.append((reset_seed, action_seed))
            return dict(J=-10.+(differences[pair] if is_teacher else 0.),
                        success=float(is_teacher), episode_steps=length)
        out = self.root/'controlled'
        with patch.object(teacher_probe, 'episode', side_effect=fake), redirect_stdout(StringIO()):
            main(self.base+['--output', str(out), '--evaluation-episodes', '4'])
        meta = json.loads((out/'manifest.json').read_text())
        summary = json.loads((out/'summary.json').read_text())
        self.assertEqual(len(calls), 8)
        self.assertEqual(meta['evaluation_steps'], 16)
        self.assertLess(meta['evaluation_steps'], meta['maximum_evaluation_steps'])
        self.assertEqual(summary['return_gain'], .25)
        self.assertEqual((summary['positive_pairs'], summary['negative_pairs'], summary['zero_pairs']), (1, 2, 1))
        self.assertEqual(summary['teacher_success_count'], 4)
        draws = np.random.default_rng(meta['bootstrap']['seed']).choice(np.array(differences), (50000, 4), replace=True).mean(1)
        expected = np.quantile(draws, [.025, .975])
        self.assertEqual([summary['gain_scene_ci_low'], summary['gain_scene_ci_high']], list(expected))
        self.assertEqual([r['pair_id'] for r in summary['worst_loss_pairs']], [1, 3])
        self.assertEqual(len(records(out/'paired_gains.csv')), 4)
        EVIDENCE['teacher_evaluation_statistics'] = dict(predeclared_pair_count_completed=True,
            primary_metric='mean paired realized discounted return difference',
            percentile_bootstrap_repeats=50000, independent_recomputation_matches=True,
            completed_episode_steps=16, early_success_charged_actual_steps=True)

    def test_invalid_overrides_dry_run_and_snapshot_integrity(self):
        out = self.root/'dry'
        with patch.object(teacher_probe, 'episode', side_effect=AssertionError('dry-run executed')), redirect_stdout(StringIO()):
            plan = run(parser().parse_args(self.base+['--output', str(out), '--dry-run']))
        self.assertEqual(plan['evaluation_pairs'], 100)
        self.assertEqual(plan['maximum_evaluation_steps'], 1600)
        self.assertFalse(out.exists())
        for flags in [['--eta', '.2'], ['--actor-lrs', '.0003'], ['--direction-updates', '4'],
                      ['--fit-epochs', '2'], ['--fit-trace-every-updates', '0'], ['--target-kl', '.0001'],
                      ['--seeds', '43'], ['--batch-episodes', '2'], ['--validation-episodes', '2'],
                      ['--direction-epochs', '2'], ['--sample-sizes', '2'], ['--init-seed', '40'],
                      ['--init-seeds', '40'], ['--prepare-only'], ['--evaluation-episodes', '1'],
                      ['--evaluation-seed', '20261018'], ['--evaluation-seed', '-1']]:
            with self.subTest(flags=flags), self.assertRaises(ValueError), redirect_stdout(StringIO()):
                main(self.base+['--output', str(out), *flags])
        without_seed = self.base[:4]+self.base[6:]
        with self.assertRaises(ValueError), redirect_stdout(StringIO()):
            main(without_seed+['--output', str(out)])
        with self.assertRaises(ValueError), redirect_stdout(StringIO()):
            main(['--mode', 'teacher-eval', '--checkpoint', str(self.source), '--output', str(out), '--evaluation-seed', '5'])
        corrupt = deepcopy(self.packet)
        next(iter(corrupt['actor'].values())).add_(.1)
        broken = self.root/'broken.pt'; save_pt(broken, corrupt)
        with self.assertRaises(ValueError), redirect_stdout(StringIO()):
            main(self.base+['--fixed-target', str(broken), '--output', str(out)])
        self.assertFalse(out.exists())
        out.mkdir(); (out/'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError), redirect_stdout(StringIO()):
            main(self.base+['--output', str(out)])
        self.assertEqual((out/'keep.txt').read_text(), 'keep')
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        EVIDENCE['teacher_evaluation_validation'] = dict(dry_run_no_execution=True, fixed_count_default=100,
            requires_new_explicit_scene_seed=True, training_and_eta_overrides_rejected=True,
            corrupt_snapshot_rejected=True, existing_output_preserved=True)

    def test_interrupted_execution_preserves_partial_cost_without_final_inference(self):
        calls = []
        def interrupted(policy, config, reset_seed, action_seed, on_step):
            calls.append(type(policy).__name__)
            for _ in range(2 if isinstance(policy, Target) else 3): on_step()
            if isinstance(policy, Target): raise KeyboardInterrupt('test interruption')
            return dict(J=-1., success=0., episode_steps=3)
        out = self.root/'interrupted'
        with patch.object(teacher_probe, 'episode', side_effect=interrupted), redirect_stdout(StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                main(self.base+['--output', str(out), '--evaluation-episodes', '4'])
        meta = json.loads((out/'manifest.json').read_text())
        self.assertEqual(meta['status'], 'interrupted')
        self.assertEqual(meta['evaluation_steps'], 5)
        self.assertEqual(meta['completed_pairs'], 0)
        self.assertEqual(meta['completed_policy_episodes'], 1)
        self.assertFalse((out/'diagnostics.csv').exists())
        self.assertFalse((out/'summary.json').exists())
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        EVIDENCE['teacher_evaluation_interruption'] = dict(partial_execution_steps_preserved=5,
            completed_pairs=0, partial_episode_cost_included=True, final_inference_not_emitted=True)


if __name__ == '__main__':
    unittest.main()
