"""Branch state, fresh budgets, paired starts and interrupted recovery."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from manifold_project.experiments.continue_navigation import (
    ContinuationRunner, Runner, digest, load_pt, main, make_plan)
from manifold_project.experiments.cooperative_navigation.configs import load_config
import torch


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[4]
        temporary_root = workspace/'tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='nav-continuation-', dir=temporary_root)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(workspace)
        c = load_config('smoke')
        c.update(horizon=5, history=2, hidden=8, train_episodes=2, minibatch_episodes=1,
                 direction_epochs=4, actor_epochs=1, critic_epochs=1, device='cpu',
                 budget=20, eval_fractions=[0, 1], eval_episodes=2, final_episodes=2)
        with redirect_stdout(io.StringIO()):
            Runner(self.root/'parent', c, 'no_checks', 40).run()
        self.checkpoint = self.root/'parent/checkpoints/final.pt'
        self.parent = load_pt(self.checkpoint)
        self.before = digest(self.checkpoint)
        self.plan = make_plan(self.checkpoint, self.parent, [4, 1], 30, 'cpu', 2, 90300)

    def tearDown(self):
        self.temp.cleanup()  # Target was resolved and checked under the workspace in setUp.

    def equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.equal(left, right)
        else:
            self.assertEqual(a, b)

    def branch(self, name, index=0, resume=False):
        job = self.plan['jobs'][index]
        return ContinuationRunner(self.root/name, job['config'], self.plan['source'],
                                  job['variant'], self.plan['evaluation_seed'],
                                  parent=self.parent, resume=resume)

    def test_inherited_state_new_costs_and_exact_resume(self):
        first = self.branch('paused')
        self.equal(first.actor.state_dict(), self.parent['actor'])
        self.equal(first.critic.state_dict(), self.parent['critic'])
        self.equal(first.critic_optimizer.state_dict(), self.parent['critic_optimizer'])
        self.equal(first.actor_optimizer.state_dict(), self.parent['actor_optimizer'])
        self.assertEqual(first.sampler.counts, self.parent['counts'])
        self.assertEqual(first.round, self.parent['round'])
        self.assertEqual(first.sampler.used, 0)
        self.assertEqual(first.optim_steps, {})
        self.assertEqual(first.curves, [])
        with redirect_stdout(io.StringIO()):
            first.run(max_rounds=self.parent['round']+1)
            resumed = self.branch('paused', resume=True)
            actual = resumed.run()
            continuous = self.branch('continuous')
            expected = continuous.run()
        self.equal(resumed.actor.state_dict(), continuous.actor.state_dict())
        self.equal(resumed.critic.state_dict(), continuous.critic.state_dict())
        self.equal(resumed.critic_optimizer.state_dict(), continuous.critic_optimizer.state_dict())
        self.equal(resumed.curves, continuous.curves)
        for key in ('rounds', 'source_steps', 'total_source_steps', 'AUC', 'J', 'costs', 'optimizer_steps'):
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(actual['rounds'], 3)
        self.assertEqual(actual['total_rounds'], 5)
        self.assertEqual(actual['source_steps'], 30)
        self.assertEqual(actual['total_source_steps'], 50)
        self.assertEqual(actual['costs'], {'train': 30})
        self.assertEqual(actual['evaluation_steps'], 11*2*5)
        events = [json.loads(line) for line in (self.root/'paused/events.jsonl').read_text().splitlines()]
        sample = next(r['record'] for r in events if r['stream'] == 'sampling')
        self.assertEqual(sample['episode'], self.parent['counts']['train'])
        self.assertEqual(sample['round'], self.parent['round']+1)
        self.assertEqual(digest(self.checkpoint), self.before)
        self.equal(self.parent['critic_optimizer'], load_pt(self.checkpoint)['critic_optimizer'])

    def test_only_epochs_differ_and_pair_has_identical_initial_evaluation(self):
        left, right = [j['config'] for j in self.plan['jobs']]
        self.assertEqual([k for k in left if left[k] != right[k]], ['direction_epochs'])
        a, b = self.branch('a'), self.branch('b', 1)
        with redirect_stdout(io.StringIO()):
            sa, sb = a.run(), b.run()
        for key in ('J', 'coverage', 'distance', 'J_se'):
            self.assertEqual(a.curves[0][key], b.curves[0][key])
        self.assertEqual(sa['source_steps'], sb['source_steps'])
        self.assertEqual(sa['optimizer_steps_by_module']['direction'], 4*sb['optimizer_steps_by_module']['direction'])
        for module in ('critic', 'actor_fit'):
            self.assertEqual(sa['optimizer_steps_by_module'][module], sb['optimizer_steps_by_module'][module])
        # Initial and final use the same evaluation stream when the Actor is unchanged.
        probe = self.branch('evaluation')
        with redirect_stdout(io.StringIO()):
            probe.evaluate_node(probe.actor, 0, 0)
            probe.evaluate_node(probe.actor, 30, 0)
        self.assertEqual(probe.curves[0]['J'], probe.curves[1]['J'])

    def test_cli_plot_resume_and_protection(self):
        args = ['--checkpoint', str(self.checkpoint), '--direction-epochs', '4', '1',
                '--additional-budget', '30', '--eval-episodes', '2', '--device', 'cpu',
                '--output', str(self.root/'suite')]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args+['--dry-run']), 0)
        self.assertFalse((self.root/'suite').exists())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args+['--plot', '--plot-every', '1']), 0)
            self.assertEqual(main(['--resume-suite', str(self.root/'suite')]), 0)
        suite = self.root/'suite'
        self.assertTrue((suite/'overview.png').stat().st_size > 10000)
        self.assertTrue((suite/'training_monitor.png').stat().st_size > 10000)
        self.assertEqual(len(list(suite.glob('*/seed_40/checkpoints/final.pt'))), 2)
        with self.assertRaises(FileExistsError):
            main(args)
        with self.assertRaises(ValueError):
            main(args[:-1]+[str(self.root/'parent'/'new_output')])
        manifest_path = suite/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['metadata']['continuation_sha256'] = 'wrong'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ValueError):
            main(['--resume-suite', str(suite)])
        self.assertEqual(digest(self.checkpoint), self.before)


if __name__ == '__main__':
    unittest.main()
