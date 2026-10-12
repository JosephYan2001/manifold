"""CPU integration checks for the public resume/extend entry point."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from manifold_project.experiments.common.runtime import configure_runtime
configure_runtime(True)
import torch
from manifold_project.experiments.cli import main
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt


def same(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
    return left == right


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        # Every automatically cleaned directory is resolved inside the workspace.
        workspace = Path(__file__).resolve().parents[3]
        self.tmp = tempfile.TemporaryDirectory(prefix='recovery_test_', dir=workspace)
        self.root = Path(self.tmp.name).resolve()
        assert self.root.is_relative_to(workspace) and self.root != workspace
        self.config = self.root/'settings.json'
        self.config.write_text(json.dumps(dict(direction_check=False, return_check=False,
            save_fit_diagnostics=False, source_eval_episodes=1, evaluation_episodes=1,
            hidden=8, heads=1, relation_layers=0, direction_steps=1,
            actor_fit_steps=1, critic_steps=1)), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, args):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return main(args + ['--allow-duplicate-openmp'])

    def fresh(self, name):
        return ['--experiments','N-C','--profile','smoke','--methods','AN','--device','cpu',
                '--budget','256','--config',str(self.config),'--output',str(self.root/name)]

    def checkpoint(self, name):
        return self.root/name/'N-C_source_learning/full/AN_s40/checkpoints/final.pt'

    def test_interruption_matches_uninterrupted_and_extension_recovers(self):
        self.assertEqual(self.invoke(self.fresh('reference')), 0)
        original_update = Runner.update
        calls = 0
        def interrupt(runner):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('simulated interruption after a committed round')
            return original_update(runner)
        with patch.object(Runner, 'update', interrupt):
            self.assertEqual(self.invoke(self.fresh('interrupted')), 2)
        before = load_pt(self.checkpoint('interrupted'))
        self.assertEqual(before['round'], 1)
        self.assertFalse(before['complete'])
        self.assertEqual(self.invoke(['--resume',str(self.root/'interrupted')]), 0)
        reference, recovered = [load_pt(self.checkpoint(n)) for n in ('reference','interrupted')]
        for key in ('actor','critic','actor_optimizer','critic_optimizer','costs','counts','optim_steps','torch_rng'):
            self.assertTrue(same(reference[key],recovered[key]), key)
        source_bytes = self.checkpoint('interrupted').read_bytes()
        source_events = (self.checkpoint('interrupted').parent.parent/'events.jsonl').read_bytes()
        extend = ['--resume',str(self.root/'interrupted'),'--budget','512','--output',str(self.root/'extended')]
        with patch.object(Runner, 'update', side_effect=RuntimeError('extension interrupted')):
            self.assertEqual(self.invoke(extend),2)
        partial = load_pt(self.checkpoint('extended'))
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['round'], recovered['round'])
        self.assertEqual(self.invoke(['--resume',str(self.root/'extended')]),0)
        extended = load_pt(self.checkpoint('extended'))
        self.assertTrue(extended['complete'])
        self.assertEqual(extended['config']['budget'],512)
        self.assertGreater(extended['round'],recovered['round'])
        self.assertGreater(sum(extended['costs'].values()),sum(recovered['costs'].values()))
        self.assertLessEqual(sum(extended['costs'].values()),512)
        self.assertEqual(extended['curves'][:len(recovered['curves'])], recovered['curves'])
        self.assertEqual(source_bytes,self.checkpoint('interrupted').read_bytes())
        self.assertTrue((self.checkpoint('extended').parent.parent/'events.jsonl').read_bytes().startswith(source_events))
        self.assertTrue((self.root/'extended/summary.csv').is_file())
        self.assertTrue((self.checkpoint('extended').parent.parent/'final_evaluation/evaluation_episodes.csv').is_file())

    def test_invalid_changes_and_dry_run_leave_source_untouched(self):
        self.assertEqual(self.invoke(self.fresh('source')),0)
        source = self.root/'source'
        before = {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
        for extra in (['--budget','128'], ['--budget','512'], ['--device','cuda'],
                      ['--methods','SA'], ['--config',str(self.config)]):
            self.assertEqual(self.invoke(['--resume',str(source)]+extra),2)
        target = self.root/'planned'
        self.assertEqual(self.invoke(['--resume',str(source),'--budget','512','--output',str(target),'--dry-run']),0)
        self.assertFalse(target.exists())
        after = {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_unstarted_extension_run_keeps_its_source(self):
        # Multiple seeds: recovery must not cold-start an extension that had not begun.
        fresh = self.fresh('multi')+['--seeds','40','41']
        self.assertEqual(self.invoke(fresh),0)
        from manifold_project.experiments.cooperative_navigation import current_protocol
        original_train = current_protocol.train
        def stop_before_second(environment, method, config, output, **kwargs):
            if config['seed'] == 41:
                raise RuntimeError('second extension not started')
            return original_train(environment, method, config, output, **kwargs)
        target = self.root/'multi_extended'
        with patch.object(current_protocol, 'train', stop_before_second):
            self.assertEqual(self.invoke(['--resume',str(self.root/'multi'),'--budget','512',
                                          '--output',str(target)]),2)
        second = target/'N-C_source_learning/full/AN_s41/checkpoints/final.pt'
        self.assertFalse(second.exists())
        self.assertEqual(self.invoke(['--resume',str(target)]),0)
        source = load_pt(self.root/'multi/N-C_source_learning/full/AN_s41/checkpoints/final.pt')
        result = load_pt(second)
        self.assertTrue(result['complete'])
        self.assertGreater(result['round'],source['round'])
        self.assertEqual(result['curves'][:len(source['curves'])],source['curves'])


if __name__ == '__main__':
    unittest.main()
