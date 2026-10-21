"""Target-only ablation preserves scale/orders/start and common evaluation."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from manifold_project.experiments.common.runtime import configure_runtime
configure_runtime(True)
import torch
from manifold_project.experiments.common.config import load_config
from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
from manifold_project.experiments.cooperative_navigation.diagnose import main
from manifold_project.experiments.cooperative_navigation.training.critic_probe import file_hash, fit_variant
from manifold_project.experiments.cooperative_navigation.training.fixed_target import fingerprint
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt, save_pt


class CriticTargetProbeTests(unittest.TestCase):
    def test_one_fit_common_targets_zero_collection_and_immutable_reference(self):
        workspace = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix='critic_target_test_', dir=workspace) as tmp:
            root = Path(tmp)
            public = load_config('navigation', 'smoke')
            public.update(hidden=8, heads=1, relation_layers=0, history=1, horizon=8,
                          evaluation_horizon=8, device='cpu', batch_episodes=2,
                          direction_steps=1, actor_fit_steps=1, critic_steps=1,
                          minibatch_size=64, seed=40, budget=256, threads=1)
            config = runner_config(public)
            runner = Runner(root/'source', config, 'no_checks', 40)
            runner.checkpoint(); runner.sampler.close()
            source = root/'source/checkpoints/final.pt'
            state = load_pt(source);state['current_method'] = 'AN';save_pt(source, state)
            with redirect_stdout(StringIO()):
                main(['--mode','critic','--checkpoint',str(source),'--output',str(root/'scale'),
                      '--seeds','45','--batch-episodes','2','--validation-episodes','2',
                      '--critic-epochs','2','--device','cpu','--allow-duplicate-openmp'])
            packet_path = root/'scale/critic_data.pt'
            reference_path = root/'scale/candidates/fresh_fixed_normalized.pt'
            packet = load_pt(packet_path)
            hashes = {p: file_hash(p) for p in (source, packet_path, reference_path)}
            base = ['--mode','critic-target','--critic-data',str(packet_path),'--device','cpu','--allow-duplicate-openmp']
            with redirect_stdout(StringIO()):
                main(base+['--output',str(root/'dry'),'--dry-run'])
            self.assertFalse((root/'dry').exists())
            with patch('manifold_project.experiments.cooperative_navigation.training.critic_target_probe.fit_variant', wraps=fit_variant) as fitting, \
                 patch('manifold_project.experiments.cooperative_navigation.training.critic_probe.Collector', side_effect=AssertionError), \
                 patch('manifold_project.experiments.cooperative_navigation.training.critic_probe.Actor', side_effect=AssertionError), \
                 redirect_stdout(StringIO()):
                main(base+['--output',str(root/'target')])
            self.assertEqual(fitting.call_count, 1)
            start, train, validation, actual_config, orders, norm, variant = fitting.call_args.args[:7]
            self.assertEqual(fingerprint(start.state_dict()), fingerprint(packet['initial']['fresh']))
            self.assertEqual(orders, packet['orders'])
            self.assertEqual(norm, packet['normalization'])
            self.assertEqual(variant, 'fixed_normalized')
            self.assertTrue(torch.equal(train['value_target'], packet['batch']['frozen_old_values']+packet['batch']['frozen_gae']))
            self.assertTrue(torch.equal(validation['value_target'], packet['validation']['value_target']))
            meta = json.loads((root/'target/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['status'], 'completed')
            self.assertEqual(meta['critic_optimizer_updates'], 2)
            for key in ('collection_steps','evaluation_steps','actor_optimizer_updates','direction_optimizer_updates'):
                self.assertEqual(meta[key], 0)
            summary = json.loads((root/'target/summary.json').read_text(encoding='utf-8'))
            a, b = summary['final_candidates']
            self.assertTrue(a['reused'])
            self.assertEqual(a['optimizer_updates'], 0)
            self.assertEqual(a['original_optimizer_updates'], 2)
            self.assertAlmostEqual(a['long_validation_constant_train_mean_mse'], b['long_validation_constant_train_mean_mse'])
            self.assertEqual(summary['descriptive_screen_passed'],
                             b['long_validation_mse'] <= .9*b['long_validation_constant_train_mean_mse'] and b['long_validation_explained_variance'] > .05)
            for p,h in hashes.items(): self.assertEqual(file_hash(p), h)
            with self.assertRaises(FileExistsError): main(base+['--output',str(root/'target')])
            with self.assertRaises(ValueError): main(base+['--output',str(root/'bad'),'--critic-epochs','3'])
            reference = load_pt(reference_path);reference['frozen_data_sha256'] = 'wrong';save_pt(reference_path,reference)
            with self.assertRaises(ValueError): main(base+['--output',str(root/'bad_reference')])
            self.assertFalse((root/'bad_reference').exists())


if __name__=='__main__': unittest.main()
