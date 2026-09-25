"""Exercise the original Runner through the new protocol adapter."""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
import io
import unittest
import torch

from manifold_project.experiments.common.config import load_config
from manifold_project.experiments.cooperative_navigation.current_protocol import train
from manifold_project.experiments.applications.evaluation import load_actor, evaluate_checkpoint


class RestoredBackendTests(unittest.TestCase):
    def config(self, environment):
        c = load_config(environment, 'smoke')
        c.update(horizon=1 if environment == 'pair_entities' else 4, budget=80,
                 batch_episodes=2, source_eval_episodes=1, evaluation_episodes=1,
                 eval_every_steps=1000, target_sizes=[2,6], save_fit_diagnostics=False,
                 direction_steps=1, actor_fit_steps=1, critic_steps=1, seed=40)
        return c

    def test_original_runner_updates_all_methods_and_frozen_variable_size(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            c = self.config('navigation')
            for method in ('AN','SA','DA','MAPPO-E','IPPO-E'):
                result = train('navigation', method, c, Path(tmp)/method)
                self.assertEqual(result['records'][0]['training_backend'], 'restored_cooperative_navigation.Runner')
                self.assertLessEqual(result['source_steps_total'], c['budget'])
                saved = torch.load(result['checkpoint'], weights_only=False)
                self.assertIn('actor_optimizer', saved)
                self.assertIn('torch_rng', saved)
                actor, normalized = load_actor(result['checkpoint'])
                self.assertEqual(normalized['method'], method)
                evaluated = evaluate_checkpoint(result['checkpoint'], c, Path(tmp)/(method+'_eval'))
                self.assertEqual({r['target_n'] for r in evaluated['records']}, {2,4,6})
                self.assertTrue(evaluated['frozen_actor_verified'])

    def test_additional_tasks_use_same_runner(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            for environment, method in [('warehouse','AN'), ('warehouse','MAPPO-E'), ('pair_entities','AN')]:
                c = self.config(environment)
                result = train(environment, method, c, Path(tmp)/(environment+method))
                evaluated = evaluate_checkpoint(result['checkpoint'], c, Path(tmp)/(environment+method+'_eval'))
                self.assertTrue(evaluated['frozen_actor_verified'])
                self.assertNotIn('success_rate', evaluated['records'][0])

    def test_diagnostics_do_not_change_source_training(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            c = self.config('navigation')
            plain = train('navigation', 'AN', c, Path(tmp)/'plain')
            diagnostic = dict(c, save_fit_diagnostics=True,
                              fit_snapshot_fractions=[.5,1.], fit_steps=[1,2])
            probed = train('navigation', 'AN', diagnostic, Path(tmp)/'probe')
            self.assertEqual(plain['source_steps_total'], probed['source_steps_total'])
            self.assertGreater(probed['diagnostic_steps'], 0)
            left = torch.load(plain['checkpoint'], weights_only=False)['actor']
            right = torch.load(probed['checkpoint'], weights_only=False)['actor']
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]), key)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_original_runner_entity_backward(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            c = dict(self.config('navigation'), device='cuda')
            for method in ('AN','MAPPO-E'):
                result = train('navigation', method, c, Path(tmp)/method)
                self.assertLessEqual(result['source_steps_total'], c['budget'])
                self.assertGreater(result['records'][0]['rounds'], 0)

    def test_frozen_native_replay(self):
        from PIL import Image
        from manifold_project.experiments.visualize import render_checkpoint
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            c = self.config('navigation')
            result = train('navigation', 'AN', c, Path(tmp)/'train')
            output = Path(tmp)/'replay.gif'
            rendered = render_checkpoint(result['checkpoint'], output, n_agents=6)
            self.assertTrue(rendered['frozen_actor_verified'])
            with Image.open(output) as gif:
                self.assertEqual(gif.n_frames, rendered['frames'])
                self.assertGreater(gif.width, 100)

    def test_current_protocol_resume_preserves_parameters_and_cost(self):
        from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
        from manifold_project.experiments.cooperative_navigation.training.runner import Runner
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            c = runner_config(self.config('navigation'))
            for condition in ('ours', 'mappo', 'ippo'):
                full = Runner(Path(tmp)/(condition+'_full'), c, condition, 40)
                full.run()
                paused = Runner(Path(tmp)/(condition+'_resume'), c, condition, 40)
                paused.run(max_rounds=1)
                resumed = Runner(paused.directory, c, condition, 40, resume=True)
                resumed.run()
                torch.testing.assert_close(full.actor.state_dict(), resumed.actor.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(full.critic.state_dict(), resumed.critic.state_dict(), rtol=0, atol=0)
                self.assertEqual(full.sampler.costs, resumed.sampler.costs)
                self.assertEqual(full.optim_steps, resumed.optim_steps)
                if full.author_ppo:
                    torch.testing.assert_close(full.author_ppo.normalizer_state(),
                                               resumed.author_ppo.normalizer_state(), rtol=0, atol=0)

    def test_author_source_integrity(self):
        import hashlib
        import json
        root = Path(__file__).resolve().parents[1]/'cooperative_navigation/vendor/mappo'
        source = json.loads((root/'UPSTREAM.json').read_text())
        for name, expected in source['sha256'].items():
            content = (root/name).read_bytes().replace(b'\r\n', b'\n').replace(
                b'from manifold_project.experiments.cooperative_navigation.vendor.mappo.onpolicy.',
                b'from onpolicy.')
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected, name)

    def test_navigation_trial_variants_account_for_checks_and_freeze(self):
        root = Path(__file__).resolve().parents[1]/'configs/navigation_trial'
        cases = [('an_core','AN',False,False), ('an_checked','AN',True,True),
                 ('da_checked','DA',False,True), ('mappo','MAPPO-E',False,False)]
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            for name, method, direction, returns in cases:
                c = load_config('navigation', 'pilot', root/f'{name}.json')
                self.assertGreater(c['critic_lr'], c['actor_lr'])
                # Shorten only for engineering verification, not research evidence.
                c.update(seed=40, device='cpu', horizon=4, budget=80,
                         hidden=16, heads=2, relation_layers=1, batch_episodes=2,
                         minibatch_size=16, direction_steps=1, actor_fit_steps=1,
                         critic_steps=1, ppo_epochs=1, check_episodes=1,
                         source_eval_episodes=1, evaluation_episodes=1,
                         eval_every_steps=1000, target_sizes=[2,6,8])
                result = train('navigation', method, c, Path(tmp)/name)
                saved = torch.load(result['checkpoint'], weights_only=False)
                costs = saved['costs']
                if not direction:
                    self.assertEqual(costs.get('direction_check', 0), 0)
                if not returns:
                    self.assertEqual(costs.get('return_old', 0)+costs.get('return_candidate', 0), 0)
                self.assertLessEqual(sum(costs.values()), c['budget'])
                evaluation = evaluate_checkpoint(result['checkpoint'], c, Path(tmp)/(name+'_eval'))
                self.assertEqual({r['target_n'] for r in evaluation['records']}, {2,4,6,8})
                self.assertTrue(evaluation['frozen_actor_verified'])


if __name__ == '__main__':
    unittest.main()
