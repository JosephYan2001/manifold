"""Integration contracts against the pinned upstream PPO implementation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from manifold_project.experiments.cooperative_navigation.tests.test_protocol import tiny
from manifold_project.experiments.cooperative_navigation.configs import condition_config, load_config, validate
from manifold_project.experiments.cooperative_navigation.models import Actor, load_actor
from manifold_project.experiments.cooperative_navigation.training.author_ppo import AuthorPPO, arguments
from manifold_project.experiments.cooperative_navigation.vendor.mappo.onpolicy.config import get_config
from manifold_project.experiments.cooperative_navigation.training.collector import Collector, episode
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt


class AuthorPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = dict(tiny(), task_mode='continuing', budget=20, eval_fractions=[0, 1])

    def test_upstream_files_only_have_import_namespace_patch(self):
        root = Path(__file__).resolve().parents[1]/'vendor/mappo'
        source = json.loads((root/'UPSTREAM.json').read_text())
        for name, expected in source['sha256'].items():
            content = (root/name).read_bytes().replace(b'\r\n', b'\n').replace(
                b'from manifold_project.experiments.cooperative_navigation.vendor.mappo.onpolicy.',
                b'from onpolicy.')
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected, name)

    def test_update_equals_direct_author_trainer_and_uses_native_probabilities(self):
        for condition in ('mappo', 'ippo'):
            with self.subTest(condition=condition):
                self.check_author_update(condition)

    def check_author_update(self, condition, config=None):
        c = self.c if config is None else config
        model = AuthorPPO(44, 64, c, condition)
        batch = Collector(c, condition, 8).collect(model.actor, 2, 'train')
        if condition == 'ippo':
            # Even diagnostics must not read centralized inputs in the IPPO path.
            del batch['state'], batch['final_state']
        oracle = deepcopy(model)
        buffer = oracle.make_buffer(batch)
        rng = torch.get_rng_state()
        oracle.trainer.train(buffer)
        torch.set_rng_state(rng)
        with patch.object(model.trainer, 'train', wraps=model.trainer.train) as train:
            info = model.update(batch)
        train.assert_called_once()
        torch.testing.assert_close(model.actor.state_dict(), oracle.actor.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(model.policy.critic.state_dict(), oracle.policy.critic.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(model.normalizer_state(), oracle.normalizer_state(), rtol=0, atol=0)
        self.assertTrue(np.isfinite(info['critic_mse']))
        # No beta/5 floor: force one action to near-zero probability.
        with torch.no_grad():
            model.policy.actor.act.action_out.linear.bias[0] = -30
            p = model.actor(batch['x'])
            native = model.policy.actor.act.get_probs(model.policy.actor.base(batch['x']))
        torch.testing.assert_close(p, native)
        self.assertLess(float(p[..., 0].max()), self.c['beta']/5)

    def test_author_gae_bootstraps_actual_final_state_and_keeps_terminal_reward(self):
        for condition in ('mappo', 'ippo'):
            with self.subTest(condition=condition):
                self.check_gae_boundaries(condition)

    def check_gae_boundaries(self, condition):
        for normalized in (False, True):
            for mode in ('continuing', 'finite_horizon'):
                c = dict(self.c, task_mode=mode, ppo_value_normalization=normalized)
                model = AuthorPPO(44, 64, c, condition)
                e, t, n = 2, c['horizon'], c['n_agents']
                batch = dict(x=torch.zeros(e,t,n,44), state=torch.zeros(e,t,64),
                             final_x=torch.zeros(e,n,44), final_state=torch.zeros(e,64),
                             actions=torch.zeros(e,t,n,dtype=torch.long),
                             reward=torch.arange(1,t+1,dtype=torch.float32).expand(e,t).clone(),
                             terminated=torch.zeros(e,t,dtype=torch.bool),
                             truncated=torch.zeros(e,t,dtype=torch.bool))
                batch['state'][..., 0] = torch.arange(t)*.2
                batch['final_state'][:, 0] = torch.tensor([-3., -7.])
                batch['x'][..., 0] = torch.arange(t)[:,None]*.2 + torch.arange(n)[None,:]*.1
                batch['final_x'][..., 0] = -torch.arange(1,e*n+1).reshape(e,n)
                batch['truncated'][:, -1] = True
                batch['terminated'][1, -1] = True  # terminal takes priority over timeout
                if normalized:
                    model.trainer.value_normalizer.update(torch.tensor([[-20.], [-10.]]))
                with patch.object(model, 'values', side_effect=lambda state: state[..., :1]):
                    b = model.make_buffer(batch)
                v = b.value_preds.copy()
                if normalized:
                    v = model.trainer.value_normalizer.denormalize(v)
                expected = np.zeros_like(b.returns[:-1])
                advantage = np.zeros_like(v[0])
                for step in reversed(range(t)):
                    mask = b.masks[step+1]
                    delta = b.rewards[step]+c['gamma']*mask*v[step+1]-v[step]
                    advantage = delta+c['gamma']*c['gae_lambda']*mask*advantage
                    expected[step] = advantage+v[step]
                np.testing.assert_allclose(b.returns[:-1], expected, atol=2e-6)
                np.testing.assert_allclose(b.rewards[:, :, 0, 0], batch['reward'].T)
                if condition == 'ippo':
                    np.testing.assert_array_equal(b.share_obs[:-1], batch['x'].transpose(0,1))
                    np.testing.assert_array_equal(b.share_obs[-1], batch['final_x'])
                self.assertEqual(float(b.returns[-2,1,0,0]), t)
                if mode == 'continuing':
                    self.assertAlmostEqual(float(b.returns[-2,0,0,0]), t+c['gamma']*v[-1,0,0,0], places=4)
                    self.assertLess(float(b.returns[-2,0,0,0]), t)
                    if condition == 'ippo':
                        # No broadcasting one agent's bootstrap to its peers.
                        self.assertEqual(len(set(b.returns[-2,0,:,0].tolist())), n)
                else:
                    self.assertEqual(float(b.returns[-2,0,0,0]), t)
                batch['truncated'][0, 1] = True
                with self.assertRaises(ValueError):
                    model.make_buffer(batch)

    def test_resume_restores_optimizer_normalizer_and_actor_only_evaluation(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for activation in ('relu', 'tanh'):
            for condition in ('mappo', 'ippo'):
                with self.subTest(condition=condition, activation=activation):
                    self.check_resume_and_evaluation(condition, devices, activation)

    def check_resume_and_evaluation(self, condition, devices, activation):
        with tempfile.TemporaryDirectory() as folder:
            for device in devices:
                c = dict(self.c, device=device, ppo_activation=activation)
                full = Runner(Path(folder)/(device+'_full'), c, condition, 40)
                full.run()
                paused = Runner(Path(folder)/(device+'_resume'), c, condition, 40)
                paused.run(max_rounds=1)
                saved = load_pt(paused.directory/'checkpoints/final.pt')
                self.assertEqual(set(saved['value_normalizer']), {'running_mean','running_mean_sq','debiasing_term'})
                resumed = Runner(paused.directory, c, condition, 40, resume=True)
                resumed.run()
                torch.testing.assert_close(full.actor.state_dict(), resumed.actor.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(full.critic.state_dict(), resumed.critic.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(full.author_ppo.normalizer_state(), resumed.author_ppo.normalizer_state(), rtol=0, atol=0)
                torch.testing.assert_close(full.actor_optimizer.state_dict(), resumed.actor_optimizer.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(full.critic_optimizer.state_dict(), resumed.critic_optimizer.state_dict(), rtol=0, atol=0)
                self.assertEqual(full.curves, resumed.curves)
                for name in ('best', 'final'):
                    state = load_pt(full.directory/f'checkpoints/{name}.pt')
                    actor = load_actor(state)
                    probe = torch.linspace(-1, 1, full.input_dim).repeat(2, 1)
                    with torch.no_grad():
                        expected = deepcopy(full.actor).cpu()
                        expected.load_state_dict(state['actor'])
                        # Tanh/ReLU have identical state_dict shapes: compare behavior.
                        torch.testing.assert_close(actor(probe), expected(probe), rtol=0, atol=0)
                    before = deepcopy(actor.state_dict())
                    self.assertEqual(state['actor_kind'], f'author_{condition}_v1')
                    self.assertEqual(next(actor.parameters()).device.type, 'cpu')
                    for n in (3,4,6,8):
                        result = episode(actor, dict(c, horizon=7), 42, 43, n=n)
                        self.assertTrue(np.isfinite(result['J']))
                    torch.testing.assert_close(before, actor.state_dict(), rtol=0, atol=0)

    def test_legacy_models_remain_readable_and_cannot_resume_as_author(self):
        for condition in ('mappo', 'ippo'):
            with self.subTest(condition=condition):
                self.check_legacy_restore(condition)

    def check_legacy_restore(self, condition):
        c = dict(self.c, mappo_backend='local', ippo_backend='local')
        with tempfile.TemporaryDirectory() as folder:
            legacy = Runner(Path(folder)/'local', c, condition, 40)
            legacy.run()
            state = load_pt(legacy.directory/'checkpoints/final.pt')
            state.pop('actor_kind')
            self.assertIsInstance(load_actor(state), Actor)
            new = Runner(Path(folder)/'author', self.c, condition, 40)
            with self.assertRaisesRegex(ValueError, '旧本地模型'):
                new.restore(state)
        with self.assertRaises(ValueError):
            condition_config(dict(self.c, ppo_label='mc'), condition)
        root = Path(__file__).resolve().parents[1]/'configs'
        aligned = load_config('smoke', root/'label_alignment.json')
        self.assertEqual(condition_config(aligned, condition)[f'{condition}_backend'], 'local')

    def test_author_baselines_share_policy_settings_but_not_critic_input(self):
        torch.manual_seed(7)
        mappo = AuthorPPO(44, 64, self.c, 'mappo')
        torch.manual_seed(7)
        ippo = AuthorPPO(44, 64, self.c, 'ippo')
        torch.testing.assert_close(mappo.actor.state_dict(), ippo.actor.state_dict(), rtol=0, atol=0)
        a, b = vars(mappo.args), vars(ippo.args)
        self.assertEqual({k for k in a if a[k] != b[k]}, {'algorithm_name','use_centralized_V'})
        self.assertEqual(mappo.state_space.shape, (64,))
        self.assertEqual(ippo.state_space.shape, (44,))

    def test_old_override_names_load_to_shared_ppo_keys_and_old_author_actor_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.json'
            path.write_text(json.dumps({'mappo_value_normalization':False}), encoding='utf-8')
            c = load_config('smoke', path)
            self.assertFalse(c['ppo_value_normalization'])
            self.assertNotIn('mappo_value_normalization', c)
            path.write_text(json.dumps({'mappo_value_normalization':False,'ppo_value_normalization':True}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config('smoke', path)
        legacy = dict(self.c)
        legacy.pop('ippo_backend')
        legacy.pop('ppo_activation')
        for suffix in ('value_normalization','value_clipping','huber_loss'):
            legacy['mappo_'+suffix] = legacy.pop('ppo_'+suffix)
        model = AuthorPPO(44,64,legacy)
        actor = load_actor(dict(actor_kind='author_mappo_v1',input_dim=44,config=legacy,actor=model.actor.state_dict()))
        torch.testing.assert_close(actor.state_dict(), model.actor.state_dict(), rtol=0, atol=0)

    def test_author_mpe_config_matches_reference_optimizer_and_full_batch_update(self):
        path = Path(__file__).resolve().parents[1]/'configs/learning_20m/author_mpe.json'
        config = load_config('pilot', path)
        # These switches come from the pinned train_mpe_spread.sh, including its
        # counterintuitive store_false --use_ReLU flag (Tanh in that scenario).
        upstream = get_config().parse_args(['--lr','7e-4','--critic_lr','7e-4',
                                          '--ppo_epoch','10','--num_mini_batch','1','--use_ReLU'])
        for condition in ('mappo', 'ippo'):
            c = condition_config(config, condition)
            actual = arguments(c, condition)
            for key in ('lr','critic_lr','ppo_epoch','num_mini_batch','use_ReLU',
                        'max_grad_norm','value_loss_coef','clip_param','entropy_coef',
                        'gamma','gae_lambda','use_gae','use_valuenorm','use_clipped_value_loss',
                        'use_huber_loss','huber_delta','opti_eps','gain','use_linear_lr_decay'):
                self.assertEqual(getattr(actual, key), getattr(upstream, key), key)
            self.assertFalse(actual.use_recurrent_policy)
            self.assertFalse(actual.use_naive_recurrent_policy)
            self.assertEqual(c['train_episodes']*c['horizon'], 1600)
            small = dict(c, horizon=5, history=2, hidden=8, train_episodes=2,
                         minibatch_episodes=2, device='cpu', budget=20, eval_fractions=[0,1])
            with self.subTest(condition=condition):
                self.check_author_update(condition, small)
        with self.assertRaisesRegex(ValueError, 'ppo_activation'):
            validate(dict(config, ppo_activation='sigmoid'))


if __name__ == '__main__':
    unittest.main()
