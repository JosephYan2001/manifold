"""核对 PyTorch 迁移的数值、设备与实际更新。"""
import unittest
import numpy as np
import torch
from manifold_project.experiments.pair_coordination.models.mlp import MLPActor, MLPDirection, load_actor
from manifold_project.experiments.pair_coordination.models.torch_models import TorchModel, resolve_device
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig, sample_episodes
from manifold_project.experiments.pair_coordination.training.direction_loss import loss_and_gradient, fit_direction
from manifold_project.experiments.pair_coordination.training.actor_fit import kl_and_gradient, fit_actor
from manifold_project.experiments.pair_coordination.training.torch_fit import direction_loss, actor_loss


class TorchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.actor = MLPActor.initial(2, width=4, depth=2)
        self.batch = sample_episodes(PairConfig(), self.actor.table(), 13, seed=7)
        self.labels = self.batch["team_rewards"]

    def test_old_checkpoint_output_and_export(self):
        model = TorchModel(self.actor, "cpu")
        np.testing.assert_allclose(model.table(), self.actor.table(), atol=1e-14)
        np.testing.assert_array_equal(load_actor(model.state()).parameters, self.actor.parameters)

    def check_gradients(self, device):
        direction = MLPDirection(2, width=4, depth=2)
        direction.parameters[:] += .1
        for method in ("analytic", "sampled"):
            model = TorchModel(direction, device)
            expected_loss, grad = loss_and_gradient(direction.table(), self.batch, self.labels, method)
            actual = direction_loss(model, self.batch, self.labels, method)
            actual.backward()
            self.assertAlmostEqual(actual.item(), expected_loss, places=12)
            np.testing.assert_allclose(model.flat.grad.cpu().numpy(), direction.parameter_gradient(grad), atol=1e-12)
        model = TorchModel(self.actor, device)
        target = np.array([[.8, .2], [.3, .7]])
        weights = np.bincount(self.batch["local_types"].ravel(), weights=self.batch["weights"].ravel(), minlength=2)
        expected_loss, grad = kl_and_gradient(self.actor, target, weights)
        actual = actor_loss(model, target, self.batch)
        actual.backward()
        self.assertAlmostEqual(actual.item(), expected_loss, places=12)
        np.testing.assert_allclose(model.flat.grad.cpu().numpy(), self.actor.parameter_gradient(grad), atol=1e-12)

    def test_cpu_gradients(self):
        self.check_gradients("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 不可用")
    def test_cuda_gradients_and_updates(self):
        self.check_gradients("cuda")
        self.check_updates("cuda")

    def check_updates(self, device):
        direction = TorchModel(MLPDirection(2, width=4, depth=2), device)
        before = direction.flat.detach().cpu().numpy().copy()
        logs = []
        fit_direction(direction, self.batch, self.labels, "analytic", 2, .01, 1,
                      batch_size=5, epochs=2, update_callback=logs.append)
        self.assertEqual(len(logs), 6)
        self.assertFalse(np.array_equal(before[:8], direction.flat.detach().cpu().numpy()[:8]))
        old = TorchModel(self.actor, device)
        candidate, metrics = fit_actor(old, np.array([[.8, .2], [.3, .7]]), self.batch,
                                      10, .01, batch_size=5, epochs=2)
        self.assertEqual(metrics["fit_steps"], 6)
        self.assertLess(metrics["fit_kl"], metrics["fit_initial_kl"])
        np.testing.assert_array_equal(old.state()["network"]["parameters"], self.actor.parameters)
        self.assertEqual(candidate.flat.device.type, device)

    def test_cpu_updates(self):
        self.check_updates("cpu")

    def test_device_validation(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        with self.assertRaises(ValueError):
            resolve_device("meta")
