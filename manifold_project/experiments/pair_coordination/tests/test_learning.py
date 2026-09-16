import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig, sample_episodes
from manifold_project.experiments.pair_coordination.models.actor import TableActor
from manifold_project.experiments.pair_coordination.models.direction import TableDirection
from manifold_project.experiments.pair_coordination.models.critic import TableCritic
from manifold_project.experiments.pair_coordination.training.labels import make_labels
from manifold_project.experiments.pair_coordination.training.direction_loss import loss_and_gradient, fit_direction, episode_scores
from manifold_project.experiments.pair_coordination.training.actor_fit import kl_and_gradient, fit_actor
from manifold_project.experiments.pair_coordination.training.acceptance import check_mean, alpha_per_check, direction_check
from manifold_project.experiments.pair_coordination.training.config import TrainConfig
from manifold_project.experiments.pair_coordination.training.runner import run_training
from manifold_project.experiments.pair_coordination.evaluation.exact_pair import exact_evaluate, exponential_target


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.config = PairConfig()
        self.actor = TableActor.initial(2)
        self.batch = sample_episodes(self.config, self.actor.table(), 256, seed=12)
        self.labels = make_labels(self.batch)

    def test_actor_copy_bounds_and_serialization(self):
        copy = self.actor.copy()
        copy.logits[:] = [1000, -1000]
        self.assertTrue(np.min(copy.table()) >= .009999999)
        np.testing.assert_allclose(self.actor.table(), [[.65, .35], [.35, .65]])
        np.testing.assert_array_equal(TableActor.from_state(copy.state()).table(), copy.table())

    def test_direction_gradient(self):
        model = TableDirection(2)
        model.parameters[:] = [.2, -.3]
        for method in ("analytic", "sampled"):
            _, table_grad = loss_and_gradient(model.table(), self.batch, self.labels, method)
            grad = model.parameter_gradient(table_grad)
            for i in range(2):
                original = model.parameters[i]
                model.parameters[i] = original + 1e-6
                plus = loss_and_gradient(model.table(), self.batch, self.labels, method)[0]
                model.parameters[i] = original - 1e-6
                minus = loss_and_gradient(model.table(), self.batch, self.labels, method)[0]
                model.parameters[i] = original
                self.assertAlmostEqual(grad[i], (plus-minus)/2e-6, places=7)

    def test_actor_gradient(self):
        target = np.array([[.8, .2], [.2, .8]])
        weights = np.array([.6, .4])
        _, grad = kl_and_gradient(self.actor, target, weights)
        for i in range(2):
            plus, minus = self.actor.copy(), self.actor.copy()
            plus.logits[i] += 1e-6
            minus.logits[i] -= 1e-6
            finite = (kl_and_gradient(plus, target, weights)[0]-kl_and_gradient(minus, target, weights)[0])/2e-6
            self.assertAlmostEqual(grad[i], finite, places=7)

    def test_finite_data_objectives_differ(self):
        batch = {"local_types": np.array([[0]]), "actions": np.array([[0]]),
                 "old_probs": np.array([[[.8, .2]]]), "weights": np.ones((1, 1))}
        q = np.array([[1., -1.]])
        self.assertAlmostEqual(loss_and_gradient(q, batch, np.ones(1), "analytic")[0], -.16)
        self.assertAlmostEqual(loss_and_gradient(q, batch, np.ones(1), "sampled")[0], -.64)

    def test_learning_reduces_error_without_changing_data(self):
        copies = {k: v.copy() for k, v in self.batch.items()}
        model = TableDirection(2)
        initial = exact_evaluate(self.config, self.actor.table(), model.table())["direction_error"]
        fit_direction(model, self.batch, self.labels, "analytic", 100, .05, 10)
        final = exact_evaluate(self.config, self.actor.table(), model.table())["direction_error"]
        self.assertLess(final, initial/5)
        for key in copies:
            np.testing.assert_array_equal(copies[key], self.batch[key])
        np.testing.assert_allclose(model.table().sum(axis=1), 0)

    def test_fit_reduces_kl_without_changing_old_actor(self):
        old = self.actor.table()
        target = exponential_target(old, np.array([[.4, -.4], [-.4, .4]]), .2)
        candidate, metrics = fit_actor(self.actor, target, self.batch, 100, .05)
        self.assertLess(metrics["fit_kl"], metrics["fit_initial_kl"]*.01)
        np.testing.assert_array_equal(self.actor.table(), old)
        self.assertFalse(np.array_equal(candidate.table(), old))

    def test_critic_and_labels_bounded(self):
        critic = TableCritic(4, 2, self.config.reward_bound)
        training = sample_episodes(self.config, self.actor.table(), 128, seed=13)
        critic.fit(training["local_types"], training["team_rewards"])
        labels = make_labels(self.batch, critic)
        self.assertLessEqual(np.max(np.abs(labels)), 2*self.config.reward_bound)
        self.assertFalse(labels.flags.writeable)

    def test_episode_score_bounds(self):
        scores = episode_scores(np.array([[-1., 1.], [1., -1.]]), self.batch, self.labels)
        self.assertEqual(scores.shape, (256,))
        direction_check(scores, self.config.reward_bound, 1., .01, "hoeffding")

    def test_conservative_and_empirical_checks(self):
        self.assertFalse(check_mean([.1, .1], -1, 1, .05)["accepted"])
        self.assertTrue(check_mean([.1, .1], -1, 1, .05, "empirical")["accepted"])
        self.assertAlmostEqual(alpha_per_check(.05, 5, 2)*15, .05)
        with self.assertRaises(ValueError):
            check_mean([2], -1, 1, .05)

    def test_config_validation(self):
        for values in ({"budget": -1}, {"eta": float("nan")}, {"method": "oracle"}):
            with self.assertRaises(ValueError):
                TrainConfig(**values)

    def test_runner_reproducible_budgeted_and_snapshot_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = TrainConfig(save_direction_snapshots=True, actor_model="table", direction_model="table", mode="policy", acceptance="empirical", baseline="table",
                                   episodes=128, critic_episodes=64, check_episodes=128,
                                   rounds=2, direction_steps=30, fit_steps=30, budget=2000)
            first = Path(tmp)/"first"
            result = run_training(self.config, settings, first)
            # Changing all exact diagnostic outputs cannot change the learned actor.
            with patch("manifold_project.experiments.pair_coordination.training.runner.diagnostics", return_value={}):
                second = run_training(self.config, settings, Path(tmp)/"second")
            self.assertEqual(result["actor"], second["actor"])
            records = [json.loads(s) for s in (first/"batches.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["episodes"] for r in records), result["source_episodes"])
            self.assertLessEqual(result["source_episodes"], settings.budget)
            self.assertEqual(len({r["seed"] for r in records}), len(records))
            snapshot = json.loads((first/"round_001.json").read_text())
            self.assertEqual(snapshot["reference_actor"], self.actor.state())
            self.assertEqual(snapshot["direction"]["kind"], "bounded_type_direction")
            with self.assertRaises(FileExistsError):
                run_training(self.config, settings, first)

    def test_direction_mode_keeps_actor_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_training(self.config, TrainConfig(actor_model="table", direction_model="table", episodes=32, direction_steps=3), Path(tmp)/"run")
            self.assertEqual(result["actor"], self.actor.state())
            self.assertEqual(result["source_episodes"], 32)


if __name__ == "__main__":
    unittest.main()
