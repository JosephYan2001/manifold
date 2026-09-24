"""Numerical checks for the new finite protocol, not legacy result fixtures."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from manifold_project.experiments.finite import run_experiment
from manifold_project.experiments.finite.diagnostics import actor_realization, empirical_update, estimation, information, settings, temporal_variance, transfer
from manifold_project.experiments.finite.environments import PairGame, TwoStepGame, labels_for
from manifold_project.experiments.finite.numerics import (direction_gradient, direction_terms, episode_scores,
                                                        fit_actor, kl, tilted)
from manifold_project.experiments.finite.training import load_policy


class FiniteProtocolTests(unittest.TestCase):
    def test_label_freezing_migrates_legacy_immutability(self):
        env = TwoStepGame()
        batch = env.sample(env.initial_policy, 2, np.random.default_rng(3))
        labels = labels_for(batch, "mc_raw", env, env.initial_policy)
        with self.assertRaises(ValueError):
            labels[0, 0] = 123

    def test_pair_closed_forms_agree_with_joint_enumeration(self):
        for env in (PairGame(), PairGame(2, variant="fixed-edge"), PairGame(3, 0.4, "composition")):
            policy = env.initial_policy
            batch, mass = env.enumerate(policy)
            truth = env.oracle(policy)
            self.assertAlmostEqual(float(mass.sum()), 1.0, places=12)
            self.assertAlmostEqual(float(mass @ batch["returns"][:, 0]), truth["expected_return"], places=12)
            moment = np.zeros(2)
            for state in range(2):
                term = (batch["exact_advantage"][..., None] * 2 * (batch["actions"] - batch["old"])
                        * (batch["x"] == state)).mean(axis=(1, 2))
                moment[state] = mass @ term
            coefficient = moment / (env.type_probabilities * 4 * policy * (1 - policy))
            np.testing.assert_allclose(coefficient, truth["local_direction"], atol=1e-12)

    def test_pair_scale_protocol_changes_actual_direction(self):
        old = PairGame.initial_policy
        source = PairGame().oracle(old)
        normalized = PairGame(2).oracle(old)
        fixed = PairGame(2, variant="fixed-edge").oracle(old)
        np.testing.assert_allclose(source["local_direction"], normalized["local_direction"])
        self.assertAlmostEqual(source["local_direction"][0], 0.06)
        self.assertAlmostEqual(fixed["local_direction"][0], -0.0466666666666667)
        self.assertGreater(source["local_direction"][0], 0)
        self.assertLess(fixed["local_direction"][0], 0)

    def test_true_derivative_matches_probability_path_difference(self):
        for env in (PairGame(), PairGame(2, variant="fixed-edge"), TwoStepGame()):
            policy = env.initial_policy
            c = np.linspace(-0.4, 0.7, env.n_states)
            eps = 1e-5
            finite = (env.oracle(tilted(policy, c, eps))["expected_return"]
                      - env.oracle(tilted(policy, c, -eps))["expected_return"]) / (2 * eps)
            self.assertAlmostEqual(finite, env.oracle(policy, c)["return_derivative"], places=8)

    def test_an_sa_equal_at_uniform_binary_but_not_generally(self):
        env, rng = PairGame(), np.random.default_rng(2)
        c = np.array([0.3, -0.6])
        batch = env.sample(np.full(2, 0.5), 16, rng)
        labels = batch["returns"]
        an, grad_an, _ = direction_terms(c, batch, labels, "AN")
        sa, grad_sa, _ = direction_terms(c, batch, labels, "SA")
        np.testing.assert_allclose(an, sa)
        np.testing.assert_allclose(grad_an, grad_sa)
        nonuniform = env.sample(env.initial_policy, 16, rng)
        an, _, _ = direction_terms(c, nonuniform, nonuniform["returns"], "AN")
        sa, _, _ = direction_terms(c, nonuniform, nonuniform["returns"], "SA")
        self.assertGreater(np.max(abs(an - sa)), 0.01)

    def test_direction_gradient_matches_loss_difference(self):
        env, c = TwoStepGame(), np.array([0.2, -0.3, 0.1])
        batch = env.sample(env.initial_policy, 17, np.random.default_rng(7))
        for method in ("AN", "SA"):
            _, gradient = direction_gradient(c, batch, batch["returns"], method)
            numerical = np.zeros_like(c)
            for index in range(len(c)):
                increment = np.zeros_like(c)
                increment[index] = 1e-6
                positive = direction_gradient(c + increment, batch, batch["returns"], method)[0]
                negative = direction_gradient(c - increment, batch, batch["returns"], method)[0]
                numerical[index] = (positive - negative) / 2e-6
            np.testing.assert_allclose(gradient, numerical, atol=1e-9)

    def test_forward_kl_fit_and_capacity_are_separate(self):
        old = PairGame.initial_policy.copy()
        weights = PairGame().type_probabilities
        target = tilted(old, np.array([-0.4, 0.8]), 0.4)
        fitted = fit_actor(old, target, weights, steps=500, lr=0.03)
        restricted = fit_actor(old, target, weights, steps=500, lr=0.03, restricted=True)
        self.assertLess(kl(target, fitted, weights), 1e-12)
        self.assertGreater(kl(target, restricted, weights), 0.01)
        np.testing.assert_array_equal(old, PairGame.initial_policy)

    def test_two_step_label_protocol_and_time_normalization(self):
        env = TwoStepGame()
        batch, mass = env.enumerate(env.initial_policy)
        self.assertAlmostEqual(float(mass.sum()), 1.0)
        self.assertAlmostEqual(env.oracle(env.initial_policy)["expected_return"], 0.319014)
        self.assertAlmostEqual(float(batch["weights"].sum()), 1.0)
        exact = labels_for(batch, "exact_advantage", env, env.initial_policy)
        mc = labels_for(batch, "mc_exact_baseline", env, env.initial_policy)
        self.assertGreater(np.max(abs(exact[:, 0] - mc[:, 0])), 0.1)
        np.testing.assert_allclose(exact[:, 1], mc[:, 1])
        for direction in (np.array([0.4, 0.9, -0.6]), np.ones(3)):
            exact_score = float(mass @ episode_scores(direction, batch, exact, "AN"))
            self.assertAlmostEqual(exact_score, env.oracle(env.initial_policy, direction)["score"], places=12)

    def test_temporal_variance_label_change_reverses_ordering(self):
        rows = temporal_variance({"sample_sizes": [8], "data_seeds": [10], "direction_steps": 2})
        variance = {(r["label"], r["method"]): r["exact_raw_episode_score_variance"] for r in rows}
        self.assertAlmostEqual(variance[("mc_exact_baseline", "AN")], 0.17966338018542272, places=12)
        self.assertAlmostEqual(variance[("mc_exact_baseline", "SA")], 0.21671631723222268, places=12)
        self.assertLess(variance[("mc_exact_baseline", "AN")], variance[("mc_exact_baseline", "SA")])
        self.assertGreater(variance[("exact_advantage", "AN")], variance[("exact_advantage", "SA")])

    def test_information_loss_decomposition_and_nuisance_removal(self):
        rows = {row["method"]: row for row in information({})}
        self.assertAlmostEqual(rows["full_multiset"]["preprocessing_loss"], 0.0)
        self.assertAlmostEqual(rows["drop_independent_nuisance"]["preprocessing_loss"], 0.0)
        self.assertGreater(rows["mean_and_count"]["preprocessing_loss"], 0.0)
        self.assertGreater(rows["sorted_first_two"]["preprocessing_loss"], 0.0)
        for row in rows.values():
            self.assertLess(row["decomposition_residual"], 1e-14)

    def test_source_empirical_selection_never_calls_oracle(self):
        env, old = PairGame(), PairGame.initial_policy.copy()
        batch = env.sample(old, 16, np.random.default_rng(3))
        cfg = settings({"direction_steps": 3, "fit_steps": [3], "evaluation_episodes": 4})
        with patch.object(PairGame, "oracle", side_effect=AssertionError("oracle selection forbidden")):
            for method in ("AN", "SA", "DA"):
                empirical_update(env, old, batch, batch["returns"], "mc_raw", None, method, cfg,
                                 np.random.default_rng(4))

    def test_flat_schema_aliases_filter_methods_and_share_one_fitted_q(self):
        config = {"methods": ["SA"], "labels": ["exact"], "sample_sizes": [8], "data_seeds": [2],
                  "direction_steps": 2, "fit_steps": [1, 3], "evaluation_episodes": 2}
        rows = estimation(config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "SA")
        self.assertEqual(rows[0]["label"], "exact_advantage")
        fits = actor_realization(config)
        self.assertEqual({row["method"] for row in fits}, {"SA"})
        self.assertEqual(len({(row["learned_q_0"], row["learned_q_1"]) for row in fits}), 1)
        self.assertEqual({row["fit_steps"] for row in fits}, {1, 3})

    def test_check_switches_charge_only_checks_that_execute(self):
        env, old = PairGame(), PairGame.initial_policy
        batch = env.sample(old, 8, np.random.default_rng(3))
        cfg = settings({"direction_check": False, "return_check": False,
                        "direction_steps": 2, "fit_steps": [200], "actor_fit_steps": 2,
                        "check_episodes": 3})
        _, _, costs, details = empirical_update(env, old, batch, batch["returns"], "mc_raw", None,
                                               "AN", cfg, np.random.default_rng(8))
        self.assertEqual(sum(costs.values()), 0)
        self.assertEqual(details["reason"], "accepted_return_check_disabled")

    def test_transport_bounds_and_frozen_actor(self):
        rows = transfer({"sample_sizes": [32], "direction_steps": 20, "fit_steps": [20], "evaluation_episodes": 16})
        for method in ("AN", "SA", "DA"):
            group = [row for row in rows if row["method"] == method]
            self.assertEqual(len({(r["frozen_actor_p_0"], r["frozen_actor_p_1"]) for r in group}), 1)
            for row in group:
                if method != "DA":
                    self.assertLessEqual(row["target_direction_error"], row["transport_error_upper_bound"] + 1e-12)
                    self.assertGreaterEqual(row["target_normalized_path_derivative"], row["normalized_derivative_lower_bound"] - 1e-12)

    def test_training_costs_complete_episodes_and_checkpoints(self):
        config = {"sample_sizes": [8], "batch_episodes": 8, "critic_episodes": 8,
                  "check_episodes": 4, "monitor_episodes": 4, "evaluation_episodes": 4,
                  "direction_steps": 2, "fit_steps": [2], "step_sizes": [0.2],
                  "budget": 128, "seed": 9, "verbose": False}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for experiment in ("P-L", "T-L"):
                result = run_experiment(experiment, config, root / experiment)
                self.assertEqual(result["status"], "completed")
                for record in result["records"]:
                    if "final_checkpoint" not in record:
                        continue
                    total = sum(record[f"source_steps_{purpose}"] for purpose in (
                        "training", "critic", "direction_check", "return_check", "monitor"))
                    self.assertEqual(total, record["source_steps_total"])
                    self.assertLessEqual(total, config["budget"])
                    if experiment == "T-L":
                        self.assertEqual(total % 2, 0)
                    if record["method"] == "DA":
                        self.assertEqual(record["source_steps_direction_check"], 0)
                    policy = load_policy(root / experiment / record["final_checkpoint"])
                    self.assertTrue(np.all((policy > 0) & (policy < 1)))

    def test_report_only_evaluation_cannot_select_or_change_actor(self):
        config = {"methods": ["AN"], "learning_labels": ["mc"], "sample_sizes": [8],
                  "batch_episodes": 8, "critic_episodes": 8, "check_episodes": 4,
                  "source_eval_episodes": 4, "evaluation_episodes": 2,
                  "direction_steps": 2, "actor_fit_steps": 2, "fit_steps": [99],
                  "step_sizes": [0.2], "budget": 128, "seed": 9, "verbose": False}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = run_experiment("P-L", config, root / "short")
            long = run_experiment("P-L", {**config, "evaluation_episodes": 11}, root / "long")
            a, b = short["records"][0], long["records"][0]
            self.assertEqual(a["source_steps_total"], b["source_steps_total"])
            self.assertEqual(a["source_steps_monitor"], b["source_steps_monitor"])
            self.assertEqual(a["source_report_only_steps"], 2)
            self.assertEqual(b["source_report_only_steps"], 11)
            np.testing.assert_array_equal(load_policy(root / "short" / a["final_checkpoint"]),
                                          load_policy(root / "long" / b["final_checkpoint"]))
            np.testing.assert_array_equal(load_policy(root / "short" / a["best_source_checkpoint"]),
                                          load_policy(root / "long" / b["best_source_checkpoint"]))


if __name__ == "__main__":
    unittest.main()
