"""Environment lifecycle, sampling and mathematical regression tests."""
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from manifold_project.experiments.pair_coordination.envs.pair_coordination import (
    PairConfig, PairCoordinationEnv, sample_episodes, validate_policy,
)
from manifold_project.experiments.pair_coordination.evaluation.exact_pair import (
    closed_form_local_direction, exact_evaluate, exponential_target,
)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.config = PairConfig()
        self.policy = np.array([[0.65, 0.35], [0.35, 0.65]])

    def test_json_config(self):
        path = Path(__file__).resolve().parents[1] / "configs/pair_source.json"
        self.assertEqual(PairConfig.from_json(path), self.config)

    def test_known_reward_and_lifecycle(self):
        env = PairCoordinationEnv(self.config)
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 1, 1])
        obs, info = env.reset(types=[0, 0, 1, 1])
        self.assertTrue(info["diagnostic_reset"])
        _, reward, terminated, truncated, _ = env.step([0, 0, 1, 1])
        self.assertAlmostEqual(reward, 1.0 + 0.7 * 4 / 3)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 1, 1])
        obs[:] = 1
        np.testing.assert_array_equal(env.state(), [0, 0, 1, 1])

    def test_seed_reproducibility_and_snapshot(self):
        a = sample_episodes(self.config, self.policy, 30, seed=101)
        b = sample_episodes(self.config, self.policy, 30, seed=101)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        old = a["old_probs"].copy()
        self.policy[:] = 0.5
        np.testing.assert_array_equal(a["old_probs"], old)
        self.assertAlmostEqual(a["weights"].sum(), 1)
        self.assertEqual(a["local_types"].shape, (30, 4))
        self.assertNotIn("advantage", a)

    def test_reward_bound_and_exchangeability(self):
        env = PairCoordinationEnv(self.config)
        x, actions = [0, 1, 0, 1], [1, 0, 0, 1]
        order = [2, 0, 3, 1]
        self.assertAlmostEqual(env.reward(x, actions),
                               env.reward(np.array(x)[order], np.array(actions)[order]))
        batch = sample_episodes(self.config, self.policy, 100, seed=5)
        self.assertLessEqual(abs(batch["team_rewards"]).max(), self.config.reward_bound)

    def test_invalid_actions_do_not_consume_episode(self):
        env = PairCoordinationEnv()
        env.reset(seed=0)
        for bad in ([0, 1], [0, 0, 1, 2], [0., 0., 1., 1.], [True]*4):
            with self.assertRaises(ValueError):
                env.step(bad)
        env.step([0, 0, 1, 1])

    def test_invalid_configs(self):
        for kwargs in ({"n_agents": 1}, {"n_agents": True},
                       {"type_probs": [0.5, 0.6]}, {"type_probs": [-1, 2]},
                       {"type_probs": [float("nan"), 1]}, {"local_bias": [0]},
                       {"interaction_strength": -1}, {"pair_payoff": [[1, 2], [0, 1]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PairConfig(**kwargs)

    def test_invalid_policies(self):
        for policy in ([[1, 0], [0.5, 0.5]], [[0.6, 0.6]]*2,
                       [[0.5, 0.5]], [[float("nan"), 0.5]]*2):
            with self.assertRaises(ValueError):
                validate_policy(self.config, policy)

    def test_config_copies_input(self):
        bias = [-0.2, 0.3]
        config = PairConfig(local_bias=bias)
        bias[0] = 99
        self.assertEqual(config.local_bias, (-0.2, 0.3))

    def test_sampling_matches_expectations(self):
        batch = sample_episodes(self.config, self.policy, 6000, seed=43)
        truth = exact_evaluate(self.config, self.policy)["expected_return"]
        self.assertLess(abs(batch["team_rewards"].mean() - truth), 0.05)
        self.assertLess(abs((batch["local_types"] == 0).mean() - 0.6), 0.02)


class ExactTests(unittest.TestCase):
    def setUp(self):
        self.config = PairConfig()
        self.policy = np.array([[0.65, 0.35], [0.35, 0.65]])
        self.q = np.array([[0.1, -0.1], [-0.2, 0.2]])
        self.result = exact_evaluate(self.config, self.policy, self.q)

    def test_mass_and_closed_form(self):
        r = self.result
        self.assertAlmostEqual(r["probability_mass"], 1)
        np.testing.assert_allclose(r["local_type_probs"], self.config.type_probs, atol=1e-12)
        np.testing.assert_allclose(r["local_direction"],
                                   closed_form_local_direction(self.config, self.policy), atol=1e-12)
        np.testing.assert_allclose(np.einsum("xij,xj->xi", r["fisher"], r["local_direction"]),
                                   r["conditional_m"], atol=1e-12)

    def test_variational_and_regression(self):
        r = self.result
        self.assertAlmostEqual(r["local_energy"] - r["score"], r["direction_error"])
        self.assertAlmostEqual(r["advantage_second_moment"] - r["regression_risk"], r["score"])
        self.assertGreater(r["information_gap"], 0)
        self.assertAlmostEqual(r["full_energy"], r["information_gap"] + r["local_energy"])

    def test_return_derivative(self):
        eps = 1e-5
        plus = exact_evaluate(self.config, exponential_target(self.policy, self.q, eps))
        minus = exact_evaluate(self.config, exponential_target(self.policy, self.q, -eps))
        self.assertAlmostEqual((plus["expected_return"] - minus["expected_return"])/(2*eps),
                               self.result["return_derivative"], places=8)

    def test_oracle_improves_return(self):
        target = exponential_target(self.policy, self.result["local_direction"], 0.01)
        self.assertGreater(exact_evaluate(self.config, target)["expected_return"],
                           self.result["expected_return"])

    def test_scale_invariance(self):
        for n in (2, 3, 6):
            with self.subTest(n=n):
                result = exact_evaluate(replace(self.config, n_agents=n), self.policy)
                np.testing.assert_allclose(result["local_direction"],
                                           self.result["local_direction"], atol=1e-12)

    def test_composition_shift_bound(self):
        target = replace(self.config, n_agents=3, type_probs=(0.2, 0.8))
        result = exact_evaluate(target, self.policy)
        diff = result["local_direction"] - self.result["local_direction"]
        sem = np.einsum("x,xi,xij,xj->", result["local_type_probs"], diff, result["fisher"], diff)
        measure = self.config.interaction_strength * np.abs(
            np.array(target.type_probs) - self.config.type_probs).sum()
        bound = np.max(np.abs(self.config.pair_payoff))**2 * measure**2
        self.assertGreater(sem, 0)
        self.assertLessEqual(sem, bound + 1e-12)

    def test_zero_mass_support(self):
        result = exact_evaluate(replace(self.config, type_probs=(1., 0.)), self.policy)
        np.testing.assert_array_equal(result["supported_types"], [True, False])
        np.testing.assert_array_equal(result["local_direction"][1], [0, 0])

    def test_second_order_coordination(self):
        config = PairConfig(n_agents=2, type_probs=(1.,), local_bias=(0.,),
                            pair_payoff=((1.,),), interaction_strength=1.)
        base = exact_evaluate(config, [[0.5, 0.5]], [[-0.1, 0.1]])
        np.testing.assert_allclose(base["local_direction"], 0, atol=1e-12)
        self.assertLess(base["score"], 0)
        self.assertAlmostEqual(exact_evaluate(config, [[0.4, 0.6]])["expected_return"], 0.04)

    def test_guard_and_invalid_direction(self):
        with self.assertRaises(ValueError):
            exact_evaluate(replace(self.config, n_agents=30), self.policy)
        with self.assertRaises(ValueError):
            exact_evaluate(self.config, self.policy, np.ones((2, 2)))
        with self.assertRaises(ValueError):
            exponential_target(self.policy, self.q, float("inf"))

    def test_zero_interaction(self):
        result = exact_evaluate(replace(self.config, interaction_strength=0), self.policy)
        self.assertAlmostEqual(result["information_gap"], 0, places=12)


if __name__ == "__main__":
    unittest.main()
