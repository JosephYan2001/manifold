"""Verify that displayed contributions match the actual environment."""
import unittest
import numpy as np
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig
from manifold_project.experiments.pair_coordination.visualization.render import episode_contributions


class VisualizationTests(unittest.TestCase):
    def test_mixed_sign_episode(self):
        data = episode_contributions(PairConfig(), [0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(data["team_reward"], .6)
        self.assertAlmostEqual(sum(v for _, _, v in data["pairs"]), 0)
        self.assertTrue(any(v < 0 for _, _, v in data["pairs"]))
        self.assertTrue(any(v > 0 for _, _, v in data["pairs"]))

    def test_aligned_episode(self):
        data = episode_contributions(PairConfig(), [0, 0, 1, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(data["team_reward"], 1 + .7*4/3)
        np.testing.assert_allclose(data["local"], [.2, .2, .3, .3])

    def test_validation_delegates_to_environment(self):
        with self.assertRaises(ValueError):
            episode_contributions(PairConfig(), [0, 0, 1, 1], [0, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
