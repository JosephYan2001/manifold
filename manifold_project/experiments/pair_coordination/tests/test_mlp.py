import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from manifold_project.experiments.pair_coordination.models.mlp import MLPActor, MLPDirection, load_actor
from manifold_project.experiments.pair_coordination.training.actor_fit import kl_and_gradient, fit_actor
from manifold_project.experiments.pair_coordination.training.direction_loss import loss_and_gradient, fit_direction
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig, sample_episodes
from manifold_project.experiments.pair_coordination.training.labels import make_labels
from manifold_project.experiments.pair_coordination.training.config import TrainConfig
from manifold_project.experiments.pair_coordination.training.runner import run_training
from manifold_project.experiments.pair_coordination.train import build_parser, resolve_configs


class MLPTests(unittest.TestCase):
    def test_actor_initialization_and_roundtrip(self):
        actor = MLPActor.initial(2, width=4, depth=2, seed=3)
        np.testing.assert_allclose(actor.table(), [[.65, .35], [.35, .65]], atol=1e-12)
        np.testing.assert_array_equal(load_actor(actor.state()).parameters, actor.parameters)
        other = actor.copy()
        other.parameters[:] += .1
        self.assertFalse(np.array_equal(actor.parameters, other.parameters))

    def finite_gradient(self, parameters, analytic, loss):
        for i in range(len(parameters)):
            old = parameters[i]
            parameters[i] = old+1e-6
            plus = loss()
            parameters[i] = old-1e-6
            minus = loss()
            parameters[i] = old
            self.assertAlmostEqual(analytic[i], (plus-minus)/2e-6, places=6)

    def test_all_actor_layers_gradient(self):
        actor = MLPActor.initial(2, width=3, depth=2)
        target, weights = np.array([[.8, .2], [.2, .8]]), np.array([.6, .4])
        _, grad = kl_and_gradient(actor, target, weights)
        self.finite_gradient(actor.parameters, actor.parameter_gradient(grad),
                             lambda: kl_and_gradient(actor, target, weights)[0])

    def test_all_direction_layers_gradient(self):
        actor = MLPActor.initial(2, width=3)
        batch = sample_episodes(PairConfig(), actor.table(), 32, seed=3)
        labels = make_labels(batch)
        model = MLPDirection(2, width=3)
        model.parameters[:] += .1  # exercise hidden derivatives beyond zero-head initialization
        for method in ("analytic", "sampled"):
            _, grad = loss_and_gradient(model.table(), batch, labels, method)
            self.finite_gradient(model.parameters, model.parameter_gradient(grad),
                                 lambda: loss_and_gradient(model.table(), batch, labels, method)[0])

    def test_learning_updates_hidden_layers_and_fits(self):
        actor = MLPActor.initial(2, width=4)
        batch = sample_episodes(PairConfig(), actor.table(), 512, seed=3)
        model = MLPDirection(2, width=4)
        before = model.network.arrays()[0].copy()
        fit_direction(model, batch, make_labels(batch), "analytic", 100, .01, 10)
        self.assertFalse(np.array_equal(before, model.network.arrays()[0]))
        np.testing.assert_allclose(model.table().sum(axis=1), 0)
        target = np.array([[.75, .25], [.25, .75]])
        new, fit = fit_actor(actor, target, batch, 100, .01)
        self.assertLess(fit["fit_kl"], fit["fit_initial_kl"]*.01)
        self.assertFalse(np.array_equal(actor.network.arrays()[0], new.network.arrays()[0]))

    def test_cli_overrides_json_and_expands_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.json"
            path.write_text(json.dumps({"hidden_width": 8, "seed": 99}), encoding="utf-8")
            args = build_parser().parse_args(["--training-config", str(path), "--hidden-width", "4",
                "--seeds", "1", "2", "--methods", "analytic", "sampled", "--sample-sizes", "32", "64",
                "--n-agents", "6"])
            source, configs = resolve_configs(args)
            self.assertEqual(source.n_agents, 6)
            self.assertEqual(len(configs), 8)
            self.assertTrue(all(c.hidden_width == 4 for c in configs))

    def test_mlp_runner_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_training(PairConfig(), TrainConfig(mode="policy", acceptance="empirical",
                                  hidden_width=4, episodes=128, check_episodes=128,
                                  direction_steps=30, fit_steps=30, rounds=1), Path(tmp)/"run")
            self.assertEqual(result["actor"]["kind"], "mlp_actor")
            self.assertTrue(np.isfinite(load_actor(result["actor"]).table()).all())


if __name__ == "__main__":
    unittest.main()
