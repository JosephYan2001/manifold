import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig
from manifold_project.experiments.pair_coordination.evaluation.exact_pair import (
    closed_form_expected_return, exact_evaluate,
)
from manifold_project.experiments.pair_coordination.models.actor import TableActor
from manifold_project.experiments.pair_coordination.training.checkpoints import save_checkpoint
from manifold_project.experiments.pair_coordination.training.config import TrainConfig


class BestCheckpointTests(unittest.TestCase):
    def test_score_matches_independent_enumeration(self):
        rng = np.random.default_rng(123)
        for n in (2, 3, 4):
            for probs in ((.6, .4), (0., 1.)):
                config = PairConfig(n_agents=n, type_probs=probs)
                for _ in range(3):
                    actor = TableActor(2, logits=rng.normal(size=2))
                    self.assertAlmostEqual(closed_form_expected_return(config, actor.table()),
                                           exact_evaluate(config, actor.table())["expected_return"])
        # This calculation also works beyond the joint-enumeration limit.
        large = replace(config, n_agents=100)
        self.assertAlmostEqual(closed_form_expected_return(large, actor.table()),
                               25 * closed_form_expected_return(config, actor.table()))

    def test_best_survives_regression_ties_and_copied_resume(self):
        config = PairConfig(local_bias=(1., 1.), interaction_strength=0.)
        settings = TrainConfig()
        sampler = SimpleNamespace(used=0, calls=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = None
            for round_index, logit in enumerate((-2., 2., 2., -1.)):
                actor = TableActor(2, logits=[logit, logit])
                sampler.used = round_index * 10
                state = save_checkpoint(root, config, settings, actor, round_index, sampler, True, best)
                best = state["best_checkpoint"]
            saved_best = json.loads((root/"checkpoints/best.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_best["round"], 1)
            self.assertEqual(state["round"], 3)
            self.assertGreater(saved_best["selection_score"], state["selection_score"])
            self.assertNotIn("best_checkpoint", saved_best)
            self.assertEqual({p.name for p in (root/"checkpoints").iterdir()}, {"best.json", "final.json"})
            restored = json.loads((root/"checkpoints/final.json").read_text(encoding="utf-8"))
            save_checkpoint(root/"resumed", config, settings, actor, 3, sampler, True,
                            restored["best_checkpoint"])
            self.assertEqual(json.loads((root/"resumed/checkpoints/best.json").read_text(encoding="utf-8")), saved_best)
