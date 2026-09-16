import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig, sample_episodes
from manifold_project.experiments.pair_coordination.models.actor import TableActor
from manifold_project.experiments.pair_coordination.training.batching import episode_batches
from manifold_project.experiments.pair_coordination.training.config import TrainConfig
from manifold_project.experiments.pair_coordination.training.runner import run_training
from manifold_project.experiments.pair_coordination.models.mlp import load_actor


class BatchCheckpointTests(unittest.TestCase):
    def test_grouping_coverage_and_remainder(self):
        batch = sample_episodes(PairConfig(), TableActor.initial(2).table(), 10, seed=1)
        labels = np.arange(10)
        parts = list(episode_batches(batch, labels, 4, 2, 42))
        self.assertEqual([len(p["actions"]) for _, p, _ in parts], [4, 4, 2]*2)
        for epoch in (1, 2):
            selected = [(p, ids) for e, p, ids in parts if e == epoch]
            np.testing.assert_array_equal(np.sort(np.concatenate([ids for _, ids in selected])), labels)
            for part, ids in selected:
                np.testing.assert_array_equal(part["actions"], batch["actions"][ids])
                self.assertAlmostEqual(part["weights"].sum(), 1)

    def test_resume_matches_uninterrupted(self):
        settings = TrainConfig(checkpoint_every=1, mode="policy", acceptance="empirical", hidden_width=4,
                               episodes=65, check_episodes=64, rounds=3,
                               batch_size_episodes=16, direction_epochs=3, actor_epochs=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = run_training(PairConfig(), settings, root/"full")
            run_training(PairConfig(), settings, root/"paused", stop_after_round=1)
            resumed = run_training(PairConfig(), settings, root/"resumed",
                                   resume=root/"paused/checkpoints/final.json")
            self.assertEqual(full["actor"], resumed["actor"])
            self.assertEqual(full["source_episodes"], resumed["source_episodes"])
            for name in ("best", "final"):
                path = root/f"full/checkpoints/{name}.json"
                state = json.loads(path.read_text(encoding="utf-8"))
                resumed_state = json.loads((root/f"resumed/checkpoints/{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(state, resumed_state)
                self.assertTrue(np.isfinite(load_actor(state["actor"]).table()).all())
            self.assertEqual(state["round"], 3)
            logs = [json.loads(s) for s in (root/"full/direction.jsonl").read_text().splitlines()]
            self.assertEqual(max(s["step"] for s in logs), 15)

    def test_reject_incomplete_batch_configuration(self):
        with self.assertRaises(ValueError):
            TrainConfig(batch_size_episodes=32)
