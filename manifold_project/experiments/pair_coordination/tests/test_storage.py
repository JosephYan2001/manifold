import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig
from manifold_project.experiments.pair_coordination.training.config import TrainConfig
from manifold_project.experiments.pair_coordination.training.runner import run_training
from manifold_project.experiments.pair_coordination.train import build_parser, resolve_configs


class StorageTests(unittest.TestCase):
    def test_flags(self):
        _, configs = resolve_configs(build_parser().parse_args([
            "--save-batches", "--save-direction-snapshots", "--checkpoint-every", "2"]))
        self.assertTrue(configs[0].save_batches)
        self.assertTrue(configs[0].save_direction_snapshots)
        _, configs = resolve_configs(build_parser().parse_args(["--no-save-batches"]))
        self.assertFalse(configs[0].save_batches)

    def test_storage_does_not_change_learning(self):
        settings = TrainConfig(mode="policy", acceptance="empirical", hidden_width=4,
                               rounds=2, episodes=128, check_episodes=128,
                               direction_steps=20, fit_steps=20, checkpoint_every=2)
        with tempfile.TemporaryDirectory() as tmp:
            lean, debug = Path(tmp)/"lean", Path(tmp)/"debug"
            a = run_training(PairConfig(), settings, lean)
            b = run_training(PairConfig(), replace(settings, save_batches=True,
                             save_direction_snapshots=True), debug)
            self.assertEqual(a["actor"], b["actor"])
            self.assertFalse(list(lean.glob("*.npz")))
            self.assertFalse(list(lean.glob("round_*.json")))
            self.assertTrue(list(debug.glob("*.npz")))
            self.assertTrue(list(debug.glob("round_*.json")))
            self.assertEqual({p.name for p in (lean/"checkpoints").iterdir()}, {"best.json", "final.json"})
            policies = [json.loads(s) for s in (lean/"policy.jsonl").read_text().splitlines()]
            best = json.loads((lean/"checkpoints/best.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(best["selection_score"], max(p["expected_return"] for p in policies))
            records = [json.loads(s) for s in (lean/"checks.jsonl").read_text().splitlines()]
            for record in records:
                self.assertNotIn("candidate_actor", record)
                self.assertNotIn("target", record)
                if record.get("actor_checkpoint"):
                    self.assertTrue((lean/record["actor_checkpoint"]).exists())
            batches = [json.loads(s) for s in (lean/"batches.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["file"] is None for r in batches))
