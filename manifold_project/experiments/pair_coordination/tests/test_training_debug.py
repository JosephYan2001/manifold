import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from manifold_project.experiments.pair_coordination.train import build_parser, resolve_configs
from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig
from manifold_project.experiments.pair_coordination.training.config import TrainConfig
from manifold_project.experiments.pair_coordination.training.runner import run_training
from manifold_project.experiments.pair_coordination.training.plan import training_plan


class DebugTests(unittest.TestCase):
    def test_new_names_and_conflicts(self):
        _, configs = resolve_configs(build_parser().parse_args([
            "--episodes-per-round", "65", "--max-rounds", "9", "--max-source-episodes", "9999", "--actor-lr", ".003"]))
        c = configs[0]
        self.assertEqual((c.episodes, c.rounds, c.budget, c.fit_lr), (65, 9, 9999, .003))
        self.assertEqual(c.direction_epochs, 25)
        with self.assertRaises(ValueError):
            resolve_configs(build_parser().parse_args(["--direction-epochs", "2", "--direction-steps", "10"]))
        with self.assertRaises(ValueError):
            resolve_configs(build_parser().parse_args(["--rounds", "2", "--max-rounds", "3"]))

    def test_plan_counts(self):
        plan = training_plan(PairConfig(), TrainConfig(mode="policy", episodes=65, batch_size_episodes=16,
                             direction_epochs=3, actor_epochs=4, baseline="table"))
        self.assertEqual(plan["direction_updates"], 15)
        self.assertEqual(plan["actor_updates_per_candidate"], 20)
        self.assertEqual(plan["one_candidate_cost"], 65+512+3072)

    def test_dry_run_does_not_create_directory(self):
        script = Path(__file__).resolve().parents[1]/"train.py"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"absent"
            subprocess.run([sys.executable, str(script), "--dry-run", "--output-dir", str(output)],
                           check=True, capture_output=True, timeout=30)
            self.assertFalse(output.exists())

    def test_debug_logs_and_failure_cost(self):
        settings = TrainConfig(episodes=32, direction_epochs=2, actor_epochs=2,
                               batch_size_episodes=16, hidden_width=4)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run_training(PairConfig(), settings, root/"ok")
            for name in ("requested_config.json", "resolved_config.json", "training_plan.json", "stages.jsonl", "updates.jsonl", "data.jsonl"):
                self.assertTrue((root/"ok"/name).exists())
            updates = [json.loads(line) for line in (root/"ok/updates.jsonl").read_text().splitlines()]
            self.assertEqual(len(updates), 4)
            self.assertTrue(all(row["lr"] == settings.direction_lr for row in updates))
            with patch("manifold_project.experiments.pair_coordination.training.runner.fit_direction", side_effect=FloatingPointError("test")):
                with self.assertRaises(FloatingPointError):
                    run_training(PairConfig(), settings, root/"failed")
            report = json.loads((root/"failed/failure.json").read_text(encoding="utf-8"))
            self.assertEqual(report["uncommitted_source_episodes"], 32)
            self.assertEqual(report["committed_round"], 0)
            self.assertEqual(Path(report["latest_checkpoint"]).name, "final.json")
            self.assertTrue(Path(report["latest_checkpoint"]).exists())
