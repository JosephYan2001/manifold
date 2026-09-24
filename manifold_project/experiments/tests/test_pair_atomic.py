import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from manifold_project.experiments.pair_coordination.training.checkpoints import atomic_json
from manifold_project.experiments.pair_coordination.training.progress import Progress


class AtomicCheckpointTests(unittest.TestCase):
    def test_transient_permission_error_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"state.json"
            atomic_json(path, {"round": 1})
            real_replace = os.replace
            count = [0]
            def busy_then_replace(source, target):
                count[0] += 1
                if count[0] < 3:
                    self.assertEqual(json.loads(path.read_text())["round"], 1)
                    raise PermissionError("temporarily locked")
                real_replace(source, target)
            with patch("os.replace", side_effect=busy_then_replace), patch("time.sleep") as sleep:
                atomic_json(path, {"round": 2})
            self.assertEqual(count[0], 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(path.read_text())["round"], 2)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_persistent_error_preserves_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"state.json"
            atomic_json(path, {"round": 1})
            with patch("os.replace", side_effect=PermissionError("locked")) as replace, patch("time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {"round": 2})
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(json.loads(path.read_text())["round"], 1)

    def test_other_error_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("os.replace", side_effect=OSError(28, "disk full")) as replace, patch("time.sleep") as sleep:
                with self.assertRaises(OSError):
                    atomic_json(Path(tmp)/"state.json", {})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()

    def test_progress_error_reports_attempted_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = type("Sampler", (), {"used": 123})()
            progress = Progress(tmp, sampler)
            with patch("manifold_project.experiments.pair_coordination.training.progress.atomic_json", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError) as caught:
                    progress.start(256, "冻结策略")
            self.assertEqual(caught.exception.training_stage["round"], 256)
