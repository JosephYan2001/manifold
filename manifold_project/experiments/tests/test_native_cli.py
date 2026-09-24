"""A real native-environment user flow, including source reuse and provenance."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from manifold_project.experiments.cli import main
from manifold_project.experiments.common.reporting import read_csv


class NativeEntryTests(unittest.TestCase):
    def test_source_reuse_ablation_selection_and_formal_guard(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "source"
            capture = io.StringIO()
            with redirect_stdout(capture), redirect_stderr(capture):
                code = main(["--experiments", "N-I", "N-C", "N-R", "N-Gd", "N-Gr", "N-T", "N-F",
                             "--profile", "smoke", "--methods", "AN", "--budget", "256", "--output", str(output)])
            self.assertEqual(code, 0, capture.getvalue())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["status"] == "completed" for row in manifest["experiments"].values()))
            inputs = manifest["experiments"]["N-I"]["runs"]
            self.assertTrue(next(row for row in inputs if row["config"]["observation_mode"] == "full")["source_training_reused"])
            self.assertFalse((output / "N-I_observation_ablation" / "full" / "AN_s40" / "checkpoints/final.pt").exists())
            # Primary transfer uses only N-C even though this suite contains six
            # other source actors at the same seed/environment.
            self.assertEqual(len(manifest["experiments"]["N-T"]["runs"]), 1)
            target = read_csv(output / "N-T_frozen_transfer" / "summary.csv")
            self.assertTrue(any(row["metric"] == "success_rate" for row in target))
            self.assertTrue(all(row["smoke_only"] == "True" for row in target))
            self.assertEqual(len(list(output.rglob('manifest.json'))), 1)
            self.assertTrue(all(path.name == 'events.jsonl' for path in output.rglob('*.jsonl')))
            ablated = read_csv(output / 'N-I_observation_ablation' / 'summary.csv')
            self.assertEqual({int(row['target_n']) for row in ablated if row.get('target_n')}, {2,4,6,8})
            with redirect_stdout(capture), redirect_stderr(capture):
                formal = main(["--experiments", "N-T", "--profile", "formal", "--source", str(output),
                               "--output", str(root / "invalid_formal")])
            self.assertEqual(formal, 2)
            self.assertIn("smoke checkpoint", capture.getvalue())


if __name__ == "__main__":
    unittest.main()
