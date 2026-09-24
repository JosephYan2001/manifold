"""Integration checks for reproducible experiment dispatch and seed statistics."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from manifold_project.experiments.cli import main, parser, resolve, _select_checkpoints
from manifold_project.experiments.common.config import load_config
from manifold_project.experiments.common.reporting import read_csv, summarize_records, paired_summaries
from manifold_project.experiments.common.runtime import metadata
from manifold_project.experiments.cooperative_navigation.training.storage import atomic
from manifold_project.experiments.registry import get_experiment


class EntryAndStatisticsTests(unittest.TestCase):
    def test_all_finite_entries_and_compact_artifacts(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = Path(tmp) / "suite"
            code = main(["--experiments", "P-E", "P-A", "P-T", "P-I", "P-L", "P-B", "T-V", "T-L",
                         "--profile", "smoke", "--output", str(out)])
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["experiments"]), 8)
            self.assertEqual(len(list(out.rglob('manifest.json'))), 1)
            self.assertTrue(all(path.name == 'events.jsonl' for path in out.rglob('*.jsonl')))
            rows = read_csv(out / "summary.csv")
            self.assertTrue(rows)
            self.assertTrue(all(row["smoke_only"] in ("True", "true") for row in rows))
            for checkpoint in out.rglob("final.npz"):
                self.assertLessEqual(len(list(checkpoint.parent.glob("*.npz"))), 2)
            for checkpoint in out.rglob('checkpoints/final.json'):
                self.assertLessEqual(len(list(checkpoint.parent.glob('*.json'))), 2)

    def test_existing_results_are_not_overwritten(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            marker = Path(tmp) / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            self.assertEqual(main(["--experiments", "P-I", "--profile", "smoke", "--output", tmp]), 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
            self.assertFalse((Path(tmp) / "manifest.json").exists())

    def test_data_budget_and_snapshot_axes_do_not_get_pooled(self):
        rows = []
        for sample in (32, 128):
            for mark in (.25, 1.):
                for seed in (100, 101):
                    rows.append(dict(experiment="N-F", method="AN", seed=seed, fit_steps=10,
                                     sample_episodes=sample, budget_fraction=mark, error=float(seed)))
        summary = summarize_records(rows)
        errors = [r for r in summary if r["metric"] == "error"]
        self.assertEqual(len(errors), 4)
        self.assertTrue(all(r["independent_units"] == 2 for r in errors))

    def test_seed_level_bootstrap_not_episode_pseudoreplication(self):
        rows = [dict(experiment="N-C", method="AN", seed=100, value=1.) for _ in range(100)]
        rows += [dict(experiment="N-C", method="AN", seed=101, value=3.)]
        row = summarize_records(rows)[0]
        self.assertEqual(row["independent_units"], 2)
        self.assertAlmostEqual(row["mean"], 2.)

    def test_paired_statistics_verify_shared_data_hash(self):
        rows = [dict(experiment="P-E", method="AN", data_seed=7, sample_episodes=32, direction_error=1., dataset_sha256="same"),
                dict(experiment="P-E", method="SA", data_seed=7, sample_episodes=32, direction_error=2., dataset_sha256="same")]
        self.assertEqual(paired_summaries(rows)[0]["mean"], -1.)
        rows[1]["dataset_sha256"] = "different"
        with self.assertRaises(ValueError):
            paired_summaries(rows)

    def test_source_manifest_excludes_ablation_checkpoints(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "N-C_source_learning" / "AN_s40" / "final.pt"
            expected.parent.mkdir(parents=True)
            expected.touch()
            other = root / "N-R_relation_ablation" / "AN_s40" / "final.pt"
            other.parent.mkdir(parents=True)
            other.touch()
            (root / "manifest.json").write_text(json.dumps({"experiments": {
                "N-C": {"status": "completed", "checkpoints": [str(expected.relative_to(root))]},
                "N-R": {"status": "completed", "checkpoints": [str(other.relative_to(root))]}}}), encoding="utf-8")
            args = parser().parse_args(["--experiments", "N-T", "--source", tmp])
            self.assertEqual(_select_checkpoints(get_experiment("N-T"), args, {}), [expected.resolve()])

    def test_preflight_and_profile_resolution(self):
        with self.assertRaises(ValueError):
            resolve(parser().parse_args(["--experiments", "N-T"]))
        plan = resolve(parser().parse_args(["--experiments", "N-T", "N-C", "--profile", "smoke"]))
        self.assertEqual(plan[0][0].id, "N-C")
        self.assertEqual(plan[0][1]["horizon"], 8)
        self.assertEqual(load_config("navigation", "formal")["horizon"], 200)
        self.assertEqual(load_config("pair", "formal")["learning_labels"], ["mc"])

    def test_atomic_failure_preserves_previous_artifact(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.pt"
            path.write_bytes(b"previous")
            def broken_writer(temporary):
                Path(temporary).write_bytes(b"incomplete")
                raise ValueError("serialization failed")
            with self.assertRaises(ValueError):
                atomic(path, broken_writer)
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["final.pt"])

    def test_provenance_hash_covers_actual_sources(self):
        import hashlib
        self.assertNotEqual(metadata()["experiment_code_sha256"], hashlib.sha256(b"").hexdigest())


if __name__ == "__main__":
    unittest.main()
