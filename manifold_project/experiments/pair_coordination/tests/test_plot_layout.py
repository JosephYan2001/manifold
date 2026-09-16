import json
from pathlib import Path
import tempfile
import unittest
import warnings
from manifold_project.experiments.pair_coordination.analysis.plot_training import representative_rounds, plot_training


class PlotLayoutTests(unittest.TestCase):
    def test_representative_rounds_include_endpoints(self):
        rounds = representative_rounds([{"round": r} for r in range(1, 501)])
        self.assertEqual(len(rounds), 6)
        self.assertEqual((rounds[0], rounds[-1]), (1, 500))

    def test_long_run_layout(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("可选 Matplotlib 未安装")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/"config.json").write_text(json.dumps({"training": {
                "mode": "direction", "method": "analytic", "baseline": "zero", "seed": 42}}), encoding="utf-8")
            rows = [{"round": r, "step": s, "loss": -.1*s, "direction_error": .1/(s+1), "score": .01*s}
                    for r in range(1, 501) for s in (0, 10)]
            (directory/"direction.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                plot_training(directory, detailed=True)
            self.assertFalse(any("axes sizes collapsed" in str(w.message) for w in observed))
            self.assertTrue((directory/"direction_curves.png").exists())

    def test_default_writes_only_one_overview_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/"policy.jsonl").write_text(json.dumps({
                "round": 0, "source_episodes": 0, "expected_return": .1, "accepted": False}), encoding="utf-8")
            result = plot_training(directory)
            self.assertEqual(result.name, "round_overview.png")
            self.assertEqual({p.name for p in directory.iterdir()}, {"policy.jsonl", "round_overview.png"})
