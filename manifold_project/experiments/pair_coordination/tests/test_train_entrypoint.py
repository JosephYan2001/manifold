"""检查两种启动方式及直接脚本的包上下文。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainEntrypointTests(unittest.TestCase):
    def test_script_and_module_resolve_same_configuration(self):
        root = Path(__file__).resolve().parents[4]
        folder = root / "manifold_project/experiments/pair_coordination"
        arguments = ["--config", str(folder / "configs/pair_source.json"),
                     "--training-config", str(folder / "configs/train_mlp.json"),
                     "--acceptance", "empirical", "--plot", "--print-config"]
        results = []
        for entry in ([str(folder / "train.py")],
                      ["-m", "manifold_project.experiments.pair_coordination.train"]):
            result = subprocess.run([sys.executable, *entry, *arguments], cwd=root,
                                    capture_output=True, text=True, check=True, timeout=30)
            results.append(json.loads(result.stdout))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["runs"][0]["acceptance"], "empirical")

    def test_absolute_script_outside_repository(self):
        script = Path(__file__).resolve().parents[1] / "train.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script), "--print-config"],
                                    cwd=directory, capture_output=True, text=True,
                                    check=True, timeout=30)
        self.assertEqual(json.loads(result.stdout)["source"]["n_agents"], 4)
