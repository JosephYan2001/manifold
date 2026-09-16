import json
from pathlib import Path
import tempfile
import unittest
from manifold_project.experiments.pair_coordination.analysis.round_overview import collect_overview, snapshot_records


class RoundOverviewTests(unittest.TestCase):
    def test_only_committed_rounds_and_last_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/"checkpoints").mkdir()
            (p/"checkpoints/latest.json").write_text('{"round": 2}')
            policies = [{"round": i, "source_episodes": i*10, "expected_return": 1., "accepted": False} for i in range(4)]
            direction = [{"round": 1, "step": 10, "loss": -.5, "direction_error": .1},
                         {"round": 1, "step": 0, "loss": 0., "direction_error": 1.}]
            checks = [{"round": 1, "kind": "return", "attempt": 0, "fit_kl": .1},
                      {"round": 1, "kind": "return", "attempt": 1, "fit_kl": .2}]
            for name, rows in [("policy", policies), ("direction", direction), ("checks", checks)]:
                (p/f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            rows = collect_overview(p)
            self.assertEqual([r["round"] for r in rows], [0, 1, 2])
            self.assertEqual(rows[1]["direction_final_loss"], -.5)
            self.assertEqual(rows[1]["last_candidate_fit_kl"], .2)
            self.assertIsNone(rows[2]["last_candidate_fit_kl"])
            self.assertEqual(rows[1]["expected_return"], 1.)
            # New rolling final takes precedence over an old latest file.
            (p/"checkpoints/final.json").write_text('{"round": 1}')
            self.assertEqual([r["round"] for r in collect_overview(p)], [0, 1])

    def test_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"log.jsonl"
            p.write_text('{"round":1}\n{"round":')
            self.assertEqual(snapshot_records(p), [{"round": 1}])
