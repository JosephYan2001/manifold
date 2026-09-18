"""Monitoring must leave training and sampling unchanged."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from manifold_project.experiments.cooperative_navigation.tests.test_protocol import tiny
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.evaluation.monitoring import (
    EventHistory, TrainingMonitor, plot_monitor, progress_line)


class MonitoringTests(unittest.TestCase):
    def test_plotting_preserves_training_and_refreshes_final(self):
        c = tiny()
        c['eval_fractions'] = [0, 1]
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            plain = Runner(root/'plain', c, 'mappo', 40)
            plain.run()
            output = root/'training_monitor.png'
            monitor = TrainingMonitor(output, every=2)
            refreshes = []
            def render(*args):
                refreshes.append((len(args[4]), args[6]))
                plot_monitor(*args)
            with patch('manifold_project.experiments.cooperative_navigation.evaluation.monitoring.plot_monitor',
                       side_effect=render):
                observed = Runner(root/'observed', c, 'mappo', 40, progress=monitor)
                observed.run()
            self.assertEqual(refreshes, [(0, False), (2, False), (4, False), (5, True)])
            self.assertTrue(output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertEqual(list(root.glob('*.png')), [output])
            torch.testing.assert_close(plain.actor.state_dict(), observed.actor.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(plain.critic.state_dict(), observed.critic.state_dict(), rtol=0, atol=0)
            self.assertEqual(plain.curves, observed.curves)
            self.assertEqual(plain.sampler.costs, observed.sampler.costs)
            self.assertEqual(plain.optim_steps, observed.optim_steps)
            self.assertIn('last_eval@50', progress_line('mappo', 40, 50, observed.rounds[-1], observed.curves[-1]))

    def test_old_log_enrichment_and_partial_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'events.jsonl'
            events = [
                {'stream':'train_batch', 'record':{'round':1, 'J_mean':-4.2}},
                {'stream':'optimization', 'record':{'round':1, 'module':'critic', 'value_mse':3.5}},
                {'stream':'commit', 'record':{'round':2, 'completed_round':1, 'source_steps':25,
                                            'accepted':False, 'direction_pass':False}},
            ]
            lines = [json.dumps(e).encode()+b'\n' for e in events]
            path.write_bytes(b''.join(lines[:2])+lines[2][:20])
            reader = EventHistory(path)
            self.assertEqual(reader.read(), 0)
            self.assertEqual(reader.rounds, {})
            with path.open('ab') as stream:
                stream.write(lines[2][20:])
            self.assertEqual(reader.read(), 1)
            self.assertEqual(reader.read(), 0)
            self.assertEqual(reader.rounds[1]['train_J'], -4.2)
            self.assertEqual(reader.rounds[1]['critic_mse'], 3.5)
            self.assertEqual(reader.rounds[1]['round'], 1)
            self.assertIn('direction-rejected', progress_line('ours', 40, 50, reader.rounds[1]))

    def test_plot_failure_does_not_abort_committed_training(self):
        c = tiny()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            runner = Runner(folder, c, 'ours', 40, progress=TrainingMonitor(Path(folder)/'monitor.png'))
            with patch('manifold_project.experiments.cooperative_navigation.evaluation.monitoring.plot_monitor',
                       side_effect=PermissionError('locked image')):
                runner.run()
            self.assertTrue(runner.complete)
            self.assertFalse((Path(folder)/'failure.json').exists())
            self.assertIn('train_J', runner.rounds[0])
            self.assertIn('critic_mse', runner.rounds[0])


if __name__ == '__main__':
    unittest.main()
