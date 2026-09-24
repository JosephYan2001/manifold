"""Exercise plots with conflicting cost/sample axes and nested training data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from manifold_project.experiments.common.plotting import build_figure, setup
from manifold_project.experiments.common.reporting import plot_overview, write_csv


class PlottingProtocols(unittest.TestCase):
    def tearDown(self):
        setup().close('all')

    def test_fixed_experiments_use_their_actual_axes_and_seed_intervals(self):
        for experiment, axis_key, xlabel, metric, xs in (
                ('P-E', 'sample_episodes', 'Source training episodes', 'direction_error', [32, 128]),
                ('P-A', 'fit_steps', 'Actor fitting steps', 'forward_kl_validation', [1, 10]),
                ('N-T', 'target_n', 'Target agents', 'discounted_return', [2, 6])):
            rows = [dict(experiment=experiment, method='AN', seed=seed, source_steps_total=999,
                         **{axis_key: x, metric: 1 / x + seed / 100})
                    for seed in (1, 2) for x in xs]
            with TemporaryDirectory() as directory:
                fig = build_figure(directory, rows)
                ax = next(ax for ax in fig.axes if ax.get_xlabel() == xlabel)
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), xs)
                self.assertEqual(len(ax.lines), 1)  # one mean, not interleaved seed trajectories
                self.assertEqual(len(ax.collections), 1)  # independent-seed interval
                one_seed = build_figure(directory, [row for row in rows if row['seed'] == 1])
                one_ax = next(ax for ax in one_seed.axes if ax.get_xlabel() == xlabel)
                self.assertEqual(len(one_ax.collections), 0)

    def test_nested_training_curves_survive_final_summary_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'N-C_source_learning'
            for mode in ('full', 'summary'):
                for seed in (100, 101):
                    rows = [dict(method='AN', seed=seed, source_steps_total=steps,
                                 train_discounted_return=-10 + steps / 100 + seed / 100,
                                 policy_kl_old_to_new=.01, accepted=1, wall_seconds=steps / 10,
                                 step_size=.1 if steps == 100 else .05)
                            for steps in (100, 200, 300)]
                    write_csv(root / mode / f'AN_s{seed}' / 'training.csv', rows)
            final = [dict(experiment='N-C', method='AN', seed=100, target_n=4,
                          source_steps_total=300, discounted_return=-5)]
            fig = build_figure(root, final)
            learning = next(ax for ax in fig.axes if 'train_discounted_return' in ax.get_title())
            self.assertEqual(learning.get_xlabel(), 'All source joint steps')
            self.assertEqual(len(learning.lines), 2)  # input protocols are separate groups
            for line in learning.lines:
                np.testing.assert_array_equal(line.get_xdata(), [100, 200, 300])
            image = plot_overview(root, final)
            from PIL import Image
            with Image.open(image) as png:
                self.assertGreater(png.width, 1500)
                self.assertGreater(png.height, 900)
                self.assertGreater(float(np.asarray(png).std()), 5.)
            self.assertEqual(list(root.glob('*.png')), [root / 'overview.png'])

    def test_snapshot_and_label_protocols_do_not_merge(self):
        rows = [dict(experiment='N-F', method='AN', seed=40, fit_steps=step,
                     budget_fraction=fraction, label_mode=label,
                     independent_forward_kl=1 / (step + 1))
                for fraction in (.25, 1.) for label in ('mc', 'gae') for step in (1, 10)]
        with TemporaryDirectory() as directory:
            fig = build_figure(directory, rows)
            ax = next(ax for ax in fig.axes if 'independent_forward_kl' in ax.get_title())
            self.assertEqual(len(ax.lines), 4)
            self.assertEqual(len(ax.collections), 0)

    def test_information_oracle_is_not_plotted_against_zero_sample_cost(self):
        rows = [dict(experiment='P-I', method=method, source_steps_total=0,
                     preprocessing_loss=loss, local_direction_energy=1 - loss,
                     input_states=states, decomposition_residual=0)
                for method, loss, states in [('full', 0., 20), ('mean_count', .2, 8)]]
        with TemporaryDirectory() as directory:
            fig = build_figure(directory, rows)
            self.assertTrue(any(ax.get_xlabel() == 'Declared diagnostic case' for ax in fig.axes))
            self.assertFalse(any(ax.get_xlabel() == 'All source joint steps' for ax in fig.axes))


if __name__ == '__main__':
    unittest.main()
