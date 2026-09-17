import contextlib
from dataclasses import replace
import io
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from manifold_project.experiments.pair_coordination.envs.pair_coordination import PairConfig, PairCoordinationEnv, sample_episodes
from manifold_project.experiments.pair_coordination.evaluation.exact_pair import exact_evaluate, closed_form_local_direction
from manifold_project.experiments.pair_coordination.models.actor import TableActor
from manifold_project.experiments.pair_coordination.experiments import mechanisms
from manifold_project.experiments.pair_coordination.experiments.protocol import conditions_for, selected_experiments, condition_settings
from manifold_project.experiments.pair_coordination.experiments.reporting import (bootstrap, summarize_runs,
    binomial_interval, paired_direction_summary, summarize_mechanism)
from manifold_project.experiments.pair_coordination.run_experiments import build_parser, resolve_plan, execute
from manifold_project.experiments.pair_coordination.training.config import TrainConfig
from manifold_project.experiments.pair_coordination.training.plan import training_plan
from manifold_project.experiments.pair_coordination.training.runner import run_training


class MechanismTests(unittest.TestCase):
    def test_check_progress_counts_sampling_once_without_changing_results(self):
        config = {'check_sizes':[2,4], 'data_seeds':[0,1], 'q_max':1.}
        source, training = PairConfig(), TrainConfig()
        expected = mechanisms.check_rules(source,config,training)
        progress = []
        actual = mechanisms.check_rules(source,config,training,progress=progress.append)
        self.assertEqual(actual,expected)
        self.assertEqual(progress[0]['completed_episodes'],0)
        self.assertEqual(progress[-1]['completed_episodes'],9*(2+4)*2)
        self.assertEqual(progress[-1]['total_episodes'],9*(2+4)*2)
        self.assertEqual(progress[-1]['completed_batches'],6*2*2)
        self.assertEqual(progress[-1]['eta_seconds'],0)
        self.assertEqual(len(actual),2*6*2*2)

    def test_exact_advantage_is_conditional_on_joint_types(self):
        source = PairConfig(n_agents=2)
        policy = np.array([[.8, .2], [.2, .8]])
        batch = sample_episodes(source, policy, 24, seed=5)
        labels = mechanisms.exact_advantages(source, policy, batch)
        env = PairCoordinationEnv(source)
        actions = np.array(list(product(range(2), repeat=2)))
        for i, x in enumerate(batch["local_types"]):
            probabilities = np.prod(policy[x[None, :], actions], axis=1)
            expected = probabilities @ env.reward_batch(x, actions)
            self.assertAlmostEqual(labels[i], batch["team_rewards"][i]-expected)

    def test_transport_bounds_match_enumeration(self):
        source = PairConfig()
        policy = np.array([[.8, .2], [.2, .8]])
        q = np.array([[-.15, .15], [-.35, .35]])
        for target in (replace(source, n_agents=2), replace(source, type_probs=(.3, .7)),
                       replace(source, interaction_strength=1.4)):
            metrics = mechanisms.transport_metrics(source, target, policy, q)
            self.assertTrue(metrics["bound_holds"])
            self.assertAlmostEqual(metrics["target_error"], exact_evaluate(target, policy, q)["direction_error"])
        self.assertEqual(mechanisms.transport_metrics(source, replace(source, n_agents=8), policy, q)["direction_shift"], 0.)

    def test_actor_and_second_order_identities(self):
        config = {"beta": .02, "step_sizes": [1e-6, .1], "fit_steps": [2, 4], "actor_lr": .05,
                  "epsilons": [.01, .05, .1]}
        rows = mechanisms.actor_realization(PairConfig(), config)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertGreaterEqual(row["optimization_excess_kl"], -1e-12)
            if row["eta"] == 1e-6:
                self.assertAlmostEqual(row["target_difference_quotient"], row["exact_derivative"], places=6)
        self.assertTrue(all(row["identity_holds"] for row in mechanisms.second_order(config)))

    def test_direction_pairing_and_check_nesting(self):
        args = build_parser().parse_args(["--profile", "smoke"])
        source, base, config, _ = resolve_plan(args)
        with contextlib.redirect_stdout(io.StringIO()):
            rows = mechanisms.direction_estimation(source, config["mechanisms"])
        for row in rows:
            paired = [r for r in rows if (r["actor_id"], r["sample_size"], r["seed"]) ==
                      (row["actor_id"], row["sample_size"], row["seed"])]
            self.assertEqual(len(paired), 4)
            self.assertEqual(len({r["dataset_sha256"] for r in paired}), 1)
        checks = mechanisms.check_rules(source, config["mechanisms"], base)
        pairs = {}
        for row in checks:
            key = row["kind"], row["truth_class"], row["sample_size"], row["seed"]
            pairs.setdefault(key, {})[row["mode"]] = row
            if row["truth_class"]:
                self.assertGreater(row["true_value"]*row["truth_class"], 0)
        for modes in pairs.values():
            self.assertLessEqual(modes["hoeffding"]["accepted"], modes["empirical"]["accepted"])


class ProtocolTests(unittest.TestCase):
    def test_binomial_zero_events_has_positive_upper_bound(self):
        zero = binomial_interval([0]*200)
        self.assertEqual(zero['mean'], 0)
        self.assertAlmostEqual(zero['ci_high'], 0.0188453263772666)
        one = binomial_interval([1]*200)
        self.assertAlmostEqual(one['ci_low'], 1-zero['ci_high'])

    def test_paired_direction_interval_and_dataset_guard(self):
        rows = []
        for seed, offset in enumerate([1., 10., 100.]):
            for method, error in [('analytic', offset), ('sampled', offset+2)]:
                rows.append(dict(actor_id=1, label='return', sample_size=128, seed=seed,
                                 method=method, direction_error=error, dataset_sha256=str(seed)))
        summary = paired_direction_summary(rows, 100)[0]
        self.assertEqual(summary['mean'], -2)
        self.assertEqual(summary['ci_low'], -2)
        rows[-1]['dataset_sha256'] = 'mismatched'
        with self.assertRaises(ValueError):
            paired_direction_summary(rows, 100)

    def test_paper_plans_have_independent_repetitions(self):
        root = Path(__file__).resolve().parents[1]/'configs'
        for file, experiments, key, expected in [
            ('training', ['P-C','P-A'], 'training_runs', 60),
            ('mechanisms', ['P-M1','P-M3'], 'P-M1_fits', 640),
            ('checks', ['P-H'], 'P-H_records', 9600),
        ]:
            args = build_parser().parse_args(['--suite-config',str(root/f'suite_paper_{file}.json'),
                                              '--experiments',*experiments,'--dry-run'])
            _, _, config, plan = resolve_plan(args)
            self.assertEqual(plan['counts'][key], expected)
            self.assertEqual(config['training_seeds'], list(range(100,110)))
            if file == 'mechanisms':
                self.assertEqual(plan['counts']['P-M3_records'], 180)

    def test_plan_deduplicates_and_declares_dependencies(self):
        args = build_parser().parse_args(["--profile", "formal", "--dry-run"])
        _, _, config, plan = resolve_plan(args)
        self.assertEqual(plan["counts"]["training_runs"], 60)
        self.assertEqual(plan["counts"]["P-M1_fits"], 640)
        self.assertEqual(selected_experiments(["P-M3"]), ["P-M1", "P-M3"])
        self.assertEqual(len(conditions_for(["P-C", "P-A"])), 6)
        self.assertEqual(config["protocol_status"], "candidate_not_frozen")

    def test_ablation_minimum_cost_and_fit_budget(self):
        base = TrainConfig(mode="policy", episodes=16, check_episodes=8, actor_epochs=5, direction_epochs=5)
        expected = {"ours": 40, "pg": 16, "no_direction_check": 32, "no_return_check": 24}
        for condition, cost in expected.items():
            settings = condition_settings(base, condition, 0)
            self.assertEqual(training_plan(PairConfig(), settings)["minimum_to_start_round"], cost)
        self.assertEqual(condition_settings(base, "fit_quarter", 0).actor_epochs, 2)

    def test_failure_counts_and_no_single_seed_ci(self):
        rows = [{"condition": "pg", "status": "failed"},
                {"condition": "pg", "status": "complete", "final_return": 1.}]
        row = summarize_runs(rows, 100)[0]
        self.assertEqual(row["failed_or_interrupted"], 1)
        self.assertEqual(row["count"], 1)
        self.assertIsNone(row["ci_low"])
        self.assertIsNone(bootstrap([])["mean"])

    def test_mechanism_resume_reuses_finished_source_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/"suite"
            args = build_parser().parse_args(["--experiments", "P-M3", "--output-dir", str(out)])
            values = resolve_plan(args)
            self.assertEqual(execute(args, *values), 0)
            before = (out/"P-M1/records.csv").read_bytes()
            args.resume_suite = True
            with patch.object(mechanisms, "direction_estimation", side_effect=AssertionError("should reuse")):
                self.assertEqual(execute(args, *values), 0)
            self.assertEqual(before, (out/"P-M1/records.csv").read_bytes())

    def test_single_mechanism_plot_has_only_one_png(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/"suite"
            args = build_parser().parse_args(["--experiments", "P-M4", "--output-dir", str(out), "--plot"])
            self.assertEqual(execute(args, *resolve_plan(args)), 0)
            self.assertEqual([p.name for p in out.glob("*.png")], ["overview.png"])
            self.assertFalse(list(out.rglob("*.svg")))


class TrainingVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_pg_gradient_matches_exact_team_return(self):
        import torch
        from manifold_project.experiments.pair_coordination.models.torch_models import TorchModel
        from manifold_project.experiments.pair_coordination.training.policy_gradient import pg_loss
        source = PairConfig(n_agents=2)
        actor = TorchModel(TableActor(2, logits=[-.4, .7]), "cpu")
        x, actions, labels = [], [], []
        p = actor.table()
        env = PairCoordinationEnv(source)
        for types in product(range(2), repeat=2):
            for u in product(range(2), repeat=2):
                probability = np.prod([source.type_probs[t]*p[t, a] for t, a in zip(types, u)])
                x.append(types)
                actions.append(u)
                labels.append(env.reward(np.array(types), np.array(u))*probability*16)
        loss = pg_loss(actor, {"local_types": np.array(x), "actions": np.array(actions)}, np.array(labels))
        actual = torch.autograd.grad(loss, actor.flat)[0]
        means = actor()[:, 1]-actor()[:, 0]
        weighted = actor.tensor(source.type_probs)*means
        value = source.n_agents*(weighted @ actor.tensor(source.local_bias)
                                + source.interaction_strength/2*(weighted @ actor.tensor(source.pair_payoff) @ weighted))
        expected = torch.autograd.grad(-value, actor.flat)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_pg_resume_restores_adam_and_both_models(self):
        settings = TrainConfig(algorithm="pg", mode="policy", device="cpu", hidden_width=4,
                               episodes=16, budget=64, rounds=4, save_update_logs=False)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            full = run_training(PairConfig(), settings, root/"full")
            run_training(PairConfig(), settings, root/"pause", stop_after_round=2)
            resumed = run_training(PairConfig(), settings, root/"resumed", resume=root/"pause/checkpoints/final.json")
            self.assertEqual(full["actor"], resumed["actor"])
            for name in ("best", "final"):
                a = json.loads((root/f"full/checkpoints/{name}.json").read_text(encoding="utf-8"))
                b = json.loads((root/f"resumed/checkpoints/{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(a, b)

    def test_disabled_checks_have_zero_sampling_cost(self):
        from manifold_project.experiments.pair_coordination.experiments.reporting import read_jsonl
        base = TrainConfig(mode="policy", device="cpu", hidden_width=4, episodes=16, check_episodes=16,
                           direction_steps=4, fit_steps=4, rounds=2, budget=128, save_update_logs=False)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for condition, absent in (("no_direction_check", {"direction_check"}),
                                      ("no_return_check", {"return_old", "return_candidate"}),
                                      ("pg", {"direction_train", "direction_check", "return_old", "return_candidate"})):
                directory = Path(tmp)/condition
                settings = condition_settings(base, condition, 0)
                run_training(PairConfig(), settings, directory)
                batches = read_jsonl(directory/"batches.jsonl")
                self.assertFalse({r["purpose"] for r in batches} & absent)
                self.assertLessEqual(sum(r["episodes"] for r in batches), settings.budget)
                self.assertFalse((directory/"updates.jsonl").exists())
                self.assertEqual({p.name for p in (directory/"checkpoints").iterdir()}, {"best.json", "final.json"})
