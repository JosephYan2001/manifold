"""Protocol tests for masks, input rights and real native environment endings."""
import importlib.util
import itertools
from copy import deepcopy
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from manifold_project.experiments.applications.envs import make_env


def entity_model(self_dim, entity_dim, action_dim=5, hidden=128, heads=4,
                 relation_layers=2, mode='full', probability_floor=.01, q_max=4., kind='actor'):
    # Test the actual training network, not a parallel test-only architecture.
    from manifold_project.experiments.cooperative_navigation.models import Actor, Direction
    c = dict(observation_protocol='entities_v1', entity_self_dim=self_dim,
             entity_record_dim=entity_dim, action_dim=action_dim, hidden=hidden,
             heads=heads, relation_layers=relation_layers, observation_mode=mode,
             beta=probability_floor, q_max=q_max)
    return (Actor if kind == 'actor' else Direction)(0, c)


@unittest.skipIf(torch is None, "torch is required for entity model tests")
class EntityModelTests(unittest.TestCase):
    def setUp(self):
        self.Actor = entity_model
        self.Direction = lambda *args, **kw: entity_model(*args, **kw, kind='direction')
        torch.manual_seed(17)
        self.obs = {"self": torch.randn(3, 5), "entities": torch.randn(3, 7, 6),
                    "mask": torch.ones(3, 7, dtype=torch.bool)}
        self.obs["entities"][..., :2] = 0
        self.obs["entities"][:, :4, 0] = 1
        self.obs["entities"][:, 4:, 1] = 1

    def test_permutation_and_padding_for_all_modes(self):
        for mode in ("full", "summary", "nearest2", "no_relation"):
            actor = self.Actor(5, 6, hidden=16, heads=2, relation_layers=1, mode=mode).eval()
            reference = actor(self.obs)
            permutation = torch.tensor([6, 1, 4, 0, 3, 5, 2])
            permuted = {"self": self.obs["self"], "entities": self.obs["entities"][:, permutation],
                        "mask": self.obs["mask"][:, permutation]}
            padded = {"self": permuted["self"],
                      "entities": torch.cat([permuted["entities"], torch.full((3, 4, 6), float("nan"))], 1),
                      "mask": torch.cat([permuted["mask"], torch.zeros(3, 4, dtype=torch.bool)], 1)}
            torch.testing.assert_close(reference, actor(permuted), atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(reference, actor(padded), atol=2e-7, rtol=2e-6)
            self.assertTrue(torch.isfinite(actor(padded)).all())

    def test_empty_list_and_fully_masked_list_agree(self):
        for mode in ("full", "summary", "nearest2"):
            actor = self.Actor(5, 6, hidden=16, heads=2, relation_layers=1, mode=mode)
            empty = {"self": self.obs["self"], "entities": torch.zeros(3, 0, 6),
                     "mask": torch.zeros(3, 0, dtype=torch.bool)}
            padded = {"self": self.obs["self"], "entities": torch.full((3, 2, 6), float("nan")),
                      "mask": torch.zeros(3, 2, dtype=torch.bool)}
            torch.testing.assert_close(actor(empty), actor(padded), atol=2e-7, rtol=2e-6)
            # Action-specific readout must remain expressive with no entities.
            self.assertGreater(float(actor(empty).std(-1).max()), 1e-8)
            actor(padded)[:, 0].sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))

    def test_summary_really_discards_individual_records(self):
        actor = self.Actor(5, 6, hidden=16, heads=2, mode="summary")
        other = {key: value.clone() for key, value in self.obs.items()}
        other["entities"][:, 0, 2:] += 3
        other["entities"][:, 1, 2:] -= 3
        torch.testing.assert_close(actor(self.obs), actor(other), atol=2e-7, rtol=2e-6)

    def test_same_parameters_accept_new_entity_count(self):
        actor = self.Actor(5, 6, hidden=16, heads=2)
        initial = {key: value.clone() for key, value in actor.state_dict().items()}
        for size in (1, 3, 11, 15):
            obs = {"self": torch.randn(2, 5), "entities": torch.zeros(2, size, 6),
                   "mask": torch.ones(2, size, dtype=torch.bool)}
            obs["entities"][..., 0] = 1
            probs = actor(obs)
            self.assertEqual(probs.shape, (2, 5))
            torch.testing.assert_close(probs.sum(-1), torch.ones(2))
            self.assertTrue((probs >= 0.01 / 5).all())
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    def test_direction_bound_gauge_and_independence(self):
        actor = self.Actor(5, 6, hidden=16, heads=2)
        direction = self.Direction(5, 6, hidden=16, heads=2, q_max=2)
        before = direction(self.obs).detach().clone()
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.add_(1)
        torch.testing.assert_close(direction(self.obs), before)
        self.assertTrue((before.abs() <= 2).all())
        torch.testing.assert_close(before.sum(-1), torch.zeros(3), atol=1e-6, rtol=0)

    def test_authorized_relation_features_ignore_invalid_endpoints(self):
        from manifold_project.experiments.cooperative_navigation.models.entities import RelationAttention
        positions = torch.tensor([[[0., 0.], [2., 1.], [float("nan"), float("inf")]]])
        valid = torch.tensor([[True, True, False]])
        relation, edges = RelationAttention.relation_features(positions, valid)
        torch.testing.assert_close(relation[0, 0, 1], torch.tensor([2., 1.]))
        torch.testing.assert_close(relation[0, 1, 0], torch.tensor([-2., -1.]))
        self.assertTrue(torch.isfinite(relation).all())
        self.assertEqual(int(edges.sum()), 4)
        self.assertEqual(int(torch.count_nonzero(relation[:, 2])), 0)
        layer = RelationAttention(8, 2)
        tokens = torch.randn(1, 3, 8)
        tokens[:, 2] = float("nan")
        result = layer(tokens, positions, valid)
        torch.testing.assert_close(result[:, :2], layer(tokens[:, :2], positions[:, :2], valid[:, :2]))
        self.assertEqual(int(torch.count_nonzero(result[:, 2])), 0)

    def test_relations_affect_messages_even_with_fixed_node_embeddings(self):
        from manifold_project.experiments.cooperative_navigation.models.entities import RelationAttention
        layer = RelationAttention(8, 2)
        tokens = torch.ones(1, 2, 8)
        valid = torch.ones(1, 2, dtype=torch.bool)
        colocated = torch.zeros(1, 2, 2)
        separated = torch.tensor([[[0., 0.], [1., 2.]]])
        self.assertGreater(float((layer(tokens, separated, valid) - layer(tokens, colocated, valid)).abs().max()), 1e-5)

    def test_type_readout_empty_type_is_exact_zero(self):
        from manifold_project.experiments.cooperative_navigation.models.entities import TypedActionReadout
        readout = TypedActionReadout(8)
        queries, tokens = torch.randn(2, 5, 8), torch.randn(2, 3, 8)
        tokens[0, 2] = float("nan")
        first = torch.tensor([[True, True, False], [False, False, False]])
        second = torch.tensor([[False, False, False], [True, True, True]])
        outputs = readout(queries, tokens, [first, second])
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs))
        self.assertEqual(int(torch.count_nonzero(outputs[0][1])), 0)
        self.assertEqual(int(torch.count_nonzero(outputs[1][0])), 0)
        first_only = readout(queries[:1], tokens[:1, :2], [first[:1, :2], second[:1, :2]])
        torch.testing.assert_close(outputs[0][:1], first_only[0])

    def test_no_relation_ablation_removes_only_interaction_layers(self):
        full = self.Actor(5, 6, hidden=16, heads=2, relation_layers=2)
        ablated = self.Actor(5, 6, hidden=16, heads=2, relation_layers=0)
        kept = {name: value for name, value in full.state_dict().items()
                if not name.startswith("net.relations.")}
        self.assertEqual(set(kept), set(ablated.state_dict()))
        ablated.load_state_dict(kept)
        manually_disabled = deepcopy(full)
        manually_disabled.net.relations = torch.nn.ModuleList()
        torch.testing.assert_close(ablated(self.obs), manually_disabled(self.obs))

    def test_nearest_two_does_not_expose_discarded_entity_count(self):
        actor = self.Actor(5, 6, hidden=16, heads=2, mode="nearest2")
        base = {"self": self.obs["self"][:1], "entities": torch.zeros(1, 4, 6),
                "mask": torch.ones(1, 4, dtype=torch.bool)}
        base["entities"][0, :2, 0] = 1
        base["entities"][0, 2:, 1] = 1
        base["entities"][0, :, 2] = torch.tensor([1., 2., 1., 2.])
        extra = torch.tensor([[[1., 0., 20., 0., 3., 4.], [0., 1., 30., 0., 5., 6.]]])
        expanded = {"self": base["self"], "entities": torch.cat([base["entities"], extra], 1),
                    "mask": torch.ones(1, 6, dtype=torch.bool)}
        torch.testing.assert_close(actor(base), actor(expanded), atol=2e-7, rtol=2e-6)


@unittest.skipIf(importlib.util.find_spec("mpe2") is None, "install mpe2==1.1.1 for real navigation tests")
class NavigationProtocolTests(unittest.TestCase):
    def test_variable_entity_shapes_reward_and_deadline(self):
        for n in (2, 4, 8):
            env = make_env("navigation", {"n_agents": n, "horizon": 1})
            try:
                obs, info = env.reset(seed=71)
                self.assertFalse(env.initial_done)
                self.assertEqual(obs["entities"].shape, (n, 2 * n - 1, 6))
                self.assertEqual(env.state().shape, (env.state_dim,))
                self.assertEqual(int(obs["mask"].sum()), n * (2 * n - 1))
                next_obs, reward, terminal, truncated, info = env.step(np.zeros(n, dtype=int))
                self.assertTrue(terminal)
                self.assertFalse(truncated)
                self.assertEqual(info["end_reason"], "deadline")
                self.assertAlmostEqual(reward, info["reward_distance"] + info["reward_collision"])
                np.testing.assert_array_equal(next_obs["self"][:, -1], 0)
                with self.assertRaises(RuntimeError):
                    env.step(np.zeros(n, dtype=int))
            finally:
                env.close()

    def test_initial_success_stops_without_action(self):
        env = make_env("navigation", {"n_agents": 2, "coverage_radius": 100})
        try:
            _, info = env.reset(seed=2)
            self.assertTrue(env.initial_done)
            self.assertEqual(info["completion_time_capped"], 0)
            self.assertEqual(info["end_reason"], "success")
            with self.assertRaises(RuntimeError):
                env.step([0, 0])
        finally:
            env.close()

    def test_peer_velocity_is_not_added_to_actor(self):
        env = make_env("navigation", {"n_agents": 4})
        try:
            before, _ = env.reset(seed=9)
            env.world.agents[1].state.p_vel[:] = [4, -3]
            raw = {name: env.env.unwrapped.observe(name) for name in env.names}
            after = env._observation(raw)
            np.testing.assert_array_equal(before["self"][0], after["self"][0])
            np.testing.assert_array_equal(before["entities"][0], after["entities"][0])
            self.assertTrue(np.any(env.state() == 4))
        finally:
            env.close()


@unittest.skipIf(importlib.util.find_spec("rware") is None, "install rware==2.0.0 for real warehouse tests")
class WarehouseProtocolTests(unittest.TestCase):
    def test_native_tiny_layout_and_local_grid_preservation(self):
        from rware.warehouse import Warehouse as Native, RewardType
        from manifold_project.experiments.applications.envs.warehouse import TINY_LAYOUT
        native = Native(3, 8, 1, 4, 0, 1, 4, None, None, RewardType.GLOBAL)
        adapter = make_env("warehouse", {"n_agents": 4, "horizon": 1})
        try:
            native.reset(seed=123)
            obs, _ = adapter.reset(seed=123)
            np.testing.assert_array_equal(native.highways, adapter.env.highways)
            self.assertEqual(adapter.env.grid_size, (11, 10))
            self.assertEqual(TINY_LAYOUT.count("x"), 32)
            self.assertEqual(obs["entities"].shape, (4, 9, 11))
            self.assertEqual(int(obs["mask"].sum()), 36)
            self.assertEqual(adapter.state().shape, (adapter.state_dim,))
            for i, agent in enumerate(adapter.env.agents):
                raw = adapter.env._make_obs(agent)
                for j, cell in enumerate(raw["sensors"]):
                    self.assertEqual(obs["entities"][i, j, 4], cell["has_agent"][0])
                    self.assertEqual(obs["entities"][i, j, 9], cell["has_shelf"][0])
                    self.assertEqual(obs["entities"][i, j, 10], cell["shelf_requested"][0])
            _, reward, terminal, truncated, info = adapter.step([0] * 4)
            self.assertEqual(reward, 0)
            self.assertTrue(terminal)
            self.assertFalse(truncated)
            self.assertEqual(info["end_reason"], "deadline")
            self.assertEqual(info["longest_no_delivery_streak"], 1)
            self.assertEqual(info["blocked_forward_actions"], 0)
        finally:
            native.close()
            adapter.close()

    def test_global_delivery_reward_counted_once(self):
        env = make_env("warehouse", {"n_agents": 4, "horizon": 4})
        try:
            env.reset(seed=45)
            requested = env.env.request_queue[0]
            requested.x, requested.y = env.env.goals[0]
            env.env._recalc_grid()
            _, reward, terminal, _, info = env.step([0] * 4)
            self.assertEqual(reward, 1)
            self.assertEqual(info["deliveries"], 1)
            self.assertEqual(info["deliveries_total"], 1)
            self.assertEqual(info["first_delivery_step"], 1)
            self.assertFalse(terminal)
        finally:
            env.close()


class PairEntityProtocolTests(unittest.TestCase):
    def test_reward_matches_analytic_expectation(self):
        for variant in ("normalized", "fixed-edge"):
            for n in (2, 4, 6):
                env = make_env("pair_entities", {"pair_variant": variant}, n_agents=n)
                value = 0.0
                for values in itertools.product((-1, 1), repeat=n):
                    types = np.asarray(values)
                    mass = np.prod(np.where(types > 0, 0.6, 0.4))
                    # Deterministic own-type policy: signed action equals type.
                    value += mass * env.team_reward(types, types)
                expected = 0.16 * n + env.edge_weight * n * (n - 1) / 2 * 0.2 ** 2
                self.assertAlmostEqual(value, expected, places=12)

    def test_complete_list_size_mask_and_ending(self):
        for n in (2, 4, 6):
            env = make_env("pair_entities", {}, n_agents=n)
            obs, _ = env.reset(seed=7)
            self.assertEqual(obs["self"].shape, (n, 3))
            self.assertEqual(obs["entities"].shape, (n, n - 1, 6))
            self.assertEqual(int(obs["mask"].sum()), n * (n - 1))
            self.assertEqual(env.state().shape, (2 * n + 1,))
            for i in range(n):
                expected_positive = int((env.types > 0).sum()) - int(env.types[i] > 0)
                self.assertEqual(int(obs["entities"][i, :, 5].sum()), expected_positive)
            after, reward, terminated, truncated, info = env.step([1] * n)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            np.testing.assert_array_equal(after["self"][:, -1], 0)
            self.assertAlmostEqual(info["per_agent_reward"], reward / n)
            with self.assertRaises(RuntimeError):
                env.step([1] * n)

    @unittest.skipIf(torch is None, "torch is required for cross-size actor execution")
    def test_frozen_two_action_actor_executes_other_sizes(self):
        from manifold_project.experiments.cooperative_navigation.models.entities import tensor_observation
        actor = entity_model(3, 6, action_dim=2, hidden=16, heads=2).eval()
        before = {key: value.clone() for key, value in actor.state_dict().items()}
        for n in (4, 2, 6):
            env = make_env("pair_entities", {}, n_agents=n)
            obs, _ = env.reset(seed=73)
            with torch.no_grad():
                probabilities = actor(tensor_observation(obs))
            self.assertEqual(probabilities.shape, (n, 2))
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(n))
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
