"""Read a frozen actor (or a matched direction snapshot); never update it."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .envs.pair_coordination import PairConfig, sample_episodes
from .models.mlp import load_actor
from .training.runner import diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--n-agents", type=int)
    parser.add_argument("--config", type=Path, help="Frozen evaluation environment")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    state = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    config = PairConfig.from_json(args.config) if args.config else PairConfig(**state["source"])
    if args.n_agents is not None:
        config = replace(config, n_agents=args.n_agents)
    actor = load_actor(state.get("actor", state.get("reference_actor")))
    if config.n_types != actor.n_types:
        raise ValueError("Target type vocabulary must match the frozen actor")
    q = np.asarray(state["direction"]["table"]) if "direction" in state else None
    data = sample_episodes(config, actor.table(), args.episodes, args.seed)
    mean = float(data["team_rewards"].mean())
    exact = diagnostics(config, actor.table(), q)
    if q is None:
        exact = {k: v for k, v in exact.items() if k in ("expected_return", "exact_skipped")}
    if "expected_return" in exact:
        exact["exact_return_per_agent"] = exact["expected_return"]/config.n_agents
    print(json.dumps({"n_agents": config.n_agents, "episodes": args.episodes, "seed": args.seed,
                      "sample_team_return": mean, "sample_return_per_agent": mean/config.n_agents,
                      "matched_direction": q is not None,
                      **exact}, indent=2))


if __name__ == "__main__":
    main()
