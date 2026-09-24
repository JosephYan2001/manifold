"""The executable matrix corresponding to docs/总体实验方案.md."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    environment: str
    kind: str
    methods: tuple[str, ...]
    source_experiment: str = ""


CORE = ("AN", "SA", "DA")
APPLICATION = CORE + ("MAPPO-E", "IPPO-E")
EXPERIMENTS = {item.id: item for item in (
    Experiment("P-E", "direction_estimation", "pair", "finite", ("AN", "SA")),
    Experiment("P-A", "actor_realization", "pair", "finite", CORE),
    Experiment("P-T", "conditional_transfer", "pair", "finite", CORE),
    Experiment("P-I", "information_loss", "pair", "finite", ()),
    Experiment("P-L", "source_learning", "pair", "finite", CORE),
    Experiment("P-B", "boundary_cases", "pair", "finite", ()),
    Experiment("P-TB", "entity_extrapolation", "pair_entities", "train_transfer", CORE),
    Experiment("T-V", "trajectory_variance", "two_step", "finite", ("AN", "SA")),
    Experiment("T-L", "sequential_learning", "two_step", "finite", CORE),
    Experiment("N-C", "source_learning", "navigation", "train", APPLICATION),
    Experiment("N-T", "frozen_transfer", "navigation", "evaluate", APPLICATION, "N-C"),
    Experiment("N-F", "actor_realization", "navigation", "replay", ("AN",), "N-C"),
    Experiment("N-I", "observation_ablation", "navigation", "input_train", ("AN", "MAPPO-E")),
    Experiment("N-R", "relation_ablation", "navigation", "train", ("AN",)),
    Experiment("N-Gd", "without_direction_check", "navigation", "train", ("AN",)),
    Experiment("N-Gr", "without_return_check", "navigation", "train", ("AN",)),
    Experiment("N-D", "initial_density_control", "navigation", "evaluate", APPLICATION, "N-C"),
    Experiment("W-C", "source_learning", "warehouse", "train", APPLICATION),
    Experiment("W-T", "frozen_transfer", "warehouse", "evaluate", APPLICATION, "W-C"),
    Experiment("W-L", "request_load_control", "warehouse", "evaluate", APPLICATION, "W-C"),
    Experiment("W-F", "actor_realization", "warehouse", "replay", ("AN",), "W-C"),
)}


def get_experiment(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError:
        raise ValueError(f"Unknown experiment {name!r}; choose from {', '.join(EXPERIMENTS)}") from None
