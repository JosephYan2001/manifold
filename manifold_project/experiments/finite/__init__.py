"""Finite direction-learning experiments from the 2026-09-24 protocol.

Public integration API: run_experiment(experiment, config, output) -> dict.
The legacy ZIP is a reference, not an import/runtime dependency.
"""
from pathlib import Path

from .diagnostics import (actor_realization, boundaries, estimation, information,
                          temporal_variance, transfer)
from .numerics import write_csv
from .training import train
from .linked_realization import linked_realization

EXPERIMENTS = ("P-E", "P-A", "P-EA", "P-T", "P-I", "P-L", "P-B", "T-V", "T-L")


def run_experiment(experiment: str, config: dict, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if experiment in ("P-L", "T-L"):
        records, training_rows, evaluation_rows = train(config, output, temporal=experiment == "T-L")
        return {"records": records, "status": "completed", "training_rows": len(training_rows),
                "evaluation_episodes": len(evaluation_rows), "checkpoint_format": "legacy_json_resumable" if experiment == 'P-L' else "npz_actor_only",
                "execution_backend": "restored_pair_runner" if experiment == 'P-L' else "numpy_cpu",
                "evaluation_role": "report_only_never_checkpoint_selection"}
    runners = {"P-E": estimation, "P-A": actor_realization, "P-EA": linked_realization, "P-T": transfer,
               "P-I": information, "P-B": boundaries, "T-V": temporal_variance}
    if experiment not in runners:
        raise ValueError(f"Finite experiment {experiment!r} is not implemented; available: {', '.join(EXPERIMENTS)}")
    records = runners[experiment](dict(config))
    write_csv(output / "diagnostics.csv", records)
    return {"records": records, "status": "completed", "diagnostics_rows": len(records),
            "execution_backend": "numpy_cpu",
            "implementation_scope": "exact_input_mapping_only" if experiment == "P-I" else "finite_protocol_v1"}
