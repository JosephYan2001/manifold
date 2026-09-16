"""Budgets fixed before training; no target-environment configuration."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainConfig:
    device: str = "auto"
    algorithm: str = "direction"
    direction_check_enabled: bool = True
    return_check_enabled: bool = True
    save_update_logs: bool = True
    save_batches: bool = False
    save_direction_snapshots: bool = False
    checkpoint_every: int = 5  # Legacy config/checkpoint compatibility; no longer controls retention.
    actor_model: str = "mlp"
    direction_model: str = "mlp"
    hidden_width: int = 32
    hidden_depth: int = 2
    mode: str = "direction"
    method: str = "analytic"
    baseline: str = "zero"
    acceptance: str = "hoeffding"
    seed: int = 42
    episodes: int = 1024
    batch_size_episodes: int = 0
    direction_epochs: int = 0
    actor_epochs: int = 0
    critic_episodes: int = 512
    check_episodes: int = 1024
    direction_steps: int = 200
    fit_steps: int = 100
    log_every: int = 10
    rounds: int = 5
    attempts: int = 2
    budget: int = 50000
    direction_lr: float = .05
    fit_lr: float = .05
    q_max: float = 1.
    beta: float = .02
    eta: float = .2
    alpha: float = .05

    def __post_init__(self):
        if self.algorithm not in ("direction", "pg"):
            raise ValueError("algorithm must be direction or pg")
        for name in ("direction_check_enabled", "return_check_enabled", "save_update_logs"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.algorithm == "pg" and self.mode != "policy":
            raise ValueError("PG requires mode=policy")
        if not isinstance(self.device, str) or not (
                self.device in ("auto", "cpu", "cuda") or
                (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("device 请选择 auto、cpu 或 cuda:编号")
        if type(self.save_batches) is not bool or type(self.save_direction_snapshots) is not bool:
            raise ValueError("保存开关必须为布尔值")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 1:
            raise ValueError("checkpoint_every 必须为正整数")
        for key in ("batch_size_episodes", "direction_epochs", "actor_epochs"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.batch_size_episodes and (not self.direction_epochs or not self.actor_epochs):
            raise ValueError("Mini-batches require positive direction_epochs and actor_epochs")
        for key, choices in {"actor_model": ("mlp", "table"),
                             "direction_model": ("mlp", "table"),
                             "mode": ("direction", "policy"),
                             "method": ("analytic", "sampled"),
                             "baseline": ("zero", "table"),
                             "acceptance": ("hoeffding", "empirical")}.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"Invalid {key}")
        for key in ("episodes", "critic_episodes", "check_episodes", "direction_steps",
                    "fit_steps", "log_every", "rounds", "attempts", "budget",
                    "hidden_width", "hidden_depth"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("direction_lr", "fit_lr", "q_max", "beta", "eta", "alpha"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        if self.beta >= .7 or self.alpha >= 1:
            raise ValueError("Require beta < .7 and alpha < 1")
