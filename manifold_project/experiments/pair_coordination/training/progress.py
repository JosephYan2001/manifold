"""阶段日志与耗时；每个采样批次的成本由独立账目记录。"""
from time import perf_counter
from pathlib import Path
from .logging import append_json
from .checkpoints import atomic_json


class Progress:
    def __init__(self, directory, sampler):
        self.directory, self.sampler = Path(directory), sampler
        self.active = None

    def start(self, round_index, stage, attempt=None):
        self.finish()
        self.active = {"round": round_index, "stage": stage, "attempt": attempt,
                       "source_episodes_start": self.sampler.used}
        self.started = perf_counter()
        try:
            atomic_json(self.directory/"progress.json", self.active)
        except OSError as error:
            error.training_stage = self.active.copy()
            raise

    def finish(self):
        if self.active is not None:
            append_json(self.directory/"stages.jsonl", {**self.active, "seconds": perf_counter()-self.started,
                        "source_episodes": self.sampler.used-self.active["source_episodes_start"]})
            self.active = None
