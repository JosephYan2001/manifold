"""Inspectable JSON records and independent, budgeted source batches."""
import json
from pathlib import Path
import numpy as np
from ..envs.pair_coordination import sample_episodes


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


class SourceSampler:
    def __init__(self, config, settings, directory):
        self.config, self.settings, self.directory = config, settings, Path(directory)
        self.used = 0
        self.calls = 0

    def sample(self, policy, count, purpose, round_index):
        if self.used + count > self.settings.budget:
            raise ValueError("Source episode budget exhausted")
        seed = int(np.random.SeedSequence([self.settings.seed, self.calls]).generate_state(1)[0])
        batch = sample_episodes(self.config, policy, count, seed=seed)
        filename = f"batch_{self.calls:04d}.npz" if self.settings.save_batches else None
        if filename:
            np.savez_compressed(self.directory/filename, **batch,
                                episode_ids=np.arange(self.used, self.used+count))
        self.used += count
        self.calls += 1
        # 崩溃或中断后可核对已采集但尚未提交轮次的交互成本。
        from .checkpoints import atomic_json
        atomic_json(self.directory/"sampling_progress.json", {"source_episodes": self.used, "sampler_calls": self.calls})
        append_json(self.directory/"batches.jsonl", {
            "file": filename, "purpose": purpose, "round": round_index,
            "seed": seed, "episodes": count, "cumulative_episodes": self.used,
            "policy": policy.tolist()})
        for array in batch.values():
            array.setflags(write=False)
        return batch
