"""Inspectable JSON records and independent, budgeted source batches."""
import json
from pathlib import Path
import numpy as np
from ..envs.pair_coordination import sample_episodes

LOG_STREAMS = ('policy', 'rounds', 'checks', 'batches', 'direction',
               'actor_fit', 'data', 'stages', 'updates')


def read_records(path, tolerate_partial=False):
    """Read a legacy stream or its records inside the unified event log."""
    path = Path(path)
    unified = not path.exists() and (path.parent/'events.jsonl').exists()
    source = path.parent/'events.jsonl' if unified else path
    if not source.exists():
        return []
    rows = []
    lines = source.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_partial and index == len(lines)-1:
                break
            raise
        if unified:
            if row['stream'] != path.stem:
                continue
            row = row['record']
        rows.append(row)
    return rows


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def append_json(path, value):
    path = Path(path)
    if path.stem in LOG_STREAMS and (path.parent/'events.jsonl').exists():
        value = {'stream': path.stem, 'record': value}
        path = path.parent/'events.jsonl'
    with path.open("a", encoding="utf-8") as stream:
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
