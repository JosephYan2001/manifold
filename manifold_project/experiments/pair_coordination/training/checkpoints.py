"""轮次边界 checkpoint；临时文件写完后原子替换。"""
import json
import os
from pathlib import Path
import tempfile
import time
import warnings
from dataclasses import asdict
from ..evaluation.exact_pair import closed_form_expected_return


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows 文件监视器、同步或扫描程序可能短暂占用目标文件。
        # 保留原文件并重试原子替换；持续拒绝访问时仍报告失败。
        for attempt in range(8):
            try:
                os.replace(name, path)
                break
            except OSError as error:
                transient = isinstance(error, PermissionError) or getattr(error, "winerror", None) in (5, 32, 33)
                if not transient or attempt == 7:
                    if hasattr(error, "add_note"):
                        error.add_note(f"原子替换失败：{path}；已尝试 {attempt+1} 次，原文件未主动删除。")
                    raise
                time.sleep(min(.05*2**attempt, .8))
    finally:
        if os.path.exists(name):
            try:
                os.unlink(name)
            except OSError as cleanup_error:
                warnings.warn(f"临时文件暂时无法清理：{name}；{cleanup_error}", RuntimeWarning)


def save_checkpoint(directory, source, settings, actor, round_index, sampler, accepted, best=None, optimizer_state=None, *, legacy_alias=True):
    state = {"format_version": 1, "boundary": "completed_round", "actor": actor.state(),
             "source": asdict(source), "training": asdict(settings), "round": round_index,
             "source_episodes": sampler.used, "sampler_calls": sampler.calls, "accepted": accepted,
             "selection_metric": "source_exact_expected_return",
             "selection_score": closed_form_expected_return(source, actor.table())}
    folder = Path(directory)/"checkpoints"
    if optimizer_state is not None:
        state["optimizer_state"] = optimizer_state
    improved = best is None or state["selection_score"] > best["selection_score"]
    if improved:
        best = state.copy()
    # Keep the best snapshot inside final so resuming a copied file also
    # preserves the historical best. The snapshot contains no nested history.
    state["best_checkpoint"] = best
    atomic_json(folder/"final.json", state)
    if improved or not (folder/"best.json").exists():
        atomic_json(folder/"best.json", best)
    if legacy_alias:
        atomic_json(Path(directory)/"actor.json", state)
    return state
