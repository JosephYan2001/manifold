"""Configure CUDA reproducibility before importing torch; inspect dependencies lazily."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path


def configure_runtime(allow_duplicate_openmp: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if allow_duplicate_openmp:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def metadata() -> dict:
    versions = {}
    for name in ("numpy", "torch", "matplotlib", "mpe2", "rware", "gymnasium"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "experiments").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "versions": versions, "experiment_code_sha256": digest.hexdigest(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "duplicate_openmp_allowed": os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() == "TRUE"}


def configure_torch(config: dict) -> None:
    configure_runtime(bool(config.get("allow_duplicate_openmp", False)))
    import torch
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    deterministic = bool(config.get("deterministic", True))
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    if deterministic and hasattr(torch.backends, "cuda"):
        # Deterministic training uses the math attention implementation.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    device = str(config.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python/PyTorch installation")
