import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import torch


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        writer(name)
        for attempt in range(8):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.05*2**attempt, .8))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_json(path, value):
    atomic(path, lambda p: Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                                       allow_nan=False), encoding='utf-8'))


def save_pt(path, value):
    atomic(path, lambda p: torch.save(value, p))


def load_pt(path, device='cpu'):
    # Only load checkpoints created locally by this runner.
    return torch.load(path, map_location=device, weights_only=False)


def event(path, stream, **record):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps({'stream': stream, 'record': record}, ensure_ascii=False, allow_nan=False)+'\n')
