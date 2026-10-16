"""Self-contained fixed targets: no environment sampling or direction refit on replay."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .actor_fit import minibatch_orders
from .diagnostics import cpu_copy
from .storage import save_pt

FORMAT = 'navigation-fixed-actor-target-v1'


def model_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root/'models').glob('*.py')) + [root/'training/ppo.py']
    return {path.relative_to(root).as_posix(): hashlib.sha256(
        path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for path in paths}


class Target(nn.Module):
    def __init__(self, old, direction, eta):
        super().__init__()
        self.old, self.direction, self.eta = old, direction, eta

    def forward(self, x):
        return torch.softmax(self.old(x).log()+self.eta*self.direction(x), -1)


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            digest.update(str((str(data.dtype), tuple(data.shape))).encode())
            digest.update(data.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else deepcopy(value)
            for key, value in batch.items()}


def save_fixed_target(path, runner, old, critic, direction, batch, validation, *, eta,
                      direction_epochs, fit_epochs, provenance, evaluation_seed, evaluation_episodes):
    packet = cpu_copy(dict(format=FORMAT, config=runner.c, data_seed=runner.seed,
        round=runner.round, input_dim=runner.input_dim, state_dim=runner.state_dim,
        actor=old.state_dict(), critic=critic.state_dict(), direction=direction.state_dict(),
        batch=batch, validation=validation, eta=eta, direction_epochs=direction_epochs,
        direction_updates=direction_epochs*((len(batch['reward'])+runner.c['minibatch_episodes']-1)//runner.c['minibatch_episodes']),
        fit_epochs=fit_epochs, fit_orders=minibatch_orders(runner.seed, runner.round+1, len(batch['reward']), fit_epochs),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        provenance=provenance, evaluation_seed=evaluation_seed, evaluation_episodes=evaluation_episodes,
        optimizer_initialization='fresh Adam with zero moments for each learning rate',
        weight_rule='discounted valid transitions, fixed episode-horizon denominator',
        model_code_sha256=model_fingerprint(),
        torch_version=str(torch.__version__)))
    packet['config_sha256'] = hashlib.sha256(json.dumps(packet['config'],sort_keys=True).encode()).hexdigest()
    packet['frozen_sha256'] = fingerprint({k:packet[k] for k in ('actor','critic','direction','batch','validation','eta','fit_orders')})
    save_pt(path, packet)
    return packet


def validate_packet(packet):
    if packet.get('format') != FORMAT:
        raise ValueError('Expected a fixed Actor target snapshot, not a training checkpoint')
    if packet['model_code_sha256'] != model_fingerprint():
        raise ValueError('Model/weighting code changed; replay requires the original fixed-target semantics')
    config_hash = hashlib.sha256(json.dumps(packet['config'],sort_keys=True).encode()).hexdigest()
    if config_hash != packet['config_sha256']:
        raise ValueError('Fixed target configuration checksum mismatch')
    if fingerprint({k:packet[k] for k in ('actor','critic','direction','batch','validation','eta','fit_orders')}) != packet['frozen_sha256']:
        raise ValueError('Fixed target tensors or minibatch orders have changed')
    count = len(packet['batch']['reward'])
    if len(packet['fit_orders']) != packet['fit_epochs'] or any(sorted(order)!=list(range(count)) for order in packet['fit_orders']):
        raise ValueError('Invalid saved Actor minibatch schedule')
