"""Offline Critic probes and chronology; no environment, training, or evaluation.

Synthetic inputs are sensitivity probes, not archived rollout states, new task
samples, or a measurement of value accuracy. Checkpoints remain read-only.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

HERE = Path(__file__).resolve().parent
NAV = HERE.parent.parent
PATHS = {'old_base': NAV/'navigation_diagnosis_20261010_2m_s40_v1/an_base',
         **{f'n{n}_{s}': HERE.parent/f'n{n}_{s}'
            for n in (16, 32, 64) for s in ('fixed', 'linear')}}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def probe(weights, x):
    w = {k: v.double() for k, v in weights.items()}
    assert w['net.0.weight'].shape[1] == 24
    h1 = (x @ w['net.0.weight'].T + w['net.0.bias']).tanh()
    z2 = h1 @ w['net.2.weight'].T + w['net.2.bias']
    h2 = z2.tanh()
    y = h2 @ w['net.4.weight'].T + w['net.4.bias']
    derivative = 1-h2.square()
    assert torch.isfinite(y).all()
    return dict(value_mean=float(y.mean()), value_std=float(y.std(unbiased=False)),
                second_preactivation_abs_median=float(z2.abs().median()),
                second_tanh_abs_gt_095_fraction=float((h2.abs() > .95).double().mean()),
                second_tanh_derivative_median=float(derivative.median()),
                second_tanh_derivative_lt_1e_minus_6_fraction=float((derivative < 1e-6).double().mean()))


def main():
    # Same positions for both probes; second probe adds bounded synthetic velocity.
    g = torch.Generator().manual_seed(20261022)
    bounded = (2*torch.rand((4096, 24), generator=g)-1).double()
    zero_velocity = bounded.clone()
    zero_velocity[:, 8:16] = 0
    result = dict(date='2026-10-22', torch_version=torch.__version__, dtype='float64',
                  device='cpu', seed=20261022, states_per_probe=4096,
                  input_layout='8 agent position, 8 agent velocity, 8 landmark position fields',
                  input_distribution='Independent U[-1,1] synthetic coordinates; velocity zeroed in first probe',
                  limitation='Sensitivity probes only: no rollout states, targets, value-accuracy estimate, or causal intervention',
                  added_environment_steps=0, optimizer_updates=0, runs={}, source_sha256={})
    for name, base in PATHS.items():
        root = base/'N-C_source_learning/full/AN_s40'
        checkpoint = root/'checkpoints/final.pt'
        log = root/'training.csv'
        for p in (checkpoint, log):
            result['source_sha256'][str(p.relative_to(NAV))] = sha(p)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        with log.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        chronology = [{k: r.get(k) for k in
                       ('round', 'source_steps', 'train_J', 'critic_mse',
                        'explained_variance', 'critic_postfit_value_variance',
                        'critic_postfit_explained_variance', 'fit_kl_before')}
                      for r in rows if int(r['round']) in (1, 5, 10, 20, 40, 60, 80, len(rows))]
        result['runs'][name] = dict(zero_velocity=probe(state['critic'], zero_velocity),
                                   bounded_velocity=probe(state['critic'], bounded),
                                   chronology=chronology)
    for rel, before in result['source_sha256'].items():
        assert sha(NAV/rel) == before, rel
    (HERE/'mechanism_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, run in result['runs'].items():
        p = run['zero_velocity']
        print(name, 'value_std=', f"{p['value_std']:.6g}",
              'fraction_derivative_lt_1e-6=', f"{p['second_tanh_derivative_lt_1e_minus_6_fraction']:.6g}")


if __name__ == '__main__':
    main()
