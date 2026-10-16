"""One fixed-target fitting path for training and frozen-checkpoint diagnostics."""
from copy import deepcopy
import math
import time

import numpy as np
import torch

from .diagnostics import categorical_kl
from .storage import seed_for


SELECTIONS = ('last', 'best_train_kl')


def validate_fit_config(config):
    if config.get('actor_fit_selection', 'last') not in SELECTIONS:
        raise ValueError('actor_fit_selection must be last or best_train_kl')
    for key, default in (('actor_fit_selection_atol', 1e-12), ('actor_fit_selection_rtol', 1e-6)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')


def minibatch_orders(seed, round_number, episodes, epochs):
    rng = np.random.default_rng(seed_for(seed, round_number, 'actor_fit', 'minibatches'))
    return [rng.permutation(episodes).tolist() for _ in range(epochs)]


def state_changed(left, right):
    return any(not torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items())


@torch.no_grad()
def target_kl(runner, model, batch):
    return float(runner.weighted(categorical_kl(batch['target'], model(batch['x'])), batch))


def fit_actor(runner, old, batch, epochs, *, lr=None, selection=None, orders=None,
              observer=None, trace_every_updates=0):
    """Finish the entire trajectory before selecting; validation never selects.

    Selection compares normalized FP64 full-training-batch KL at epoch boundaries.
    The optimization loss, fresh Adam, clipping and minibatch order are unchanged.
    The returned best is the minimum finite epoch, unless its improvement over
    epoch zero is within the declared numerical tolerance. Ties favor earlier epochs.
    Adam is discarded after fitting; neither selected candidate resumes its moments.
    """
    validate_fit_config(runner.c)
    rule = selection or runner.c.get('actor_fit_selection', 'last')
    if rule not in SELECTIONS:
        raise ValueError(f'Unknown Actor fit selection: {rule}')
    started = time.perf_counter()
    candidate = deepcopy(old)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=runner.c['actor_lr'] if lr is None else lr)
    orders = orders if orders is not None else minibatch_orders(
        runner.seed, runner.round+1, len(batch['reward']), epochs)
    history = []
    initial = target_kl(runner, candidate, batch)
    if not math.isfinite(initial):
        raise FloatingPointError('Nonfinite initial Actor fit KL')
    best, minimum, best_epoch = deepcopy(candidate), initial, 0

    def observe(epoch, updates, scope):
        nonlocal best, minimum, best_epoch
        if scope != 'post_optimizer_step':
            loss = target_kl(runner, candidate, batch)
            finite = math.isfinite(loss)
            history.append(dict(epoch=epoch, optimizer_updates=updates,
                                train_target_kl=loss if finite else None, finite=finite))
            if finite and loss < minimum:
                best, minimum, best_epoch = deepcopy(candidate), loss, epoch
        if observer is not None and (scope != 'post_optimizer_step' or
                                    trace_every_updates and updates % trace_every_updates == 0):
            observer(candidate, epoch, updates, scope)

    observe(0, 0, 'initial')
    def objective(mb):
        p = candidate(mb['x'])
        return runner.weighted((mb['target']*(mb['target'].log()-p.log())).sum(-1), mb), {}
    optimization = runner.optimize(candidate, optimizer, batch, epochs, 'actor_fit', objective,
                                   observer=observe, orders=orders)
    final_kl = target_kl(runner, candidate, batch)
    if rule == 'last' and not math.isfinite(final_kl):
        raise FloatingPointError('Nonfinite final Actor fit KL')
    tolerance = runner.c.get('actor_fit_selection_atol', 1e-12) + runner.c.get('actor_fit_selection_rtol', 1e-6)*abs(initial)
    minimum_epoch = best_epoch
    if initial-minimum <= tolerance:
        best, best_epoch = deepcopy(old), 0
    best_kl = initial if best_epoch == 0 else minimum
    selected = candidate if rule == 'last' else best
    selected_epoch = epochs if rule == 'last' else best_epoch
    info = dict(actor_fit_semantics_version='actor-fit-selection-v1', actor_fit_selection=rule,
                actor_fit_selection_atol=runner.c.get('actor_fit_selection_atol', 1e-12),
                actor_fit_selection_rtol=runner.c.get('actor_fit_selection_rtol', 1e-6),
                fit_initial_train_kl=initial, fit_minimum_train_kl=minimum,
                fit_minimum_epoch=minimum_epoch, fit_best_train_kl=best_kl,
                fit_final_train_kl=final_kl if math.isfinite(final_kl) else None,
                fit_selected_train_kl=final_kl if rule == 'last' else best_kl,
                fit_best_epoch=best_epoch, fit_selected_epoch=selected_epoch,
                fit_optimizer_updates=optimization['updates'], fit_seconds=time.perf_counter()-started,
                fit_candidate_changed=state_changed(selected, old),
                fit_selected_zero_update=selected_epoch == 0,
                fit_optimizer_state_policy='fresh_Adam_per_candidate_discard_after_selection')
    return dict(selected=selected, final=candidate, best=best, info=info, history=history, orders=orders)
