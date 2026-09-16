"""Forward-KL fitting, starting from an independent copy of the old actor."""
import numpy as np
from ..models.actor import sigmoid
from .optim import Adam
from .batching import episode_batches


def kl_and_gradient(actor, target, type_weights):
    target, weights = np.asarray(target, float), np.asarray(type_weights, float)
    p = actor.table()
    if (target.shape != p.shape or weights.shape != (actor.n_types,)
            or not np.isfinite(target).all() or np.any(target <= 0)
            or not np.allclose(target.sum(axis=1), 1)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)):
        raise ValueError("Invalid fitting target or type weights")
    raw = sigmoid(actor.logits)
    derivative = (1-actor.beta)*raw*(1-raw)
    gradient = weights*(p[:, 1]-target[:, 1])/(p[:, 0]*p[:, 1])*derivative
    loss = np.sum(weights[:, None]*target*(np.log(target)-np.log(p)))
    return float(loss), gradient


def fit_actor(old_actor, target, batch, steps, lr, batch_size=0, epochs=0, seed=0,
              callback=None, update_callback=None):
    if hasattr(old_actor, "flat"):
        from .torch_fit import fit_actor as torch_fit
        return torch_fit(old_actor, target, batch, steps, lr, batch_size, epochs, seed,
                         callback, update_callback)
    candidate = old_actor.copy()
    weights = np.bincount(batch["local_types"].ravel(), weights=batch["weights"].ravel(),
                          minlength=candidate.n_types)
    optimizer = Adam(candidate.parameters.shape, lr)
    initial, _ = kl_and_gradient(candidate, target, weights)
    batches = (part for _, part, _ in episode_batches(batch, None, batch_size, epochs, seed)) if epochs else (batch for _ in range(steps))
    updates = 0
    per_epoch = int(np.ceil(len(batch["local_types"])/(batch_size or len(batch["local_types"]))))
    if callback:
        callback({"step": 0, "epoch": 0, "fit_kl": initial})
    for part in batches:
        part_weights = np.bincount(part["local_types"].ravel(), weights=part["weights"].ravel(), minlength=candidate.n_types)
        batch_loss, grad = kl_and_gradient(candidate, target, part_weights)
        gradient = candidate.parameter_gradient(grad)
        optimizer.step(candidate.parameters, gradient)
        updates += 1
        if update_callback:
            update_callback({"step": updates, "batch_loss": batch_loss, "gradient_norm": float(np.linalg.norm(gradient)),
                             "batch_episodes": len(part["local_types"]), "lr": lr})
        if callback and (updates % per_epoch == 0 if epochs else updates % 10 == 0):
            callback({"step": updates, "epoch": updates//per_epoch if epochs else None,
                      "fit_kl": kl_and_gradient(candidate, target, weights)[0]})
    final, _ = kl_and_gradient(candidate, target, weights)
    if callback and not epochs and updates % 10:
        callback({"step": updates, "epoch": None, "fit_kl": final})
    return candidate, {"fit_initial_kl": initial, "fit_kl": final,
                       "fit_steps": updates, "actor_epochs": epochs,
                       "target_min_probability": float(np.min(target))}
