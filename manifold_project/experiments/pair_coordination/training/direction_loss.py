"""Two empirical direction objectives and their analytic gradients."""
import numpy as np
from .optim import Adam
from .batching import episode_batches


def record_terms(table, batch, labels, method="analytic"):
    if method not in ("analytic", "sampled"):
        raise ValueError("Unknown direction objective")
    x, u, p = batch["local_types"], batch["actions"], batch["old_probs"]
    q = np.asarray(table, float)[x]
    a = np.asarray(labels, float)
    if a.shape != (len(x),):
        raise ValueError("Labels must have one entry per episode")
    scores = np.eye(2)[u] - p
    f = np.sum(scores*q, axis=-1)
    fq = p*(q - np.sum(p*q, axis=-1, keepdims=True))
    if method == "analytic":
        quadratic = np.sum(q*fq, axis=-1)
        grad = 2*fq - 2*a[:, None, None]*scores
    else:
        quadratic = f*f
        grad = 2*(f-a[:, None])[..., None]*scores
    losses = quadratic - 2*a[:, None]*f
    return losses, grad


def loss_and_gradient(table, batch, labels, method="analytic"):
    losses, gradients = record_terms(table, batch, labels, method)
    weights = batch["weights"]
    result = np.zeros_like(table, dtype=float)
    np.add.at(result, batch["local_types"].ravel(),
              (gradients * weights[..., None]).reshape(-1, 2))
    return float(np.sum(weights*losses)), result


def episode_scores(table, batch, labels):
    """Independent statistical units are whole episodes, averaging their agents."""
    losses, _ = record_terms(table, batch, labels, "analytic")
    return -losses.mean(axis=1)


def fit_direction(model, batch, labels, method, steps, lr, log_every, callback=None,
                  batch_size=0, epochs=0, seed=0, update_callback=None):
    if hasattr(model, "flat"):
        from .torch_fit import fit_direction as torch_fit
        return torch_fit(model, batch, labels, method, steps, lr, log_every, callback,
                         batch_size, epochs, seed, update_callback)
    optimizer = Adam(model.parameters.shape, lr)
    if epochs:
        if callback:
            callback(0, loss_and_gradient(model.table(), batch, labels, method)[0], model.table())
        updates = 0
        batches = int(np.ceil(len(labels)/(batch_size or len(labels))))
        for epoch, part, part_labels in episode_batches(batch, labels, batch_size, epochs, seed):
            batch_loss, grad = loss_and_gradient(model.table(), part, part_labels, method)
            gradient = model.parameter_gradient(grad)
            optimizer.step(model.parameters, gradient)
            updates += 1
            if update_callback:
                update_callback({"step": updates, "epoch": epoch, "batch_episodes": len(part_labels),
                                 "batch_loss": batch_loss, "gradient_norm": float(np.linalg.norm(gradient)), "lr": lr})
            if callback and (updates % log_every == 0 or updates % batches == 0):
                callback(updates, loss_and_gradient(model.table(), batch, labels, method)[0], model.table())
        return model
    for step in range(steps+1):
        table = model.table()
        loss, grad = loss_and_gradient(table, batch, labels, method)
        if callback is not None and (step % log_every == 0 or step == steps):
            callback(step, loss, table.copy())
        if step < steps:
            gradient = model.parameter_gradient(grad)
            optimizer.step(model.parameters, gradient)
            if update_callback:
                update_callback({"step": step+1, "epoch": None, "batch_episodes": len(labels),
                                 "batch_loss": loss, "gradient_norm": float(np.linalg.norm(gradient)), "lr": lr})
    return model
