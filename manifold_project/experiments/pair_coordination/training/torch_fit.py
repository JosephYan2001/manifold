"""在指定设备上通过自动微分和 Adam 训练方向函数及 actor。"""
import numpy as np
import torch
from .batching import episode_batches


def direction_loss(model, batch, labels, method):
    if method not in ("analytic", "sampled"):
        raise ValueError("Unknown direction objective")
    x = model.tensor(batch["local_types"], torch.long)
    u = model.tensor(batch["actions"], torch.long)
    p = model.tensor(batch["old_probs"])
    a = model.tensor(labels)
    if a.shape != (len(x),):
        raise ValueError("Labels must have one entry per episode")
    q = model()[x]
    scores = torch.nn.functional.one_hot(u, 2).to(p.dtype)-p
    f = (scores*q).sum(-1)
    quadratic = ((p*q*q).sum(-1)-(p*q).sum(-1).square()
                 if method == "analytic" else f.square())
    return (model.tensor(batch["weights"])*(quadratic-2*a[:, None]*f)).sum()


def actor_loss(model, target, batch):
    x = model.tensor(batch["local_types"], torch.long)
    target = model.tensor(target)
    p = model()
    per_type = (target*(target.log()-p.log())).sum(-1)
    return (model.tensor(batch["weights"])*per_type[x]).sum()


def update(model, optimizer, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("训练损失出现非有限值")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.linalg.vector_norm(model.flat.grad)
    if not torch.isfinite(norm):
        raise FloatingPointError("训练梯度出现非有限值")
    optimizer.step()
    if not torch.isfinite(model.flat).all():
        raise FloatingPointError("训练参数出现非有限值")
    return float(norm.detach().cpu())


def fit_direction(model, batch, labels, method, steps, lr, log_every, callback=None,
                  batch_size=0, epochs=0, seed=0, update_callback=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    count = int(np.ceil(len(labels)/(batch_size or len(labels))))
    stream = (episode_batches(batch, labels, batch_size, epochs, seed) if epochs
              else ((None, batch, labels) for _ in range(steps)))

    def report(step):
        if callback:
            with torch.no_grad():
                value = float(direction_loss(model, batch, labels, method).cpu())
            callback(step, value, model.table())

    report(0)
    updates = 0
    for epoch, part, part_labels in stream:
        loss = direction_loss(model, part, part_labels, method)
        norm = update(model, optimizer, loss)
        updates += 1
        if update_callback:
            update_callback({"step": updates, "epoch": epoch, "batch_episodes": len(part_labels),
                             "batch_loss": float(loss.detach().cpu()), "gradient_norm": norm, "lr": lr})
        if updates % log_every == 0 or (epochs and updates % count == 0):
            report(updates)
    if not (updates % log_every == 0 or (epochs and updates % count == 0)):
        report(updates)
    return model


def fit_actor(old_actor, target, batch, steps, lr, batch_size=0, epochs=0, seed=0,
              callback=None, update_callback=None):
    model = old_actor.copy()
    target = np.asarray(target, float)
    if (target.shape != (model.n_types, 2) or not np.isfinite(target).all()
            or np.any(target <= 0) or not np.allclose(target.sum(-1), 1)):
        raise ValueError("Invalid fitting target")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    count = int(np.ceil(len(batch["local_types"])/(batch_size or len(batch["local_types"]))))

    def value():
        with torch.no_grad():
            return float(actor_loss(model, target, batch).cpu())

    initial = value()
    if callback:
        callback({"step": 0, "epoch": 0, "fit_kl": initial})
    stream = (episode_batches(batch, None, batch_size, epochs, seed) if epochs
              else ((None, batch, None) for _ in range(steps)))
    updates = 0
    for epoch, part, _ in stream:
        loss = actor_loss(model, target, part)
        norm = update(model, optimizer, loss)
        updates += 1
        if update_callback:
            update_callback({"step": updates, "batch_loss": float(loss.detach().cpu()),
                             "gradient_norm": norm, "batch_episodes": len(part["local_types"]), "lr": lr})
        if callback and (updates % count == 0 if epochs else updates % 10 == 0):
            callback({"step": updates, "epoch": epoch, "fit_kl": value()})
    final = value()
    if callback and not epochs and updates % 10:
        callback({"step": updates, "epoch": None, "fit_kl": final})
    return model, {"fit_initial_kl": initial, "fit_kl": final, "fit_steps": updates,
                   "actor_epochs": epochs, "target_min_probability": float(target.min())}
