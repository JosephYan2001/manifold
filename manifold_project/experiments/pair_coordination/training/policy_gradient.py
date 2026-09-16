"""One on-policy Adam update per fresh batch; no stale-batch PG epochs."""
import torch
from .torch_fit import update


def pg_loss(actor, batch, labels):
    x = actor.tensor(batch["local_types"], torch.long)
    actions = actor.tensor(batch["actions"], torch.long)
    log_probs = actor()[x].log().gather(-1, actions[..., None]).squeeze(-1)
    # Derivative of team return: sum agent scores, average independent episodes.
    return -(actor.tensor(labels).detach() * log_probs.sum(-1)).mean()


def make_optimizer(actor, lr, state=None):
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr, foreach=False)
    if state:
        optimizer.state[actor.flat] = {
            "step": torch.tensor(float(state["step"])),
            "exp_avg": actor.tensor(state["exp_avg"]),
            "exp_avg_sq": actor.tensor(state["exp_avg_sq"]),
        }
    return optimizer


def optimizer_snapshot(optimizer, actor):
    state = optimizer.state.get(actor.flat)
    return ({key: value.detach().cpu().tolist() for key, value in state.items()}
            if state else {})


def step(actor, optimizer, batch, labels):
    loss = pg_loss(actor, batch, labels)
    norm = update(actor, optimizer, loss)
    return {"loss": float(loss.detach().cpu()), "gradient_norm": norm, "updates": 1}
