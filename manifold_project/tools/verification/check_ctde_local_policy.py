"""Exact finite-model checks for the current CTDE manuscript; no training."""
import itertools
import numpy as np


MU = np.array([[0.2, 0.3, 0.5], [0.55, 0.45, 0.0], [1.0, 0.0, 0.0]])
MASK = MU > 0
F = np.array([np.diag(p) - np.outer(p, p) for p in MU])
FP = np.array([np.linalg.pinv(f, rcond=1e-12) for f in F])
MAX_RESIDUAL = 0.0


def close(label, actual, expected, tol=1e-9):
    global MAX_RESIDUAL
    residual = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    MAX_RESIDUAL = max(MAX_RESIDUAL, residual)
    assert residual < tol, (label, residual, actual, expected)
    print(f"{label}: residual={residual:.3e}")


def tilted(q, eta):
    p = MU * np.exp(eta * q)
    return p / p.sum(axis=1, keepdims=True)


def kl(p, q):
    active = p > 0
    return float(np.sum(p[active] * np.log(p[active] / q[active])))


def model(n, ph, shift=0.0):
    mh = np.zeros((len(ph), n, 3))
    nu = np.zeros(3)
    records = []
    tables = []
    for h, hp in enumerate(ph):
        xs = [(h + i) % 3 for i in range(n)]
        joints = list(itertools.product(*[np.flatnonzero(MASK[x]) for x in xs]))
        probs = np.array([np.prod([MU[x, a] for x, a in zip(xs, acts)])
                          for acts in joints])
        rewards = np.array([
            sum((h + 1) * (i + 1) * (a - 0.7) for i, a in enumerate(acts))
            + 0.8 * sum(acts[i] == acts[i + 1] for i in range(n - 1))
            + shift * sum(a == 0 for a in acts)
            for acts in joints
        ])
        value = probs @ rewards
        tables.append((hp, xs, joints, rewards))
        for acts, prob, reward in zip(joints, probs, rewards):
            advantage = reward - value
            for i, (x, a) in enumerate(zip(xs, acts)):
                score = np.eye(3)[a] - MU[x]
                mh[h, i] += prob * advantage * score
                records.append((hp * prob / n, x, advantage, score, h, i))
        for x in xs:
            nu[x] += hp / n
    vf = np.zeros_like(mh)
    ml = np.zeros((3, 3))
    for h, hp in enumerate(ph):
        for i in range(n):
            x = (h + i) % 3
            vf[h, i] = FP[x] @ mh[h, i]
            ml[x] += hp / n * mh[h, i]
    ml /= nu[:, None]
    vl = np.array([FP[x] @ ml[x] for x in range(3)])
    return dict(n=n, ph=np.array(ph), nu=nu, mh=mh, vf=vf, ml=ml,
                vl=vl, records=records, tables=tables)


def quantities(m, q):
    ef = info = norm = d = mse = advantage2 = 0.0
    for w, x, advantage, score, h, i in m["records"]:
        full, local = m["vf"][h, i], m["vl"][x]
        ef += w * full @ F[x] @ full
        info += w * (full - local) @ F[x] @ (full - local)
        norm += w * q[x] @ F[x] @ q[x]
        d += w * advantage * score @ q[x]
        mse += w * (advantage - score @ q[x]) ** 2
        advantage2 += w * advantage ** 2
    el = sum(w * v @ f @ v for w, v, f in zip(m["nu"], m["vl"], F))
    delta = sum(w * (v - z) @ f @ (v - z)
                for w, v, z, f in zip(m["nu"], m["vl"], q, F))
    fullerr = sum(w * (m["vf"][h, i] - q[x]) @ F[x] @
                  (m["vf"][h, i] - q[x])
                  for w, x, _, _, h, i in m["records"])
    return dict(ef=ef, info=info, norm=norm, d=d, b=2*d-norm,
                el=el, delta=delta, fullerr=fullerr,
                mse=mse, advantage2=advantage2)


def value(m, policy):
    result = 0.0
    for hp, xs, joints, rewards in m["tables"]:
        result += hp * sum(np.prod([policy[x, a] for x, a in zip(xs, acts)]) * r
                           for acts, r in zip(joints, rewards))
    return result


source = model(2, [0.15, 0.25, 0.35, 0.25])
target = model(3, [0.25, 0.2, 0.15, 0.4], shift=0.3)
q = source["vl"] + np.array([[0.1, -0.25, 0.15], [0.12, -0.12, 0], [0, 0, 0]])
for label, m in [("source", source), ("target", target)]:
    result = quantities(m, q)
    close(label + "_projection", result["fullerr"], result["info"] + result["delta"])
    close(label + "_energy", result["ef"], result["info"] + result["el"])
    close(label + "_variational", result["el"] - result["b"], result["delta"])
    close(label + "_regression", result["b"], result["advantage2"] - result["mse"])
    close(label + "_normal_equation", np.einsum("xij,xj->xi", F, m["vl"]), m["ml"])
    epsilon = 1e-6
    derivative = (value(m, tilted(q, epsilon)) - value(m, tilted(q, -epsilon))) / (2*epsilon)
    close(label + "_return_derivative", derivative, m["n"] * result["d"], tol=2e-8)
    eta = 0.04
    newp = tilted(q, eta)
    joint_kl = 0.0
    for hp, xs, joints, _ in m["tables"]:
        oldjoint = np.array([np.prod([MU[x, a] for x, a in zip(xs, acts)]) for acts in joints])
        newjoint = np.array([np.prod([newp[x, a] for x, a in zip(xs, acts)]) for acts in joints])
        joint_kl += hp * kl(oldjoint, newjoint)
    local_kl = sum(w * kl(oldp, np_) for w, oldp, np_ in zip(m["nu"], MU, newp))
    close(label + "_joint_kl", joint_kl, m["n"] * local_kl)

rs, rt = quantities(source, q), quantities(target, q)
kappa = max(target["nu"] / source["nu"])
sem = sum(w * (a - b) @ f @ (a - b)
          for w, a, b, f in zip(target["nu"], target["vl"], source["vl"], F))
bound = rt["info"] + (np.sqrt(kappa * rs["delta"]) + np.sqrt(sem)) ** 2
assert rt["fullerr"] <= bound + 1e-10
print(f"transport_bound: actual={rt['fullerr']:.8f}, upper={bound:.8f}")

sem_via_m = sum(w * (a - b) @ fp @ (a - b)
                for w, a, b, fp in zip(target["nu"], target["ml"], source["ml"], FP))
close("semantic_pseudoinverse", sem, sem_via_m)
for x in range(3):
    if MASK[x].sum() > 1:
        eig = np.linalg.eigvalsh(F[x])
        assert min(eig[eig > 1e-10]) >= MU[x, MASK[x]].min() - 1e-10
    else:
        close("singleton_mask", F[x], np.zeros((3, 3)))

qbest = source["vl"]
assert value(source, tilted(qbest, 1e-4)) > value(source, MU)
print("source_small_step_improvement: PASS")

# Two-step model with action-dependent random termination tests discounted Z.
gamma = 0.83
n = 2
initial = np.array([0.3, 0.7])
joints2 = list(itertools.product(range(2), repeat=n))
p2 = np.array([[0.35, 0.65], [0.6, 0.4]])
q2 = np.array([[0.3, -0.3], [-0.2, 0.2]])


def reward(s, acts):
    return 0.4*s + (1 + s)*acts[0] - 0.6*acts[1] + 0.5*(acts[0] == acts[1])


def survival(s, acts):
    return 0.25 + 0.25 * acts[0] + 0.15 * s


def transition(s, acts):
    pnext = 0.2 + 0.2*s + 0.15*acts[0] + 0.1*acts[1]
    return np.array([1-pnext, pnext])


def jp(policy, s, acts):
    return np.prod([policy[(s+i) % 2, a] for i, a in enumerate(acts)])


def finite_values(policy):
    v1 = np.array([sum(jp(policy, s, a)*reward(s, a) for a in joints2) for s in range(2)])
    q0 = {(s, a): reward(s, a) + gamma*survival(s, a)*(transition(s, a) @ v1)
          for s in range(2) for a in joints2}
    v0 = np.array([sum(jp(policy, s, a)*q0[s, a] for a in joints2) for s in range(2)])
    return initial @ v0, v0, v1, q0


_, v0, v1, q0 = finite_values(p2)
occupancy1 = np.zeros(2)
for s in range(2):
    for acts in joints2:
        occupancy1 += initial[s]*jp(p2, s, acts)*survival(s, acts)*transition(s, acts)
Z = 1 + gamma*occupancy1.sum()
numerator = 0.0
for time, weights in [(0, initial), (1, gamma*occupancy1)]:
    for s, weight in enumerate(weights):
        for acts in joints2:
            advantage = (q0[s, acts]-v0[s]) if time == 0 else (reward(s, acts)-v1[s])
            score_q = sum(q2[(s+i) % 2, a]-p2[(s+i) % 2] @ q2[(s+i) % 2]
                          for i, a in enumerate(acts))
            numerator += weight*jp(p2, s, acts)*advantage*score_q
D = numerator/(n*Z)
eps = 1e-6
plus = p2*np.exp(eps*q2)
plus /= plus.sum(axis=1, keepdims=True)
minus = p2*np.exp(-eps*q2)
minus /= minus.sum(axis=1, keepdims=True)
derivative = (finite_values(plus)[0]-finite_values(minus)[0])/(2*eps)
close("variable_length_discounted_gradient", derivative, n*Z*D, tol=2e-9)
print(f"discounted_mass_Z={Z:.8f}, derivative={derivative:.8f}")
print(f"ALL CHECKS PASSED; maximum residual={MAX_RESIDUAL:.3e}")
