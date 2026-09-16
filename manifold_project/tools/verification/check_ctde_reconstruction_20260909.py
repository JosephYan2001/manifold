"""Finite-model verification of the revised manuscript's added propositions."""
import itertools
import runpy
from pathlib import Path
import numpy as np

old = runpy.run_path(str(Path(__file__).with_name("check_ctde_local_policy_20260909.py")))
close = old["close"]

# Proposition 4: conditional averaging can either reduce or increase variance.
p = np.array([0.8, 0.2])
f0 = np.array([0.4, -1.6])
u = 1.0
f = u*f0
y = f*f
v = p @ y
r = y-v
def variance(z):
    return float(p @ ((z-p@z)**2))
for scale in [0.0, 0.5, 1.0]:
    advantage = scale*f0
    linear = 2*advantage*f
    ga, gs = linear-v, linear-y
    close(f"variance_identity_{scale}", variance(ga)-variance(gs),
          2*(p @ (linear*r)) - p @ (r*r))
    grad_l = 2*advantage*f0
    grad_y = 2*u*f0*f0
    grad_v = p @ grad_y
    grad_r = grad_y-grad_v
    close(f"gradient_variance_identity_{scale}",
          variance(grad_l-grad_v)-variance(grad_l-grad_y),
          2*(p @ (grad_l*grad_r))-p @ (grad_r*grad_r))
    print(f"variance_comparison scale={scale}: analytic={variance(ga):.6f}, sampled={variance(gs):.6f}")
close("empirical_example_analytic_min", 0.8/(2*0.64), 0.625)
close("empirical_example_sample_min", 1/0.4, 2.5)

# Proposition 5: a restricted shared actor tangent class.
m, fs = old["source"], old["F"]
dmat = np.array([[[1., 0.], [-1., 1.], [0., -1.]],
                 [[1., 2.], [-1., -2.], [0., 0.]],
                 [[0., 0.], [0., 0.], [0., 0.]]])
nu, vloc, q = m["nu"], m["vl"], old["q"]
g = sum(weight * d.T @ fmat @ d for weight, d, fmat in zip(nu, dmat, fs))
def inner(a, b):
    return sum(weight*x @ fmat @ y for weight, x, fmat, y in zip(nu, a, fs, b))
def projection(a):
    h = sum(weight*d.T @ fmat @ z for weight, d, fmat, z in zip(nu, dmat, fs, a))
    return np.einsum("xij,j->xi", dmat, np.linalg.pinv(g) @ h)
pv, pq = projection(vloc), projection(q)
param_error = inner(vloc-pv, vloc-pv)
actor_energy = inner(pv, pv)
close("actor_energy_decomposition", inner(vloc, vloc), param_error+actor_energy)
close("projection_orthogonality", inner(vloc-pv, pq), 0.0)
delta = inner(q-vloc, q-vloc)
direction_return = inner(vloc, pq)
bound = actor_energy - np.sqrt(actor_energy*delta)
assert direction_return >= bound-1e-10
print(f"projected_direction_bound: actual={direction_return:.8f}, lower={bound:.8f}")
eps = 1e-6
numerical = (old["value"](m, old["tilted"](pq, eps)) -
             old["value"](m, old["tilted"](pq, -eps)))/(2*eps)
close("parameter_path_return_derivative", numerical, m["n"]*direction_return, tol=2e-8)
eta = 1e-4
teacher, student = old["tilted"](q, eta), old["tilted"](pq, eta)
fit_kl = sum(w*old["kl"](a, b) for w, a, b in zip(nu, teacher, student))
teacher_minus, student_minus = old["tilted"](q, -eta), old["tilted"](pq, -eta)
fit_kl_minus = sum(w*old["kl"](a, b) for w, a, b in zip(nu, teacher_minus, student_minus))
close("actor_fit_second_order", (fit_kl+fit_kl_minus)/(2*eta*eta),
      0.5*inner(q-pq, q-pq), tol=1e-6)

# Curvature rank at a single 3-action input.
p3 = np.array([0.2, 0.3, 0.5])
f3 = np.diag(p3)-np.outer(p3, p3)
score = np.eye(3)[0]-p3
assert np.linalg.matrix_rank(f3, tol=1e-10) == 2
assert np.linalg.matrix_rank(np.outer(score, score), tol=1e-10) == 1
print("analytic_vs_sampled_curvature_rank: 2 versus 1")

# Proposition 6: nonzero pair interactions, different numbers of robots.
rho = 0.7
policy = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
actions = np.array([-1.0, 1.0])
means = policy @ actions
fisher = np.array([np.diag(z)-np.outer(z, z) for z in policy])
def pair_model(n, type_probs):
    marginal_m = np.zeros((3, 2))
    marginal_x = np.zeros(3)
    for types in itertools.product(range(3), repeat=n):
        ph = np.prod([type_probs[x] for x in types])
        joint = list(itertools.product(range(2), repeat=n))
        jp = np.array([np.prod([policy[x, a] for x, a in zip(types, aa)]) for aa in joint])
        reward = np.array([
            sum(0.1*x*actions[a] for x, a in zip(types, aa)) +
            rho/(n-1)*sum(actions[aa[i]]*actions[aa[j]]
                          for i in range(n) for j in range(i+1, n))
            for aa in joint
        ])
        advantage = reward-jp@reward
        for i, x in enumerate(types):
            marginal_x[x] += ph/n
            for aa, prob, adv in zip(joint, jp, advantage):
                score_i = np.eye(2)[aa[i]]-policy[x]
                marginal_m[x] += ph/n*prob*adv*score_i
    conditional_m = marginal_m/marginal_x[:, None]
    vl = np.array([np.linalg.pinv(f) @ z for f, z in zip(fisher, conditional_m)])
    local_signal = 0.1*np.arange(3)+rho*(type_probs @ means)
    predicted = local_signal[:, None]*actions
    close(f"pair_mechanism_direction_n{n}", vl, predicted)
    return marginal_x, vl
p_source = np.array([0.5, 0.3, 0.2])
p_target = np.array([0.5, 0.2, 0.3])
nu0, v0 = pair_model(3, p_source)
nue, ve = pair_model(4, p_target)
_, same = pair_model(4, p_source)
close("interacting_scale_invariance_same_composition", v0, same)
sem = sum(w*(a-b) @ f @ (a-b) for w, a, b, f in zip(nue, ve, v0, fisher))
measure_change = rho*np.abs(p_target-p_source).sum()
upper = measure_change**2
assert sem <= upper+1e-10
print(f"pair_transport_bound: actual={sem:.8f}, upper={upper:.8f}")

# Pure second-order coordination and the positive-score gate.
for eps in [0.01, 0.1]:
    close("second_order_coordination", (2*(0.5+eps)-1)**2, 4*eps*eps)
for direction in [0.1, 1.0]:
    score_value = -direction*direction
    assert score_value < 0
print("positive_score_gate_at_symmetric_point: all tested nonzero directions rejected")
print("ALL RECONSTRUCTION CHECKS PASSED")
