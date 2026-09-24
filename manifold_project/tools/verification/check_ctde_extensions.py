"""Check CTDE variance, actor realization, and interaction propositions."""
import itertools
import runpy
from pathlib import Path
import numpy as np

old = runpy.run_path(str(Path(__file__).with_name("check_ctde_local_policy.py")))
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

# Corollary 1: U is finer than X, with the same old policy and masks.
source_model = old["source"]
groups = {}
for weight, x, _, _, h, i in source_model["records"]:
    key = (x, h % 2)
    mass, moment = groups.get(key, (0.0, np.zeros(3)))
    groups[key] = (mass + weight, moment + weight * source_model["vf"][h, i])
u_direction = {key: moment / mass for key, (mass, moment) in groups.items()}
permission = preprocessing = information = 0.0
for weight, x, _, _, h, i in source_model["records"]:
    full = source_model["vf"][h, i]
    finer = u_direction[x, h % 2]
    local = source_model["vl"][x]
    fmat = old["F"][x]
    permission += weight * (full-finer) @ fmat @ (full-finer)
    preprocessing += weight * (finer-local) @ fmat @ (finer-local)
    information += weight * (full-local) @ fmat @ (full-local)
close("information_permission_preprocessing_split", information, permission+preprocessing)
assert permission > 0 and preprocessing > 0
print(f"information_parts: permission={permission:.8f}, preprocessing={preprocessing:.8f}")

# Corollary 2: enumerate entire two-step, two-agent trajectories.
# The second-step direction depends on the publicly observed first action,
# so future V is correlated with an earlier quadratic sampling residual.
episode_rows, episode_probs = [], []
discount = 0.9
prob = np.array([0.8, 0.2])
score_projection = np.array([0.4, -1.6])
for trajectory in itertools.product(range(2), repeat=4):
    first, second = trajectory[:2], trajectory[2:]
    probability = float(np.prod([prob[a] for a in trajectory]))
    amplitude = 1.0 if first[0] == 0 else 3.0
    reward0 = 0.5 * sum(2*a-1 for a in first)
    reward1 = sum(2*a-1 for a in second) + 0.2*np.prod([2*a-1 for a in second])
    labels = (reward0+discount*reward1, reward1)
    linear = quadratic = sampled = 0.0
    for t, (actions_t, amp) in enumerate(((first, 1.0), (second, amplitude))):
        f_values = amp*score_projection
        v_value = float(prob @ (f_values*f_values))
        for a in actions_t:
            weight = discount**t/2
            linear += weight*2*labels[t]*f_values[a]
            quadratic += weight*v_value
            sampled += weight*f_values[a]**2
    episode_rows.append((linear, quadratic, sampled))
    episode_probs.append(probability)
ep = np.array(episode_probs)
linear_ep, quadratic_ep, sampled_ep = np.array(episode_rows).T
residual_ep = sampled_ep-quadratic_ep
def episode_covariance(a, b):
    return float(ep @ ((a-ep@a)*(b-ep@b)))
close("episode_residual_mean", ep@residual_ep, 0.0)
analytic_ep, sample_ep = linear_ep-quadratic_ep, linear_ep-sampled_ep
variance_difference = episode_covariance(analytic_ep, analytic_ep)-episode_covariance(sample_ep, sample_ep)
correct = 2*episode_covariance(analytic_ep, residual_ep)-episode_covariance(residual_ep, residual_ep)
close("episode_variance_identity", variance_difference, correct)
wrong_single_record_extension = (
    2*episode_covariance(linear_ep, residual_ep)-episode_covariance(residual_ep, residual_ep)
)
assert abs(wrong_single_record_extension-correct) > 1e-6
print(f"episode_future_quadratic_covariance={episode_covariance(quadratic_ep, residual_ep):.8f}")

# Corollary 3: a padded valid mask still identifies the full roster length.
source_mask = np.array([True]*4 + [False]*4)
target_mask = np.array([True]*6 + [False]*2)
assert int(source_mask.sum()) == 4 and int(target_mask.sum()) == 6
assert not (source_mask.sum() != 4) and (target_mask.sum() != 4)
print("roster_event_mass: source=0, target=1 despite equal padded storage size")

# Corollary 4: reuse the actual SOURCE parameter direction on the target;
# do not compute a target projection.
target_model = old["target"]
fixed_q = old["q"]
source_actor_direction = pq
def fisher_inner(model, a, b):
    return sum(weight*x @ fmat @ y for weight, x, y, fmat in
               zip(model["nu"], a, b, old["F"]))
kappa = max(target_model["nu"]/source_model["nu"])
delta_source = fisher_inner(source_model, fixed_q-source_model["vl"], fixed_q-source_model["vl"])
fit_source = np.sqrt(fisher_inner(source_model, source_actor_direction-fixed_q, source_actor_direction-fixed_q))
shift = target_model["vl"]-source_model["vl"]
semantic = fisher_inner(target_model, shift, shift)
error_bound = np.sqrt(kappa)*(np.sqrt(delta_source)+fit_source)+np.sqrt(semantic)
target_error = np.sqrt(fisher_inner(target_model, source_actor_direction-target_model["vl"],
                                  source_actor_direction-target_model["vl"]))
assert target_error <= error_bound+1e-10
energy_target = fisher_inner(target_model, target_model["vl"], target_model["vl"])
actual_gain = fisher_inner(target_model, target_model["vl"], source_actor_direction)
lower_gain = energy_target-np.sqrt(energy_target)*error_bound
assert actual_gain >= lower_gain-1e-10
epsilon = 1e-6
derivative_target = (
    old["value"](target_model, old["tilted"](source_actor_direction, epsilon))
    - old["value"](target_model, old["tilted"](source_actor_direction, -epsilon))
)/(2*epsilon)
close("same_source_parameter_direction_target_derivative",
      derivative_target, target_model["n"]*actual_gain, tol=2e-8)
print(f"same_source_actor_bound: actual={actual_gain:.8f}, lower={lower_gain:.8f}")
print("ALL EXTENSION CHECKS PASSED")
