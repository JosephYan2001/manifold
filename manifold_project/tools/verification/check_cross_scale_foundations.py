"""Check self-contained finite scale, information, and identification examples.

Run with Python's standard library only.  This script writes no artifacts and
does not import or run the training code.  Actions are ordered (-1, +1).
"""

from itertools import product
from math import exp, fsum, isclose, prod


ACTIONS = (-1, 1)
EXACT_ABS_TOL = 2e-12
FD_STEP = 1e-5
FD_ABS_TOL = 1e-8
FD_REL_TOL = 2e-7


def check_close(actual, expected, label, *, finite_difference=False):
    """Fail explicitly even if Python was started with optimization enabled."""
    abs_tol = FD_ABS_TOL if finite_difference else EXACT_ABS_TOL
    rel_tol = FD_REL_TOL if finite_difference else 0.0
    if not isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol):
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def make_events(n_agents, type_probs, reward):
    """Enumerate joint types and actions; retain type probabilities only."""
    events = []
    for types in product(range(len(type_probs)), repeat=n_agents):
        type_probability = prod(type_probs[x] for x in types)
        for actions in product(ACTIONS, repeat=n_agents):
            events.append((types, actions, type_probability, reward(types, actions)))
    return events


def action_probability(types, actions, plus_probs):
    return prod(
        plus_probs[x] if a == 1 else 1.0 - plus_probs[x]
        for x, a in zip(types, actions)
    )


def expected_reward(events, plus_probs):
    return fsum(
        type_probability * action_probability(types, actions, plus_probs) * reward
        for types, actions, type_probability, reward in events
    )


def tilted_probabilities(plus_probs, directions, eta):
    """Apply the exponential policy update, not a return-derivative formula."""
    updated = []
    for plus_probability, (q_minus, q_plus) in zip(plus_probs, directions):
        minus_weight = (1.0 - plus_probability) * exp(eta * q_minus)
        plus_weight = plus_probability * exp(eta * q_plus)
        updated.append(plus_weight / (minus_weight + plus_weight))
    return tuple(updated)


def return_derivative(events, plus_probs, directions):
    positive = expected_reward(
        events, tilted_probabilities(plus_probs, directions, FD_STEP)
    )
    negative = expected_reward(
        events, tilted_probabilities(plus_probs, directions, -FD_STEP)
    )
    return (positive - negative) / (2.0 * FD_STEP)


def conditional_score_moments(events, type_probs, plus_probs):
    """Compute E[(one_hot(A_I)-mu(X_I))*R | X_I=x] directly.

    I is a uniformly sampled robot.  An action-before baseline has zero score
    moment, so the team reward is sufficient for this one-step calculation.
    """
    terms = [[[], []] for _ in type_probs]
    for types, actions, type_probability, reward in events:
        weight = (
            type_probability
            * action_probability(types, actions, plus_probs)
            * reward
            / len(types)
        )
        for x, a in zip(types, actions):
            probability = plus_probs[x]
            terms[x][0].append(weight * (float(a == -1) - (1.0 - probability)))
            terms[x][1].append(weight * (float(a == 1) - probability))
    return tuple(
        tuple(fsum(component) / type_probs[x] for component in row)
        for x, row in enumerate(terms)
    )


def fisher_energy(type_probs, plus_probs, directions):
    return fsum(
        weight * probability * (1.0 - probability) * (q_plus - q_minus) ** 2
        for weight, probability, (q_minus, q_plus) in zip(
            type_probs, plus_probs, directions
        )
    )


def score_direction_product(type_probs, moments, directions):
    return fsum(
        weight * fsum(m * q for m, q in zip(moment, direction))
        for weight, moment, direction in zip(type_probs, moments, directions)
    )


def check_p0():
    type_probs = (0.6, 0.4)
    biases = (-0.2, 0.3)
    rho = 0.7
    coupling = ((1.0, -0.5), (-0.5, 1.0))
    plus_probs = (0.25, 0.75)
    expected_g = (-0.48, 0.545)
    directions = tuple((-g, g) for g in expected_g)
    expected_energy = 0.1927875

    for n_agents in (2, 4, 6):
        def reward(types, actions):
            individual = fsum(biases[x] * a for x, a in zip(types, actions))
            pairs = fsum(
                coupling[types[i]][types[j]] * actions[i] * actions[j]
                for i in range(n_agents)
                for j in range(i + 1, n_agents)
            )
            return individual + rho * pairs / (n_agents - 1)

        events = make_events(n_agents, type_probs, reward)
        moments = conditional_score_moments(events, type_probs, plus_probs)
        recovered_g = []
        for x, (minus_moment, plus_moment) in enumerate(moments):
            check_close(minus_moment + plus_moment, 0.0, "P0 score centering")
            probability = plus_probs[x]
            g = plus_moment / (2.0 * probability * (1.0 - probability))
            check_close(g, expected_g[x], f"P0 n={n_agents}, g[{x}]")
            recovered_g.append(g)
        per_robot_return = expected_reward(events, plus_probs) / n_agents
        energy = fisher_energy(type_probs, plus_probs, directions)
        direction_product = score_direction_product(type_probs, moments, directions)
        derivative = return_derivative(events, plus_probs, directions)
        check_close(per_robot_return, 0.1865, "P0 J/n")
        check_close(energy, expected_energy, "P0 E")
        check_close(direction_product, expected_energy, "P0 D")
        check_close(
            derivative, n_agents * expected_energy,
            "P0 finite-difference dJ=nE", finite_difference=True,
        )
        print(
            f"P0 n={n_agents}: J/n={per_robot_return:.10f}, "
            f"g=({recovered_g[0]:.6f},{recovered_g[1]:.6f}), "
            f"D=E={energy:.10f}, dJ(fd)={derivative:.10f}"
        )


def check_f1():
    type_probs = (0.5, 0.5)
    type_values = (-1, 1)
    directions = ((1.0, -1.0), (-1.0, 1.0))
    for n_agents in (4, 6):
        events = make_events(
            n_agents, type_probs,
            lambda types, actions: sum(type_values[x] * a for x, a in zip(types, actions)),
        )
        # The same constant policy is used at both types: the encoder hides x.
        for probability in (0.2, 0.5, 0.8):
            check_close(
                expected_reward(events, (probability, probability)), 0.0,
                "F1 constant actor return",
            )
        moments = conditional_score_moments(events, type_probs, (0.5, 0.5))
        for action_index in range(2):
            compressed_moment = fsum(
                weight * moment[action_index]
                for weight, moment in zip(type_probs, moments)
            )
            check_close(compressed_moment, 0.0, "F1 compressed score moment")
        derivative = return_derivative(events, (0.5, 0.5), directions)
        check_close(
            derivative, float(n_agents), "F1 visible-type dJ=n",
            finite_difference=True,
        )
        print(
            f"F1 n={n_agents}: constant-actor J=0, compressed score=0, "
            f"type-actor dJ(fd)={derivative:.10f}"
        )


def check_f2():
    mechanisms = ((-2, 1), (4, -1))

    def reward(n_agents, actions, mechanism):
        bias, interaction = mechanism
        return (bias + (n_agents - 1) * interaction) * sum(actions)

    for actions in product(ACTIONS, repeat=4):
        left = reward(4, actions, mechanisms[0])
        right = reward(4, actions, mechanisms[1])
        if left != right or left != sum(actions):
            raise AssertionError("F2 source mechanisms differ on a joint action")
    for actions in product(ACTIONS, repeat=6):
        check_close(reward(6, actions, mechanisms[0]), 3 * sum(actions), "F2 target +")
        check_close(reward(6, actions, mechanisms[1]), -sum(actions), "F2 target -")
    # The maximizers are unique; all source joint actions above are exhaustive.
    for mechanism, optimum in zip(mechanisms, ((1,) * 6, (-1,) * 6)):
        optimum_reward = reward(6, optimum, mechanism)
        for actions in product(ACTIONS, repeat=6):
            if actions != optimum and reward(6, actions, mechanism) >= optimum_reward:
                raise AssertionError("F2 target optimum is not unique")
    print("F2 n=4: all 16 joint-action rewards identical; n=6: coefficients=(3,-1), opposite unique optima")


def check_f3():
    type_probs = (1.0,)
    plus_probs = (0.75,)
    source_direction = ((0.5, -0.5),)
    for n_agents, expected_g, expected_d in ((4, -0.5, 0.1875), (6, 0.5, -0.1875)):
        def reward(_types, actions):
            return -2 * sum(actions) + sum(
                actions[i] * actions[j]
                for i in range(n_agents)
                for j in range(i + 1, n_agents)
            )

        events = make_events(n_agents, type_probs, reward)
        moments = conditional_score_moments(events, type_probs, plus_probs)
        g = moments[0][1] / (2.0 * plus_probs[0] * (1.0 - plus_probs[0]))
        direction_product = score_direction_product(type_probs, moments, source_direction)
        derivative = return_derivative(events, plus_probs, source_direction)
        check_close(g, expected_g, "F3 g")
        check_close(direction_product, expected_d, "F3 D")
        check_close(
            derivative, n_agents * expected_d, "F3 finite-difference dJ=nD",
            finite_difference=True,
        )
        print(
            f"F3 n={n_agents}: g={g:.6f}, D={direction_product:.10f}, "
            f"dJ(fd)={derivative:.10f}"
        )


def main():
    check_p0()
    check_f1()
    check_f2()
    check_f3()
    print(
        "PASS: all exhaustive and finite-difference checks; "
        f"exact atol={EXACT_ABS_TOL:g}, FD h={FD_STEP:g}, "
        f"atol={FD_ABS_TOL:g}, rtol={FD_REL_TOL:g}."
    )


if __name__ == "__main__":
    main()
