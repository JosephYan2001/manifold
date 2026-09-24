"""Native application adapters with a common CTDE interface.

Actor observations are dictionaries of ``self``, ``entities`` and valid-item
``mask`` arrays. Team reward is a scalar; a deadline is a true terminal in both
finite-horizon protocols. Collector batch cuts are not environment terminations.
"""


def make_env(environment, config, n_agents=None, render_mode=None):
    if environment == "pair_entities":
        from .pair_entities import PairEntities
        return PairEntities(config, n_agents=n_agents, render_mode=render_mode)
    if environment == "navigation":
        from .navigation import Navigation
        return Navigation(config, n_agents=n_agents, render_mode=render_mode)
    if environment == "warehouse":
        from .warehouse import Warehouse
        return Warehouse(config, n_agents=n_agents, render_mode=render_mode)
    raise ValueError(f"Unknown application environment: {environment}")
