"""Method objectives and acceptance gates are independent settings."""


def direction_check_enabled(condition, config=None):
    if condition in ('mappo', 'ippo', 'direct'):
        return False
    return (config or {}).get('direction_gate', condition not in ('no_direction_check', 'no_checks'))


def return_check_enabled(condition, config=None):
    if condition in ('mappo', 'ippo'):
        return False
    return (config or {}).get('return_gate', condition not in ('no_return_check', 'no_checks'))


def objective_kind(config, condition):
    return config.get('direction_objective', 'sampled_square' if condition == 'sampled' else 'analytic_fisher')


def effective_settings(config, condition):
    inferred = {'direct':'DA', 'mappo':'MAPPO-E', 'ippo':'IPPO-E', 'sampled':'SA'}.get(condition,'AN')
    return dict(requested_method=config.get('requested_method', inferred),
                effective_objective=(objective_kind(config, condition) if condition not in ('direct','mappo','ippo')
                                     else {'direct':'direct_surrogate','mappo':'ppo','ippo':'ppo'}[condition]),
                direction_gate=direction_check_enabled(condition, config),
                return_gate=return_check_enabled(condition, config),
                semantics_version=config.get('semantics_version', 'legacy'))


def validate_legacy_resume(saved, method=None):
    """Never silently relabel a legacy buggy run as a corrected experiment."""
    c = saved['config']
    if c.get('semantics_version'):
        return
    public = c.get('current_config', {})
    method = method or saved.get('current_method')
    if method is None and len(public.get('methods', [])) == 1:
        method = public['methods'][0]
    bad_sa = method == 'SA' and saved['condition'] != 'sampled'
    bad_da = method == 'DA' and public.get('return_check') is False
    if bad_sa or bad_da:
        raise ValueError('Legacy SA/DA checkpoint used incompatible objective/gate semantics; '
                         'keep it as historical evidence and start a new corrected run.')
