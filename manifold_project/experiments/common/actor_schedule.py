"""Actor fitting rate, clocked before sampling and persisted across recovery."""
import math


def validate_actor_schedule(config):
    kind = config.get('actor_lr_schedule', 'fixed')
    if kind not in ('fixed', 'linear'):
        raise ValueError('actor_lr_schedule must be fixed or linear')
    final = config.get('actor_lr_final')
    if final is not None and (isinstance(final, bool) or not isinstance(final, (int, float))
                              or not math.isfinite(final) or not 0 < final <= config['actor_lr']):
        raise ValueError('actor_lr_final must be positive and no larger than actor_lr')
    end = config.get('actor_lr_decay_steps')
    if end is not None and (type(end) is not int or end <= 0):
        raise ValueError('actor_lr_decay_steps must be a positive integer')
    if kind == 'linear' and (final is None or end is None):
        raise ValueError('linear requires explicit actor_lr_final and actor_lr_decay_steps')
    if kind == 'linear' and config.get('environment', 'navigation') != 'navigation':
        raise ValueError('Actor rate scheduling currently supports navigation AN only')


def initial_schedule(config):
    validate_actor_schedule(config)
    return dict(version='actor-source-clock-v1', schedule=config.get('actor_lr_schedule', 'fixed'),
                initial_lr=config['actor_lr'], final_lr=config.get('actor_lr_final'),
                end_source_steps=config.get('actor_lr_decay_steps'), start_source_steps=None)


def restore_schedule(config, saved):
    expected = initial_schedule(config)
    if saved is None:
        if expected['schedule'] != 'fixed':
            raise ValueError('Linear schedule checkpoint is missing its persisted clock')
        return expected  # Legacy fixed-rate checkpoints retain fixed-rate behavior.
    for key in expected.keys() - {'start_source_steps'}:
        if saved.get(key) != expected[key]:
            raise ValueError(f'Cannot change Actor schedule on recovery: {key}')
    start = saved.get('start_source_steps')
    if start is not None and (type(start) is not int or start < 0):
        raise ValueError('Invalid saved Actor schedule start')
    if saved['schedule'] == 'linear' and start is not None and start >= saved['end_source_steps']:
        raise ValueError('Actor schedule endpoint must follow its start')
    return dict(saved)


def rate_at(state, source_steps):
    """Initialize s0 once; caller checkpoints it before first sampling starts."""
    if type(source_steps) is not int or source_steps < 0:
        raise ValueError('source_steps must be a nonnegative integer')
    if state['start_source_steps'] is None:
        state['start_source_steps'] = source_steps
    start, end = state['start_source_steps'], state['end_source_steps']
    if source_steps < start:
        raise ValueError('Source clock cannot precede the saved schedule start')
    progress = None
    if end is not None and end > start:
        progress = min(1., max(0., (source_steps-start)/(end-start)))
    lr = state['initial_lr']
    if state['schedule'] == 'linear':
        if progress is None:
            raise ValueError('Actor schedule endpoint must follow its start')
        lr += (state['final_lr']-lr)*progress
    return dict(actor_lr_schedule=state['schedule'], actor_lr_used=lr,
                actor_lr_source_steps=source_steps, actor_lr_start_source_steps=start,
                actor_lr_end_source_steps=end, actor_lr_progress=progress)
