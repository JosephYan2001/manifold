"""One experiment-aware dashboard, adapted from the archived navigation plots.

Source: legacy_experiments_20260923.zip / cooperative_navigation/evaluation/
plotting.py. Retains lazy Agg/font setup, panels, step curves and independent-
seed bootstrap. New IDs, variable training paths and explicit x axes replace
the old fixed CSV names/conditions. No environment or torch imports are needed.
"""
from pathlib import Path
import json
import math
import re
import textwrap
import numpy as np

from .statistics import bootstrap


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams['font.sans-serif'] = [name for name in
        ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans') if name in available]
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _context(path, cache):
    """Training CSVs predate suite tags; recover only declared run context."""
    context = {}
    for part in path.parts:
        match = re.match(r'^([PTNW]-[A-Za-z]+)(?:_|$)', part)
        if match:
            context['experiment'] = match.group(1)
        if part in ('full', 'summary', 'nearest2'):
            context['observation_mode'] = part
    for parent in path.parents:
        manifest_path = parent / 'manifest.json'
        if manifest_path.exists():
            if manifest_path not in cache:
                cache[manifest_path] = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
            entry = cache[manifest_path].get('experiments', {}).get(context.get('experiment'), {})
            config = entry.get('config', {})
            for key in ('environment', 'relation_layers', 'label_mode', 'direction_check', 'return_check'):
                if key in config:
                    context[key] = config[key]
            break
    return context


def _read_data(directory, records):
    from .reporting import read_csv
    directory, cache = Path(directory), {}
    curves = []
    for path in sorted(directory.rglob('training.csv')):
        context = _context(path, cache)
        curves.extend({**context, **row, '_family': 'learning'} for row in read_csv(path))
    rows = list(records or [])
    if not rows:
        for path in sorted(directory.rglob('diagnostics.csv')):
            context = _context(path, cache)
            rows.extend({**context, **row} for row in read_csv(path))
        # Source monitoring already appears in training.csv; do not count its
        # episode rows a second time. Standalone transfer plots can use raw CSV.
        if not curves:
            for path in sorted(directory.rglob('evaluation_episodes.csv')):
                context = _context(path, cache)
                rows.extend({**context, **row} for row in read_csv(path))
    has_training_curves = bool(curves)
    for original in rows:
        row = dict(original)
        if _number(row.get('fit_steps')) is not None:
            family = 'fit'
        elif _number(row.get('sample_episodes', row.get('sample_size'))) is not None:
            family = 'estimate'
        elif _number(row.get('target_n')) is not None:
            family = 'transfer'
        elif (_number(row.get('source_steps_total')) is not None and not has_training_curves
              and any(_number(row.get(key)) is not None for key in
                      ('train_discounted_return', 'training_return', 'monitor_return', 'eval_discounted_return'))):
            family = 'learning'
        else:
            family = 'categorical'
        row['_family'] = family
        curves.append(row)
    return curves


AXES = {'learning': ('source_steps_total', 'All source joint steps'),
        'estimate': ('sample_episodes', 'Source training episodes'),
        'fit': ('fit_steps', 'Actor fitting steps'),
        'transfer': ('target_n', 'Target agents'),
        'categorical': (None, 'Declared diagnostic case')}
PANELS = {
    'learning': [
        ('Source return', ('eval_discounted_return', 'monitor_return', 'train_discounted_return', 'training_return')),
        ('Source task', ('eval_success', 'eval_throughput', 'train_success')),
        ('Direction', ('direction_error', 'direction_score', 'direction_check_score')),
        ('Actor realization', ('independent_forward_kl', 'forward_kl_training', 'train_forward_kl', 'policy_kl_old_to_new', 'old_new_kl')),
        ('Update acceptance', ('accepted',)),
        ('Training cost', ('wall_seconds', 'seconds'))],
    'estimate': [
        ('Direction estimation error', ('direction_error', 'learned_direction_error')),
        ('Episode score variance', ('validation_score_variance', 'sample_score_variance', 'exact_episode_score_variance')),
        ('Independent direction score', ('validation_score', 'validation_score_mean', 'sample_score_mean', 'score', 'direction_score')),
        ('Direction magnitude', ('direction_norm',)),
        ('Optimization cost', ('optimization_seconds', 'seconds', 'wall_seconds', 'optimizer_steps')),
        ('Gradient noise / time dependence', ('validation_gradient_trace_variance', 'cross_time_covariance', 'label_mse_against_exact_joint_advantage'))],
    'fit': [
        ('Forward KL: target to Actor', ('independent_forward_kl', 'forward_kl_validation', 'forward_kl_exact', 'forward_kl_training')),
        ('Actual return improvement', ('actual_return_gain', 'source_return_gain', 'return_gain')),
        ('Realization residual', ('direction_realization_residual', 'source_realization_residual')),
        ('Realized path derivative', ('realized_path_derivative',)),
        ('Old to new policy KL', ('old_new_kl',)),
        ('Fitting cost', ('fit_seconds', 'seconds'))],
    'transfer': [
        ('Frozen policy return', ('discounted_return', 'return', 'frozen_actor_return_diagnostic', 'expected_return')),
        ('Task completion / throughput', ('success_rate', 'success', 'throughput', 'coverage')),
        ('Finite update improvement', ('return_gain',)),
        ('Target direction drift', ('semantic_drift', 'target_direction_error')),
        ('Target derivative', ('target_path_derivative',)),
        ('Task cost', ('completion_time', 'first_delivery_time', 'collisions_per_step', 'evaluation_steps'))],
    'categorical': [
        ('Information loss', ('preprocessing_loss', 'information_loss', 'local_energy')),
        ('Direction / energy', ('local_direction_energy', 'direction_error', 'exact_episode_score_variance', 'variance')),
        ('Actual effect', ('return_gain', 'candidate_return', 'return_derivative', 'score', 'final_source_return_diagnostic')),
        ('Numerical residual', ('decomposition_residual', 'score_max_difference', 'variance_identity_residual', 'residual')),
        ('Input distinctions', ('input_states',)),
        ('Mechanism change', ('normalized_target_minus_coefficient', 'true_first_derivative'))]
}


def _select_panels(rows):
    families = [family for family in PANELS if any(row['_family'] == family for row in rows)]
    choices = []
    for family in families:
        subset = [row for row in rows if row['_family'] == family]
        selections = []
        for title, fields in PANELS[family]:
            metric = next((field for field in fields if any(_number(row.get(field)) is not None for row in subset)), None)
            if metric:
                selections.append((family, title, metric, subset))
        # At most three families share a 3x3 figure. A single experiment gets
        # six useful panels; mixed suites allocate panels across experiment axes.
        choices.append(selections)
    panels = []
    maximum = 6 if len(families) == 1 else 9
    while len(panels) < maximum and any(choices):
        for selections in choices:
            if selections and len(panels) < maximum:
                panels.append(selections.pop(0))
    return panels


def _series(rows, metric, axis_key):
    from .reporting import GROUP_KEYS
    groups = {}
    for row in rows:
        y = _number(row.get(metric))
        x = _number(row.get(axis_key)) if axis_key else 0.
        if axis_key == 'sample_episodes' and x is None:
            x = _number(row.get('sample_size'))
        if y is None or x is None:
            continue
        excluded = {axis_key, 'smoke_only', 'source_smoke_only'}
        if axis_key == 'sample_episodes':
            excluded.update(('sample_size', 'samples', 'n_samples', 'n_episodes'))
        if axis_key == 'target_n':
            excluded.update(('n_agents', 'target_agents', 'target_size'))
        if axis_key == 'source_steps_total':
            # The selected step is an update outcome, not a new training run.
            # A backtrack must not split one seed's learning curve into fragments.
            excluded.update(('step_size', 'fit_steps', 'budget_fraction'))
        tags = tuple((key, str(row[key])) for key in ('environment', *GROUP_KEYS)
                     if key not in excluded and row.get(key) not in (None, ''))
        unit = (str(row.get('seed', row.get('run_seed', 'exact'))),
                str(row.get('data_seed', row.get('batch_seed', 'exact'))))
        group = groups.setdefault(tags, {})
        group.setdefault(unit, {}).setdefault(x, []).append(y)
    return groups


def _plot_panel(ax, family, title, metric, rows):
    axis_key, xlabel = AXES[family]
    groups = _series(rows, metric, axis_key)
    keys = set(key for tags in groups for key, _ in tags)
    by_method = {}
    for tags in groups:
        by_method.setdefault(dict(tags).get('method', ''), []).append(dict(tags))
    varying = {key for key in keys if any(len({tags.get(key, '') for tags in cohort}) > 1
                                         for cohort in by_method.values())}
    for index, (tags, units) in enumerate(groups.items()):
        fields = [(key, value) for key, value in tags if key == 'method' or key in varying]
        label = ' | '.join(value if key in ('method', 'experiment', 'label', 'variant', 'actor_class', 'observation_mode')
                           else f'{key}={value}' for key, value in fields) or dict(tags).get('input_mode', 'exact')
        label = textwrap.fill(label, width=42)
        histories = {unit: {x: float(np.mean(ys)) for x, ys in history.items()} for unit, history in units.items()}
        grid = np.array(sorted({x for history in histories.values() for x in history}), float)
        if family == 'learning' and len(grid) > 120:
            grid = grid[np.unique(np.linspace(0, len(grid) - 1, 120).astype(int))]
        mean, low, high, counts = [], [], [], []
        for x in grid:
            values = []
            for history in histories.values():
                if x in history:
                    values.append(history[x])
                elif family == 'learning' and min(history) <= x <= max(history):
                    # Hold the last observed value inside a seed's actual range;
                    # never extrapolate a short run into a longer run's tail.
                    values.append(history[max(t for t in history if t <= x)])
            stats = bootstrap(values, repeats=1000, seed=913)
            mean.append(stats['mean'])
            low.append(np.nan if stats['ci_low'] is None else stats['ci_low'])
            high.append(np.nan if stats['ci_high'] is None else stats['ci_high'])
            counts.append(stats['count'])
        if family == 'categorical':
            grid = np.full(len(grid), index, float)
        line, = ax.plot(grid, mean, '.-', label=label,
            drawstyle='steps-post' if family == 'learning' else 'default', linewidth=1.5, markersize=4)
        if any(count > 1 for count in counts):
            ax.fill_between(grid, low, high, alpha=.14, color=line.get_color(),
                            step='post' if family == 'learning' else None)
        if len(grid) == 1 and counts[0] > 1:
            ax.errorbar(grid, mean, yerr=[[mean[0] - low[0]], [high[0] - mean[0]]],
                        fmt='none', color=line.get_color(), capsize=3)
    if family == 'categorical':
        ax.set_xticks([])
    if family in ('fit', 'estimate'):
        values = [x for units in groups.values() for history in units.values() for x in history]
        if values and min(values) > 0 and max(values) / min(values) >= 8:
            ax.set_xscale('log')
            ax.set_xticks(sorted(set(values)))
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())
    if family == 'transfer':
        values = sorted({x for units in groups.values() for history in units.values() for x in history})
        ax.set_xticks(values)
    ax.set(title=f'{title}\n{metric}', xlabel=xlabel)
    ax.grid(alpha=.18)
    if groups:
        ax.legend(fontsize=6.5 if len(groups) <= 8 else 5.5, ncol=1 if len(groups) <= 5 else 2,
                  loc='best', framealpha=.8)


def build_figure(directory, records=None):
    """Build a reviewable figure; callers own closing it."""
    rows = _read_data(directory, records)
    panels = _select_panels(rows)
    if not panels:
        return None
    plt = setup()
    nrows = 2 if len(panels) <= 6 else 3
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 4.3 * nrows), layout='constrained')
    for ax, panel in zip(axes.ravel(), panels):
        _plot_panel(ax, *panel)
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    fig.suptitle(f'{Path(directory).name}\nMeans and 95% bootstrap intervals across independent seeds/batches; one seed has no interval', fontsize=11)
    return fig


def plot_overview(directory, records=None):
    fig = build_figure(directory, records)
    if fig is None:
        return None
    path = Path(directory) / 'overview.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    setup().close(fig)
    return path
