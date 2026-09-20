"""Read-only training diagnostics; no evaluation, model access or training RNG use."""
import json
from pathlib import Path
import numpy as np
from .plotting import setup, LABELS
from ..training.storage import atomic


def number(value, precision=4):
    return '-' if value is None else f'{value:.{precision}g}'


def progress_line(condition, seed, budget, row, evaluation=None):
    if condition in ('mappo', 'ippo'):
        decision = 'PPO-update'
    elif row.get('accepted'):
        decision = 'accepted'
    elif row.get('direction_pass') is False:
        decision = 'direction-rejected'
    elif row.get('return_difference') is not None:
        decision = 'return-rejected'
    else:
        decision = 'no-candidate'
    used = row['source_steps']
    fields = [f'{condition} seed={seed} round={row["round"]}',
              f'steps={used}/{budget} ({used / budget:.1%})',
              f'train_J={number(row.get("train_J"))}',
              f'critic_mse={number(row.get("critic_mse"))}', decision]
    if condition in ('mappo', 'ippo'):
        fields += [f'ppo_KL={number(row.get("ppo_kl"))}',
                   f'clip={number(row.get("clip_fraction"))}']
    else:
        fields += [f'dir_score={number(row.get("direction_score"))}',
                   f'return_delta={number(row.get("return_difference"))}',
                   f'fit_KL={number(row.get("fit_kl_after"))}',
                   f'candidates={row.get("candidates", 0)}']
    fields.append(f'update_s={number(row.get("update_seconds"))}')
    if evaluation:
        fields.append(f'last_eval@{evaluation["budget_checkpoint"]}: '
                      f'J={number(evaluation["J"])} '
                      f'coverage={evaluation["coverage"]:.1%} '
                      f'distance={number(evaluation["distance"])}')
        if 'tail_coverage' in evaluation:
            fields.append(f'last_eval_tail_coverage={evaluation["tail_coverage"]:.1%}')
    return ' | '.join(fields)


def plot_monitor(path, condition, seed, budget, rounds, curves, complete=False):
    """Overwrite one 3x3 dashboard, using committed diagnostics only."""
    plt = setup()
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), layout='constrained')
    axes = axes.ravel()
    try:
        def series(ax, rows, key, label, xkey='source_steps', **kwargs):
            values = [(r[xkey], r.get(key)) for r in rows]
            # Keep missing points as gaps, rather than implying a candidate existed.
            if any(v is not None for _, v in values):
                kwargs.setdefault('marker', '.')
                kwargs.setdefault('markersize', 4)
                ax.plot([x for x, _ in values],
                        [float(v) if v is not None else np.nan for _, v in values],
                        label=label, **kwargs)

        series(axes[0], rounds, 'train_J', '逐轮旧策略训练批回报', alpha=.4)
        if rounds:
            smooth = []
            for i, row in enumerate(rounds):
                values = [r['train_J'] for r in rounds[max(0, i-19):i+1] if r.get('train_J') is not None]
                smooth.append(dict(source_steps=row['source_steps'], mean=float(np.mean(values)) if values else None))
            series(axes[0], smooth, 'mean', '最近20轮均值')
        for ax, key, label in zip(axes[1:4], ('J', 'coverage', 'distance'),
                                 ('独立评价回报', '独立评价覆盖率', '独立评价终点距离')):
            series(ax, curves, key, label, 'budget_checkpoint', marker='o', drawstyle='steps-post')
            errors = [r for r in curves if r.get(key + '_se') is not None]
            if errors:
                ax.errorbar([r['budget_checkpoint'] for r in errors], [r[key] for r in errors],
                            yerr=[1.96*r[key+'_se'] for r in errors], fmt='none', capsize=2, alpha=.5)
        series(axes[2], curves, 'mean_coverage', '窗口平均覆盖率', 'budget_checkpoint', linestyle='--')
        series(axes[2], curves, 'tail_coverage', '后半窗口覆盖率', 'budget_checkpoint', linestyle=':')
        author = any('author_stats' in row for row in rounds)
        series(axes[4], rounds, 'critic_mse',
               'Critic更新后全训练批原尺度MSE' if author else 'Critic最后训练小批次MSE')
        for key, label in (('fit_kl_after', '目标拟合残差'), ('policy_step_kl', '候选与旧策略KL'),
                           ('ppo_kl', 'PPO更新后全训练批KL' if author else 'PPO最后小批次KL')):
            series(axes[5], rounds, key, label)
        rates = [dict(source_steps=row['source_steps'], rate=float(np.mean(
            [r['accepted'] for r in rounds[max(0, i-19):i+1]]))) for i, row in enumerate(rounds)]
        series(axes[6], rates, 'rate', '最近20轮接受/更新比例')
        series(axes[7], rounds, 'direction_score', '方向检查分数')
        series(axes[8], rounds, 'return_difference', '最后候选回报检查差')
        for ax in axes[7:]:
            ax.axhline(0, color='gray', linewidth=.7, linestyle='--')
        axes[2].set_ylim(-.03, 1.03)
        axes[6].set_ylim(-.03, 1.03)
        # Large initialization errors otherwise make nonzero late errors look like zero.
        # Symlog keeps true zeros visible; values above 1 use a logarithmic scale.
        axes[4].set_yscale('symlog', linthresh=1)
        titles = ('训练批回报（非独立评价）', '独立源回报（已有评价节点）', '终点覆盖率',
                  '终点距离', 'Critic训练MSE（对数刻度，0附近线性）', 'KL诊断（统计口径不同）',
                  '策略更新比例', '方向检查（正值通过）', '回报检查（正值通过）')
        for ax, title in zip(axes, titles):
            ax.set(title=title, xlabel='累计源团队步')
            ax.grid(alpha=.15)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, '暂无数据 / 不适用', transform=ax.transAxes, ha='center')
        used = rounds[-1]['source_steps'] if rounds else 0
        fig.suptitle(f'{LABELS.get(condition, condition)} | seed={seed} | '
                     f'已提交轮数={len(rounds)} | 源步={used:,}/{budget:,} | '
                     f'{"已完成" if complete else "最近已提交数据"}\n'
                     '评价误差条为回合均值的约95%误差范围；不是训练种子区间。训练与评价不混合。')
        atomic(path, lambda temp: fig.savefig(temp, format='png', dpi=120))
    finally:
        plt.close(fig)


class TrainingMonitor:
    def __init__(self, path, every=10):
        if every < 1:
            raise ValueError('plot interval must be positive')
        self.path, self.every = Path(path), every
        self.last_round, self.last_evaluations, self.finished = -1, -1, False

    def __call__(self, runner):
        if (self.last_round < 0 or runner.round-self.last_round >= self.every or
                len(runner.curves) != self.last_evaluations or runner.complete != self.finished):
            try:
                plot_monitor(self.path, runner.condition, runner.seed, runner.c['budget'],
                             runner.rounds, runner.curves, runner.complete)
            except Exception as error:
                # Reporting errors (e.g. a locked PNG) must not invalidate a committed update.
                print(f'[monitor] 图像更新失败，下次重试: {error}', flush=True)
            self.last_round = runner.round
            self.last_evaluations = len(runner.curves)
            self.finished = runner.complete


class EventHistory:
    """Incrementally tail JSONL, including logs written before live monitoring existed."""
    def __init__(self, path):
        self.path, self.offset = Path(path), 0
        self.rounds, self.curves, self.pending = {}, {}, {}

    def read(self):
        if not self.path.exists():
            return 0
        if self.path.stat().st_size < self.offset:
            self.offset = 0
            self.rounds.clear()
            self.curves.clear()
            self.pending.clear()
        changes = 0
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    self.offset = position  # retry a partially written final line on next poll
                    break
                self.offset = stream.tell()
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue  # tolerate interrupted historical lines
                record = event['record']
                kind = event['stream']
                index = record.get('round')
                if kind == 'train_batch':
                    self.pending[index] = {'train_J':record['J_mean']}
                elif kind == 'optimization' and record.get('module') == 'critic':
                    self.pending.setdefault(index, {})['critic_mse'] = record.get('value_mse')
                elif kind == 'commit':
                    index = record['completed_round']
                    row = {**self.pending.pop(index, {}), **record, 'round':index}
                    self.rounds[index] = row
                    changes += 1
                elif kind == 'evaluation':
                    self.curves[record['budget_checkpoint']] = record
                    changes += 1
        return changes
