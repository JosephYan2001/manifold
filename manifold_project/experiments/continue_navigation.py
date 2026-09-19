"""Fork a completed no_checks run into matched direction-epoch continuations."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from manifold_project.experiments.cooperative_navigation.configs import validate
from manifold_project.experiments.cooperative_navigation.run_experiments import ROOT, metadata, new_directory
from manifold_project.experiments.cooperative_navigation.training.runner import Runner, minimum_cost
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt, save_pt, save_json
from manifold_project.experiments.cooperative_navigation.evaluation.evaluate import evaluate
from manifold_project.experiments.cooperative_navigation.evaluation.reporting import write_csv, summaries, read_csv
from manifold_project.experiments.cooperative_navigation.evaluation.monitoring import TrainingMonitor
from manifold_project.experiments.cooperative_navigation.evaluation.plotting import setup

KIND = 'navigation_direction_reuse_continuation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(checkpoint, state, epochs, budget, device=None, eval_episodes=256, eval_seed=90300):
    if not state.get('complete') or state.get('condition') != 'no_checks':
        raise ValueError('父模型必须是已完成no_checks训练的final.pt，不支持best.pt或未完成运行')
    if state['config']['direction_label'] != 'mc':
        raise ValueError('本实验固定MC标签，请使用MC no_checks父模型')
    if not epochs or len(set(epochs)) != len(epochs) or any(e <= 0 for e in epochs):
        raise ValueError('direction-epochs必须是互不重复的正整数')
    config = deepcopy(state['config'])
    config.update(budget=budget, device=device or config['device'], seeds=[state['seed']],
                  eval_fractions=[i/10 for i in range(11)], eval_episodes=eval_episodes,
                  final_episodes=eval_episodes, protocol_status='candidate_not_frozen', method_overrides={})
    validate(config)
    if budget < minimum_cost(config, 'no_checks'):
        raise ValueError('新增预算不足以完成一轮')
    # Nested continuations remain traceable without counting inherited costs twice.
    inherited = state.get('branch_origin', {}).get('parent_source_steps', 0)
    origin = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest(checkpoint),
                  seed=state['seed'], condition=state['condition'], parent_rounds=state['round'],
                  parent_source_steps=inherited+sum(state['costs'].values()),
                  parent_config=deepcopy(state['config']))
    parent_manifest = Path(checkpoint).resolve().parents[3]/'manifest.json'
    if parent_manifest.exists():
        origin['parent_metadata'] = json.loads(parent_manifest.read_text(encoding='utf-8')).get('metadata')
    jobs = []
    for e in epochs:
        branch = deepcopy(config)
        branch['direction_epochs'] = e
        validate(branch)
        variant = f'direction_epochs_{e}'
        jobs.append(dict(variant=variant, condition='no_checks', seed=state['seed'],
                         path=f'{variant}/seed_{state["seed"]}', config=branch))
    return dict(format_version=1, kind=KIND, config=config, source=origin,
                evaluation_seed=eval_seed, jobs=jobs,
                protocol='Fresh branch ledgers; inherited model/optimizer/RNG/round and sampling indices; '
                         'common evaluation scenes at 0..100% of additional budget, final included.')


class ContinuationRunner(Runner):
    def __init__(self, directory, config, origin, variant, evaluation_seed, *, parent=None,
                 resume=False, progress=None):
        self.origin, self.variant, self.evaluation_seed = origin, variant, evaluation_seed
        if not resume and parent is None:
            raise ValueError('新分支缺少父状态')
        super().__init__(directory, config, 'no_checks', origin['seed'], resume=resume, progress=progress)
        if not resume:
            self.actor.load_state_dict(parent['actor'])
            self.critic.load_state_dict(parent['critic'])
            self.actor_optimizer.load_state_dict(deepcopy(parent['actor_optimizer']))
            self.critic_optimizer.load_state_dict(deepcopy(parent['critic_optimizer']))
            # Keep absolute rounds for direction initialization/minibatch RNG and
            # episode counters for fresh on-policy sampling. Costs/history start at zero.
            self.round = parent['round']
            self.sampler.counts = Counter(parent['counts'])
            torch.set_rng_state(parent['torch_rng'].cpu())
            if config['device'] == 'cuda' and parent.get('cuda_rng') is not None:
                if len(parent['cuda_rng']) != torch.cuda.device_count():
                    raise ValueError('父CUDA随机状态的设备数与当前设备数不一致；请选择一致环境或CPU分支')
                torch.cuda.set_rng_state_all([s.cpu() for s in parent['cuda_rng']])
            self.checkpoint()  # Recovery before the initial evaluation is valid.

    def restore(self, state):
        if (state.get('branch_origin') != self.origin or state.get('branch_variant') != self.variant
                or state.get('branch_evaluation_seed') != self.evaluation_seed):
            raise ValueError('分支checkpoint与manifest的来源或评价协议不一致')
        super().restore(state)

    def checkpoint(self):
        state = {name: getattr(self, name) for name in ('round', 'curves', 'rounds', 'snapshots',
                 'train_seconds', 'eval_seconds', 'eval_steps', 'best', 'complete')}
        state.update(format_version=1, config=self.c, condition=self.condition, seed=self.seed,
                     input_dim=self.input_dim, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                     actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                     costs=dict(self.sampler.costs), counts=dict(self.sampler.counts), optim_steps=dict(self.optim_steps),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     branch_origin=self.origin, branch_variant=self.variant,
                     branch_evaluation_seed=self.evaluation_seed)
        save_pt(self.directory/'checkpoints/final.pt', state)
        if self.best is not None:
            save_pt(self.directory/'checkpoints/best.pt', self.best)

    def evaluate_node(self, actor, budget_node, actor_cost):
        started = time.perf_counter()
        count = self.c['eval_episodes']
        print(f'  [eval] {self.variant} added_steps={budget_node} episodes={count}', flush=True)
        # All branches and all nodes, including final, use the same fresh bank.
        metrics, episodes = evaluate(actor, self.c, self.condition, self.evaluation_seed,
                                     f'continuation-source-{self.evaluation_seed}', count)
        for key in ('J', 'coverage', 'distance'):
            values = np.asarray([e[key] for e in episodes], dtype=np.float64)
            metrics[key+'_se'] = float(values.std(ddof=1)/np.sqrt(count)) if count > 1 else None
        self.eval_seconds += time.perf_counter()-started
        self.eval_steps += count*self.c['horizon']
        row = dict(condition=self.variant, algorithm_condition=self.condition, seed=self.seed,
                   budget_checkpoint=budget_node, actor_source_steps=actor_cost,
                   total_actor_source_steps=self.origin['parent_source_steps']+actor_cost,
                   episodes=count, **metrics)
        self.curves.append(row)
        self.log('evaluation', **row)
        print(f'  [eval] {self.variant} J={metrics["J"]:.4f} coverage={metrics["coverage"]:.1%}', flush=True)
        if self.best is None or metrics['J'] > self.best['score']:
            self.best = dict(actor=deepcopy(actor.state_dict()), config=self.c, input_dim=self.input_dim,
                             score=metrics['J'], selection='branch_evaluation_diagnostic_only',
                             budget_checkpoint=budget_node, actor_source_steps=actor_cost,
                             branch_origin=self.origin)

    def summary(self):
        result = super().summary()
        result.update(condition=self.variant, algorithm_condition=self.condition,
                      rounds=len(self.rounds), total_rounds=self.round,
                      parent_rounds=self.origin['parent_rounds'],
                      parent_source_steps=self.origin['parent_source_steps'],
                      total_source_steps=self.origin['parent_source_steps']+self.sampler.used,
                      initial_J=self.curves[0]['J'], delta_J=self.curves[-1]['J']-self.curves[0]['J'])
        return result


def monitor_for(path, every, variant):
    monitor = TrainingMonitor(path, every)
    def update(runner):
        # Only the display name changes; the algorithm condition stays no_checks.
        monitor(SimpleNamespace(condition=variant, seed=runner.seed, c=runner.c, round=runner.round,
                                rounds=runner.rounds, curves=runner.curves, complete=runner.complete))
    return update


def tables(directory, manifest):
    runs, curves = [], []
    for job in manifest['jobs']:
        path = directory/job['path']
        summary = path/'summary.json'
        if summary.exists():
            row = json.loads(summary.read_text(encoding='utf-8'))
            row.update({f'cost_{k}': v for k, v in row.pop('costs').items()})
        else:
            row = dict(condition=job['variant'], seed=job['seed'], status='pending')
            if (path/'failure.json').exists():
                row['status'] = 'failed'
        runs.append(row)
        checkpoint = path/'checkpoints/final.pt'
        if checkpoint.exists():
            curves.extend(load_pt(checkpoint)['curves'])
    write_csv(directory/'training_results.csv', runs)
    write_csv(directory/'training_summary.csv', summaries(runs, manifest['config']))
    write_csv(directory/'learning_curves.csv', curves)


def plot_overview(directory, manifest):
    rows = read_csv(directory/'learning_curves.csv')
    if not rows:
        return
    plt = setup()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    try:
        for job in manifest['jobs']:
            group = [r for r in rows if r['condition'] == job['variant']]
            if not group:
                continue
            x = [int(r['budget_checkpoint']) for r in group]
            for ax, key, title in zip(axes.flat, ('J', 'coverage', 'distance', 'delta_J'),
                                      ('独立评价回报', '终点覆盖率', '终点最近距离', '相对续训起点的回报变化')):
                values = [float(r['J'])-float(group[0]['J']) if key == 'delta_J' else float(r[key]) for r in group]
                line, = ax.step(x, values, where='post', marker='.', label=job['variant'])
                if key != 'delta_J':
                    valid = [r for r in group if r.get(key+'_se') not in (None, '')]
                    ax.errorbar([int(r['budget_checkpoint']) for r in valid], [float(r[key]) for r in valid],
                                yerr=[1.96*float(r[key+'_se']) for r in valid], fmt='none', alpha=.25,
                                color=line.get_color())
                ax.set(title=title, xlabel='新增源团队步')
                ax.grid(alpha=.2)
        axes[0, 0].legend()
        axes[0, 1].set_ylim(-.02, 1.02)
        axes[1, 1].axhline(0, color='gray', linestyle='--')
        fig.suptitle(f'同一final模型分支续训 | 已有源步={manifest["source"]["parent_source_steps"]:,}\n'
                     '误差条为评价回合均值的不确定性，不是训练种子区间；起点差值不作显著性结论')
        from manifold_project.experiments.cooperative_navigation.training.storage import atomic
        atomic(directory/'overview.png', lambda p: fig.savefig(p, format='png', dpi=130))
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='新建分支必须显式指定已完成的 no_checks MC final 模型')
    parser.add_argument('--direction-epochs', type=int, nargs='+')
    parser.add_argument('--additional-budget', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--eval-seed', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume-suite', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-every', type=int, default=10)
    args = parser.parse_args(argv)
    if args.plot_every < 0:
        parser.error('plot-every must be nonnegative')
    parent = None
    if args.resume_suite:
        if any(getattr(args, key) is not None for key in ('checkpoint', 'direction_epochs', 'additional_budget',
                                                         'device', 'eval_episodes', 'eval_seed', 'output')):
            parser.error('恢复分支套件时不接受来源、配置或预算覆盖')
        directory = args.resume_suite.resolve()
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('kind') != KIND:
            raise ValueError('只支持恢复本入口创建的分支套件')
    else:
        if args.checkpoint is None:
            parser.error('新建分支必须提供 --checkpoint；旧导航结果已清理，不再使用默认父模型')
        checkpoint = args.checkpoint.resolve()
        parent = load_pt(checkpoint)
        manifest = make_plan(checkpoint, parent, args.direction_epochs or [16, 4],
                             args.additional_budget if args.additional_budget is not None else 1000000,
                             args.device, args.eval_episodes if args.eval_episodes is not None else 256,
                             args.eval_seed if args.eval_seed is not None else 90300)
        directory = args.output.resolve() if args.output else None
    if args.dry_run:
        print(json.dumps(dict(kind=KIND, source=manifest['source'], jobs=manifest['jobs'],
                              evaluation_seed=manifest['evaluation_seed'],
                              total_additional_source_budget=sum(j['config']['budget'] for j in manifest['jobs']),
                              evaluation_steps_per_branch=len(manifest['config']['eval_fractions'])*
                              manifest['config']['eval_episodes']*manifest['config']['horizon']),
                         ensure_ascii=False, indent=2))
        return 0
    if manifest['config']['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('配置要求CUDA，但当前解释器无法使用CUDA')
    current = metadata()
    current['continuation_sha256'] = digest(__file__)
    if args.resume_suite:
        for key in ('code_sha256', 'continuation_sha256', 'versions', 'runtime_environment'):
            if current[key] != manifest['metadata'][key]:
                raise ValueError(f'分支恢复的{key}与原运行不一致，请使用原代码和运行环境')
    else:
        if directory is not None:
            source = Path(manifest['source']['checkpoint'])
            if source.is_relative_to(directory) or directory.is_relative_to(source.parents[1]):
                raise ValueError('输出目录不能包含或写入父模型目录')
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f'输出目录非空: {directory}；恢复请使用本脚本的--resume-suite')
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = new_directory('continue_reuse')
        manifest['metadata'] = current
        save_json(directory/'manifest.json', manifest)
    print(f'分支套件: {directory}\n原有源步: {manifest["source"]["parent_source_steps"]:,}; '
          'round沿用原编号，steps和图表预算只统计新增部分。', flush=True)
    status_path = directory/'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'jobs': {}}
    failed = False
    for job in manifest['jobs']:
        path = directory/job['path']
        if (path/'summary.json').exists():
            status['jobs'][job['path']] = 'complete'
            continue
        saved = (path/'checkpoints/final.pt').exists()
        if not saved and parent is None:
            source = Path(manifest['source']['checkpoint'])
            if digest(source) != manifest['source']['checkpoint_sha256']:
                raise ValueError('父checkpoint已变化，无法初始化尚未开始的分支')
            parent = load_pt(source)
        if not saved and (path/'config.json').exists():
            raise RuntimeError(f'分支初始化未完成且没有checkpoint: {path}；请新建输出目录')
        print(f'开始 {job["variant"]}: epochs={job["config"]["direction_epochs"]}, '
              f'新增预算={job["config"]["budget"]:,}', flush=True)
        status['jobs'][job['path']] = 'running'
        save_json(status_path, status)
        try:
            progress = monitor_for(directory/'training_monitor.png', args.plot_every, job['variant']) \
                if args.plot and args.plot_every else None
            runner = ContinuationRunner(path, job['config'], manifest['source'], job['variant'],
                                        manifest['evaluation_seed'], parent=parent, resume=saved, progress=progress)
            runner.run()
            status['jobs'][job['path']] = 'complete'
        except KeyboardInterrupt:
            status['jobs'][job['path']] = 'interrupted'
            save_json(status_path, status)
            tables(directory, manifest)
            raise
        except Exception as error:
            failed = True
            status['jobs'][job['path']] = 'failed'
            if not (path/'failure.json').exists():
                save_json(path/'failure.json', dict(type=type(error).__name__, message=str(error)))
            traceback.print_exc()
        save_json(status_path, status)
        tables(directory, manifest)
        if args.plot:
            plot_overview(directory, manifest)
    save_json(status_path, status)
    tables(directory, manifest)
    if args.plot:
        plot_overview(directory, manifest)
    print(f'完成：{directory / "training_results.csv"}；重点比较delta_J、J、coverage与新增成本。', flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
