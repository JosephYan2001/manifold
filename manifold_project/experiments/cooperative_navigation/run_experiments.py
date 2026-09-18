"""Run deduplicated source jobs, optional frozen transfer and independent audits."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# CUDA deterministic matrix products require this before CUDA/cuBLAS initializes.
# Preserve an explicitly configured workspace (for example, :16:8).
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from manifold_project.experiments.cooperative_navigation.configs import load_config, validate, condition_config, EXPERIMENTS

ROOT = Path(__file__).resolve().parent


def fingerprint():
    files = sorted([*ROOT.rglob('*.py'),* (ROOT/'configs').glob('*.json')])
    digest = hashlib.sha256()
    for path in files:
        if any(part in ('results','tests','__pycache__') for part in path.relative_to(ROOT).parts):
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def metadata():
    import torch
    versions = {name:importlib.metadata.version(name) for name in
                ('mpe2','pettingzoo','gymnasium','numpy','torch','matplotlib','pygame-ce','scipy')}
    result = subprocess.run(['git','-c',f'safe.directory={ROOT.parents[2].as_posix()}','rev-parse','HEAD'],
                            cwd=ROOT,capture_output=True,text=True)
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                processor=platform.processor(),versions=versions,code_sha256=fingerprint(),
                git_head=result.stdout.strip() if result.returncode==0 else None,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                runtime_environment={key:os.environ.get(key) for key in
                                     ('KMP_DUPLICATE_LIB_OK','MKL_THREADING_LAYER','OMP_NUM_THREADS',
                                      'CUBLAS_WORKSPACE_CONFIG')},
                dtype='float32; episode/statistical aggregation float64')


def new_directory(profile, tag=''):
    parent = ROOT/'results'
    parent.mkdir(exist_ok=True)
    for i in range(1,10000):
        path = parent/f'nav_{i:02d}_{profile}{tag}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('结果编号空间已满')

#  python manifold/manifold_project/experiments/cooperative_navigation/run_experiments.py --profile pilot --conditions ours --config manifold/manifold_project/experiments/cooperative_navigation/configs/study_v3/reference.json --device cuda --output manifold/manifold_project/experiments/cooperative_navigation/results/nav_v3_ours_reference_s40_10m --plot
def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['smoke','pilot','formal'],default=None)
    p.add_argument('--experiments',nargs='+',choices=sorted(EXPERIMENTS),default=['N-C','N-A'])
    p.add_argument('--conditions', nargs='+', choices=sorted({c for v in EXPERIMENTS.values() for c in v}),
                   help='仅新建pilot/smoke：从实验计划中选取训练条件，例如 ours')
    p.add_argument('--config',type=Path,help='覆盖 source + profile 的 JSON 配置')
    p.add_argument('--budget',type=int)
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--threads',type=int)
    p.add_argument('--output',type=Path,help='新目录；已有运行用 --resume-suite')
    p.add_argument('--resume-suite',type=Path,help='严格按原配置恢复，或只做 N-T/N-D')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--plot',action='store_true')
    p.add_argument('--plot-every',type=int,default=10,metavar='ROUNDS',
                   help='配合 --plot，每隔若干已提交轮次更新 training_monitor.png；0 仅生成最终图')
    p.add_argument('--audit',action='store_true',help='训练前启用固定轮次候选快照，结束后评价')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.plot_every < 0:
        raise ValueError('--plot-every 必须为非负整数')
    if args.resume_suite:
        if args.output or args.config or args.budget or args.seeds or args.device or args.threads or args.profile or args.conditions:
            raise ValueError('--resume-suite 使用原配置，不接受新的 profile/config/预算/设备参数')
        directory = args.resume_suite.resolve()
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        config,profile = manifest['config'],manifest['profile']
        if manifest['metadata']['code_sha256'] != fingerprint():
            raise ValueError('代码指纹已变化；不得将不同实现混入原套件，请创建新版本运行')
        if (args.audit or 'N-D' in args.experiments) and not manifest['audit']:
            raise ValueError('原训练未预先启用审计，无法补回未保存的候选；请新建 --audit 套件')
        requested = set(c for e in args.experiments for c in EXPERIMENTS[e])
        if args.experiments == manifest['experiments'] and 'selected_conditions' in manifest:
            requested = set(manifest['selected_conditions'])
        available = {j['condition'] for j in manifest['jobs']}
        if not requested <= available:
            raise ValueError(f'套件未规划条件: {sorted(requested-available)}，请新建完整套件')
    else:
        profile = args.profile or 'pilot'
        config = load_config(profile,args.config)
        for key in ('budget','seeds','device','threads'):
            if getattr(args,key) is not None:
                config[key] = getattr(args,key)
        if 'N-L' in args.experiments:
            if any(e not in ('N-L',) for e in args.experiments):
                raise ValueError('N-L 必须单独运行，避免标签对齐与主结果混合')
            if not args.config:
                config.update(json.loads((ROOT/'configs/label_alignment.json').read_text(encoding='utf-8')))
            if config['direction_label'] != config['ppo_label'] or config['ppo_normalize_advantage']:
                raise ValueError('N-L 要求方向/PPO 同为 mc 或 gae，且关闭 PPO 优势标准化')
        validate(config)
        conditions = list(dict.fromkeys(c for e in args.experiments for c in EXPERIMENTS[e]))
        if args.conditions:
            if profile == 'formal':
                raise ValueError('--conditions 仅用于pilot/smoke开发，不改变正式实验计划')
            if not set(args.conditions) <= set(conditions):
                raise ValueError('--conditions 必须属于 --experiments 对应的条件')
            conditions = [c for c in conditions if c in args.conditions]
        if not conditions:
            raise ValueError('N-T 只复用已有 final，请指定 --resume-suite；或者与 N-C 一同规划')
        audit = args.audit or 'N-D' in args.experiments
        jobs = [dict(condition=c,seed=s,path=f'{c}/seed_{s}') for c in conditions for s in config['seeds']]
        manifest = dict(format_version=1,profile=profile,config=config,experiments=args.experiments,
                        audit=audit,jobs=jobs)
        if args.conditions:
            manifest['selected_conditions'] = conditions
        directory = args.output
    # Planning works without importing torch or MPE2 and does not create directories.
    if args.dry_run:
        print(json.dumps(dict(profile=profile,config=config,jobs=manifest['jobs'],audit=manifest['audit'],
                              total_source_budget=len(manifest['jobs'])*config['budget']),ensure_ascii=False,indent=2))
        return 0
    if profile == 'formal' and config['protocol_status'] != 'frozen':
        raise ValueError('formal 配置仍为候选。完成 pilot 后用 --config 提供 protocol_status=frozen 的参数冻结文件；预算也需一起确认。')
    from manifold_project.experiments.cooperative_navigation.training.storage import save_json
    from manifold_project.experiments.cooperative_navigation.training.runner import Runner,minimum_cost
    from manifold_project.experiments.cooperative_navigation.evaluation.reporting import source_tables
    from manifold_project.experiments.cooperative_navigation.evaluation.transfer import transfer_suite,audit_suite
    from manifold_project.experiments.cooperative_navigation.evaluation.plotting import plot_suite,plot_transfer
    from manifold_project.experiments.cooperative_navigation.evaluation.monitoring import TrainingMonitor
    import torch
    torch.set_num_threads(config['threads'])
    if any(config['budget'] < minimum_cost(config,job['condition']) for job in manifest['jobs']):
        raise ValueError('预算不足以完成计划中某条件的一轮；请增加预算')
    if args.resume_suite and metadata()['versions'] != manifest['metadata']['versions']:
        raise ValueError('依赖版本已变化，请使用原环境恢复或新建运行')
    if not args.resume_suite:
        manifest['metadata'] = metadata()
        if directory is None:
            directory = new_directory(profile,'_labels' if 'N-L' in args.experiments else '')
        else:
            directory = directory.resolve()
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f'输出目录非空: {directory}')
            directory.mkdir(parents=True,exist_ok=True)
        save_json(directory/'manifest.json',manifest)
    print(f'套件: {directory}',flush=True)
    if args.plot and args.plot_every:
        print(f'训练监控: {directory / "training_monitor.png"}（每{args.plot_every}轮及新评价后刷新）', flush=True)
    status_path = directory/'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'jobs':{}}
    evaluate_only = args.resume_suite and set(args.experiments) <= {'N-T','N-D'}
    failed = False
    if not evaluate_only:
        for job in manifest['jobs']:
            path = directory/job['path']
            if (path/'summary.json').exists():
                status['jobs'][job['path']] = 'complete'
                continue
            status['jobs'][job['path']] = 'running'
            save_json(status_path,status)
            try:
                runner = Runner(path,condition_config(config,job['condition']),job['condition'],job['seed'],audit=manifest['audit'],
                                resume=(path/'checkpoints/final.pt').exists(),
                                progress=TrainingMonitor(directory/'training_monitor.png',args.plot_every)
                                if args.plot and args.plot_every else None)
                runner.run()
                status['jobs'][job['path']] = 'complete'
            except KeyboardInterrupt:
                status['jobs'][job['path']] = 'interrupted'
                save_json(status_path,status)
                source_tables(directory,config)
                raise
            except Exception as error:
                failed = True
                status['jobs'][job['path']] = 'failed'
                if not (path/'failure.json').exists():
                    save_json(path/'failure.json',dict(type=type(error).__name__,message=str(error)))
                traceback.print_exc()
            save_json(status_path,status)
            source_tables(directory,config)
    if 'N-T' in args.experiments:
        rows = transfer_suite(directory,manifest)
        failed |= any(r['status']!='complete' for r in rows)
        status['transfer'] = 'partial' if any(r['status']!='complete' for r in rows) else 'complete'
        if args.plot:
            plot_transfer(directory)
    if args.audit or 'N-D' in args.experiments:
        audit_suite(directory,manifest)
        status['audit'] = 'complete_with_missing_candidates_reported'
    if args.plot:
        plot_suite(directory)
    save_json(status_path,status)
    print(f'完成。先查看 training_summary.csv 和 overview.png：{directory}',flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
