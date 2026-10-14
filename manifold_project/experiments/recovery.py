"""Public suite recovery: reuse the existing training, reporting and checkpoint paths."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def restored_config(saved, public):
    """Preserve runner settings; only an explicit larger cumulative budget may change."""
    from .cooperative_navigation.training.semantics import validate_legacy_resume
    validate_legacy_resume(saved)
    c = deepcopy(saved['config'])
    old_public = c['current_config']
    changed = [key for key in set(public) | set(old_public)
               if key != 'budget' and public.get(key) != old_public.get(key)]
    if changed:
        raise ValueError(f'Recovery cannot change configuration fields: {sorted(changed)}')
    budget, old_budget = int(public['budget']), int(c['budget'])
    if budget < old_budget:
        raise ValueError('Recovery cannot reduce the cumulative budget.')
    if budget == old_budget:
        return c
    if not saved.get('complete'):
        raise ValueError('Finish the interrupted source run before extending its budget.')
    from .cooperative_navigation.configs import evaluation_horizon
    from .cooperative_navigation.training.runner import minimum_cost
    nodes = [int(row['budget_checkpoint']) for row in saved['curves']]
    if not nodes or nodes[0] != 0 or nodes[-1] != old_budget or nodes != sorted(set(nodes)):
        raise ValueError('The completed source has an invalid monitoring history.')
    interval = int(old_public['eval_every_steps'])
    extra = [n for n in range(interval, budget, interval) if n > old_budget] + [budget]
    reserve = ((len(extra)-1)*c['eval_episodes']+c['final_episodes'])*evaluation_horizon(c)
    if sum(saved['costs'].values()) + minimum_cost(c, saved['condition']) + reserve > budget:
        raise ValueError('Added budget cannot cover one complete update and source monitoring.')
    c['budget'] = c['current_config']['budget'] = budget
    c['eval_fractions'] = [n/budget for n in nodes+extra]
    return c


def prepare(args):
    """Validate every run before creating or changing any output (also used by dry-run)."""
    from .registry import get_experiment
    from .common.runtime import configure_runtime
    source = args.resume.resolve()
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('experiments'):
        raise ValueError('No initialized experiments in the source suite.')
    if args.config or args.source or args.checkpoints:
        raise ValueError('--resume inherits configuration; do not combine it with --config, --source or --checkpoints.')
    for key in ('data_seeds', 'sample_sizes', 'fit_steps'):
        if getattr(args, key) is not None:
            raise ValueError(f'--{key.replace("_", "-")} is not a recovery option.')
    if args.experiments and set(args.experiments) != set(manifest['experiments']):
        raise ValueError('Resume the whole suite; --experiments must match the saved experiments.')
    plan = []
    for name, entry in manifest['experiments'].items():
        exp = get_experiment(name)
        if exp.environment != 'navigation' or exp.kind not in ('train', 'train_transfer'):
            raise ValueError('Public recovery currently supports navigation training suites, not finite/replay/input-ablation suites.')
        c = deepcopy(entry['config'])
        if c.get('save_fit_diagnostics'):
            raise ValueError('FitProbe state is not checkpointed; recovery of save_fit_diagnostics runs is unsupported.')
        for argument, field in [('profile','profile'), ('methods','methods'), ('seeds','seeds'),
                                ('device','device'), ('threads','torch_threads'), ('plot_every','plot_every'),
                                ('evaluation_episodes','evaluation_episodes'), ('evaluation_horizon','evaluation_horizon')]:
            value = getattr(args, argument)
            if value is not None and value != c.get(field):
                raise ValueError(f'--{argument.replace("_", "-")} must match the saved configuration; omit it to inherit.')
        if args.plot and not c.get('plot'):
            raise ValueError('Recovery inherits plotting settings; do not change --plot.')
        if args.budget is not None:
            if args.budget < c['budget']:
                raise ValueError('Recovery cannot reduce the cumulative budget.')
            c['budget'] = args.budget
        plan.append((exp,c))
    extending = any(c['budget'] != manifest['experiments'][e.id]['config']['budget'] for e,c in plan)
    output = (args.output or source).resolve()
    if extending:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('A larger --budget requires a separate new --output, outside the source suite.')
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Extension output is nonempty: {output}')
    elif output != source:
        raise ValueError('Same-budget interruption recovery writes to the original suite; omit --output.')
    configure_runtime(args.allow_duplicate_openmp or any(c['allow_duplicate_openmp'] for _,c in plan))
    from .cooperative_navigation.training.storage import load_pt
    recovery, inventory = {}, []
    for exp,c in plan:
        relative = Path(manifest['experiments'][exp.id]['directory'])
        expected = Path(f'{exp.id}_{exp.title}')
        if relative != expected:
            raise ValueError('Unexpected experiment directory in manifest.')
        for method in c['methods']:
            for seed in c['seeds']:
                suffix = relative/c['observation_mode']/f'{method}_s{seed}'
                run_dir, target = source/suffix, output/suffix
                checkpoint = run_dir/'checkpoints/final.pt'
                fork_pending = None
                if not checkpoint.exists() and not run_dir.exists():
                    for history in reversed(manifest.get('recovery_history', [])):
                        if history['mode'] == 'extend':
                            fork_pending = next((r for r in history['runs'] if r['output'] == str(run_dir)), None)
                            if fork_pending:
                                checkpoint = Path(fork_pending['source'])
                                break
                record = dict(method=method, seed=seed, source=str(checkpoint), output=str(target))
                if checkpoint.is_file():
                    saved = load_pt(checkpoint)
                    from .cooperative_navigation.training.semantics import validate_legacy_resume
                    validate_legacy_resume(saved,method)
                    required = ('actor','critic','actor_optimizer','critic_optimizer','costs','counts',
                                'optim_steps','torch_rng','collector_state','rounds','curves','complete')
                    if any(key not in saved for key in required):
                        raise ValueError(f'Not a resumable current navigation checkpoint: {checkpoint}')
                    if (extending or fork_pending) and not saved['complete']:
                        raise ValueError('Finish the interrupted source run before extending its budget.')
                    if fork_pending and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != fork_pending['checkpoint_sha256']:
                        raise ValueError('The source checkpoint for a pending extension has changed.')
                    public = dict(c, seed=seed, observation_mode=c['observation_mode'])
                    restored_config(saved, public)
                    identity = json.loads((checkpoint.parent.parent/'config.json').read_text(encoding='utf-8'))
                    if identity != dict(config=saved['config'], condition=saved['condition'], seed=saved['seed'], audit=False):
                        raise ValueError(f'Checkpoint/config identity mismatch: {run_dir}')
                    if saved['seed'] != seed:
                        raise ValueError(f'Checkpoint seed mismatch: {checkpoint}')
                    recovery[str(target)] = {'resume_from': str(checkpoint)} if extending or fork_pending else {'resume': True}
                    record.update(round=saved['round'], complete=saved['complete'],
                                  source_steps=sum(saved['costs'].values()),
                                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                elif extending or fork_pending or run_dir.exists():
                    raise FileNotFoundError(f'No committed checkpoint at {checkpoint}; cannot safely recover an initialized run.')
                else:
                    record['not_started'] = True
                inventory.append(record)
    return source, output, manifest, plan, recovery, inventory, extending


def resume_suite(args):
    from .cli import _run_train, _write_analysis, _require_dependencies
    from .common.reporting import write_json, write_csv, summarize_records, paired_summaries, plot_overview
    from .common.runtime import metadata, configure_torch
    source, output, old, plan, recovery, inventory, extending = prepare(args)
    print(json.dumps(dict(mode='extend' if extending else 'resume', source=str(source),
                         output=str(output), budgets={e.id:c['budget'] for e,c in plan}, runs=inventory),
                     ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    _require_dependencies(plan)
    configure_torch(plan[0][1])
    now = datetime.now(timezone.utc).isoformat()
    current = metadata()
    if current['experiment_code_sha256'] != old['metadata'].get('experiment_code_sha256'):
        print('NOTE: source/current code fingerprints differ; both are recorded in recovery history.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = deepcopy(old)
    manifest.setdefault('recovery_history', []).append(dict(time_utc=now, mode='extend' if extending else 'resume',
        source=str(source), source_metadata=old['metadata'], metadata=current, runs=inventory,
        command_arguments=vars(args).copy()))
    manifest.update(status='running', metadata=current, command_arguments=vars(args).copy())
    manifest.pop('completed_utc', None)
    if extending:
        manifest['created_utc'] = now
        manifest['experiments'] = {e.id: dict(status='pending', config=c, directory=f'{e.id}_{e.title}') for e,c in plan}
    write_json(output/'manifest.json', manifest)
    args._recovery_runs = recovery
    rows_all = []
    for exp,c in plan:
        directory = output/f'{exp.id}_{exp.title}'
        directory.mkdir(parents=True, exist_ok=True)
        entry = dict(status='running', config=c, directory=str(directory.relative_to(output)))
        manifest['experiments'][exp.id] = entry
        write_json(output/'manifest.json', manifest)
        started = time.perf_counter()
        try:
            rows,runs,checkpoints = _run_train(exp,c,directory,args,{})
            summary = summarize_records(rows)+paired_summaries(rows)
            write_csv(directory/'summary.csv',summary)
            _write_analysis(directory,exp,rows,summary,c)
            if c['plot']:
                plot_overview(directory,rows)
            rows_all.extend(rows)
            entry.update(status='completed', runs=runs, elapsed_seconds=time.perf_counter()-started,
                         records=len(rows), checkpoints=[str(p.relative_to(output)) for p in checkpoints])
        except BaseException as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            manifest['status']='failed'
            write_json(output/'manifest.json',manifest)
            raise
        write_json(output/'manifest.json',manifest)
    write_csv(output/'summary.csv',summarize_records(rows_all)+paired_summaries(rows_all))
    if any(c['plot'] for _,c in plan):
        plot_overview(output,rows_all)
    manifest.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'manifest.json',manifest)
    print(f'Completed: {output}',flush=True)
    return 0
