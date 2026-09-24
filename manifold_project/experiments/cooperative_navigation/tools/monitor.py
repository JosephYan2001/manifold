"""Monitor an existing navigation suite from JSONL without changing its training code."""
import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--watch', action='store_true', help='持续读取，Ctrl+C仅停止监控')
    parser.add_argument('--interval', type=float, default=30, help='监控刷新秒数，默认30')
    args = parser.parse_args(argv)
    if not 0 < args.interval < float('inf'):
        parser.error('--interval must be finite and positive')
    # Set runtime defaults before importing the plotting/storage dependencies.
    from manifold_project.experiments.cooperative_navigation import run_experiments
    from manifold_project.experiments.cooperative_navigation.evaluation.monitoring import EventHistory, plot_monitor, progress_line
    manifest = json.loads((args.suite/'manifest.json').read_text(encoding='utf-8'))
    jobs = manifest['jobs']
    readers = [(job, EventHistory(args.suite/job['path']/'events.jsonl')) for job in jobs]
    output = args.suite/'live_monitor.png'  # distinct from a new runner's own dashboard
    print(f'只读日志监控；覆盖更新: {output}', flush=True)
    first = True
    try:
        while True:
            changed = []
            for job, reader in readers:
                previous_round = max(reader.rounds, default=0)
                count = reader.read()
                if count:
                    changed.append((job, reader))
                    new_rounds = [reader.rounds[k] for k in sorted(reader.rounds) if k > previous_round]
                    if first:
                        new_rounds = new_rounds[-1:]  # do not replay thousands of old terminal lines
                    latest_eval = reader.curves[max(reader.curves)] if reader.curves else None
                    for row in new_rounds:
                        print(progress_line(job['condition'], job['seed'], manifest['config']['budget'],
                                            row, latest_eval), flush=True)
            # One image follows the latest updated job; no per-round image files.
            if changed:
                job, reader = max(changed, key=lambda item: item[1].path.stat().st_mtime_ns)
                rounds = [reader.rounds[k] for k in sorted(reader.rounds)]
                curves = [reader.curves[k] for k in sorted(reader.curves)]
                complete = (args.suite/job['path']/'summary.json').exists()
                try:
                    plot_monitor(output, job['condition'], job['seed'], manifest['config']['budget'], rounds, curves, complete)
                except Exception as error:
                    print(f'图像更新失败: {error}', flush=True)
            elif first:
                print('暂无已写入的训练数据，等待日志。', flush=True)
            first = False
            if not args.watch:
                break
            if all((args.suite/j['path']/'summary.json').exists() for j in jobs):
                print('所有训练任务已完成，监控结束。', flush=True)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('监控已停止，训练进程不受影响。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
