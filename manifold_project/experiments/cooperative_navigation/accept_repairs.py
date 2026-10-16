"""Reproducible CPU repair acceptance; no return-based pass criteria or formal training."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest


MODULES = ('test_direction_remediation','test_repair_acceptance',
           'test_navigation_diagnose','test_recovery','test_actor_fit_selection')


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.passed = []
    def addSuccess(self,test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():parser.error('Use a new acceptance output directory; existing evidence is never overwritten.')
    root = Path(__file__).resolve().parents[3]
    sources = {}
    for directory in ('cooperative_navigation','common','tests'):
        for path in sorted((root/'manifold_project/experiments'/directory).rglob('*.py')):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    git = subprocess.run(['git','-c',f'safe.directory={root.as_posix()}','rev-parse','HEAD'],
                         cwd=root,capture_output=True,text=True)
    output.mkdir(parents=True)
    started = time.perf_counter()
    with (output/'tests.log').open('w',encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            'manifold_project.experiments.tests.'+name for name in MODULES])
        result = unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordedResult).run(suite)
    import torch
    from manifold_project.experiments.tests import test_repair_acceptance, test_direction_remediation, test_actor_fit_selection
    report = dict(scope='CPU engineering acceptance, not a training-performance experiment',
        created_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,
        python_version=platform.python_version(),torch_version=torch.__version__,device='cpu',
        git_head=git.stdout.strip() if git.returncode==0 else None,source_sha256=sources,
        elapsed_seconds=time.perf_counter()-started,tests_run=result.testsRun,
        verified=result.wasSuccessful() and not result.skipped,
        passed=result.passed,failures=[dict(test=t.id(),traceback=trace) for t,trace in result.failures],
        errors=[dict(test=t.id(),traceback=trace) for t,trace in result.errors],
        skipped=[dict(test=t.id(),reason=why) for t,why in result.skipped],
        evidence={**test_direction_remediation.EVIDENCE,**test_repair_acceptance.EVIDENCE,
                  **test_actor_fit_selection.EVIDENCE},
        limits=['No formal training launched; no reward threshold used.',
                'CPU tiny fixtures only; CUDA and full-scale performance are not certified.',
                'Frozen-data aggregation invariance does not imply optimizer minibatch invariance.'])
    (output/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('verified','tests_run','elapsed_seconds')},ensure_ascii=False))
    print(f'Evidence: {output / "acceptance.json"}')
    print(f'Test log: {output / "tests.log"}')
    return 0 if report['verified'] else 1


if __name__ == '__main__':raise SystemExit(main())
