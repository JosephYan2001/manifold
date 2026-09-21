"""Evaluate first simultaneous coverage using existing frozen navigation actors."""
import os
import sys
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manifold_project.experiments.cooperative_navigation.evaluation.arrival import main

if __name__ == '__main__':
    raise SystemExit(main())
