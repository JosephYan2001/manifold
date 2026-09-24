"""Versioned experiments; importing this package does not import optional environments."""

__version__ = "1.0.0"

# Preserve the explicitly requested legacy Windows runtime settings before
# NumPy/PyTorch load; an explicit caller setting remains authoritative.
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
