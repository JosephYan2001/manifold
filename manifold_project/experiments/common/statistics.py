"""Original pair bootstrap and Wilson intervals, shared by current reports."""
import math
import numpy as np


def bootstrap(values, repeats=2000, seed=20260916):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError('Bootstrap values must be finite independent summaries')
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = float(values.mean())
    if len(values) == 1:
        return {"count": 1, "mean": mean, "std": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(-1)
    return {"count": len(values), "mean": mean, "std": float(values.std(ddof=1)),
            "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975))}


def binomial_interval(values):
    """Two-sided 95% Wilson interval, including zero/all-event samples."""
    values = np.asarray(values, float)
    n = len(values)
    if not n or not np.isin(values, [0, 1]).all():
        raise ValueError("Binomial observations must be a nonempty binary sample")
    p = float(values.mean())
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return {"count": n, "events": int(values.sum()), "mean": p,
            "std": float(values.std(ddof=1)) if n > 1 else None,
            "ci_low": max(0., center-half), "ci_high": min(1., center+half),
            "ci_method": "wilson_95"}
