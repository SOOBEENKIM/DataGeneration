from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def _finite_by_label(
    values: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    return tuple(
        values[(labels == label) & np.isfinite(values)] for label in (0, 1)
    )


def association_delta(values: np.ndarray, labels: np.ndarray) -> float:
    y0, y1 = _finite_by_label(values, labels)
    return float(y1.mean() - y0.mean())


def association_statistics(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    real_delta: float | None = None,
) -> dict[str, float | bool | None]:
    y0, y1 = _finite_by_label(values, labels)
    delta = float(y1.mean() - y0.mean())
    pooled_variance = (
        ((len(y0) - 1) * y0.var(ddof=1) + (len(y1) - 1) * y1.var(ddof=1))
        / (len(y0) + len(y1) - 2)
    )
    pooled_sd = float(np.sqrt(pooled_variance))
    standardized = delta / pooled_sd if pooled_sd > 0 else float("nan")
    correction = 1 - 3 / (4 * (len(y0) + len(y1)) - 9)
    hedges_g = correction * standardized
    result: dict[str, float | bool | None] = {
        "n_y0": len(y0),
        "n_y1": len(y1),
        "mean_y0": float(y0.mean()),
        "mean_y1": float(y1.mean()),
        "std_y0": float(y0.std(ddof=1)),
        "std_y1": float(y1.std(ddof=1)),
        "delta_joint": delta,
        "standardized_delta": float(standardized),
        "hedges_g": float(hedges_g),
        "label_conditional_wasserstein": float(wasserstein_distance(y0, y1)),
    }
    if real_delta is None:
        result.update(
            association_recovery_error=None,
            recovery_ratio=None,
            sign_consistent=None,
        )
    else:
        result.update(
            association_recovery_error=abs(real_delta - delta),
            recovery_ratio=delta / real_delta if real_delta != 0 else None,
            sign_consistent=bool(
                np.sign(delta) == np.sign(real_delta)
                or (delta == 0 and real_delta == 0)
            ),
        )
    return result


def bootstrap_delta_interval(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    y0, y1 = _finite_by_label(values, labels)
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    chunk = 100
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        mean0 = y0[
            rng.integers(0, len(y0), size=(stop - start, len(y0)))
        ].mean(1)
        mean1 = y1[
            rng.integers(0, len(y1), size=(stop - start, len(y1)))
        ].mean(1)
        boot[start:stop] = mean1 - mean0
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)
