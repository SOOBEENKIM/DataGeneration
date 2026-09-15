from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind


def welch_and_hedges(a: np.ndarray, b: np.ndarray) -> dict:
    test = ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    d = (a.mean()-b.mean())/pooled if pooled else 0.0
    correction = 1 - 3/(4*(len(a)+len(b))-9)
    return {"welch_t": float(test.statistic), "welch_p": float(test.pvalue), "hedges_g": float(d*correction)}
