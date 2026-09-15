from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def row_fidelity(real, synth) -> dict:
    return {
        "amount_ks": float(ks_2samp(real.x_num[real.valid_mask, 0], synth.x_num[synth.valid_mask, 0]).statistic),
        "amount_wasserstein": float(wasserstein_distance(real.x_num[real.valid_mask, 0], synth.x_num[synth.valid_mask, 0])),
    }
