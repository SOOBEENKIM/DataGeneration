"""Observable-information oracle for the existing joint semi-Markov DGP.

No data loading, learned model, CUDA, or realized oracle latents are used here.
The augmented hidden state is (regime, remaining duration). Current-gap
response means conditional prediction, not a causal do(gap) intervention.
"""
from __future__ import annotations

import numpy as np

from benchmarks.semi_markov import calibrated_duration_pmf, residual_pmf
from benchmarks.temporal_coupling_v2 import BenchmarkConfig


def repeat_probability(copy_probability: np.ndarray, category_count: int) -> np.ndarray:
    """A fresh uniform draw can accidentally equal the previous mark."""
    if category_count < 2:
        raise ValueError("at least two mark categories are required")
    return copy_probability + (1.0 - copy_probability) / category_count


class SemiMarkovCopyOracle:
    def __init__(self, config: BenchmarkConfig, upper_bounds: np.ndarray):
        if config.scenario != "joint_semimarkov_v2b" or config.kappa not in (0, 1):
            raise ValueError("oracle supports only joint semi-Markov kappa=0/1")
        self.config = config
        self.upper = np.asarray(upper_bounds, dtype=np.float64)
        if (self.upper.ndim != 1 or not len(self.upper)
                or not np.isposinf(self.upper[-1])
                or not np.all(np.isfinite(self.upper[:-1]))
                or np.any(np.diff(self.upper) <= 0) or np.any(self.upper <= 0)):
            raise ValueError("positive increasing bin boundaries ending at +inf required")
        if config.n_receiver_categories < 2 or not 0 <= config.q_low < config.q_high < 1:
            raise ValueError("invalid copy/mark parameters")
        self.pmfs = np.stack([
            calibrated_duration_pmf(mean, config.duration_sigma, config.duration_max)[1]
            for mean in (config.duration_normal_mean, config.duration_burst_mean)
        ])
        self.scales = np.array([config.gap_normal_scale, config.gap_burst_scale])
        if np.any(self.scales <= 0):
            raise ValueError("gap scales must be positive")
        self.copy_by_state = np.array([config.q_low, config.q_high])
        self.repeat_by_state = repeat_probability(self.copy_by_state, config.n_receiver_categories)
        lower = np.r_[0.0, self.upper[:-1]]
        self.bin_mass = np.exp(-lower[None, :] / self.scales[:, None]) * (
            -np.expm1(-(self.upper - lower)[None, :] / self.scales[:, None])
        )
        if np.any(self.bin_mass.sum(0) <= 0):
            raise ValueError("gap bins must have representable positive mass")

    def initial(self, n: int) -> np.ndarray:
        probabilities = np.stack([residual_pmf(p) for p in self.pmfs])
        probabilities *= np.array([1-self.config.pi_burst, self.config.pi_burst])[:, None]
        return np.broadcast_to(probabilities, (n, *probabilities.shape)).copy()

    def advance(self, posterior: np.ndarray) -> np.ndarray:
        result = np.zeros_like(posterior)
        result[:, :, :-1] = posterior[:, :, 1:]
        result[:, 0] += posterior[:, 1, 0, None] * self.pmfs[0]
        result[:, 1] += posterior[:, 0, 0, None] * self.pmfs[1]
        return result

    def copy_curve(self, prior_burst: np.ndarray, active: bool) -> np.ndarray:
        prior = np.asarray(prior_burst, dtype=float)
        if np.any(~np.isfinite(prior)) or np.any((prior < 0) | (prior > 1)):
            raise ValueError("invalid prior probability")
        if active:
            mass = ((1-prior[:, None])*self.bin_mass[0]
                    + prior[:, None]*self.bin_mass[1])
            burst = prior[:, None]*self.bin_mass[1] / mass
        else:
            burst = np.broadcast_to(prior[:, None], (len(prior), len(self.upper)))
        return self.config.q_low + (self.config.q_high-self.config.q_low)*burst

    def filter_batch(self, gaps: np.ndarray, marks: np.ndarray,
                     lengths: np.ndarray, *, active: bool) -> dict[str, np.ndarray]:
        """Return predictions made before the current mark is observed.

        First gap must be missing. Future marks/gaps, absolute time origin,
        realized regimes and current mark never enter the prediction. Past
        continuous gaps do enter the active-state filter, matching SAF history.
        """
        gaps, marks, lengths = np.asarray(gaps), np.asarray(marks), np.asarray(lengths)
        if gaps.ndim != 2 or marks.shape != gaps.shape or lengths.shape != (len(gaps),):
            raise ValueError("invalid padded batch shapes")
        if np.any((lengths < 2) | (lengths > gaps.shape[1])):
            raise ValueError("sequences must contain at least two events")
        if not np.isnan(gaps[:, 0]).all():
            raise ValueError("first gap must be missing; time-origin information is excluded")
        valid = np.arange(gaps.shape[1])[None, :] < lengths[:, None]
        if np.any(~np.isfinite(gaps[:, 1:][valid[:, 1:]])) or np.any(gaps[:, 1:][valid[:, 1:]] < 0):
            raise ValueError("nonfirst gaps must be finite and nonnegative")
        if np.any((marks[valid] < 0) | (marks[valid] >= self.config.n_receiver_categories)):
            raise ValueError("marks outside production category range")
        prior = self.advance(self.initial(len(gaps)))
        fields = {key: [] for key in ("entity_index", "event_index", "prior_burst",
                                      "copy_probability", "repeat_probability",
                                      "observed_repeat", "response_range", "no_gap_copy_probability")}
        for t in range(1, gaps.shape[1]):
            ids = np.flatnonzero(valid[:, t])
            if not len(ids):
                break
            p = prior[ids].sum(-1)[:, 1]
            curves = self.copy_curve(p, active)
            bins = np.searchsorted(self.upper, gaps[ids, t], side="left")
            q = curves[np.arange(len(ids)), bins]
            observed = marks[ids, t] == marks[ids, t-1]
            items = (ids, np.full(len(ids), t), p, q,
                     repeat_probability(q, self.config.n_receiver_categories),
                     observed.astype(float), np.ptp(curves, axis=1),
                     self.config.q_low+(self.config.q_high-self.config.q_low)*p)
            for key, value in zip(fields, items):
                fields[key].append(value)
            # Assimilate current observations only AFTER saving the prediction.
            log_weight = np.log(np.where(observed[:, None], self.repeat_by_state,
                                         1-self.repeat_by_state))
            if active:
                log_weight += -np.log(self.scales)-gaps[ids, t, None]/self.scales
            weight = np.exp(log_weight-log_weight.max(1, keepdims=True))
            posterior = prior[ids]*weight[:, :, None]
            posterior /= posterior.sum((1, 2), keepdims=True)
            prior[ids] = self.advance(posterior)
        return {key: np.concatenate(value) for key, value in fields.items()}


def summarize_context(predictions: dict[str, np.ndarray], n_entities: int,
                      category_count: int) -> dict:
    entity = predictions["entity_index"].astype(int)
    counts = np.bincount(entity, minlength=n_entities)
    if np.any(counts == 0):
        raise ValueError("every audited entity must have a transition")
    def entity_means(values):
        return np.bincount(entity, weights=values, minlength=n_entities)/counts
    residuals = entity_means(predictions["observed_repeat"]-predictions["repeat_probability"])
    ranges = entity_means(predictions["response_range"])
    return {
        "entity_count": n_entities,
        "transition_count": int(counts.sum()),
        "mean_copy_probability_range": float(ranges.mean()),
        "max_copy_probability_range": float(predictions["response_range"].max()),
        "mean_repeat_probability_range": float(ranges.mean()*(1-1/category_count)),
        "repeat_calibration_mean_residual": float(residuals.mean()),
        "repeat_calibration_entity_cluster_se": float(residuals.std(ddof=1)/np.sqrt(n_entities)),
        "mean_predicted_repeat": float(entity_means(predictions["repeat_probability"]).mean()),
        "mean_observed_repeat": float(entity_means(predictions["observed_repeat"]).mean()),
    }


def decide_oracle(cells: dict, criteria: dict) -> dict:
    expected = {(str(k), str(y)) for k in (0, 1) for y in (0, 1)}
    if {(k, y) for k, groups in cells.items() for y in groups} != expected:
        raise ValueError("both kappas and both contexts are mandatory")
    active = cells["1"]["1"]["mean_copy_probability_range"]
    null = max(cells[k][y]["mean_copy_probability_range"]
               for k, y in expected if (k, y) != ("1", "1"))
    checks = {
        "active_response_material": active >= criteria["active_cell_min_mean_range"],
        "null_response_small": null <= criteria["noncausal_cell_max_mean_range"],
        "selectivity": active >= criteria["active_to_largest_noncausal_min_ratio"]*null,
        "sufficient_context_entities": all(cells[k][y]["entity_count"] >= criteria["minimum_entities_per_context"] for k, y in expected),
        "observable_repeat_calibrated": all(
            abs(cells[k][y]["repeat_calibration_mean_residual"]) <= max(
                criteria["repeat_calibration_absolute_tolerance"],
                criteria["repeat_calibration_cluster_se_multiplier"]*cells[k][y]["repeat_calibration_entity_cluster_se"])
            for k, y in expected),
    }
    return {"checks": checks, "decision": "PASS" if all(checks.values()) else "FAIL",
            "active_mean_range": active, "largest_null_mean_range": null,
            "selectivity_ratio": active/null if null > 0 else None,
            "selectivity_ratio_note": "unbounded_when_null_zero" if null == 0 else "finite"}
