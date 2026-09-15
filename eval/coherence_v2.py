from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CoherenceReference:
    thresholds_by_channel: Mapping[str, np.ndarray]
    n_bins_requested: int
    n_bins_effective: Mapping[str, int]
    min_bin_count: int


def fit_coherence_reference(
    g_real: Mapping[str, np.ndarray],
    y_real: np.ndarray,
    *,
    n_bins: int,
    min_bin_count: int,
) -> CoherenceReference:
    del y_real
    thresholds, effective = {}, {}
    for channel, values in g_real.items():
        valid = np.isfinite(values)
        edges = np.unique(np.quantile(values[valid], np.linspace(0, 1, n_bins + 1)))
        thresholds[channel] = edges
        effective[channel] = len(edges) - 1
    return CoherenceReference(thresholds, n_bins, effective, min_bin_count)


def _rates(values: np.ndarray, labels: np.ndarray, edges: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    bins = np.digitize(values, edges[1:-1])
    rates, counts = [], []
    for index in range(len(edges) - 1):
        selected = (bins == index) & np.isfinite(values)
        count = int(selected.sum())
        counts.append(count)
        rates.append(np.nan if count < minimum else (labels[selected].sum() + 0.5) / (count + 1))
    return np.asarray(rates), np.asarray(counts)


def coherence_report(
    g_real: Mapping[str, np.ndarray],
    y_real: np.ndarray,
    g_synth: Mapping[str, np.ndarray],
    y_synth: np.ndarray,
    reference: CoherenceReference,
) -> Mapping[str, Any]:
    report = {}
    for channel, edges in reference.thresholds_by_channel.items():
        real, real_n = _rates(g_real[channel], y_real, edges, reference.min_bin_count)
        synth, synth_n = _rates(g_synth[channel], y_synth, edges, reference.min_bin_count)
        valid = np.isfinite(real) & np.isfinite(synth)
        gap = np.abs(real - synth)
        report[channel] = {
            "macro_gap": float(np.mean(gap[valid])) if valid.any() else None,
            "weighted_gap": float(np.average(gap[valid], weights=real_n[valid])) if valid.any() else None,
            "real_counts": real_n.tolist(),
            "synth_counts": synth_n.tolist(),
            "valid": bool(valid.all()),
        }
    return report


def coherence_bin_diagnostics(
    real_values: np.ndarray,
    real_labels: np.ndarray,
    synth_values: np.ndarray,
    synth_labels: np.ndarray,
    edges: np.ndarray,
    minimum: int,
    *,
    invalid_worst_gap: float = 1.0,
) -> dict[str, Any]:
    """Expose bin accounting so missing support cannot improve a score."""
    real_rates, real_counts = _rates(real_values, real_labels, edges, minimum)
    synth_rates, synth_counts = _rates(synth_values, synth_labels, edges, minimum)
    valid = np.isfinite(real_rates) & np.isfinite(synth_rates)
    gaps = np.abs(real_rates - synth_rates)
    dropped_macro = float(np.mean(gaps[valid])) if valid.any() else None
    worst = np.where(valid, gaps, invalid_worst_gap)
    occupancy_penalty = float(np.mean(~valid))
    rows = []
    real_bins = np.digitize(real_values, edges[1:-1])
    synth_bins = np.digitize(synth_values, edges[1:-1])
    for index in range(len(edges) - 1):
        real_selected = (real_bins == index) & np.isfinite(real_values)
        synth_selected = (synth_bins == index) & np.isfinite(synth_values)
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "real_count": int(real_counts[index]),
                "synth_count": int(synth_counts[index]),
                "real_count_y0": int(np.sum(real_selected & (real_labels == 0))),
                "real_count_y1": int(np.sum(real_selected & (real_labels == 1))),
                "synth_count_y0": int(np.sum(synth_selected & (synth_labels == 0))),
                "synth_count_y1": int(np.sum(synth_selected & (synth_labels == 1))),
                "real_prevalence": (
                    float(real_labels[real_selected].mean())
                    if real_selected.any() else None
                ),
                "synth_prevalence": (
                    float(synth_labels[synth_selected].mean())
                    if synth_selected.any() else None
                ),
                "absolute_gap": float(gaps[index]) if valid[index] else None,
                "effective": bool(valid[index]),
                "excluded_from_dropped_macro": bool(not valid[index]),
            }
        )
    return {
        "rows": rows,
        "dropped_bin_macro_gap": dropped_macro,
        "worst_score_macro_gap": float(np.mean(worst)),
        "occupancy_penalized_gap": (
            None if dropped_macro is None
            else float(dropped_macro + occupancy_penalty)
        ),
        "invalid_bin_count": int(np.sum(~valid)),
        "all_bins_valid": bool(valid.all()),
    }
