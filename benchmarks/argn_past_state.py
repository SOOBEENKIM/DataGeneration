"""Causal, bounded numeric summaries shared by training and free generation.

Summaries use the DIGIT tokens already seen by ARGN. Feature caps sit below
the codec's protected upper tail, so the decoder's random tail replacement
cannot change a summary. Generated amounts/gaps themselves are NOT capped here.
"""
from __future__ import annotations

import numpy as np

FEATURES = (
    "has_history", "previous_fraud", "previous_label_unknown",
    "log_run_count", "log_run_elapsed", "last_log_amount", "mean_log_amount",
    "ema_log_amount", "std_log_amount", "ema_log_gap",
)
STATE_COLUMN = "past_state_features"
AMOUNT_CAP = 10_000.0
DURATION_CAP = 7 * 86_400.0
ALPHA = 1 / 16
# count, previous code, run count, elapsed, mean, M2, EMA amount, EMA gap, last amount
MEMORY_SIZE = 9


class PastState:
    def __init__(self, stats: dict):
        self.stats = stats
        self.columns = stats["columns"]
        self.prefixes = {
            c: f"{s['argn_processor']}:{s['argn_table']}/{s['argn_column']}"
            for c, s in self.columns.items()
        }
        self.codes = self.columns["event_is_fraud"]["codes"]
        assert self.codes["0"] != self.codes["1"]
        for column, cap in [("gap", DURATION_CAP), ("amount_or_numeric_value", AMOUNT_CAP)]:
            s = self.columns[column]
            assert s["encoding_type"] == "TABULAR_NUMERIC_DIGIT"
            assert not s["has_nan"] and not s["has_neg"]
            assert max(s["min5"]) == min(s["min5"])
            assert cap <= min(s["max5"]), "state cap must be below protected tail"

    def empty(self, size: int) -> np.ndarray:
        return np.zeros((size, MEMORY_SIZE), dtype=np.float64)

    def features(self, memory: np.ndarray) -> np.ndarray:
        n, previous, run, elapsed, mean, m2, ema, gap_ema, last = memory.T
        present = n > 0
        amount_scale = np.log1p(AMOUNT_CAP)
        gap_scale = np.log1p(DURATION_CAP)
        result = np.stack([
            present, present & (previous == self.codes["1"]),
            present & ~np.isin(previous, [self.codes["0"], self.codes["1"]]),
            np.log1p(run) / np.log1p(self.stats["seq_len"]["max"]),
            np.log1p(elapsed) / gap_scale,
            last / amount_scale, mean / amount_scale, ema / amount_scale,
            np.sqrt(np.maximum(0, m2 / np.maximum(n, 1))) / amount_scale,
            gap_ema / gap_scale,
        ], axis=-1)
        return result.astype(np.float32)

    def numeric(self, event: dict, name: str) -> np.ndarray:
        s = self.columns[name]
        prefix = self.prefixes[name] + "__"
        value = sum(
            (np.asarray(event[prefix + f"E{d}"], dtype=np.float64).reshape(-1)
             + s["min_digits"][f"E{d}"]) * 10.0 ** d
            for d in range(s["min_decimal"], s["max_decimal"] + 1)
        )
        # All protected lower values equal this lower endpoint in the frozen codec.
        return np.maximum(value, s["min5"][0])

    def advance(self, memory: np.ndarray, event: dict) -> np.ndarray:
        """Consume one event per customer; features BEFORE this call predict it."""
        m = memory.copy()
        n, previous, run, elapsed, mean, m2, ema, gap_ema, _ = memory.T
        label = np.asarray(event[self.prefixes["event_is_fraud"] + "__cat"]).reshape(-1)
        log_amount = np.log1p(np.minimum(self.numeric(event, "amount_or_numeric_value"), AMOUNT_CAP))
        gap = np.minimum(self.numeric(event, "gap"), DURATION_CAP)
        same = (n > 0) & (previous == label)
        delta = log_amount - mean
        new_mean = mean + delta / (n + 1)
        m[:, 0] = n + 1
        m[:, 1] = label
        m[:, 2] = np.where(same, run + 1, 1)
        m[:, 3] = np.where(same, np.minimum(elapsed + gap, DURATION_CAP), 0)
        m[:, 4] = new_mean
        m[:, 5] = m2 + delta * (log_amount - new_mean)
        m[:, 6] = np.where(n == 0, log_amount, (1 - ALPHA) * ema + ALPHA * log_amount)
        # The first event has no inter-event gap, regardless of its sampled token.
        log_gap = np.log1p(gap)
        m[:, 7] = np.where(n == 0, 0, np.where(n == 1, log_gap, (1 - ALPHA) * gap_ema + ALPHA * log_gap))
        m[:, 8] = log_amount
        return m

    def sequence(self, record: dict) -> np.ndarray:
        label_key = self.prefixes["event_is_fraud"] + "__cat"
        size = len(record[label_key])
        memory = self.empty(1)
        result = np.empty((size, len(FEATURES)), dtype=np.float32)
        required = [k for k in record if k.startswith(tuple(
            self.prefixes[c] + "__" for c in ["gap", "amount_or_numeric_value", "event_is_fraud"]
        ))]
        for index in range(size):
            result[index] = self.features(memory)[0]
            memory = self.advance(memory, {k: [record[k][index]] for k in required})
        return result
