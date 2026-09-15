from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchmarks.types import SequenceBatch


def categorical_tvd(a: np.ndarray, b: np.ndarray, n_classes: int) -> float:
    pa = np.bincount(a, minlength=n_classes) / len(a)
    pb = np.bincount(b, minlength=n_classes) / len(b)
    return float(np.abs(pa - pb).sum() / 2)


def row_matrix(batch: SequenceBatch) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(batch.y_entity, batch.lengths)
    return np.column_stack(
        [batch.x_num[batch.valid_mask, 0], batch.dt_bin[batch.valid_mask], batch.x_cat[batch.valid_mask, 0]]
    ), labels


def row_classifier_auc(train: SequenceBatch, test: SequenceBatch) -> dict[str, float]:
    x_train, y_train = row_matrix(train)
    x_test, y_test = row_matrix(test)
    transformer = ColumnTransformer(
        [("continuous", StandardScaler(), [0]), ("categorical", OneHotEncoder(handle_unknown="ignore"), [1, 2])]
    )
    logistic = make_pipeline(transformer, LogisticRegression(max_iter=300, class_weight="balanced"))
    logistic.fit(x_train, y_train)
    hist = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=42)
    hist.fit(x_train, y_train)
    return {
        "logistic": float(roc_auc_score(y_test, logistic.predict_proba(x_test)[:, 1])),
        "hist_gradient_boosting": float(roc_auc_score(y_test, hist.predict_proba(x_test)[:, 1])),
    }


def stratified_entity_bootstrap_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster bootstrap AUC for one preregistered row per entity.

    Sorting is shared across resamples. A deterministic tiny jitter gives the
    correct half-credit expectation for tied classifier scores without a
    per-resample O(N log N) sort.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-1e-12, 1e-12, len(scores))
    order = np.argsort(scores + jitter)
    ordered_labels = labels[order]
    by_label = [np.flatnonzero(labels == label) for label in (0, 1)]
    values = np.empty(resamples)
    chunk = 100
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        weights = np.zeros((stop - start, len(labels)), dtype=np.int16)
        for label, indices in enumerate(by_label):
            draws = rng.multinomial(
                len(indices),
                np.full(len(indices), 1 / len(indices)),
                size=stop - start,
            )
            weights[:, indices] = draws
        ordered = weights[:, order]
        negative = ordered * (ordered_labels == 0)
        positive = ordered * (ordered_labels == 1)
        negative_before = np.cumsum(negative, axis=1) - negative
        concordant = np.sum(positive * negative_before, axis=1)
        denominator = positive.sum(1) * negative.sum(1)
        values[start:stop] = concordant / denominator
    point = float(
        roc_auc_score(labels, scores)
    )
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def context_features(batch: SequenceBatch, m: int, short_cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for i, length in enumerate(batch.lengths):
        gaps = batch.x_num[i, :length, 0] * 0  # allocate correct length
        gaps[:] = batch.dt_bin[i, :length]
        cats = batch.x_cat[i, :length, 0]
        amounts = batch.x_num[i, :length, 0]
        for end in range(m - 1, int(length)):
            start = end - m + 1
            if m == 1:
                features.append([amounts[end], gaps[end], cats[end]])
            else:
                repeat = (cats[start + 1 : end + 1] == cats[start:end]).astype(float)
                short = (gaps[start + 1 : end + 1] <= short_cutoff).astype(float)
                features.append([
                    amounts[start : end + 1].mean(), gaps[start : end + 1].mean(),
                    repeat.mean(), short.mean(), np.mean(repeat * short),
                    float(cats[end] == cats[end - 1]),
                    float(gaps[end] <= short_cutoff) * float(cats[end] == cats[end - 1]),
                ])
            labels.append(batch.y_entity[i])
    return np.asarray(features), np.asarray(labels)


def context_auc_curve(train: SequenceBatch, test: SequenceBatch, short_cutoff: float) -> dict[int, float]:
    result = {}
    for m in (1, 2, 4, 8):
        x_train, y_train = context_features(train, m, short_cutoff)
        x_test, y_test = context_features(test, m, short_cutoff)
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.07, random_state=42,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        result[m] = float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
    return result


def marginal_metrics(batch: SequenceBatch, raw_gap: np.ndarray) -> dict[str, float]:
    valid = batch.valid_mask
    row_y = np.repeat(batch.y_entity, batch.lengths)
    return {
        "ks_amount": float(ks_2samp(batch.x_num[valid, 0][row_y == 0], batch.x_num[valid, 0][row_y == 1]).statistic),
        "ks_raw_gap": float(ks_2samp(raw_gap[valid][row_y == 0], raw_gap[valid][row_y == 1]).statistic),
        "tvd_receiver": categorical_tvd(
            batch.x_cat[valid, 0][row_y == 0], batch.x_cat[valid, 0][row_y == 1],
            int(batch.x_cat.max()) + 1,
        ),
        "tvd_dt_bin": categorical_tvd(
            batch.dt_bin[valid][row_y == 0], batch.dt_bin[valid][row_y == 1],
            int(batch.dt_bin.max()) + 1,
        ),
    }


def gate_status(metrics: dict[str, float], row_auc: dict[str, float], context: dict[int, float], dynamic_range: float) -> dict[str, str]:
    single = all(value <= 0.02 for value in metrics.values()) and all(
        abs(value - 0.5) <= 0.02 for value in row_auc.values()
    )
    context_ok = abs(context[1] - 0.5) <= 0.02 and context[2] > 0.52
    context_ok &= context[4] >= context[2] - 0.02 and context[8] >= context[2] - 0.02
    return {
        "gate_a_single_row_leakage": "PASS" if single else "FAIL",
        "gate_a_context_positive_control": "PASS" if context_ok else "FAIL",
        "gate_d_dynamic_range": "PASS" if dynamic_range >= 0.02 else "FAIL",
    }
