"""fraud_utility_v1: train-synthetic-test-real fraud detection with strict-past features.

Features never use labels. Two detectors: row-only and row+history. Evaluation is always the
real development table; training tables are generated outputs or real subsets.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

AMOUNT = "amount_or_numeric_value"
MERCHANT = "receiver_or_mark"
LABEL = "event_is_fraud"
CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness",
              "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"]
CHANNELS = []  # IBM use_chip levels; empty for Sparkov
ROW = ["log_amount", "log_gap", "seen_merchant", "log_position"] + [f"cat_{c}" for c in CATEGORIES]
HISTORY = ["amount_vs_med20", "gap_vs_med20", "mean_log_amount_5", "seen_merchants_5", "same_category_5",
           "log_amount_lag1", "log_amount_lag2", "log_amount_lag3"]


def configure(real_frame, max_categories=40):
    """2026-11-08: derive the one-hot vocabulary from the real evaluation table (IBM has 110 MCC codes;
    the most frequent `max_categories` get a column, the rest share none) and add use_chip when present.
    Sparkov keeps its original 14-name list, so earlier results are unchanged."""
    global CATEGORIES, CHANNELS, ROW
    cats = real_frame.category.astype(str).value_counts().index[:max_categories].tolist()
    if set(cats) != set(CATEGORIES):
        CATEGORIES = cats
    CHANNELS = sorted(real_frame.use_chip.astype(str).unique().tolist()) if "use_chip" in real_frame else []
    ROW = ["log_amount", "log_gap", "seen_merchant", "log_position"] + [f"cat_{c}" for c in CATEGORIES] + [f"chan_{c}" for c in CHANNELS]
    return dict(categories=CATEGORIES, channels=CHANNELS)


def features(frame):
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    g = d.groupby("entity_id", sort=False)
    amount = pd.to_numeric(d[AMOUNT], errors="coerce").clip(lower=0).fillna(0)
    gap = pd.to_numeric(d.gap, errors="coerce").clip(lower=0).fillna(0)
    d["log_amount"] = np.log1p(amount); d["log_gap"] = np.log1p(gap)
    d["seen_merchant"] = d.duplicated(["entity_id", MERCHANT]).astype(int)
    d["log_position"] = np.log1p(d.event_index.astype(float))
    cat = d.category.astype(str)
    for c in CATEGORIES:
        d[f"cat_{c}"] = cat.eq(c).astype(int)
    for c in CHANNELS:
        d[f"chan_{c}"] = d.use_chip.astype(str).eq(c).astype(int) if "use_chip" in d else 0
    la = d.log_amount; lg = d.log_gap
    med_a = g.log_amount.transform(lambda x: x.shift().rolling(20, min_periods=5).median())
    med_g = g.log_gap.transform(lambda x: x.shift().rolling(20, min_periods=5).median())
    d["amount_vs_med20"] = (la - med_a).fillna(0)
    d["gap_vs_med20"] = (lg - med_g).fillna(0)
    d["mean_log_amount_5"] = g.log_amount.transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    d["seen_merchants_5"] = g.seen_merchant.transform(lambda x: x.shift().rolling(5, min_periods=1).sum()).fillna(0)
    same = pd.Series(0.0, index=d.index)
    for k in range(1, 6):
        same += g.category.shift(k).eq(d.category).astype(float)
    d["same_category_5"] = same
    for k in (1, 2, 3):
        d[f"log_amount_lag{k}"] = g.log_amount.shift(k).fillna(0)
    label = pd.to_numeric(d[LABEL], errors="coerce")
    d["y"] = label.where(label.isin([0, 1]), 0).astype(int)
    return d


DETECTORS = {
    # registered 2026-11-08 (fraud_utility_v1): diverges at IBM prevalence .0018 (in-sample AUPRC .44 -> .002 by 300 iterations)
    "registered": dict(max_iter=300, learning_rate=0.1, early_stopping=False),
    # amendment 2026-11-08 (ibm_tabformer_v1 utility): regularised, same iteration budget, no early stopping
    "stable": dict(max_iter=300, learning_rate=0.05, early_stopping=False, l2_regularization=1.0, min_samples_leaf=50),
    # ibm_merchant_memory_v1 amendment 3 (2026-11-10): a different classifier family (standardised logistic regression), auxiliary only
    "logistic": dict(C=1.0, max_iter=500),
}
DETECTOR = "registered"
# ibm_merchant_memory_v1 amendment 3 (2026-11-10): feature-set variants of the same detector, auxiliary only ("all" = registered features)
FEATURE_SETS = {"all": (), "no_merchant": ("seen_merchant", "seen_merchants_5"), "no_channel": ("chan_",)}


def select_columns(columns, feature_set="all"):
    drop = FEATURE_SETS[feature_set]
    return [c for c in columns if not any(c == d or (d.endswith("_") and c.startswith(d)) for d in drop)]


def fit(train, columns, seed=20261108, detector=None):
    detector = detector or DETECTOR
    if detector == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **DETECTORS[detector]))
        # `same_category_5` is NaN on the first 5 rows per customer when `category` has pandas string dtype (ARGN outputs;
        # object-dtype frames give 0 there). HGB treats it as missing (registered behaviour, unchanged); here NaN -> 0.
        clf.fit(np.nan_to_num(train[columns].to_numpy(dtype=float), nan=0.0), train.y.to_numpy())
        return clf
    clf = HistGradientBoostingClassifier(random_state=seed, **DETECTORS[detector])
    clf.fit(train[columns].to_numpy(), train.y.to_numpy())
    return clf


def episode_metrics(test, score, top=0.01):
    """Episode recall and early detection with a top-1% alert threshold on the real test."""
    thr = np.quantile(score, 1 - top)
    alert = score >= thr
    f = test.y.eq(1)
    boundary = test.entity_id.ne(test.entity_id.shift()) | f.ne(f.shift())
    run = boundary.cumsum()
    age = test.groupby(run, sort=False).cumcount() + 1
    fr = pd.DataFrame(dict(run=run[f], alert=alert[f], early=(alert & (age <= 2))[f]))
    per = fr.groupby("run").agg(any_alert=("alert", "any"), early=("early", "any"))
    return dict(episode_recall_top1=float(per.any_alert.mean()), early_detection_top1=float(per.early.mean()),
                episodes=int(len(per)), alerts=int(alert.sum()))


def evaluate(train_frame, test_feat, variants=("row", "history"), detector=None, feature_set="all"):
    tr = features(train_frame)
    if tr.y.sum() < 2 or tr.y.sum() == len(tr):
        return {f"{v}_{k}": np.nan for v in variants for k in ["auprc", "auroc", "episode_recall_top1", "early_detection_top1"]}
    out = dict(train_events=len(tr), train_fraud=int(tr.y.sum()))
    for v in variants:
        cols = select_columns(list(ROW) if v == "row" else list(ROW) + HISTORY, feature_set)
        clf = fit(tr, cols, detector=detector)
        score = clf.predict_proba(test_feat[cols].to_numpy())[:, 1]
        out[f"{v}_auprc"] = float(average_precision_score(test_feat.y, score))
        out[f"{v}_auroc"] = float(roc_auc_score(test_feat.y, score))
        for k, val in episode_metrics(test_feat, score).items():
            out[f"{v}_{k}"] = val
    return out
