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
ROW = ["log_amount", "log_gap", "seen_merchant", "log_position"] + [f"cat_{c}" for c in CATEGORIES]
HISTORY = ["amount_vs_med20", "gap_vs_med20", "mean_log_amount_5", "seen_merchants_5", "same_category_5",
           "log_amount_lag1", "log_amount_lag2", "log_amount_lag3"]


def features(frame):
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    g = d.groupby("entity_id", sort=False)
    amount = pd.to_numeric(d[AMOUNT], errors="coerce").clip(lower=0).fillna(0)
    gap = pd.to_numeric(d.gap, errors="coerce").clip(lower=0).fillna(0)
    d["log_amount"] = np.log1p(amount); d["log_gap"] = np.log1p(gap)
    d["seen_merchant"] = d.duplicated(["entity_id", MERCHANT]).astype(int)
    d["log_position"] = np.log1p(d.event_index.astype(float))
    for c in CATEGORIES:
        d[f"cat_{c}"] = d.category.eq(c).astype(int)
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


def fit(train, columns, seed=20261108):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=False, random_state=seed)
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


def evaluate(train_frame, test_feat, variants=("row", "history")):
    tr = features(train_frame)
    if tr.y.sum() < 2 or tr.y.sum() == len(tr):
        return {f"{v}_{k}": np.nan for v in variants for k in ["auprc", "auroc", "episode_recall_top1", "early_detection_top1"]}
    out = dict(train_events=len(tr), train_fraud=int(tr.y.sum()))
    for v in variants:
        cols = ROW if v == "row" else ROW + HISTORY
        clf = fit(tr, cols)
        score = clf.predict_proba(test_feat[cols].to_numpy())[:, 1]
        out[f"{v}_auprc"] = float(average_precision_score(test_feat.y, score))
        out[f"{v}_auroc"] = float(roc_auc_score(test_feat.y, score))
        for k, val in episode_metrics(test_feat, score).items():
            out[f"{v}_{k}"] = val
    return out
