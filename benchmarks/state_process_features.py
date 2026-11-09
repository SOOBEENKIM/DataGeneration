"""ibm_state_first_v1: label-history features of the learned state process (numpy/pandas only, shared by both backbones)."""
import numpy as np
import pandas as pd

MAX_AGE = 21
FEATURES = ["prev_label", "run_age", "ever_fraud", "n_frauds", "since_last_fraud", "position"]
MEMORY = 6  # n, prev, age, ever, n_frauds, since_last


def label_history_features(y, g):
    """Strict-past features for every row. y: 0/1 int Series (sorted by entity, event), g: entity Series (same index)."""
    prev = y.groupby(g).shift()
    boundary = g.ne(g.shift()) | y.ne(y.shift())
    age = y.groupby(boundary.cumsum()).cumcount() + 1
    prev_age = age.groupby(g).shift().fillna(0)
    cum = y.groupby(g).cumsum() - y
    pos = g.groupby(g).cumcount()
    last_fraud_pos = pd.Series(np.where(y.eq(1), pos, np.nan), index=y.index).groupby(g).shift().groupby(g).ffill()
    since = (pos - last_fraud_pos.fillna(-1)).where(last_fraud_pos.notna(), pos)
    X = np.stack([prev.fillna(-1).to_numpy(), np.minimum(prev_age.to_numpy(), MAX_AGE), cum.gt(0).astype(int).to_numpy(),
                  np.minimum(cum.to_numpy(), 50), np.log1p(np.minimum(since.to_numpy(), 2000)), np.log1p(pos.to_numpy())], axis=1).astype(np.float32)
    return X


def memory_features(m):
    """m: (B, 6) array-like [n, prev, age, ever, n_frauds, since_last] -> (B, 6) float32 in FEATURES order."""
    m = np.asarray(m, dtype=np.float64)
    n, prev, age, ever, nf, since = m.T
    prev = np.where(n > 0, prev, -1.0); age = np.where(n > 0, np.minimum(age, MAX_AGE), 0.0)
    since = np.where(nf > 0, since, n)
    return np.stack([prev, age, ever, np.minimum(nf, 50), np.log1p(np.minimum(since, 2000)), np.log1p(n)], axis=1).astype(np.float32)


def update_memory_np(m, lab):
    """m (B,6) int array, lab (B,) {0,1} -> new memory after the row."""
    n, prev, age, ever, nf, since = [m[:, i] for i in range(MEMORY)]
    same = (prev == lab) & (n > 0)
    return np.stack([n + 1, lab, np.where(same, age + 1, 1), np.maximum(ever, lab), nf + lab, np.where(lab == 1, 0, since + 1)], axis=1)


def table_prob_fn(table):
    t = np.asarray(table)

    def f(X):
        prev, age, ever = X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2].astype(int)
        return t[np.where(prev < 0, 0, prev + 1), np.clip(age, 0, MAX_AGE), ever]
    return f
