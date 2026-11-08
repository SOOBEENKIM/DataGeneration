"""Generation-time rare-state module for the frozen native mostlyai-engine (1.0.4) SequentialModel.

Two interventions, both at generation only (weights untouched, no retraining):
  * `label_first`  -- the label column is generated first in every row, so the other heads condition on it
                      (the model was trained any-order; this only changes the generation order).
  * `transition`   -- the label sub-column's predictor output is replaced by a strict-past transition table
                      P(fraud | previous label, previous run age, ever fraud) estimated on training customers.
The per-customer memory rides on the LSTM state tuple so the engine's continue_mask filtering keeps it aligned.
Registered 2026-11-08 in docs/ibm_generation_module_v1/PROTOCOL.md.
"""
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch

from mostlyai.engine._common import get_argn_name, SLEN_SUB_COLUMN_PREFIX, SIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX

LABEL = "event_is_fraud"
MAX_AGE = 21  # ages >= 21 pooled
SMOOTH = 0.5


def fit_table(train, label=LABEL):
    """P(fraud | previous label in {none,0,1}, previous run age 1..21, ever fraud) with Jeffreys smoothing.
    Returns dict(table[3,22,2], first, counts). Unobserved cells fall back to the age-pooled rate of the same
    (previous label, ever) pair, then to the overall rate of that previous label."""
    d = train.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(d[label], errors="coerce").fillna(0).astype(int).eq(1).astype(int)
    g = d.entity_id
    prev = y.groupby(g).shift()
    boundary = g.ne(g.shift()) | y.ne(y.shift())
    age = d.groupby(boundary.cumsum()).cumcount() + 1          # age of the CURRENT event's run
    prev_age = age.groupby(g).shift()                            # age of the previous event's run
    ever = (y.groupby(g).cumsum() - y).gt(0).astype(int)        # fraud strictly before this event
    first = prev.isna()
    base = {int(p): (y[prev.eq(p)].sum() + SMOOTH) / (prev.eq(p).sum() + 2 * SMOOTH) for p in (0, 1)}
    table = np.zeros((3, MAX_AGE + 1, 2)); counts = []
    for p in (0, 1):
        for e in (0, 1):
            sel_pe = prev.eq(p) & ever.eq(e)
            pooled = (y[sel_pe].sum() + SMOOTH) / (sel_pe.sum() + 2 * SMOOTH) if sel_pe.any() else base[p]
            for a in range(1, MAX_AGE + 1):
                sel = sel_pe & (prev_age.clip(upper=MAX_AGE).eq(a))
                n = int(sel.sum()); k = int(y[sel].sum())
                table[p + 1, a, e] = (k + SMOOTH) / (n + 2 * SMOOTH) if n > 0 else pooled
                counts.append(dict(previous=p, age=a, ever=e, events=n, frauds=k, probability=float(table[p + 1, a, e]), observed=n > 0))
    first_rate = (y[first].sum() + SMOOTH) / (first.sum() + 2 * SMOOTH)
    table[0, :, :] = first_rate
    return dict(table=table.tolist(), first=float(first_rate), smoothing=SMOOTH, max_age=MAX_AGE,
                counts=counts, train_customers=int(g.nunique()), train_events=int(len(d)), train_frauds=int(y.sum()))


@contextmanager
def generation_module(stats, parameters=None, label_first=False, label=LABEL):
    """Patch mostlyai.engine._tabular.generation for one generate() call. `parameters` from fit_table enables
    the transition replacement; `label_first` moves the label column to the front of the generation order."""
    import mostlyai.engine._tabular.generation as generation
    col = stats["columns"][label]
    label_col = get_argn_name(col["argn_processor"], col["argn_table"], col["argn_column"])
    label_key = label_col + "__cat"
    codes = col["codes"]; c1, c0 = int(codes["1"]), int(codes["0"])
    parent_model = generation.SequentialModel
    parent_order = generation._resolve_gen_column_order

    def ordered(*args, **kwargs):
        order = list(parent_order(*args, **kwargs))
        if label_first and label_col in order:
            order.remove(label_col)
            fixed = [c for c in order if c.startswith((SLEN_SUB_COLUMN_PREFIX, SIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX))]
            rest = [c for c in order if c not in fixed]
            order = fixed + [label_col] + rest
        return order

    class ModuleModel(parent_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if parameters is not None:
                self.transition_table = torch.tensor(parameters["table"], device=self.device, dtype=torch.float64)
                self.predictors.register_forward_hook(self._replace_label)

        def _replace_label(self, module, args, output):
            if args[1] != label_key:
                return output
            m = self.module_memory  # (B, 4): n, prev label, run age, ever
            n, prev, age, ever = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
            row = torch.where(n.gt(0), prev + 1, torch.zeros_like(prev))
            p = self.transition_table[row, age.clamp(0, MAX_AGE), ever].clamp(1e-6, 1 - 1e-6)
            logits = torch.full_like(output, -1e9)
            logits[:, -1, c1] = p.log().to(output.dtype)
            logits[:, -1, c0] = torch.log1p(-p).to(output.dtype)
            return logits

        def forward(self, x, mode, **kwargs):
            if mode != "gen":
                return super().forward(x, mode, **kwargs)
            state = kwargs.get("history_state")
            if state is None:
                memory = torch.zeros((kwargs["batch_size"], 4), dtype=torch.long, device=self.device)
            else:
                memory = state[-1][0]
                kwargs["history_state"] = tuple(state[:-1])
            self.module_memory = memory
            outputs, history, recurrent = super().forward(x, mode, **kwargs)
            lab = outputs[label_key].reshape(-1).eq(c1).long()
            same = memory[:, 1].eq(lab) & memory[:, 0].gt(0)
            new = torch.stack([memory[:, 0] + 1, lab, torch.where(same, memory[:, 2] + 1, torch.ones_like(lab)),
                               torch.maximum(memory[:, 3], lab)], -1)
            return outputs, history, (*recurrent, new.unsqueeze(0))

    generation.SequentialModel = ModuleModel
    generation._resolve_gen_column_order = ordered
    try:
        yield dict(label_key=label_key, label_first=label_first, transition=parameters is not None)
    finally:
        generation.SequentialModel = parent_model
        generation._resolve_gen_column_order = parent_order
