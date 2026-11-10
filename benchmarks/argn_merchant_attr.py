"""ibm_merchant_attr_v1: merchant-attribute consistency on the frozen ARGN (docs/ibm_merchant_attr_v1/PROTOCOL.md).

The row's merchant (already sampled: generation order is label, gap, merchant, amount, category, channel) and its
label state decide category and channel:
    normal, known merchant : category ~ P(category | merchant),   channel ~ P(channel | merchant)      (train tables)
    normal, rare bucket    : category ~ P(category | rare, normal), channel ~ P(channel | category, rare, normal)
    fraud,  known merchant : category ~ P(category | merchant),   channel ~ P(online | fraud, merchant novel / repeated)
    fraud,  rare bucket    : category unchanged (fraud head),      channel ~ the same state table
Arms: attr_cat (category rule only), attr_chan (channel rule only), attr_normal (both rules, normal rows only),
attr_all. A merchant without a train table keeps the head's output. No training; weights, label process, gap, merchant
choice and amount unchanged. The hook is registered at the first generation step (after the routed heads' hooks).
"""
from contextlib import contextmanager
import numpy as np
import torch

LABEL, MERCHANT, CATEGORY, CHANNEL = "event_is_fraud", "receiver_or_mark", "category", "use_chip"
RARE, ONLINE, SWIPE = "_RARE_", "Online Transaction", "Swipe Transaction"
ARMS = ("attr_cat", "attr_chan", "attr_normal", "attr_all")


def fit_attr_tables(records, keys, fraud_code, rare_code, n_m, n_c, n_u):
    """records: the fit's ARGN-encoded train sequences. keys = dict(label, merchant, category, channel) sub-column names.
    Returns count-normalised tables (no smoothing: the real dependency is deterministic)."""
    has_chan = keys.get("channel") is not None  # sparkov_transfer_v1: datasets without a channel column
    n_u = n_u if has_chan else 1
    cat_m = np.zeros((n_m, n_c)); chan_m = np.zeros((n_m, n_u)); chan_cat_rare = np.zeros((n_c, n_u))
    for r in records:
        y = np.asarray(r[keys["label"]]).reshape(-1) == fraud_code
        m = np.asarray(r[keys["merchant"]]).reshape(-1).astype(int); c = np.asarray(r[keys["category"]]).reshape(-1).astype(int)
        u = np.asarray(r[keys["channel"]]).reshape(-1).astype(int) if has_chan else np.zeros(len(m), dtype=int)
        np.add.at(cat_m, (m[~(y & (m == rare_code))], c[~(y & (m == rare_code))]), 1.0)   # rare-bucket fraud rows excluded (table used for normal rare rows)
        np.add.at(chan_m, (m[~y], u[~y]), 1.0)
        sel = ~y & (m == rare_code); np.add.at(chan_cat_rare, (c[sel], u[sel]), 1.0)
    def norm(a):
        s = a.sum(1, keepdims=True); return np.divide(a, s, out=np.zeros_like(a), where=s > 0), s[:, 0]
    cat_p, cat_n = norm(cat_m); chan_p, chan_n = norm(chan_m); ccr_p, ccr_n = norm(chan_cat_rare)
    rare_chan = chan_m[rare_code] / max(chan_m[rare_code].sum(), 1.0)
    ccr_p[ccr_n == 0] = rare_chan                                                          # category unseen among rare normal rows -> rare-bucket channel marginal
    known = np.arange(n_m) != rare_code
    summary = dict(merchant_codes=int(n_m), merchants_with_category_table=int((cat_n > 0).sum()), merchants_with_channel_table=int((chan_n > 0).sum()),
                   category_purity_row_weighted=float((cat_p.max(1) * cat_n)[known].sum() / max(cat_n[known].sum(), 1)),
                   channel_purity_row_weighted=float((chan_p.max(1) * chan_n)[known].sum() / max(chan_n[known].sum(), 1)),
                   rare_rows=int(cat_n[rare_code]), rare_category_top=float(cat_p[rare_code].max()), rows=int(cat_n.sum()), customers=len(records))
    return dict(cat_by_m=cat_p, chan_by_m=chan_p, chan_by_cat_rare=ccr_p), summary


@contextmanager
def attr_module(stats, tables, channel_state_table, arm):
    """Wrap the current generation.SequentialModel (transition + merchant memory [+ gap tables]) and replace the category
    and channel predictor outputs. channel_state_table = (P(online | fraud, novel), P(online | fraud, repeat))."""
    import mostlyai.engine._tabular.generation as generation
    from mostlyai.engine._common import get_argn_name
    assert arm in ARMS

    def key(col):
        s = stats["columns"][col]; return get_argn_name(s["argn_processor"], s["argn_table"], s["argn_column"]) + "__cat"
    has_chan = CHANNEL in stats["columns"]  # sparkov_transfer_v1: without a channel column only the category rule exists
    label_key, m_key, c_key = key(LABEL), key(MERCHANT), key(CATEGORY); u_key = key(CHANNEL) if has_chan else "__no_channel__"
    c1 = int(stats["columns"][LABEL]["codes"]["1"]); rare_code = int(stats["columns"][MERCHANT]["codes"][RARE])
    on_code = int(stats["columns"][CHANNEL]["codes"][ONLINE]) if has_chan else 0; off_code = int(stats["columns"][CHANNEL]["codes"][SWIPE]) if has_chan else 0
    cat_by_m = torch.tensor(tables["cat_by_m"], dtype=torch.float64); chan_by_m = torch.tensor(tables["chan_by_m"], dtype=torch.float64)
    chan_by_cat_rare = torch.tensor(tables["chan_by_cat_rare"], dtype=torch.float64)
    n_u = chan_by_m.shape[1]; state = torch.zeros((2, n_u), dtype=torch.float64)
    for i, p in enumerate(channel_state_table if has_chan else ()):
        state[i, on_code] = p; state[i, off_code] = 1 - p
    do_cat = arm in ("attr_cat", "attr_normal", "attr_all"); do_chan = has_chan and arm in ("attr_chan", "attr_normal", "attr_all"); do_fraud = arm != "attr_normal"
    parent_model = generation.SequentialModel
    counter = dict(category_rows=0, channel_rows=0, channel_state_rows=0, steps=0)

    class AttrModel(parent_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.at_cat = cat_by_m.to(self.device); self.at_chan = chan_by_m.to(self.device); self.at_ccr = chan_by_cat_rare.to(self.device); self.at_state = state.to(self.device)
            self.at_has_cat = self.at_cat.sum(1).gt(0); self.at_has_chan = self.at_chan.sum(1).gt(0)
            self.at_label = self.at_m = self.at_c = None; self.at_hooked = False
            self.embedders.get(label_key).register_forward_pre_hook(lambda mod, a: self._at_capture("at_label", a))
            self.embedders.get(m_key).register_forward_pre_hook(lambda mod, a: self._at_capture("at_m", a))
            self.embedders.get(c_key).register_forward_pre_hook(lambda mod, a: self._at_capture("at_c", a))

        def _at_capture(self, name, args):
            out = args[0]
            if out.dim() == 2:  # generation: sampled codes (B,1)
                setattr(self, name, out.reshape(-1).long())

        def _at_replace(self, module, args, output):
            sub = args[1]
            if sub not in (c_key, u_key) or self.at_label is None or self.at_m is None:
                return output
            fraud = self.at_label.eq(c1); m = self.at_m; rare = m.eq(rare_code)
            new = output[:, -1, :].double().clone(); counter["steps"] += 1
            if sub == c_key and do_cat:
                rows = self.at_has_cat[m] & ~(fraud & rare)                      # rare-bucket fraud rows keep the fraud head
                if not do_fraud:
                    rows = rows & ~fraud
                if rows.any():
                    new[rows] = self.at_cat[m[rows]].clamp(min=1e-12).log(); counter["category_rows"] += int(rows.sum())
            elif sub == u_key and do_chan:
                normal = ~fraud
                rows = normal & ~rare & self.at_has_chan[m]
                if rows.any():
                    new[rows] = self.at_chan[m[rows]].clamp(min=1e-12).log(); counter["channel_rows"] += int(rows.sum())
                rr = normal & rare
                if rr.any() and self.at_c is not None:
                    new[rr] = self.at_ccr[self.at_c[rr]].clamp(min=1e-12).log(); counter["channel_rows"] += int(rr.sum())
                if do_fraud and fraud.any():
                    repeat = torch.gather(self.mm_counts, 1, m.unsqueeze(1)).squeeze(1).gt(0)   # merchant memory before this row
                    new[fraud] = self.at_state[repeat[fraud].long()].clamp(min=1e-12).log(); counter["channel_state_rows"] += int(fraud.sum())
            out = output.clone(); out[:, -1, :] = new.to(output.dtype)
            return out

        def forward(self, x, mode, **kwargs):
            if mode == "gen":
                self.at_label = self.at_m = self.at_c = None
                if not self.at_hooked:  # after load_model_weights -> after the routed heads' hooks
                    self.predictors.register_forward_hook(self._at_replace); self.at_hooked = True
            return super().forward(x, mode, **kwargs)

    generation.SequentialModel = AttrModel
    try:
        yield dict(arm=arm, keys=dict(label=label_key, merchant=m_key, category=c_key, channel=u_key), counter=counter)
    finally:
        generation.SequentialModel = parent_model
