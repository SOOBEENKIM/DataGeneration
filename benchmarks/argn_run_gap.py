"""ibm_run_gap_v1: gap generation indexed by the fraud-run position of the row (docs/ibm_run_gap_v1/PROTOCOL.md).

Row kind = (previous label -> current label): onset (0 -> 1), continuation (1 -> 1), exit (1 -> 0). At the binned gap
sub-column of such rows the predictor output (the fraud-only head's output for fraud rows, the native head's for exit
rows) is replaced by, or reweighted with, train tables over the gap bins:
    table_pooled : fraud rows <- P(bin | fraud)                         (control: table, no row kind)
    table_fraud  : onset / continuation rows <- P(bin | kind)
    table_kind   : onset / continuation / exit rows <- P(bin | kind)
    ratio_kind   : p(bin) ∝ head(bin) * P(bin | kind) / P(bin | base)   (base: all fraud rows, or normal -> normal rows)
Normal -> normal rows and a customer's first row are untouched. No training; weights, label process and other columns
unchanged. The hook is registered at the first generation step so that it runs after the routed fraud head's hook.
"""
from contextlib import contextmanager
import numpy as np
import torch

LABEL = "event_is_fraud"
GAP = "gap"
KINDS = ("onset", "cont", "exit")
ARMS = ("table_pooled", "table_fraud", "table_kind", "ratio_kind")
PRIOR_ROWS = 10.0


def fit_gap_tables(records, label_key, gap_key, fraud_code, n_codes):
    """records: the fit's ARGN-encoded train sequences (one dict per customer). Returns bin tables per row kind with
    a prior of PRIOR_ROWS rows of the base distribution (all fraud rows for onset/cont, normal->normal rows for exit)."""
    counts = {k: np.zeros(n_codes) for k in ("onset", "cont", "exit", "normal")}
    for r in records:
        y = (np.asarray(r[label_key]).reshape(-1) == fraud_code).astype(int); g = np.asarray(r[gap_key]).reshape(-1).astype(int)
        if len(y) < 2:
            continue
        prev, cur, gg = y[:-1], y[1:], g[1:]
        for name, sel in (("onset", (prev == 0) & (cur == 1)), ("cont", (prev == 1) & (cur == 1)), ("exit", (prev == 1) & (cur == 0)), ("normal", (prev == 0) & (cur == 0))):
            counts[name] += np.bincount(gg[sel], minlength=n_codes)
    fraud = counts["onset"] + counts["cont"]
    base = {"onset": fraud / fraud.sum(), "cont": fraud / fraud.sum(), "exit": counts["normal"] / counts["normal"].sum()}
    tables = {k: ((counts[k] + PRIOR_ROWS * base[k]) / (counts[k].sum() + PRIOR_ROWS)).tolist() for k in KINDS}
    return dict(tables=tables, base={k: base[k].tolist() for k in KINDS}, pooled_fraud=(fraud / fraud.sum()).tolist(),
                rows={k: int(v.sum()) for k, v in counts.items()}, prior_rows=PRIOR_ROWS, n_codes=int(n_codes), customers=len(records))


@contextmanager
def gap_module(stats, params, arm, label=LABEL, gap=GAP):
    """Wrap whatever generation.SequentialModel currently is (transition/label-first + merchant memory) and intervene on
    the gap bin of onset / continuation / exit rows."""
    import mostlyai.engine._tabular.generation as generation
    from mostlyai.engine._common import get_argn_name
    assert arm in ARMS
    lcol = stats["columns"][label]; label_key = get_argn_name(lcol["argn_processor"], lcol["argn_table"], lcol["argn_column"]) + "__cat"
    c1 = int(lcol["codes"]["1"])
    gcol = stats["columns"][gap]; gap_key = get_argn_name(gcol["argn_processor"], gcol["argn_table"], gcol["argn_column"]) + "__bin"
    tab = {k: torch.tensor(params["tables"][k], dtype=torch.float64) for k in KINDS}
    base = {k: torch.tensor(params["base"][k], dtype=torch.float64) for k in KINDS}
    ratio = {k: torch.where(base[k].gt(0), tab[k] / base[k].clamp(min=1e-300), torch.ones_like(tab[k])) for k in KINDS}
    pooled = torch.tensor(params["pooled_fraud"], dtype=torch.float64)
    parent_model = generation.SequentialModel
    counter = dict(gap_steps=0, onset=0, cont=0, exit=0)

    class GapModel(parent_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rg_tab = {k: v.to(self.device) for k, v in tab.items()}; self.rg_ratio = {k: v.to(self.device) for k, v in ratio.items()}
            self.rg_pooled = pooled.to(self.device); self.rg_label = None; self.rg_hooked = False
            self.embedders.get(label_key).register_forward_pre_hook(self._rg_capture)

        def _rg_capture(self, module, args):
            out = args[0]
            if out.dim() == 2:  # generation: sampled codes (B,1)
                self.rg_label = out.reshape(-1).eq(c1).long()

        def _rg_replace(self, module, args, output):
            if args[1] != gap_key or self.rg_label is None:
                return output
            mem = self.module_memory; has, prev, cur = mem[:, 0].gt(0), mem[:, 1], self.rg_label
            masks = dict(onset=has & prev.eq(0) & cur.eq(1), cont=has & prev.eq(1) & cur.eq(1), exit=has & prev.eq(1) & cur.eq(0))
            counter["gap_steps"] += 1
            for k in KINDS:
                counter[k] += int(masks[k].sum())
            new = output[:, -1, :].double().clone()
            if arm == "table_pooled":
                m = masks["onset"] | masks["cont"]
                if m.any():
                    new[m] = self.rg_pooled.clamp(min=1e-12).log()
            else:
                for k in (KINDS if arm in ("table_kind", "ratio_kind") else ("onset", "cont")):
                    m = masks[k]
                    if not m.any():
                        continue
                    if arm == "ratio_kind":
                        p = torch.softmax(new[m], -1) * self.rg_ratio[k]
                        new[m] = (p / p.sum(1, keepdim=True).clamp(min=1e-300)).clamp(min=1e-12).log()
                    else:
                        new[m] = self.rg_tab[k].clamp(min=1e-12).log()
            out = output.clone(); out[:, -1, :] = new.to(output.dtype)
            return out

        def forward(self, x, mode, **kwargs):
            if mode == "gen":
                self.rg_label = None
                if not self.rg_hooked:  # after load_model_weights -> after the routed heads' hooks
                    self.predictors.register_forward_hook(self._rg_replace); self.rg_hooked = True
            return super().forward(x, mode, **kwargs)

    generation.SequentialModel = GapModel
    try:
        yield dict(arm=arm, gap_key=gap_key, counter=counter)
    finally:
        generation.SequentialModel = parent_model
