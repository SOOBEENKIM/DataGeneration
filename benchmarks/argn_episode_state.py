"""ibm_episode_state_v1 (docs/ibm_episode_state_v1/PROTOCOL.md): fraud EPISODE state process for the label column.

The frozen transition table conditions on (previous label, previous run age <= 21, ever fraud); "ever fraud" is permanent and the
21+ cell pools "22 events after a fraud" with "800 events after", so the free run creates second and third bursts. Here:
  never fraud          -> onset hazard, constant (A1) or by calendar bin of the previous event (A2, A4)
  inside a fraud run   -> continuation by run age 1..21 (as before; A4: x cumulative fraud-count bin)
  normal after a fraud -> re-entry hazard by EVENTS SINCE THE LAST FRAUD in bins up to 201+ (A4: x cumulative fraud-count bin)
All tables are train-only with Jeffreys smoothing. A standard semi-Markov state model; no novelty is claimed for it.
"""
import numpy as np
import pandas as pd

LABEL = "event_is_fraud"
MAX_AGE = 21
SMOOTH = 0.5
D_EDGES = np.array([1, 2, 3, 4, 5, 10, 20, 50, 100, 200])        # distance bins: 1,2,3,4,5,6-10,11-20,21-50,51-100,101-200,201+
K_EDGES = np.array([3, 6, 10, 15])                                # cumulative fraud count bins: 1-3,4-6,7-10,11-15,16+
N_MONTH = 12
MONTH_SECONDS = 30.44 * 86400.0
BURST_GAP = 50                                                    # a re-entry more than 50 events after the last fraud starts a new burst
VARIANTS = ("A1", "A2", "A4")
ALL_VARIANTS = VARIANTS + ("A5",)  # amendment 1 (2026-11-11): A4 with hierarchical shrinkage toward the count-pooled cell


def dbin(d):
    return np.searchsorted(D_EDGES, d, side="left")


def kbin(k):
    return np.searchsorted(K_EDGES, k, side="left")


def mbin(seconds):
    """Calendar bin 0..11; 12 = beyond the one-year window (amendment 2: never occurs in real data, only in generated time)."""
    return np.minimum(N_MONTH, np.floor(np.asarray(seconds, dtype=float) / MONTH_SECONDS)).astype(np.int64)


def row_states(frame, label=LABEL):
    """Per-row strict-past state: y, prev (-1 for the first row), run age of the previous event, ever, events since the
    last fraud, cumulative fraud count before the row, calendar bin of the previous event (0 for the first row)."""
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True); g = d.entity_id
    y = pd.to_numeric(d[label], errors="coerce").fillna(0).astype(int).eq(1).astype(int)
    prev = y.groupby(g).shift().fillna(-1).astype(int)
    boundary = g.ne(g.shift()) | y.ne(y.shift()); age = d.groupby(boundary.cumsum()).cumcount() + 1
    prev_age = age.groupby(g).shift().fillna(0).astype(int)
    k = (y.groupby(g).cumsum() - y).astype(int); i = d.groupby(g, sort=False).cumcount()
    last = i.where(y.eq(1)).groupby(g).ffill().groupby(g).shift(); dist = (i - last).fillna(0).astype(int)
    t = pd.to_numeric(d.gap, errors="coerce").fillna(0).groupby(g).cumsum(); t_prev = t.groupby(g).shift().fillna(0)
    return d, dict(y=y.to_numpy(), prev=prev.to_numpy(), age=prev_age.to_numpy(), ever=(k.to_numpy() > 0).astype(int), dist=dist.to_numpy(), k=k.to_numpy(), month=mbin(t_prev.to_numpy()))


def _rate(k, n, fallback):
    return (k + SMOOTH) / (n + 2 * SMOOTH) if n > 0 else fallback


def fit_episode_tables(train, variant, label=LABEL, shrink=None):
    """Returns dict(onset[12], cont[22, 5], re[11, 5], counts) for variant A1 / A2 / A4 / A5.
    A5 (amendment 1): count-split cells are (k + m * pooled) / (n + m) with m = `shrink`, pooled = the A2 cell."""
    assert variant in ALL_VARIANTS, variant; assert (variant == "A5") == (shrink is not None)
    split = (lambda k, n, pooled: (k + shrink * pooled) / (n + shrink)) if variant == "A5" else _rate
    _, s = row_states(train, label); y = s["y"]; never = s["ever"] == 0; run = s["prev"] == 1; after = (s["prev"] == 0) & (s["ever"] == 1)
    nd, nk = len(D_EDGES) + 1, len(K_EDGES) + 1; counts = []
    base0 = _rate(y[never].sum(), never.sum(), 0.0); onset = np.full(N_MONTH, base0)
    if variant in ("A2", "A4", "A5"):
        for m in range(N_MONTH):
            sel = never & (s["month"] == m); onset[m] = _rate(y[sel].sum(), sel.sum(), base0); counts.append(dict(table="onset", month=m, events=int(sel.sum()), frauds=int(y[sel].sum()), probability=float(onset[m])))
    else:
        counts.append(dict(table="onset", month=-1, events=int(never.sum()), frauds=int(y[never].sum()), probability=float(base0)))
    a = np.minimum(s["age"], MAX_AGE); db = dbin(s["dist"]); kb = kbin(s["k"]); by_k = variant in ("A4", "A5")
    base_run = _rate(y[run].sum(), run.sum(), 0.5); cont = np.zeros((MAX_AGE + 1, nk)); base_after = _rate(y[after].sum(), after.sum(), 0.0); re = np.zeros((nd, nk))
    for age in range(1, MAX_AGE + 1):
        sel = run & (a == age); pooled = _rate(y[sel].sum(), sel.sum(), base_run)
        for q in range(nk):
            sq = sel & (kb == q) if by_k else sel; cont[age, q] = split(y[sq].sum(), sq.sum(), pooled) if by_k else pooled
            if by_k or q == 0:
                counts.append(dict(table="cont", age=age, kbin=q if by_k else -1, events=int(sq.sum()), frauds=int(y[sq].sum()), probability=float(cont[age, q])))
    cont[0] = cont[1]
    for b in range(nd):
        sel = after & (db == b); pooled = _rate(y[sel].sum(), sel.sum(), base_after)
        for q in range(nk):
            sq = sel & (kb == q) if by_k else sel; re[b, q] = split(y[sq].sum(), sq.sum(), pooled) if by_k else pooled
            if by_k or q == 0:
                counts.append(dict(table="re", dbin=b, kbin=q if by_k else -1, events=int(sq.sum()), frauds=int(y[sq].sum()), probability=float(re[b, q])))
    return dict(variant=variant, shrink=shrink, onset=onset.tolist(), onset_beyond=float(base0), cont=cont.tolist(), re=re.tolist(), counts=counts, smoothing=SMOOTH, d_edges=D_EDGES.tolist(), k_edges=K_EDGES.tolist(),
                month_seconds=MONTH_SECONDS, train_frauds=int(y.sum()), train_events=int(len(y)))


def hazard(tables, prev, age, ever, dist, k, month):
    """Vectorised P(fraud) for numpy state arrays (prev = -1 or 0 or 1)."""
    onset, cont, re = np.asarray(list(tables["onset"]) + [tables.get("onset_beyond", tables["onset"][-1])]), np.asarray(tables["cont"]), np.asarray(tables["re"])
    kb = kbin(k); p_run = cont[np.minimum(age, MAX_AGE), kb]; p_after = re[dbin(dist), kb]
    return np.where(ever == 0, onset[month], np.where(prev == 1, p_run, p_after))


def gap_seconds_by_code(gap_stats):
    """Expected decoded seconds of every gap bin code (the engine decodes a bin uniformly; special tokens: UNK -> 0, MIN -> first edge)."""
    bins = np.asarray(gap_stats["bins"], dtype=float); off = len(gap_stats["codes"]); out = np.zeros(off + len(bins) - 1)
    if "<<MIN>>" in gap_stats["codes"]:
        out[int(gap_stats["codes"]["<<MIN>>"])] = bins[0]
    if "<<MAX>>" in gap_stats["codes"]:
        out[int(gap_stats["codes"]["<<MAX>>"])] = bins[-1]
    out[off:] = (bins[:-1] + bins[1:]) / 2
    return out


def episode_module(stats, tables, label=LABEL, gap="gap"):
    """Generation-time episode state process. Enter right after generation_module(..., transition table, label_first=True):
    the base module keeps (n, prev label, run age, ever); this layer adds (events since the last fraud, cumulative fraud
    count, cumulative generated seconds) on the LSTM state tuple and overrides the label logits with the episode tables."""
    from contextlib import contextmanager
    import torch

    @contextmanager
    def ctx():
        import mostlyai.engine._tabular.generation as generation
        from mostlyai.engine._common import get_argn_name
        lcol = stats["columns"][label]; label_key = get_argn_name(lcol["argn_processor"], lcol["argn_table"], lcol["argn_column"]) + "__cat"; c1, c0 = int(lcol["codes"]["1"]), int(lcol["codes"]["0"])
        gcol = stats["columns"][gap]; gap_key = get_argn_name(gcol["argn_processor"], gcol["argn_table"], gcol["argn_column"]) + "__bin"
        onset = torch.tensor(list(tables["onset"]) + [tables["onset_beyond"]], dtype=torch.float64); cont = torch.tensor(tables["cont"], dtype=torch.float64); re = torch.tensor(tables["re"], dtype=torch.float64)
        sec = torch.tensor(gap_seconds_by_code(gcol), dtype=torch.float64); d_edges = torch.tensor(D_EDGES); k_edges = torch.tensor(K_EDGES)
        parent_model = generation.SequentialModel; counter = dict(label_steps=0, rows=0, expected_frauds=0.0, beyond_window_rows=0, never_rows=0, run_rows=0, after_rows=0)

        class EpisodeModel(parent_model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ep = {k: v.to(self.device) for k, v in dict(onset=onset, cont=cont, re=re, sec=sec, d_edges=d_edges, k_edges=k_edges).items()}
                self.predictors.register_forward_hook(self._ep_label)   # registered after the base table hook -> overrides it

            def _ep_label(self, module, args, output):
                if args[1] != label_key:
                    return output
                base, mem = self.module_memory, self.ep_memory          # base: n, prev, age, ever ; mem: dist, k, seconds
                prev, age, ever = base[:, 1], base[:, 2], base[:, 3]; dist, k, seconds = mem[:, 0], mem[:, 1], mem[:, 2]
                month = torch.clamp(torch.div(seconds, int(MONTH_SECONDS), rounding_mode="floor"), max=N_MONTH)
                kb = torch.bucketize(k, self.ep["k_edges"]); db = torch.bucketize(dist, self.ep["d_edges"])
                p = torch.where(ever.eq(0), self.ep["onset"][month], torch.where(prev.eq(1), self.ep["cont"][age.clamp(0, MAX_AGE), kb], self.ep["re"][db, kb])).clamp(1e-9, 1 - 1e-9)
                counter["label_steps"] += 1; counter["rows"] += int(p.numel()); counter["expected_frauds"] += float(p.sum()); counter["beyond_window_rows"] += int((ever.eq(0) & month.eq(N_MONTH)).sum())
                counter["never_rows"] += int(ever.eq(0).sum()); counter["run_rows"] += int((ever.eq(1) & prev.eq(1)).sum()); counter["after_rows"] += int((ever.eq(1) & prev.eq(0)).sum())
                logits = torch.full_like(output, -1e9); logits[:, -1, c1] = p.log().to(output.dtype); logits[:, -1, c0] = torch.log1p(-p).to(output.dtype)
                return logits

            def forward(self, x, mode, **kwargs):
                if mode != "gen":
                    return super().forward(x, mode, **kwargs)
                state = kwargs.get("history_state")
                if state is None:
                    mem = torch.zeros((kwargs["batch_size"], 3), dtype=torch.long, device=self.device)
                else:
                    mem = state[-1][0]; kwargs["history_state"] = tuple(state[:-1])
                self.ep_memory = mem
                outputs, history, recurrent = super().forward(x, mode, **kwargs)
                lab = outputs[label_key].reshape(-1).eq(c1).long(); code = outputs[gap_key].reshape(-1).long().clamp(0, len(self.ep["sec"]) - 1)
                new = torch.stack([torch.where(lab.eq(1), torch.zeros_like(lab), mem[:, 0]) + 1, mem[:, 1] + lab, mem[:, 2] + self.ep["sec"][code].round().long()], -1)
                return outputs, history, (*recurrent, new.unsqueeze(0))

        generation.SequentialModel = EpisodeModel
        try:
            yield dict(variant=tables["variant"], shrink=tables.get("shrink"), label_key=label_key, gap_key=gap_key, counter=counter)
        finally:
            generation.SequentialModel = parent_model
    return ctx()
