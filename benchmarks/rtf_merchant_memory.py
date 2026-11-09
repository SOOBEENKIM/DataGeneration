"""ibm_merchant_memory_v1, REaLTabFormer side (docs/ibm_merchant_memory_v1/PROTOCOL.md §1, arm `rtf_full`).

The same three-part component as benchmarks/argn_merchant_memory.py, realised as a LogitsProcessor at the merchant
token position of every generated row:
    P(m) = g * novel(m) + (1 - g) * repeat(m)
  g      = P(new merchant | row label, position bin)                 (train table, shared with the ARGN fit)
  repeat = lam * frequency(past) + (1 - lam) * recency(past)         (customer's past merchants: earlier windows + this one)
  novel  = normal: backbone merchant distribution with past merchants masked, native top-k=50 decoding kept on the
                   remaining merchants (so the backbone's decoding rule is unchanged, only restricted to unseen merchants)
           fraud : pi * shared fraud pool (train) + (1 - pi) * that masked backbone distribution
RTF has one token per train merchant (25,088; no rare bucket). The customer's past is tracked as per-merchant counts
and last positions carried over across the customer's windows (like the transition state), plus the merchant tokens
already generated in the current window, read from `input_ids`.

Native generation applies HF's top-k=50 warper AFTER all custom processors, which would truncate the mixture; the
sampler is therefore called with `top_k=0` and this processor applies the identical top-k filter itself at every
non-merchant position. Backbone weights, the label process and all other columns are untouched.
"""
import math

import numpy as np
import torch
from transformers import LogitsProcessor

from realtabformer.data_utils import SPECIAL_COL_SEP
from benchmarks.rtf_ibm_module import original_column

MERCHANT = "receiver_or_mark"
TOP_K = 50


def top_k_filter(scores, k=TOP_K):
    """Identical to transformers' TopKLogitsWarper(top_k=k, min_tokens_to_keep=1)."""
    k = min(max(k, 1), scores.shape[-1])
    thr = torch.topk(scores, k)[0][..., -1, None]
    return scores.masked_fill(scores < thr, -math.inf)


class MerchantGeometry:
    """Merchant sub-column of a fitted relational model: its tokens, values and compact indexing."""

    def __init__(self, geom, merchant=MERCHANT):
        orig = [original_column(c) for c in geom.cols]
        js = [j for j, o in enumerate(orig) if o == merchant]
        assert len(js) == 1, (orig, merchant)
        self.j = js[0]
        assert self.j > geom.j_label, "the row's label must precede the merchant"
        self.col = geom.cols[self.j]
        prefix = self.col + SPECIAL_COL_SEP
        ids = list(geom.col_ids[self.j])  # sorted token ids
        toks = [geom.i2t[t] for t in ids]
        assert all(t.startswith(prefix) for t in toks)
        self.values = [t[len(prefix):] for t in toks]
        self.codes = {v: i for i, v in enumerate(self.values)}
        self.V = len(ids)
        self.token_ids = torch.tensor(ids, dtype=torch.long)
        self.compact = torch.full((geom.V,), -1, dtype=torch.long)
        self.compact[self.token_ids] = torch.arange(self.V)

    def pool_tensor(self, pool_by_value):
        p = torch.zeros(self.V, dtype=torch.float64)
        for v, w in pool_by_value.items():
            p[self.codes[v]] = float(w)
        return p / p.sum()


def empty_state(n_customers, V):
    """Per-customer merchant memory carried across windows: counts (N, V) float64, last positions (N, V) long (-1 = never)."""
    return torch.zeros((n_customers, V), dtype=torch.float64), torch.full((n_customers, V), -1, dtype=torch.long)


class MerchantMemoryProcessor(LogitsProcessor):
    def __init__(self, geom, mg, params, pool, init_counts, init_last, arm="full", top_k=TOP_K):
        assert arm in ("full", "gate_only", "repeat_blind", "fraud_only")  # fraud_only (amendment 2, 2026-11-10): normal rows native
        self.g, self.m, self.arm, self.top_k = geom, mg, arm, top_k
        dev = init_counts.device
        self.counts0, self.last0 = init_counts, init_last                          # (B, Vm)
        self.gate = torch.tensor(params["gate"], dtype=torch.float64, device=dev)
        self.gate_blind = torch.tensor(params["gate_blind"], dtype=torch.float64, device=dev)
        self.edges = torch.tensor(params["position_edges"], dtype=torch.float64, device=dev)
        self.lam = float(params["lam"]); self.pi = float(params["pi"]) if arm in ("full", "fraud_only") else 0.0
        self.pool = pool.to(dev)                                                    # (Vm,) float64
        self.token_ids = mg.token_ids.to(dev); self.compact = mg.compact.to(dev)
        self.n_merchant_steps = 0; self.rows = 0; self.replaced_rows = 0; self.novel_mass = 0.0; self.repeat_mass = 0.0; self.pool_mass = 0.0

    def __call__(self, input_ids, scores):
        g = self.g
        idx = input_ids.shape[1] - 2
        if idx >= 0:
            k, c = divmod(idx, g.S + 2)
            if c - 1 == self.m.j:
                return self._merchant(input_ids, scores, k)
        return top_k_filter(scores, self.top_k) if self.top_k else scores

    def _merchant(self, input_ids, scores, k):
        g, m = self.g, self.m
        B = input_ids.shape[0]; dev = scores.device
        counts = self.counts0.clone(); last = self.last0.clone(); n0 = counts.sum(1).long()
        if k > 0:  # merchants of this window's earlier rows
            pos = 3 + m.j + torch.arange(k, device=dev) * (g.S + 2)
            comp = self.compact[input_ids[:, pos]]                                   # (B, k)
            valid = comp.ge(0); ci = comp.clamp(min=0)
            counts.scatter_add_(1, ci, valid.double())
            rpos = torch.where(valid, n0.unsqueeze(1) + torch.arange(k, device=dev), torch.full_like(comp, -1))
            last.scatter_reduce_(1, ci, rpos, reduce="amax")
        n = counts.sum(1)                                                           # events so far (B,) float64
        label = input_ids[:, 3 + g.j_label + k * (g.S + 2)].eq(g.fraud_id).long()
        has_past = counts.gt(0)
        b = torch.bucketize(n, self.edges, right=True)
        gate = self.gate_blind[b] if self.arm == "repeat_blind" else self.gate[label, b]
        gate = torch.where(n.gt(0), gate, torch.ones_like(gate))
        # repeat branch
        freq = counts / counts.sum(1, keepdim=True).clamp(min=1)
        rec = torch.where(has_past, 1.0 / (n.unsqueeze(1) - last).clamp(min=1).double(), torch.zeros_like(counts))
        rec = rec / rec.sum(1, keepdim=True).clamp(min=1e-12)
        repeat = self.lam * freq + (1 - self.lam) * rec
        # novel branch: backbone over unseen merchants, native top-k decoding kept on them
        sc = scores[:, self.token_ids].double().masked_fill(has_past, -math.inf)
        nb = torch.softmax(top_k_filter(sc, self.top_k) if self.top_k else sc, -1)
        novel = nb
        if self.pi > 0:
            pp = self.pool.unsqueeze(0).expand(B, -1).masked_fill(has_past, 0.0); pz = pp.sum(1, keepdim=True)
            pp = torch.where(pz.gt(1e-12), pp / pz.clamp(min=1e-12), nb)
            fraud = label.eq(1).unsqueeze(1)
            novel = torch.where(fraud, self.pi * pp + (1 - self.pi) * nb, nb)
            self.pool_mass += float((gate * self.pi)[label.eq(1)].sum())
        probs = gate.unsqueeze(1) * novel + (1 - gate).unsqueeze(1) * repeat
        probs = probs / probs.sum(1, keepdim=True).clamp(min=1e-12)
        self.n_merchant_steps += 1; self.rows += B; self.novel_mass += float(gate.sum()); self.repeat_mass += float((1 - gate).sum())
        new = torch.full_like(scores, -math.inf)
        new[:, self.token_ids] = probs.clamp(min=1e-12).log().to(scores.dtype)
        if self.arm == "fraud_only":  # normal rows keep the native scores (with the native top-k); only fraud rows are replaced
            fraud = label.eq(1)
            self.replaced_rows += int(fraud.sum())
            native = top_k_filter(scores, self.top_k) if self.top_k else scores
            return torch.where(fraud.unsqueeze(1), new, native)
        self.replaced_rows += B
        return new


def update_state(samples, kept_rows, geom, mg, counts, last):
    """After a window batch: samples (B, T) generated token ids (numpy), kept_rows (B,) rows kept per window;
    counts/last (B, Vm) CPU tensors of the carried state -> updated in place with the kept rows' merchants."""
    compact = mg.compact.numpy()
    for b in range(samples.shape[0]):
        r = int(kept_rows[b]); pos = 3 + mg.j + np.arange(r) * (geom.S + 2); pos = pos[pos < samples.shape[1]]
        comp = compact[samples[b, pos]]; valid = comp >= 0; comp = comp[valid]
        n0 = int(counts[b].sum().item())
        counts[b] += torch.from_numpy(np.bincount(comp, minlength=mg.V).astype(np.float64))
        rpos = torch.from_numpy(n0 + np.arange(len(pos))[valid])
        last[b].scatter_reduce_(0, torch.from_numpy(comp), rpos, reduce="amax")
