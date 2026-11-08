"""Two module components on a frozen REaLTabFormer relational fit (protocol: docs/ibm_realtabformer_v1/PROTOCOL.md).

 (i) transition-table label decoding: at the label column's token position the model logits are replaced by the
     strict-past transition table P(fraud | previous label, previous run age, ever fraud) estimated on training
     customers (ibm_generation_module_v1), with the run state carried over across a customer's windows.
(ii) label-routed fraud-only output head: a copy of the decoder's last-block `ln_2 + mlp`, `ln_f` and the rows of
     `lm_head` that belong to the routed columns' tokens. Token positions are deterministic in the relational
     layout `[EOS][BOS]([BMEM] c_0 .. c_{S-1} [EMEM])* [EOS]`, so a position is routed iff its row's label token
     (an earlier column of the same row) is the fraud token and its column is a routed column.

Both are realised outside the model forward: a forward pre-hook captures the input of the last block's `ln_2`
(the residual stream after attention), and a `LogitsProcessor` rewrites the scores at the affected positions.
The native parameters are never modified. Runs with the realtabformer venv (torch 2.5.1, transformers 4.46.3).
"""
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessor

from realtabformer.data_utils import SPECIAL_COL_SEP, SpecialTokens, decode_processed_column

LABEL = "event_is_fraud"
ROUTED = ("gap", "amount_or_numeric_value", "category", "use_chip")
MAX_AGE = 21  # as in benchmarks/argn_generation_module.py


def original_column(proc_col):
    """Processed sub-column name -> original column (numeric partitions `..._00` are folded back)."""
    name = decode_processed_column(proc_col)
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and f"{SPECIAL_COL_SEP}NUMERIC{SPECIAL_COL_SEP}" in proc_col:
        return parts[0]
    return name


class Geometry:
    """Token-position arithmetic for one fitted relational model."""

    def __init__(self, rtf, label=LABEL, routed=ROUTED):
        self.cols = list(rtf.processed_columns)
        self.S = len(self.cols)
        assert self.S == rtf.relational_col_size
        vocab = rtf.vocab["decoder"]
        self.t2i, self.i2t = vocab["token2id"], vocab["id2token"]
        self.V = len(self.i2t)
        orig = [original_column(c) for c in self.cols]
        lab = [j for j, o in enumerate(orig) if o == label]
        assert len(lab) == 1, (orig, label)
        self.j_label = lab[0]
        self.label_col = self.cols[self.j_label]
        self.fraud_id = self.t2i[f"{self.label_col}{SPECIAL_COL_SEP}1"]
        self.normal_id = self.t2i[f"{self.label_col}{SPECIAL_COL_SEP}0"]
        # only columns that FOLLOW the label can be routed (the row's label must already be in the prefix)
        self.routed_j = [j for j, o in enumerate(orig) if o in routed and j > self.j_label]
        self.unroutable = [self.cols[j] for j, o in enumerate(orig) if o in routed and j <= self.j_label]
        self.col_ids = {j: sorted(vocab["column_token_ids"][c]) for j, c in enumerate(self.cols)}
        self.routed_ids = sorted(set(i for j in self.routed_j for i in self.col_ids[j]))  # union, size K
        pos = {t: k for k, t in enumerate(self.routed_ids)}
        self.compact = torch.full((self.V,), -1, dtype=torch.long)
        for t, k in pos.items():
            self.compact[t] = k
        # per routed column: boolean mask over the compact K
        self.allowed = torch.zeros((self.S, len(self.routed_ids)), dtype=torch.bool)
        for j in self.routed_j:
            for t in self.col_ids[j]:
                self.allowed[j, pos[t]] = True
        self.bos = self.t2i[SpecialTokens.BOS]; self.eos = self.t2i[SpecialTokens.EOS]
        self.pad = self.t2i[SpecialTokens.PAD]; self.bmem = self.t2i[SpecialTokens.BMEM]; self.emem = self.t2i[SpecialTokens.EMEM]

    # labels layout: labels[0]=BOS; row k occupies labels[1+k(S+2) .. 1+k(S+2)+S+1]: BMEM, c_0..c_{S-1}, EMEM
    def label_index(self, k, j):
        """Index in `labels` of sub-column j of row k (decoder_input_ids index is +1)."""
        return 2 + j + k * (self.S + 2)

    def row_col_of_label_index(self, i):
        """labels index -> (row k, col j) or (k, -1 BMEM / S EMEM / None for BOS/EOS)."""
        if i <= 0:
            return None, None
        idx = i - 1
        k, c = divmod(idx, self.S + 2)
        return k, c - 1

    def decoder_input_structure(self, labels):
        """For a (B, T) labels tensor: tensors row (B,T), col (B,T) with col in [-1..S] (−2 for BOS/EOS/PAD)."""
        T = labels.shape[1]
        i = torch.arange(T, device=labels.device)
        idx = i - 1
        k = torch.div(idx, self.S + 2, rounding_mode="floor")
        c = idx - k * (self.S + 2) - 1
        c = torch.where(i.eq(0), torch.full_like(c, -2), c)
        return k.unsqueeze(0).expand_as(labels), c.unsqueeze(0).expand_as(labels)

    def row_labels(self, labels):
        """(B, T) labels -> (B, T) long: fraud flag (1/0) of the row each position belongs to; -1 outside rows."""
        B, T = labels.shape
        k, c = self.decoder_input_structure(labels)
        lab_pos = (1 + self.j_label + k.clamp(min=0) * (self.S + 2) + 1).clamp(max=T - 1)  # labels index of row k's label
        lab_tok = torch.gather(labels, 1, lab_pos)
        flag = lab_tok.eq(self.fraud_id).long()
        flag = torch.where(lab_tok.eq(self.normal_id) | lab_tok.eq(self.fraud_id), flag, torch.full_like(flag, -1))
        valid = c.ge(0) & c.lt(self.S) & ~labels.eq(self.pad)
        return torch.where(valid, flag, torch.full_like(flag, -1)), k, c


class FraudHead(nn.Module):
    """Fraud-only copy of the decoder's output path: last block `ln_2 + mlp`, `ln_f`, sliced `lm_head`."""

    def __init__(self, decoder, routed_ids):
        super().__init__()
        blk = decoder.transformer.h[-1]
        self.ln_2 = copy.deepcopy(blk.ln_2)
        self.mlp = copy.deepcopy(blk.mlp)
        self.ln_f = copy.deepcopy(decoder.transformer.ln_f)
        ids = torch.as_tensor(routed_ids, dtype=torch.long)
        W = decoder.lm_head.weight.detach()[ids.to(decoder.lm_head.weight.device)].clone()
        self.out = nn.Linear(W.shape[1], W.shape[0], bias=False)
        self.out.weight.data.copy_(W)
        self.register_buffer("routed_ids", ids)

    def forward(self, x_mid):
        h = x_mid + self.mlp(self.ln_2(x_mid))
        return self.out(self.ln_f(h))  # (N, K) logits over the routed token union


class Capture:
    """Forward pre-hook on the last block's ln_2: stores the residual stream after attention."""

    def __init__(self, decoder):
        self.x = None
        self.handle = decoder.transformer.h[-1].ln_2.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.x = args[0]

    def remove(self):
        self.handle.remove()


def transition_probability(table, n, prev, age, ever):
    """table (3, MAX_AGE+1, 2) float64 tensor; n, prev, age, ever (B,) long -> P(fraud) (B,) float64."""
    row = torch.where(n.gt(0), prev + 1, torch.zeros_like(prev))
    return table[row, age.clamp(0, MAX_AGE), ever].clamp(1e-6, 1 - 1e-6)


def run_state(flags, init):
    """flags (B, k) {0,1} labels generated so far in this window; init (B, 4) long [n, prev, age, ever] carried over.
    Returns (n, prev, age, ever) after those k rows."""
    B, k = flags.shape
    n, prev, age, ever = init[:, 0].clone(), init[:, 1].clone(), init[:, 2].clone(), init[:, 3].clone()
    for t in range(k):
        lab = flags[:, t]
        same = prev.eq(lab) & n.gt(0)
        age = torch.where(same, age + 1, torch.ones_like(age))
        prev = lab; n = n + 1; ever = torch.maximum(ever, lab)
    return n, prev, age, ever


class ModuleProcessor(LogitsProcessor):
    """Rewrites scores at (i) the label position (transition table) and (ii) routed positions of fraud rows."""

    def __init__(self, geom, table=None, init_state=None, head=None, capture=None):
        self.g = geom
        self.table = table
        self.init = init_state
        self.head = head
        self.capture = capture
        self.n_label_steps = 0; self.n_routed_rows = 0

    def __call__(self, input_ids, scores):
        g = self.g
        n = input_ids.shape[1]
        idx = n - 2
        if idx < 0:
            return scores
        k, c = divmod(idx, g.S + 2)
        col = c - 1
        if col < 0 or col >= g.S:
            return scores
        B = input_ids.shape[0]
        if self.table is not None and col == g.j_label:
            self.n_label_steps += 1
            if k > 0:
                pos = 3 + g.j_label + torch.arange(k, device=input_ids.device) * (g.S + 2)
                flags = input_ids[:, pos].eq(g.fraud_id).long()
            else:
                flags = torch.zeros((B, 0), dtype=torch.long, device=input_ids.device)
            init = self.init if self.init is not None else torch.zeros((B, 4), dtype=torch.long, device=input_ids.device)
            n_, prev, age, ever = run_state(flags, init)
            p = transition_probability(self.table, n_, prev, age, ever)
            new = torch.full_like(scores, -math.inf)
            new[:, g.fraud_id] = p.log().to(scores.dtype)
            new[:, g.normal_id] = torch.log1p(-p).to(scores.dtype)
            return new
        if self.head is not None and col in g.routed_j and col > g.j_label:
            lab = input_ids[:, 3 + g.j_label + k * (g.S + 2)].eq(g.fraud_id)
            if lab.any():
                self.n_routed_rows += int(lab.sum())
                x = self.capture.x[:, -1]
                fl = self.head(x[lab])  # (Nf, K)
                full = torch.full((int(lab.sum()), scores.shape[1]), -math.inf, device=scores.device, dtype=scores.dtype)
                full[:, self.head.routed_ids] = fl.to(scores.dtype)
                full = torch.where(torch.isinf(scores[lab]), scores[lab], full)  # keep the column-vocab constraint
                new = scores.clone()
                new[lab] = full
                return new
        return scores


def final_state(label_flags, init):
    """label_flags: list/array of {0,1} for the rows of one window (decoded), init: (4,) -> (4,) carried state."""
    f = torch.as_tensor(np.asarray(label_flags, dtype=np.int64)).unsqueeze(0)
    n, prev, age, ever = run_state(f, torch.as_tensor(init, dtype=torch.long).unsqueeze(0))
    return torch.stack([n, prev, age, ever], -1)[0]


# ----------------------------------------------------------------------------------------- teacher-forced heads
def routed_targets(geom, labels):
    """labels (B, T) -> mask (B, T) of positions whose target token is a routed column of a fraud row, plus col."""
    flag, k, c = geom.row_labels(labels)
    routed = torch.zeros_like(c, dtype=torch.bool)
    for j in geom.routed_j:
        routed |= c.eq(j)
    return flag.eq(1) & routed, c


def substitute_inputs(geom, labels, dec_in, rng):
    """routed_subst: replace every non-label field of each fraud row in `dec_in` with the fields of a random normal
    row of the same window (targets untouched). dec_in (B, T) = shift_right(labels). Returns a new tensor."""
    out = dec_in.clone()
    B, T = labels.shape
    flag, k, c = geom.row_labels(labels)
    S = geom.S
    for b in range(B):
        rows = int(torch.div(T - 2, S + 2, rounding_mode="floor").item()) + 1
        lab_idx = [geom.label_index(r, geom.j_label) for r in range(rows)]
        lab_idx = [i for i in lab_idx if i < T]
        toks = labels[b, lab_idx]
        fraud_rows = [r for r, t in enumerate(toks.tolist()) if t == geom.fraud_id]
        normal_rows = [r for r, t in enumerate(toks.tolist()) if t == geom.normal_id]
        if not fraud_rows or not normal_rows:
            continue
        for r in fraud_rows:
            src = normal_rows[int(rng.integers(len(normal_rows)))]
            for j in range(S):
                if j == geom.j_label:
                    continue
                # decoder input index = labels index + 1
                out[b, geom.label_index(r, j) + 1] = labels[b, geom.label_index(src, j)]
    return out


def head_logits_at(geom, head, x_mid, mask):
    """x_mid (B, T, H) captured residual; mask (B, T) -> compact logits (N, K) at masked positions."""
    return head(x_mid[mask])


def column_restricted_ce(geom, logits, targets, cols, reduction="none"):
    """logits (N, K) compact; targets (N,) vocab ids; cols (N,) sub-column index. CE within each column's vocab."""
    allowed = geom.allowed.to(logits.device)[cols]  # (N, K)
    masked = logits.masked_fill(~allowed, -math.inf)
    tgt = geom.compact.to(logits.device)[targets]
    assert tgt.ge(0).all(), "target outside routed union"
    return F.cross_entropy(masked, tgt, reduction=reduction)
