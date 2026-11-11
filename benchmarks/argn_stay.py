"""ibm_normal_impl_v1 (docs/ibm_normal_impl_v1/PROTOCOL.md): gap-conditional stay at the previous merchant, inside the generator.

Real normal rows that follow a normal row within minutes are at the SAME merchant (<= 1 min .79, 5-10 min .68, >= 1 h .05); the
adopted model's repeat pointer ignores the gap (.05 everywhere). Here, at the merchant sub-column of a normal row whose previous row
is normal, the probability mass that the merchant module puts on ALREADY-VISITED merchants is re-divided so that the previous row's
merchant gets the train share for the row's gap bin (never less than the pointer already gave it). The mass on novel merchants, the
fraud rows, the rare bucket and every other column are untouched. Enter AFTER merchant_module: it reads that module's per-customer
memory (mm_counts, mm_last) and the base module's (n, previous label, ...). The stay table is train-only; no training.
"""
from contextlib import contextmanager
import numpy as np
import torch

from benchmarks.argn_merchant_memory import _codes, LABEL, MERCHANT
from benchmarks.argn_episode_state import gap_seconds_by_code

GBINS = [60, 300, 600, 1800, 3600, 14400, 86400]                 # seconds: <=1m, 1-5m, 5-10m, 10-30m, 30-60m, 1-4h, 4-24h, >24h


@contextmanager
def stay_module(stats, stay, label=LABEL, merchant=MERCHANT, gap="gap"):
    import mostlyai.engine._tabular.generation as generation
    from mostlyai.engine._common import get_argn_name
    assert len(stay) == len(GBINS) + 1
    lcol = stats["columns"][label]; label_key = get_argn_name(lcol["argn_processor"], lcol["argn_table"], lcol["argn_column"]) + "__cat"; c1 = int(lcol["codes"]["1"])
    mcol = stats["columns"][merchant]; merchant_key = get_argn_name(mcol["argn_processor"], mcol["argn_table"], mcol["argn_column"]) + "__cat"
    gcol = stats["columns"][gap]; gap_key = get_argn_name(gcol["argn_processor"], gcol["argn_table"], gcol["argn_column"]) + "__bin"
    _, rare_code, V = _codes(stats)
    sec = np.asarray(gap_seconds_by_code(gcol)); by_code = torch.tensor(np.asarray(stay, dtype=float)[np.searchsorted(np.asarray(GBINS), sec, side="left")], dtype=torch.float64)
    parent_model = generation.SequentialModel
    counter = dict(merchant_steps=0, eligible_rows=0, raised_rows=0, target_mass=0.0, pointer_mass=0.0)

    class StayModel(parent_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sy_by_code = by_code.to(self.device); self.sy_label = None; self.sy_gap = None; self.sy_hooked = False
            self.embedders.get(label_key).register_forward_pre_hook(lambda mod, a: self._sy_capture("sy_label", a))
            self.embedders.get(gap_key).register_forward_pre_hook(lambda mod, a: self._sy_capture("sy_gap", a))

        def _sy_capture(self, name, args):
            out = args[0]
            if out.dim() == 2:  # generation: sampled codes (B,1)
                setattr(self, name, out.reshape(-1).long())

        def _sy_replace(self, module, args, output):
            if args[1] != merchant_key or self.sy_label is None or self.sy_gap is None:
                return output
            counts, last, base = self.mm_counts, self.mm_last, self.module_memory           # merchant memory before this row; base: n, previous label, run age, ever
            n = counts.sum(1).long(); probs = torch.softmax(output[:, -1, :].double(), -1)
            prev_m = last.eq((n - 1).unsqueeze(1)).long().argmax(1)                            # the previous row's merchant code
            has_prev = base[:, 0].gt(0) & torch.gather(last, 1, prev_m.unsqueeze(1)).squeeze(1).eq(n - 1)
            seen = counts.gt(0); seen[:, rare_code] = False                                    # the rare bucket is many merchants, never one to stay at
            R = (probs * seen).sum(1); p_prev = torch.gather(probs, 1, prev_m.unsqueeze(1)).squeeze(1); cur = (p_prev / R.clamp(min=1e-12)).clamp(0, 1)
            elig = self.sy_label.ne(c1) & base[:, 1].eq(0) & has_prev & prev_m.ne(rare_code) & R.gt(1e-9)
            target = torch.maximum(self.sy_by_code[self.sy_gap.clamp(0, len(self.sy_by_code) - 1)], cur); rows = elig & target.gt(cur) & cur.lt(1 - 1e-9)
            counter["merchant_steps"] += 1; counter["eligible_rows"] += int(elig.sum()); counter["raised_rows"] += int(rows.sum())
            counter["target_mass"] += float((R * target)[elig].sum()); counter["pointer_mass"] += float(p_prev[elig].sum())
            if not rows.any():
                return output
            scale = ((1 - target) / (1 - cur).clamp(min=1e-12)).unsqueeze(1)
            new = torch.where(seen, probs * scale, probs); new.scatter_(1, prev_m.unsqueeze(1), (R * target).unsqueeze(1))
            new = torch.where(rows.unsqueeze(1), new, probs); new = new / new.sum(1, keepdim=True).clamp(min=1e-12)
            out = output.clone(); out[:, -1, :] = new.clamp(min=1e-12).log().to(output.dtype)
            return out

        def forward(self, x, mode, **kwargs):
            if mode == "gen":
                self.sy_label = None; self.sy_gap = None
                if not self.sy_hooked:  # registered at the first generation step -> runs after the merchant module's hook and the routed heads' hooks
                    self.predictors.register_forward_hook(self._sy_replace); self.sy_hooked = True
            return super().forward(x, mode, **kwargs)

    generation.SequentialModel = StayModel
    try:
        yield dict(stay=list(map(float, stay)), merchant_key=merchant_key, gap_key=gap_key, counter=counter)
    finally:
        generation.SequentialModel = parent_model
