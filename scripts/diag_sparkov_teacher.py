"""
Diagnostic: Sparkov ablation — does timing/category add signal above amount?

Mirrors diag_amlsim_teacher.py. Purpose: verify C2b generality.
  AMLSim: amount-only 0.903, +timing +0.061, +recv +0.014
  Sparkov: if Δ(timing above amount) > 0 → C2b generalises beyond AMLSim.
           if ≈ 0 → C2b may be AMLSim-local (headline needs adjustment).

Sparkov: d_num=1 (amt), d_cat=1 (category, 14 classes), L=24, B=16
"""

import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS  = 5000
LR     = 3e-4
BS     = 256
SEED   = 42


class SequenceTeacher(nn.Module):
    """Same architecture as in coherence_teacher.py — standalone copy for diagnostic."""
    def __init__(self, n_bins=16, n_cats=14, d_bin_emb=8, d_cat_emb=16, hidden=64,
                 use_time=True, use_cat=True):
        super().__init__()
        self.use_time = use_time
        self.use_cat  = use_cat
        d_in = 1
        if use_time:
            self.bin_emb = nn.Embedding(n_bins + 1, d_bin_emb, padding_idx=n_bins)
            d_in += d_bin_emb
        if use_cat:
            self.cat_emb = nn.Embedding(n_cats, d_cat_emb)
            d_in += d_cat_emb
        self.gru  = nn.GRU(d_in, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x_num, dt_bin, x_cat, mask):
        amt = x_num[:, :, :1].float()
        parts = [amt]
        if self.use_time:
            t_emb = self.bin_emb(dt_bin.clamp(0, self.bin_emb.num_embeddings - 1))
            parts.append(t_emb)
        if self.use_cat:
            c_emb = self.cat_emb(x_cat[:, :, 0].clamp(0, self.cat_emb.num_embeddings - 1))
            parts.append(c_emb)
        x = torch.cat(parts, dim=-1)
        h, _ = self.gru(x)
        m = mask.float().unsqueeze(-1)
        h_pool = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(h_pool).squeeze(-1)   # (B,) entity logit


def train_and_eval(model, x_num, dt_bin, x_cat, y_ent, mask, steps, lr, bs, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_num.shape[0]
    x_num_d  = x_num.to(device)
    dt_bin_d = dt_bin.to(device)
    x_cat_d  = x_cat.to(device)
    y_d      = y_ent.to(device).float()
    mask_d   = mask.to(device).bool()

    losses = []
    for step in range(steps):
        idx   = torch.randint(0, N, (bs,), device=device)
        logit = model(x_num_d[idx], dt_bin_d[idx], x_cat_d[idx], mask_d[idx])
        loss  = F.binary_cross_entropy_with_logits(logit, y_d[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    model.eval()
    all_probs = []
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            logit = model(x_num_d[s:e], dt_bin_d[s:e], x_cat_d[s:e], mask_d[s:e])
            all_probs.append(torch.sigmoid(logit).cpu().numpy())
    probs  = np.concatenate(all_probs)
    y_np   = y_ent.numpy()
    auprc  = average_precision_score(y_np, probs)
    auroc  = roc_auc_score(y_np, probs)
    bce    = float(np.mean(losses[-100:]))
    return auprc, auroc, bce


def load_data():
    root = ROOT / "data" / "sparkov" / "sequences"
    with open(root / "meta.json") as f:
        meta = json.load(f)
    tr = np.load(root / "train.npz")
    x_num  = torch.from_numpy(tr["x_num"]).float()
    dt_bin = torch.from_numpy(tr["dt_bin"]).long()
    x_cat  = torch.from_numpy(tr["x_cat"]).long()
    y      = torch.from_numpy(tr["y"]).float()
    mask   = torch.from_numpy(tr["mask"]).bool()
    # Sparkov: position-level label is direct (no propagation needed — entity IS the card)
    # Entity label = max fraud in window
    y_ent  = (y * mask.float()).max(dim=1).values
    return meta, x_num, dt_bin, x_cat, y_ent, mask


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    meta, x_num, dt_bin, x_cat, y_ent, mask = load_data()
    n_bins = len(meta["tau_k"])
    n_cats = list(meta["num_classes_cat"].values())[0]   # 14

    prev = y_ent.float().mean().item()
    print(f"Sparkov train: N={len(y_ent)}  fraud={y_ent.sum():.0f}  prevalence={prev:.4f}")
    print(f"n_bins={n_bins}  n_cats={n_cats}  steps={STEPS}\n")

    configs = [
        ("amount-only",        dict(use_time=False, use_cat=False)),
        ("amount+timing",      dict(use_time=True,  use_cat=False)),
        ("amount+timing+cat",  dict(use_time=True,  use_cat=True)),
    ]

    results = {}
    for name, kwargs in configs:
        torch.manual_seed(SEED)
        model    = SequenceTeacher(n_bins=n_bins, n_cats=n_cats, **kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        auprc, auroc, bce = train_and_eval(
            model, x_num, dt_bin, x_cat, y_ent, mask,
            steps=STEPS, lr=LR, bs=BS, device=DEVICE,
        )
        elapsed = time.time() - t0
        results[name] = {"auprc": auprc, "auroc": auroc, "bce": bce}
        print(f"[{name}]  AUPRC={auprc:.4f}  AUROC={auroc:.4f}  BCE={bce:.5f}"
              f"  params={n_params}  [{elapsed:.0f}s]")

    print("\n── Incremental window contribution ──")
    ap_amt  = results["amount-only"]["auprc"]
    ap_at   = results["amount+timing"]["auprc"]
    ap_full = results["amount+timing+cat"]["auprc"]
    print(f"  timing above amount:      Δ={ap_at - ap_amt:+.4f}")
    print(f"  category above +timing:   Δ={ap_full - ap_at:+.4f}")
    print(f"  window total Δ (full−amount): {ap_full - ap_amt:+.4f}")
    print(f"  prevalence baseline:      {prev:.4f}")

    print()
    print("── AMLSim reference ──")
    print(f"  [AMLSim] amount-only: 0.9032  +timing: +0.0611  +recv: +0.0137")
    print()
    print("── C2b Generality ──")
    if ap_at - ap_amt > 0.02:
        print(f"  → timing adds Δ={ap_at-ap_amt:+.4f} on Sparkov — C2b generalises (both datasets).")
    elif ap_at - ap_amt > 0:
        print(f"  → timing adds Δ={ap_at-ap_amt:+.4f} on Sparkov — marginal C2b generalisation.")
    else:
        print(f"  → timing Δ≈0 on Sparkov — C2b may be AMLSim-specific; headline may need narrowing.")

    if ap_full - ap_at > 0.01:
        print(f"  → category adds Δ={ap_full-ap_at:+.4f} above timing (merchant type informative).")
    else:
        print(f"  → category adds Δ≈0 above timing (merchant type has marginal marginal value).")


if __name__ == "__main__":
    main()
