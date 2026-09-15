"""
Diagnostic: g-bottleneck vs thesis bottleneck.

Tests whether replacing the 4-dim g summary teacher with a full-window
sequence encoder (BiGRU over raw features) changes AUPRC.

If seq-teacher reaches high AUPRC → g was the bottleneck, not thesis.
If seq-teacher also fails → AMLSim fraud signal is fundamentally inaccessible
  from raw window features → thesis C2b/C3 loses its main testbed.

Ablation (run inside this script):
  (a) amount-only GRU      → checks if static row feature drives the signal
  (b) amount + timing GRU  → checks if temporal pattern adds above amount
  (c) full (amt+time+recv) → checks if receiver diversity adds further

Key interpretation:
  If (b)-(a) or (c)-(b) Δ AUPRC is substantial → window/temporal/relational
  component provides signal beyond amount → C2b window-behavior coupling is real.
  If (c)≈(a) → amount alone explains everything → already known (C2a), no new story.
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


# ─── Models ───────────────────────────────────────────────────────────────────

class SequenceTeacher(nn.Module):
    """
    BiGRU over raw window features → entity P(fraud).
    Sees full temporal trajectory, not just 4-dim g.
    """
    def __init__(self, n_bins=16, n_cats=256, d_bin_emb=8, d_cat_emb=16, hidden=64,
                 use_time=True, use_recv=True):
        super().__init__()
        self.use_time = use_time
        self.use_recv = use_recv
        d_in = 1  # amount
        if use_time:
            self.bin_emb = nn.Embedding(n_bins + 1, d_bin_emb, padding_idx=n_bins)
            d_in += d_bin_emb
        if use_recv:
            self.cat_emb = nn.Embedding(n_cats, d_cat_emb)
            d_in += d_cat_emb
        self.gru  = nn.GRU(d_in, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x_num, dt_bin, x_cat, mask):
        amt = x_num[:, :, :1].float()                            # (B, L, 1)
        parts = [amt]
        if self.use_time:
            t_emb = self.bin_emb(dt_bin.clamp(0, self.bin_emb.num_embeddings - 1))
            parts.append(t_emb)                                   # (B, L, d_bin)
        if self.use_recv:
            c_emb = self.cat_emb(x_cat[:, :, 0])
            parts.append(c_emb)                                   # (B, L, d_cat)
        x = torch.cat(parts, dim=-1)                              # (B, L, d_in)
        h, _ = self.gru(x)                                        # (B, L, H*2)
        m = mask.float().unsqueeze(-1)
        h_pool = (h * m).sum(1) / m.sum(1).clamp(min=1)          # (B, H*2)
        return self.head(h_pool).squeeze(-1)                       # (B,)


# ─── Training & eval ──────────────────────────────────────────────────────────

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
        idx = torch.randint(0, N, (bs,), device=device)
        logit = model(x_num_d[idx], dt_bin_d[idx], x_cat_d[idx], mask_d[idx])
        loss  = F.binary_cross_entropy_with_logits(logit, y_d[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    # Entity-level AUPRC
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
    root = ROOT / "data" / "amlsim" / "sequences"
    with open(root / "meta.json") as f:
        meta = json.load(f)
    tr = np.load(root / "train.npz")
    x_num  = torch.from_numpy(tr["x_num"]).float()
    dt_bin = torch.from_numpy(tr["dt_bin"]).long()
    x_cat  = torch.from_numpy(tr["x_cat"]).long()
    y      = torch.from_numpy(tr["y"]).float()
    mask   = torch.from_numpy(tr["mask"]).bool()
    y_ent  = (y * mask.float()).max(dim=1).values  # entity label
    return meta, x_num, dt_bin, x_cat, y_ent, mask


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    meta, x_num, dt_bin, x_cat, y_ent, mask = load_data()
    n_bins = len(meta["tau_k"])
    n_cats = list(meta["num_classes_cat"].values())[0]  # 256

    prev = y_ent.float().mean().item()
    print(f"AMLSim train: N={len(y_ent)}  fraud={y_ent.sum():.0f}  prevalence={prev:.4f}")
    print(f"n_bins={n_bins}  n_cats={n_cats}  steps={STEPS}\n")

    configs = [
        ("amount-only",       dict(use_time=False, use_recv=False)),
        ("amount+timing",     dict(use_time=True,  use_recv=False)),
        ("amount+timing+recv",dict(use_time=True,  use_recv=True)),
    ]

    results = {}
    for name, kwargs in configs:
        torch.manual_seed(SEED)
        model = SequenceTeacher(n_bins=n_bins, n_cats=n_cats, **kwargs)
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
    ap_full = results["amount+timing+recv"]["auprc"]
    print(f"  timing above amount:   Δ={ap_at - ap_amt:+.4f}")
    print(f"  receiver above +timing: Δ={ap_full - ap_at:+.4f}")
    print(f"  window total Δ (full − amount): {ap_full - ap_amt:+.4f}")
    print(f"  prevalence baseline: {prev:.4f}")
    print()
    print("── Interpretation ──")
    if ap_full >= 0.5:
        print("  → seq-teacher STRONG: g was the bottleneck, not thesis.")
        print("    C2b (window-behavior coupling) likely valid; rebuild CoherenceTeacher as seq-encoder.")
        if ap_full - ap_amt < 0.05:
            print("  [WARN] window adds <0.05 AUPRC above amount-only → mostly C2a (static amount), not C2b.")
    elif ap_full < 0.2:
        print("  → seq-teacher WEAK: AMLSim window features fundamentally insufficient.")
        print("    g bottleneck AND seq-encoder bottleneck → thesis C2b/C3 needs different testbed.")
    else:
        print(f"  → seq-teacher MODERATE (AUPRC={ap_full:.3f}): partial signal.")
        print("    Check window Δ — if timing/recv contribute, C2b has partial support.")


if __name__ == "__main__":
    main()
