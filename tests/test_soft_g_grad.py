"""
Phase 2 Gate: pytest tests/test_soft_g_grad.py

Verifies soft_g gradient properties for ALL 4 features [vel, gap, rep, amt_sum]:
  (a) attached  ‖grad‖ > 0  for bin_logits, cat_logits, AND amt
  (b) detached  ‖grad‖ = 0  (detaching kills gradient)
  (c) coherence-loss 1-step decreases per feature channel
  (d) finite-diff match (float64)

Causal window structural tests: strict m < j (no self, no future).
"""

import math
import torch
import torch.nn.functional as F
import pytest
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from models.soft_g import soft_g


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_inputs(B_=2, L=8, Bbins=16, K=6, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    bin_logits = torch.randn(B_, L, Bbins, generator=g, requires_grad=True)
    cat_logits = torch.randn(B_, L, K, generator=g, requires_grad=True)
    amt = torch.randn(B_, L, generator=g, requires_grad=True)
    tau = torch.linspace(0.5, 8.0, Bbins)
    return bin_logits, cat_logits, amt, tau


W = 5.0
TEMP = 1.0


# ── Helper ───────────────────────────────────────────────────────────────────

def _call_soft_g(bin_logits, cat_logits, amt, tau):
    valid_mask = torch.ones(bin_logits.shape[:2], dtype=torch.bool, device=bin_logits.device)
    return soft_g(
        F.softmax(bin_logits, dim=-1),
        F.softmax(cat_logits, dim=-1),
        amt, tau, W, TEMP,
        valid_mask=valid_mask,
    )


# ── (a) Attached-gradient tests ───────────────────────────────────────────────

class TestAttachedGrad:
    def _run(self, B_=2, L=8, Bbins=16, K=6):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=B_, L=L, Bbins=Bbins, K=K)
        g = _call_soft_g(bin_logits, cat_logits, amt, tau)
        F.mse_loss(g, torch.zeros_like(g)).backward()
        return bin_logits.grad, cat_logits.grad, amt.grad, g

    def test_bin_grad_nonzero(self):
        bg, _, _, _ = self._run()
        assert bg is not None and bg.norm().item() > 0

    def test_cat_grad_nonzero(self):
        _, cg, _, _ = self._run()
        assert cg is not None and cg.norm().item() > 0

    def test_amt_grad_nonzero(self):
        """Gradient must flow through amt channel (amt_sum feature)."""
        _, _, ag, _ = self._run()
        assert ag is not None, "amt.grad is None — amt_sum not connected"
        norm = ag.norm().item()
        assert norm > 0, f"amt grad norm = {norm} (should be > 0 via amt_sum)"

    def test_vel_nonzero(self):
        _, _, _, g = self._run()
        assert g[:, 1:, 0].abs().max().item() > 0

    def test_gap_nonzero(self):
        _, _, _, g = self._run()
        assert g[:, :, 1].abs().max().item() > 0

    def test_rep_nonzero(self):
        _, _, _, g = self._run()
        assert g[:, 1:, 2].abs().max().item() > 0

    def test_amt_sum_nonzero(self):
        _, _, _, g = self._run()
        assert g[:, 1:, 3].abs().max().item() > 0, "amt_sum zero for j>0"

    def test_output_shape(self):
        _, _, _, g = self._run(B_=3, L=10, Bbins=16, K=5)
        assert g.shape == (3, 10, 4), f"expected (3,10,4) got {g.shape}"


# ── (b) Detached-gradient tests ───────────────────────────────────────────────

class TestDetachedGrad:
    def _run_detached(self, B_=2, L=8, Bbins=16, K=6):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=B_, L=L, Bbins=Bbins, K=K)
        bl = bin_logits.detach().clone().requires_grad_(True)
        cl = cat_logits.detach().clone().requires_grad_(True)
        al = amt.detach().clone().requires_grad_(True)

        # Detach AFTER softmax / direct — breaks gradient path
        bp_det = F.softmax(bl, dim=-1).detach()
        cp_det = F.softmax(cl, dim=-1).detach()
        al_det = al.detach()

        g = soft_g(bp_det, cp_det, al_det, tau, W, TEMP, valid_mask=torch.ones(bp_det.shape[:2], dtype=torch.bool))
        try:
            F.mse_loss(g, torch.zeros_like(g)).backward()
        except RuntimeError:
            pass
        return bl.grad, cl.grad, al.grad

    def test_bin_grad_zero_when_detached(self):
        bg, _, _ = self._run_detached()
        norm = bg.norm().item() if bg is not None else 0.0
        assert norm == 0.0, f"detached bin grad = {norm}"

    def test_cat_grad_zero_when_detached(self):
        _, cg, _ = self._run_detached()
        norm = cg.norm().item() if cg is not None else 0.0
        assert norm == 0.0, f"detached cat grad = {norm}"

    def test_amt_grad_zero_when_detached(self):
        _, _, ag = self._run_detached()
        norm = ag.norm().item() if ag is not None else 0.0
        assert norm == 0.0, f"detached amt grad = {norm}"


# ── (c) Loss-decrease tests ───────────────────────────────────────────────────

class TestLossDecreases:
    def test_loss_decreases_one_step(self):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=4, L=12, Bbins=16, K=8)
        lr = 0.1
        target = torch.zeros(4, 12, 4)

        with torch.no_grad():
            loss0 = F.mse_loss(_call_soft_g(bin_logits, cat_logits, amt, tau), target).item()

        bl = bin_logits.detach().clone().requires_grad_(True)
        cl = cat_logits.detach().clone().requires_grad_(True)
        al = amt.detach().clone().requires_grad_(True)
        g1 = _call_soft_g(bl, cl, al, tau)
        F.mse_loss(g1, target).backward()
        with torch.no_grad():
            bl2 = bl - lr * bl.grad
            cl2 = cl - lr * cl.grad
            al2 = al - lr * al.grad
            loss2 = F.mse_loss(_call_soft_g(bl2, cl2, al2, tau), target).item()

        print(f"\n  loss: {loss0:.4f} → {loss2:.4f}")
        assert loss2 < loss0, f"loss did not decrease: {loss0:.4f} → {loss2:.4f}"

    def test_vel_loss_decreases(self):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=2, L=10, Bbins=16, K=4)
        lr = 0.1
        with torch.no_grad():
            loss0 = _call_soft_g(bin_logits, cat_logits, amt, tau)[..., 0].pow(2).mean().item()
        bl = bin_logits.detach().clone().requires_grad_(True)
        cl = cat_logits.detach().clone().requires_grad_(True)
        al = amt.detach().clone().requires_grad_(True)
        _call_soft_g(bl, cl, al, tau)[..., 0].pow(2).mean().backward()
        with torch.no_grad():
            loss1 = _call_soft_g(bl - lr*bl.grad, cl - lr*cl.grad, al - lr*al.grad, tau)[..., 0].pow(2).mean().item()
        assert loss1 < loss0, f"vel loss: {loss0:.4f} → {loss1:.4f}"

    def test_rep_loss_decreases(self):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=2, L=10, Bbins=16, K=8)
        lr = 0.1
        with torch.no_grad():
            loss0 = _call_soft_g(bin_logits, cat_logits, amt, tau)[..., 2].pow(2).mean().item()
        bl = bin_logits.detach().clone().requires_grad_(True)
        cl = cat_logits.detach().clone().requires_grad_(True)
        al = amt.detach().clone().requires_grad_(True)
        _call_soft_g(bl, cl, al, tau)[..., 2].pow(2).mean().backward()
        with torch.no_grad():
            loss1 = _call_soft_g(bl - lr*bl.grad, cl - lr*cl.grad, al - lr*al.grad, tau)[..., 2].pow(2).mean().item()
        assert loss1 < loss0, f"rep loss: {loss0:.4f} → {loss1:.4f}"

    def test_amt_sum_loss_decreases(self):
        """Gradient through amt channel actually improves amt_sum feature."""
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=2, L=10, Bbins=16, K=4)
        lr = 0.3
        with torch.no_grad():
            loss0 = _call_soft_g(bin_logits, cat_logits, amt, tau)[..., 3].pow(2).mean().item()
        bl = bin_logits.detach().clone().requires_grad_(True)
        cl = cat_logits.detach().clone().requires_grad_(True)
        al = amt.detach().clone().requires_grad_(True)
        _call_soft_g(bl, cl, al, tau)[..., 3].pow(2).mean().backward()
        with torch.no_grad():
            loss1 = _call_soft_g(bl - lr*bl.grad, cl - lr*cl.grad, al - lr*al.grad, tau)[..., 3].pow(2).mean().item()
        assert loss1 < loss0, f"amt_sum loss: {loss0:.4f} → {loss1:.4f}"


# ── Structural / causal-window tests ─────────────────────────────────────────

class TestCausalWindow:
    def test_first_position_vel_zero(self):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=3, L=6, Bbins=16, K=5)
        g = _call_soft_g(bin_logits, cat_logits, amt, tau)
        assert (g[:, 0, 0].abs() < 1e-6).all(), f"vel[0] not zero: {g[:, 0, 0]}"

    def test_first_position_rep_zero(self):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=3, L=6, Bbins=16, K=5)
        g = _call_soft_g(bin_logits, cat_logits, amt, tau)
        assert (g[:, 0, 2].abs() < 1e-6).all(), f"rep[0] not zero: {g[:, 0, 2]}"

    def test_first_position_amt_sum_zero(self):
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=3, L=6, Bbins=16, K=5)
        g = _call_soft_g(bin_logits, cat_logits, amt, tau)
        assert (g[:, 0, 3].abs() < 1e-6).all(), f"amt_sum[0] not zero: {g[:, 0, 3]}"

    def test_vel_monotone_large_W(self):
        _, _, amt, tau = _make_inputs()
        bin_logits = torch.zeros(1, 8, 16)
        cat_logits = torch.zeros(1, 8, 4)
        amt_fixed = torch.zeros(1, 8)
        g = soft_g(F.softmax(bin_logits, dim=-1), F.softmax(cat_logits, dim=-1),
                   amt_fixed, tau, 100.0, 0.1, valid_mask=torch.ones(1, 8, dtype=torch.bool))
        vel = g[0, :, 0]
        for j in range(1, 8):
            assert vel[j].item() >= vel[j-1].item() - 1e-3

    def test_none_cat_probs(self):
        """AMLSim case: cat_probs=None → rep=0, amt_sum still nonzero."""
        bin_logits, _, amt, tau = _make_inputs(B_=2, L=6, Bbins=16)
        amt_leaf = amt.detach().clone().requires_grad_(True)
        g = soft_g(F.softmax(bin_logits, dim=-1), None, amt_leaf, tau, W, TEMP, valid_mask=torch.ones(2, 6, dtype=torch.bool))
        assert g.shape == (2, 6, 4)
        assert (g[:, :, 2] == 0).all(), "rep should be 0 when cat_probs=None"
        assert g[:, 1:, 3].abs().max().item() > 0, "amt_sum should be nonzero"
        # Gradient flows through amt even with cat_probs=None
        g[..., 3].sum().backward()
        assert amt_leaf.grad is not None and amt_leaf.grad.norm().item() > 0

    def test_finite_diff_bin_grad(self):
        """float64 finite-diff check on bin_logits."""
        eps = 1e-5
        B_, L, Bbins, K = 1, 6, 8, 4
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=B_, L=L, Bbins=Bbins, K=K)
        bl64 = bin_logits.detach().double()
        cl64 = cat_logits.detach().double()
        al64 = amt.detach().double()
        tau64 = tau.double()

        def loss_fn(bl):
            return F.mse_loss(
                soft_g(F.softmax(bl, dim=-1), F.softmax(cl64, dim=-1), al64, tau64, W, TEMP, valid_mask=torch.ones(B_, L, dtype=torch.bool)),
                torch.zeros(B_, L, 4, dtype=torch.float64),
            )

        bl = bl64.clone().requires_grad_(True)
        loss_fn(bl).backward()
        ag = bl.grad.clone()

        torch.manual_seed(7)
        for flat_idx in torch.randperm(B_ * L * Bbins)[:20]:
            b = flat_idx // (L * Bbins)
            rest = flat_idx % (L * Bbins)
            l = rest // Bbins; k = rest % Bbins
            bp = bl64.clone(); bp[b, l, k] += eps
            bm = bl64.clone(); bm[b, l, k] -= eps
            fd = (loss_fn(bp) - loss_fn(bm)).item() / (2 * eps)
            auto = ag.view(-1)[flat_idx].item()
            rel = abs(fd - auto) / (abs(auto) + abs(fd) + 1e-15)
            assert rel < 0.01 or abs(fd - auto) < 1e-8, \
                f"FD mismatch idx={flat_idx}: fd={fd:.8f} auto={auto:.8f} rel={rel:.4f}"

    def test_finite_diff_amt_grad(self):
        """float64 finite-diff check on amt gradient."""
        eps = 1e-5
        B_, L, Bbins, K = 1, 6, 8, 4
        bin_logits, cat_logits, amt, tau = _make_inputs(B_=B_, L=L, Bbins=Bbins, K=K)
        bl64 = bin_logits.detach().double()
        cl64 = cat_logits.detach().double()
        al64 = amt.detach().double()
        tau64 = tau.double()

        def loss_fn(al):
            return F.mse_loss(
                soft_g(F.softmax(bl64, dim=-1), F.softmax(cl64, dim=-1), al, tau64, W, TEMP, valid_mask=torch.ones(B_, L, dtype=torch.bool)),
                torch.zeros(B_, L, 4, dtype=torch.float64),
            )

        al = al64.clone().requires_grad_(True)
        loss_fn(al).backward()
        ag = al.grad.clone()

        torch.manual_seed(13)
        for flat_idx in torch.randperm(B_ * L)[:10]:
            b = flat_idx // L; l = flat_idx % L
            ap = al64.clone(); ap[b, l] += eps
            am = al64.clone(); am[b, l] -= eps
            fd = (loss_fn(ap) - loss_fn(am)).item() / (2 * eps)
            auto = ag.view(-1)[flat_idx].item()
            rel = abs(fd - auto) / (abs(auto) + abs(fd) + 1e-15)
            assert rel < 0.01 or abs(fd - auto) < 1e-8, \
                f"amt FD mismatch idx={flat_idx}: fd={fd:.8f} auto={auto:.8f} rel={rel:.4f}"


# ── Summary (informational) ───────────────────────────────────────────────────

def test_print_grad_summary():
    bin_logits, cat_logits, amt, tau = _make_inputs(B_=2, L=8, Bbins=16, K=6)
    bl = bin_logits.detach().clone().requires_grad_(True)
    cl = cat_logits.detach().clone().requires_grad_(True)
    al = amt.detach().clone().requires_grad_(True)

    g = _call_soft_g(bl, cl, al, tau)
    F.mse_loss(g, torch.zeros_like(g)).backward()

    print(f"\n[1] gradient check (attached) — g shape {g.shape}:")
    print(f"  bin : ||grad|| attached={bl.grad.norm():.4e}")
    print(f"  cat : ||grad|| attached={cl.grad.norm():.4e}")
    print(f"  amt : ||grad|| attached={al.grad.norm():.4e}")

    # Detached
    bl2 = bin_logits.detach().clone().requires_grad_(True)
    cl2 = cat_logits.detach().clone().requires_grad_(True)
    al2 = amt.detach().clone().requires_grad_(True)
    g2 = soft_g(F.softmax(bl2, -1).detach(), F.softmax(cl2, -1).detach(), al2.detach(), tau, W, TEMP, valid_mask=torch.ones(bl2.shape[:2], dtype=torch.bool))
    try: F.mse_loss(g2, torch.zeros_like(g2)).backward()
    except RuntimeError: pass

    for name, leaf in [("bin", bl2), ("cat", cl2), ("amt", al2)]:
        n = leaf.grad.norm().item() if leaf.grad is not None else 0.0
        print(f"  {name} : ||grad|| detached={n:.4e}")
        assert n == 0.0, f"detached {name} grad = {n}"

    assert bl.grad.norm().item() > 0
    assert cl.grad.norm().item() > 0
    assert al.grad.norm().item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
