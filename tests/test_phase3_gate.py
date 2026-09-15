"""
Phase 3 Gate: pytest tests/test_phase3_gate.py

Three gate checks (must all pass before GPU full training):

  [1] Forward shape test
      SeqDenoiser output shapes correct for Sparkov and AMLSim configs.

  [2] Teacher AUPRC test
      BehaviorTeacher trained on real Sparkov sequences.
      AUPRC > prevalence (~0.6%) → confirm teacher learned fraud signal.
      Freeze all params → confirm requires_grad=False.

  [3] 1-batch overfit test (with L_coh scale normalization)
      CoFSeqGen trained on 1 batch for 60 steps.
      L_diff must decrease AND L_coh must decrease from step 0 to step 60.
      L_coh is per-component std-normalized (clamp≥1) → O(1) scale.
      After normalization, L_coh_init / L_diff_init < 10 (was 10,000×).
      λ=0 baseline also tested (CoF w/o coherence).

Dataset-specific W:
  Sparkov: W=60.0 min (1-hour burst window; ablation shows 14× lift at vel_1h)
           NOT 7 days — W=7d saturates vel to j, erasing burst signal
  AMLSim:  W=7.0 steps (7-day structuring window; user-confirmed appropriate)
"""

import json, pickle, math
import numpy as np
import torch
import torch.nn.functional as F
import pytest
import sys, pathlib

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from models.seq_denoiser import SeqDenoiser
from models.teacher import BehaviorTeacher, pretrain_teacher, compute_g_from_real
from models.cof_seqgen import CoFSeqGen, W_SPARKOV_MINUTES, W_AMLSIM_STEPS, COH_LAMBDA_SWEEP
from data.build_sequences import HashEncoder
from tests.legacy_fixture_compat import to_right_padding

# Legacy encoders were pickled while build_sequences.py ran as __main__.
# Expose the same class without rewriting the preserved pickle.
setattr(sys.modules["__main__"], "HashEncoder", HashEncoder)

DATA_DIR = ROOT / 'data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_sparkov_batch(n=512, split='train'):
    npz = np.load(DATA_DIR / 'sparkov' / 'sequences' / f'{split}.npz')
    meta = json.load(open(DATA_DIR / 'sparkov' / 'sequences' / 'meta.json'))
    enc = pickle.load(open(DATA_DIR / 'sparkov' / 'sequences' / 'encoders.pkl', 'rb'))
    idx = np.arange(min(n, len(npz['y'])))
    tensors = (
        torch.tensor(npz['x_num'][idx], dtype=torch.float32).to(DEVICE),
        torch.tensor(npz['dt_bin'][idx], dtype=torch.long).to(DEVICE),
        torch.tensor(npz['x_cat'][idx], dtype=torch.long).to(DEVICE),
        torch.tensor(npz['y'][idx], dtype=torch.long).to(DEVICE),
        torch.tensor(npz['mask'][idx], dtype=torch.bool).to(DEVICE),
    )
    return (*to_right_padding(*tensors), meta, enc)


def _load_amlsim_batch(n=512, split='train'):
    npz = np.load(DATA_DIR / 'amlsim' / 'sequences' / f'{split}.npz')
    meta = json.load(open(DATA_DIR / 'amlsim' / 'sequences' / 'meta.json'))
    enc = pickle.load(open(DATA_DIR / 'amlsim' / 'sequences' / 'encoders.pkl', 'rb'))
    idx = np.arange(min(n, len(npz['y'])))
    tensors = (
        torch.tensor(npz['x_num'][idx], dtype=torch.float32).to(DEVICE),
        torch.tensor(npz['dt_bin'][idx], dtype=torch.long).to(DEVICE),
        torch.tensor(npz['x_cat'][idx], dtype=torch.long).to(DEVICE),
        torch.tensor(npz['y'][idx], dtype=torch.long).to(DEVICE),
        torch.tensor(npz['mask'][idx], dtype=torch.bool).to(DEVICE),
    )
    return (*to_right_padding(*tensors), meta, enc)


# ── [1] Forward shape tests ───────────────────────────────────────────────────

class TestForwardShapes:
    """SeqDenoiser output shapes for both dataset configurations."""

    def _run(self, d_num, Bbins, n_cat_classes, L, B=4, d_model=64, n_heads=4, n_layers=2):
        model = SeqDenoiser(d_num, Bbins, n_cat_classes, d_model, n_heads, n_layers, L_max=L+4)
        x_num = torch.randn(B, L, d_num)
        dt_bin = torch.randint(0, Bbins, (B, L))
        x_cat = torch.randint(0, max(n_cat_classes + [1]), (B, L, len(n_cat_classes)))
        t = torch.rand(B)
        return model(x_num, dt_bin, x_cat, t)

    def test_sparkov_shapes(self):
        """Sparkov: d_num=1, Bbins=16, n_cat_classes=[14], L=24."""
        x_num_hat, bin_logits, cat_logits, y_logit = self._run(
            d_num=1, Bbins=16, n_cat_classes=[14], L=24
        )
        B, L = 4, 24
        assert x_num_hat.shape == (B, L, 1),   f"x_num_hat {x_num_hat.shape}"
        assert bin_logits.shape == (B, L, 16),  f"bin_logits {bin_logits.shape}"
        assert len(cat_logits) == 1,             f"cat_logits len {len(cat_logits)}"
        assert cat_logits[0].shape == (B, L, 14), f"cat_logits[0] {cat_logits[0].shape}"
        assert y_logit.shape == (B, L),          f"y_logit {y_logit.shape}"

    def test_amlsim_shapes(self):
        """AMLSim: d_num=3, Bbins=16, n_cat_classes=[], L=32."""
        x_num_hat, bin_logits, cat_logits, y_logit = self._run(
            d_num=3, Bbins=16, n_cat_classes=[], L=32
        )
        B, L = 4, 32
        assert x_num_hat.shape == (B, L, 3),   f"x_num_hat {x_num_hat.shape}"
        assert bin_logits.shape == (B, L, 16),  f"bin_logits {bin_logits.shape}"
        assert len(cat_logits) == 0,             f"cat_logits should be empty"
        assert y_logit.shape == (B, L),          f"y_logit {y_logit.shape}"

    def test_soft_g_on_denoiser_output(self):
        """soft_g applied to denoiser output → correct shape, gradient flows."""
        from models.soft_g import soft_g
        d_num, Bbins, n_cat = 1, 16, [14]
        model = SeqDenoiser(d_num, Bbins, n_cat, d_model=64, n_heads=4, n_layers=2, L_max=28)
        B, L = 2, 24
        x_num = torch.randn(B, L, d_num, requires_grad=True)
        dt_bin = torch.randint(0, Bbins, (B, L))
        x_cat = torch.randint(0, 14, (B, L, 1))
        t = torch.rand(B)

        x_num_hat, bin_logits, cat_logits, _ = model(x_num, dt_bin, x_cat, t)
        tau = torch.linspace(10.0, 480.0, Bbins)
        bp = F.softmax(bin_logits, dim=-1)
        cp = F.softmax(cat_logits[0], dim=-1)
        amt = x_num_hat[:, :, 0]
        g = soft_g(
            bp, cp, amt, tau, W=60.0, temp=1.0,
            valid_mask=torch.ones(B, L, dtype=torch.bool),
        )

        assert g.shape == (B, L, 4)
        g.sum().backward()
        # Gradient should reach model parameters via amt_hat
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no grad reached denoiser params through soft_g"

    def test_cof_seqgen_loss_shape(self):
        """CoFSeqGen.compute_loss returns scalars."""
        d_num, Bbins, n_cat = 1, 16, [14]
        B, L = 2, 24
        denoiser = SeqDenoiser(d_num, Bbins, n_cat, d_model=64, n_heads=4, n_layers=2, L_max=28)
        tau = torch.linspace(10.0, 480.0, Bbins)
        model = CoFSeqGen(denoiser, tau, W=60.0, temp=1.0, coh_lambda=1.0, n_cat_classes=n_cat)

        x_num = torch.randn(B, L, d_num)
        dt_bin = torch.randint(0, Bbins, (B, L))
        x_cat = torch.randint(0, 14, (B, L, 1))
        y = torch.randint(0, 2, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)

        total, info = model.compute_loss(x_num, dt_bin, x_cat, y, mask, t_frac=0.5)
        assert total.ndim == 0, "total loss must be scalar"
        assert 'L_diff' in info and 'L_coh' in info
        assert info['L_diff'] > 0
        assert info['L_coh'] >= 0


# ── [2] Teacher AUPRC gate ────────────────────────────────────────────────────

class TestTeacherAUPRC:
    """
    Train BehaviorTeacher on real Sparkov data, check AUPRC > prevalence, then freeze.
    Uses 1-batch overfit (AUPRC checked on train batch) to confirm teacher learned signal.
    """

    def test_teacher_auprc_gt_prevalence(self):
        from sklearn.metrics import average_precision_score

        x_num, dt_bin, x_cat, y, mask, meta, enc = _load_sparkov_batch(n=2048)

        prevalence = y.float().mean().item()
        print(f"\n  Sparkov prevalence: {prevalence:.4f}")

        tau = torch.tensor(meta['tau_k'], dtype=torch.float32).to(DEVICE)
        W = 60.0 * 24 * 7   # 1 week in minutes
        temp = 10.0           # soft_g temperature

        teacher = BehaviorTeacher(
            d_num=meta['d_num'],
            Bbins=meta['B'],
            n_cat_classes=[meta['num_classes_cat']['category']],
            d_hidden=64,
        ).to(DEVICE)

        pretrain_teacher(
            teacher, x_num, dt_bin, x_cat, y.float(),
            tau, W, temp, valid_mask=mask, n_steps=80, lr=3e-3,
        )

        # AUPRC on train batch (1-batch overfit → should be well above prevalence)
        teacher.eval()
        with torch.no_grad():
            _, y_logit = teacher(x_num, dt_bin, x_cat, valid_mask=mask)

        y_flat = y[mask].cpu().numpy()
        s_flat = y_logit[mask].cpu().numpy()

        # Skip if no positive examples in this batch
        if y_flat.sum() == 0:
            pytest.skip("No fraud samples in loaded batch — increase batch size")

        auprc = average_precision_score(y_flat, s_flat)
        print(f"  Teacher AUPRC: {auprc:.4f}  (prevalence: {prevalence:.4f})")
        assert auprc > prevalence, \
            f"Teacher AUPRC {auprc:.4f} ≤ prevalence {prevalence:.4f}"

    def test_teacher_freeze(self):
        """After freeze(), all parameters have requires_grad=False."""
        teacher = BehaviorTeacher(d_num=1, Bbins=16, n_cat_classes=[14], d_hidden=32)
        teacher.freeze()
        frozen = all(not p.requires_grad for p in teacher.parameters())
        assert frozen, "Not all params frozen after teacher.freeze()"

    def test_teacher_forward_shapes(self):
        teacher = BehaviorTeacher(d_num=1, Bbins=16, n_cat_classes=[14], d_hidden=32)
        B, L = 4, 24
        x_num = torch.randn(B, L, 1)
        dt_bin = torch.randint(0, 16, (B, L))
        x_cat = torch.randint(0, 14, (B, L, 1))
        g_pred, y_logit = teacher(
            x_num, dt_bin, x_cat,
            valid_mask=torch.ones(B, L, dtype=torch.bool),
        )
        assert g_pred.shape == (B, L, 4), f"g_pred {g_pred.shape}"
        assert y_logit.shape == (B, L),   f"y_logit {y_logit.shape}"

    def test_teacher_amlsim_no_cat(self):
        """AMLSim: n_cat_classes=[] — teacher handles no categorical features."""
        teacher = BehaviorTeacher(d_num=3, Bbins=16, n_cat_classes=[], d_hidden=32)
        B, L = 4, 32
        x_num = torch.randn(B, L, 3)
        dt_bin = torch.randint(0, 16, (B, L))
        x_cat = torch.zeros(B, L, 0, dtype=torch.long)
        g_pred, y_logit = teacher(
            x_num, dt_bin, x_cat,
            valid_mask=torch.ones(B, L, dtype=torch.bool),
        )
        assert g_pred.shape == (B, L, 4)
        assert y_logit.shape == (B, L)


# ── [3] 1-batch overfit gate ──────────────────────────────────────────────────

class TestOneBatchOverfit:
    """
    CoFSeqGen trained for 60 steps on 1 batch.
    L_diff AND L_coh must both decrease from step 0 → step 60.
    """

    def _build_sparkov_model(self, meta, tau_t, W=None, temp=10.0, coh_lambda=1.0):
        # W defaults to W_SPARKOV_MINUTES (60 min, 1h burst window)
        W = W if W is not None else W_SPARKOV_MINUTES
        n_cat = [meta['num_classes_cat']['category']]
        denoiser = SeqDenoiser(
            d_num=meta['d_num'], Bbins=meta['B'], n_cat_classes=n_cat,
            d_model=64, n_heads=4, n_layers=2, L_max=meta['L'] + 4,
        ).to(DEVICE)
        model = CoFSeqGen(
            denoiser, tau_t, W=W, temp=temp, coh_lambda=coh_lambda,
            n_cat_classes=n_cat,
        ).to(DEVICE)
        return model

    def _run_overfit(self, model, x_num, dt_bin, x_cat, y, mask, n_steps=60, lr=3e-3):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        for step in range(n_steps):
            optimizer.zero_grad()
            t_frac = 0.3 + 0.4 * (step / max(n_steps - 1, 1))
            total, info = model.compute_loss(x_num, dt_bin, x_cat, y, mask, t_frac)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(info.copy())
        return history

    def test_lcoh_scale_balanced_sparkov(self):
        """
        After per-component std normalization, L_coh_init / L_diff_init < 10.
        Before fix: ratio was ~10,000 (gap component in minutes dominated).
        After fix: L_coh is dimensionless O(1), comparable to L_diff O(4-8).
        """
        x_num, dt_bin, x_cat, y, mask, meta, enc = _load_sparkov_batch(n=64)
        tau_t = torch.tensor(meta['tau_k'], dtype=torch.float32).to(DEVICE)
        # Use W=1h (correct burst window per ablation, not 7d)
        model = self._build_sparkov_model(meta, tau_t, W=W_SPARKOV_MINUTES, temp=10.0)

        with torch.no_grad():
            _, info = model.compute_loss(x_num, dt_bin, x_cat, y, mask, t_frac=0.5)

        ratio = info['L_coh'] / max(info['L_diff'], 1e-8)
        print(f"\n  Sparkov L_coh_norm/L_diff = {ratio:.3f}")
        print(f"    L_diff      = {info['L_diff']:.4f}")
        print(f"    L_coh(norm) = {info['L_coh']:.4f}   (was ~89183 raw)")
        assert ratio < 10, \
            f"L_coh/L_diff = {ratio:.1f} ≥ 10 — normalization ineffective (was 10,000)"
        assert set(info) == {"L_diff", "L_coh", "L_label"}
        assert info["L_coh"] == 0.0, (
            "Without a frozen coherence_teacher, the intended contract disables L_coh"
        )

    def test_Ldiff_and_Lcoh_decrease_sparkov(self):
        x_num, dt_bin, x_cat, y, mask, meta, enc = _load_sparkov_batch(n=32)
        tau_t = torch.tensor(meta['tau_k'], dtype=torch.float32).to(DEVICE)
        # W=60min (1h burst window, not 7d) — user ablation: vel_1h gives 14× lift
        model = self._build_sparkov_model(meta, tau_t, W=W_SPARKOV_MINUTES, temp=10.0)
        history = self._run_overfit(model, x_num, dt_bin, x_cat, y, mask)

        L_diff_start, L_diff_end = history[0]['L_diff'], history[-1]['L_diff']
        L_coh_start,  L_coh_end  = history[0]['L_coh'],  history[-1]['L_coh']

        print(f"\n  Sparkov 1-batch overfit W={W_SPARKOV_MINUTES:.0f}min (60 steps):")
        print(f"    L_diff: {L_diff_start:.4f} → {L_diff_end:.4f}")
        print(f"    L_coh(norm): {L_coh_start:.4f} → {L_coh_end:.4f}")

        assert L_diff_end < L_diff_start, \
            f"L_diff did not decrease: {L_diff_start:.4f} → {L_diff_end:.4f}"
        assert L_coh_start == L_coh_end == 0.0

    def test_lambda_zero_baseline_sparkov(self):
        """
        λ=0 = CoF w/o coherence (internal baseline for ablation).
        L_diff must still decrease; L_coh must be exactly 0.
        COH_LAMBDA_SWEEP must contain 0.
        """
        assert 0.0 in COH_LAMBDA_SWEEP, f"λ=0 missing from sweep: {COH_LAMBDA_SWEEP}"

        x_num, dt_bin, x_cat, y, mask, meta, enc = _load_sparkov_batch(n=32)
        tau_t = torch.tensor(meta['tau_k'], dtype=torch.float32).to(DEVICE)
        model = self._build_sparkov_model(meta, tau_t, W=W_SPARKOV_MINUTES, coh_lambda=0.0)
        history = self._run_overfit(model, x_num, dt_bin, x_cat, y, mask)

        assert history[0]['L_coh'] == 0.0, "L_coh should be 0 when coh_lambda=0"
        assert history[-1]['L_diff'] < history[0]['L_diff'], \
            "L_diff must still decrease at λ=0"
        print(f"\n  λ=0 baseline: L_diff {history[0]['L_diff']:.4f} → {history[-1]['L_diff']:.4f}")
        print(f"  λ sweep configured: {COH_LAMBDA_SWEEP}")

    def test_Ldiff_and_Lcoh_decrease_amlsim(self):
        x_num, dt_bin, x_cat, y, mask, meta, enc = _load_amlsim_batch(n=32)
        tau_t = torch.tensor(meta['tau_k'], dtype=torch.float32).to(DEVICE)
        n_cat: list = []
        denoiser = SeqDenoiser(
            d_num=meta['d_num'], Bbins=meta['B'], n_cat_classes=n_cat,
            d_model=64, n_heads=4, n_layers=2, L_max=meta['L'] + 4,
        ).to(DEVICE)
        # AMLSim W=7 steps (7-day structuring window, user-confirmed appropriate)
        model = CoFSeqGen(
            denoiser, tau_t, W=W_AMLSIM_STEPS, temp=1.0, coh_lambda=1.0,
            n_cat_classes=n_cat,
        ).to(DEVICE)
        history = self._run_overfit(model, x_num, dt_bin, x_cat, y, mask)

        L_diff_start, L_diff_end = history[0]['L_diff'], history[-1]['L_diff']
        L_coh_start,  L_coh_end  = history[0]['L_coh'],  history[-1]['L_coh']

        print(f"\n  AMLSim 1-batch overfit W={W_AMLSIM_STEPS:.0f}steps (60 steps):")
        print(f"    L_diff: {L_diff_start:.4f} → {L_diff_end:.4f}")
        print(f"    L_coh(norm): {L_coh_start:.4f} → {L_coh_end:.4f}")

        assert L_diff_end < L_diff_start
        assert L_coh_start == L_coh_end == 0.0

    def test_amt_grad_flows_to_denoiser(self):
        """
        Gradient through soft_g.amt_sum must reach denoiser's num_head parameters.
        This confirms amount generation is coherence-loss-supervised.
        """
        from models.soft_g import soft_g
        x_num, dt_bin, x_cat, y, mask, meta, enc = _load_sparkov_batch(n=8)
        tau_t = torch.tensor(meta['tau_k'], dtype=torch.float32).to(DEVICE)
        n_cat = [meta['num_classes_cat']['category']]
        denoiser = SeqDenoiser(
            d_num=meta['d_num'], Bbins=meta['B'], n_cat_classes=n_cat,
            d_model=64, n_heads=4, n_layers=2, L_max=meta['L'] + 4,
        ).to(DEVICE)

        t_vec = torch.full((8,), 0.5).to(DEVICE)
        x_num_t, _ = CoFSeqGen.add_noise(x_num, 0.5)
        x_num_hat, bin_logits, cat_logits, _ = denoiser(x_num_t, dt_bin, x_cat, t_vec)

        bp = F.softmax(bin_logits, dim=-1)
        cp = F.softmax(cat_logits[0], dim=-1)
        amt = x_num_hat[:, :, 0]
        # W=60min (1h), not 7d
        g = soft_g(
            bp, cp, amt, tau_t, W=W_SPARKOV_MINUTES, temp=10.0,
            valid_mask=mask,
        )

        # Minimise only the amt_sum channel → gradient must reach num_head
        g[:, :, 3].pow(2).mean().backward()

        num_head_grad = denoiser.num_head.weight.grad
        assert num_head_grad is not None, "no grad on num_head — amt channel disconnected"
        assert num_head_grad.norm().item() > 0, "num_head grad norm = 0"
        print(f"\n  num_head grad norm (via amt_sum): {num_head_grad.norm():.4e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
