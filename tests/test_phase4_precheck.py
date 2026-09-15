"""
Phase 4 pre-checks before full GPU training.

TestGStdConsistency (synthetic data):
  Verifies that teacher pretraining and coherence loss operate in the same
  normalized g space via the fixed g_std buffer from compute_g_stats_from_data().

TestAMLSimTeacherAUPRC (real AMLSim data):
  Trains BehaviorTeacher on AMLSim sequences with normalized g targets.
  Evaluates AUPRC on val set. Reports vs prevalence (0.0012) and target (>0.01).
  Gate: AUPRC > max(0.01, 5 × prevalence).

  If AUPRC is weak (< 0.2), consider multi-window g extension:
    vel_7 & vel_30 concatenation for AMLSim structuring signal.
"""

import sys
import json
import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import (
    BehaviorTeacher,
    compute_g_from_real,
    compute_g_stats_from_data,
    pretrain_teacher,
)
from models.seq_denoiser import SeqDenoiser
from models.cof_seqgen import CoFSeqGen
from models.coherence_teacher import CoherenceTeacher
from tests.legacy_fixture_compat import to_right_padding


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under precision-recall curve (trapezoidal, no sklearn needed)."""
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (y_true.sum() + 1e-10)
    # Prepend (recall=0, precision=1) anchor
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def _make_sparkov_like(B=64, L=8, Bbins=16, d_num=1, seed=0):
    """Small synthetic dataset resembling Sparkov structure."""
    torch.manual_seed(seed)
    tau  = torch.linspace(0, 100, Bbins)
    x_num = torch.randn(B, L, d_num)
    dt_bin = torch.randint(0, Bbins, (B, L))
    x_cat  = torch.zeros(B, L, 0, dtype=torch.long)
    y      = (torch.rand(B, L) < 0.02).float()
    mask   = torch.ones(B, L, dtype=torch.bool)
    return x_num, dt_bin, x_cat, y, mask, tau


# ──────────────────────────────────────────────────────────────────────────────
# TestGStdConsistency
# ──────────────────────────────────────────────────────────────────────────────

class TestGStdConsistency(unittest.TestCase):
    """
    Verify that g_std flows consistently from compute_g_stats_from_data()
    into both teacher pretraining and CoFSeqGen's coherence loss.
    """

    def setUp(self):
        self.x_num, self.dt_bin, self.x_cat, self.y, self.mask, self.tau = \
            _make_sparkov_like(B=64, L=8, Bbins=16)
        self.W, self.temp = 60.0, 1.0
        self.Bbins = 16
        self.d_num = 1

        self.g_mean, self.g_std = compute_g_stats_from_data(
            self.x_num, self.dt_bin, self.x_cat,
            self.tau, self.W, self.temp, self.Bbins, [],
            valid_mask=self.mask,
        )

    # ── g_std properties ──────────────────────────────────────────────────────

    def test_g_std_shape(self):
        self.assertEqual(self.g_std.shape, (4,))

    def test_g_std_clamped(self):
        self.assertTrue((self.g_std >= 1.0).all(),
                        f"g_std not clamped: {self.g_std.tolist()}")

    def test_g_mean_shape(self):
        self.assertEqual(self.g_mean.shape, (4,))

    # ── Teacher trained in normalized space ───────────────────────────────────

    def test_teacher_targets_are_normalized(self):
        """g_real / g_std should have std ≈ 1 for components that g_std actually normalizes."""
        with torch.no_grad():
            g_real = compute_g_from_real(
                self.x_num, self.dt_bin, self.x_cat,
                self.tau, self.W, self.temp, self.Bbins, [],
                valid_mask=self.mask,
            )
        g_norm = g_real / self.g_std[None, None, :]  # (B, L, 4)
        flat = g_norm.reshape(-1, 4)
        std_per_comp = flat.std(dim=0)
        for c in range(4):
            raw_std = g_real.reshape(-1, 4)[:, c].std().item()
            # Only check components where g_std was NOT clamped (raw_std > 1.0)
            # If g_std[c] == 1.0 (clamped), normalization is a no-op → skip.
            if raw_std > 0.01 and self.g_std[c].item() > 1.01:
                self.assertAlmostEqual(std_per_comp[c].item(), 1.0, delta=0.05,
                    msg=f"Component {c}: std after /g_std = {std_per_comp[c]:.3f}, expected ≈1")

    def test_pretrain_teacher_loss_decreases(self):
        """Teacher loss should decrease when trained with normalized g targets."""
        teacher = BehaviorTeacher(self.d_num, self.Bbins, [], d_hidden=32)
        losses = pretrain_teacher(
            teacher, self.x_num, self.dt_bin, self.x_cat, self.y,
            self.tau, self.W, self.temp,
            g_std=self.g_std, valid_mask=self.mask, n_steps=100,
        )
        self.assertGreater(losses[0], losses[-1],
                           f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}")

    # ── CoFSeqGen uses same fixed g_std buffer ────────────────────────────────

    def test_cof_seqgen_g_std_buffer_registered(self):
        """CoFSeqGen.g_std buffer must equal the tensor passed at init."""
        denoiser = SeqDenoiser(self.d_num, self.Bbins, [], d_model=32, n_heads=2, n_layers=1)
        model = CoFSeqGen(
            denoiser=denoiser, tau=self.tau, W=self.W, temp=self.temp,
            coh_lambda=1.0, n_cat_classes=[], g_std=self.g_std,
        )
        self.assertIsNotNone(model.g_std)
        self.assertTrue(torch.allclose(model.g_std, self.g_std.clamp(min=1.0)),
                        f"Buffer mismatch: {model.g_std.tolist()} vs {self.g_std.tolist()}")

    def test_cof_seqgen_info_contract_and_buffer(self):
        """Loss info follows the intended contract; g_std remains a model buffer."""
        denoiser = SeqDenoiser(self.d_num, self.Bbins, [], d_model=32, n_heads=2, n_layers=1)
        model = CoFSeqGen(
            denoiser=denoiser, tau=self.tau, W=self.W, temp=self.temp,
            coh_lambda=1.0, n_cat_classes=[], g_std=self.g_std,
        )
        loss, info = model.compute_loss(
            self.x_num, self.dt_bin, self.x_cat, self.y, self.mask, t_frac=0.5
        )
        self.assertEqual(set(info), {"L_diff", "L_coh", "L_label"})
        self.assertTrue(torch.allclose(model.g_std, self.g_std.clamp(min=1.0)))

    def test_teacher_and_model_same_g_std(self):
        """
        Current seam: CoherenceTeacher consumes g_gen/g_std and the model owns
        the exact fixed train-set buffer.
        """
        teacher = CoherenceTeacher(hidden=16)
        teacher.freeze()

        denoiser = SeqDenoiser(self.d_num, self.Bbins, [], d_model=32, n_heads=2, n_layers=1)
        model = CoFSeqGen(
            denoiser=denoiser, tau=self.tau, W=self.W, temp=self.temp,
            coh_lambda=1.0, n_cat_classes=[],
            coherence_teacher=teacher, g_std=self.g_std,
        )
        self.assertTrue(torch.allclose(model.g_std, self.g_std.clamp(min=1.0)))
        loss, info = model.compute_loss(
            self.x_num, self.dt_bin, self.x_cat, self.y, self.mask, t_frac=0.5
        )
        self.assertTrue(torch.isfinite(loss), f"Loss is non-finite: {loss.item()}")
        self.assertGreaterEqual(info["L_coh"], 0.0)

    def test_no_g_std_falls_back_to_per_batch(self):
        """Legacy path (g_std=None) must still produce finite loss."""
        denoiser = SeqDenoiser(self.d_num, self.Bbins, [], d_model=32, n_heads=2, n_layers=1)
        model = CoFSeqGen(
            denoiser=denoiser, tau=self.tau, W=self.W, temp=self.temp,
            coh_lambda=1.0, n_cat_classes=[],
            # no g_std → per-batch fallback
        )
        self.assertIsNone(model.g_std)
        loss, info = model.compute_loss(
            self.x_num, self.dt_bin, self.x_cat, self.y, self.mask, t_frac=0.5
        )
        self.assertTrue(torch.isfinite(loss))


# ──────────────────────────────────────────────────────────────────────────────
# TestAMLSimTeacherAUPRC
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).parent.parent / 'data' / 'amlsim' / 'sequences'


@unittest.skipUnless(DATA_ROOT.exists(), "AMLSim data not found")
class TestAMLSimTeacherAUPRC(unittest.TestCase):
    """
    Train BehaviorTeacher on AMLSim data with normalized g targets.
    Evaluate AUPRC on val set. Gate: AUPRC > 5 × prevalence.
    """

    @classmethod
    def setUpClass(cls):
        with open(DATA_ROOT / 'meta.json') as f:
            meta = json.load(f)

        cls.tau       = torch.tensor(meta['tau_k'], dtype=torch.float32)
        cls.Bbins     = len(meta['tau_k'])
        cls.W         = 7.0
        cls.temp      = 1.0
        cls.d_num     = meta['d_num']
        cls.n_cat     = []

        tr = np.load(DATA_ROOT / 'train.npz')
        va = np.load(DATA_ROOT / 'val.npz')

        N_train = 4096   # subset for speed; covers ~13% of train set
        cls.x_num_tr  = torch.from_numpy(tr['x_num'][:N_train]).float()
        cls.dt_bin_tr = torch.from_numpy(tr['dt_bin'][:N_train]).long()
        cls.x_cat_tr  = torch.from_numpy(tr['x_cat'][:N_train]).long()
        cls.y_tr      = torch.from_numpy(tr['y'][:N_train]).float()
        cls.mask_tr   = (torch.from_numpy(tr['mask'][:N_train]).bool()
                         if 'mask' in tr else
                         torch.ones(N_train, meta['L'], dtype=torch.bool))

        cls.x_num_va  = torch.from_numpy(va['x_num']).float()
        cls.dt_bin_va = torch.from_numpy(va['dt_bin']).long()
        cls.x_cat_va  = torch.from_numpy(va['x_cat']).long()
        cls.y_va      = torch.from_numpy(va['y']).float()
        cls.mask_va   = (torch.from_numpy(va['mask']).bool()
                         if 'mask' in va else
                         torch.ones(va['y'].shape[0], meta['L'], dtype=torch.bool))

        (
            cls.x_num_tr, cls.dt_bin_tr, cls.x_cat_tr, cls.y_tr, cls.mask_tr
        ) = to_right_padding(
            cls.x_num_tr, cls.dt_bin_tr, cls.x_cat_tr, cls.y_tr, cls.mask_tr
        )
        (
            cls.x_num_va, cls.dt_bin_va, cls.x_cat_va, cls.y_va, cls.mask_va
        ) = to_right_padding(
            cls.x_num_va, cls.dt_bin_va, cls.x_cat_va, cls.y_va, cls.mask_va
        )

        cls.prevalence = float(cls.y_tr.mean())

    # ── g_std computation ─────────────────────────────────────────────────────

    def test_g_std_amlsim_shape_and_clamp(self):
        g_mean, g_std = compute_g_stats_from_data(
            self.x_num_tr, self.dt_bin_tr, self.x_cat_tr,
            self.tau, self.W, self.temp, self.Bbins, self.n_cat,
            batch_size=512,
            valid_mask=self.mask_tr,
        )
        self.assertEqual(g_std.shape, (4,))
        self.assertTrue((g_std >= 1.0).all(),
                        f"g_std not clamped: {g_std.tolist()}")
        print(f"\n[AMLSim g_std] vel={g_std[0]:.3f}  gap={g_std[1]:.3f}  "
              f"rep={g_std[2]:.3f}  amt={g_std[3]:.3f}")

    # ── AUPRC gate ────────────────────────────────────────────────────────────

    def test_teacher_auprc_on_amlsim_val(self):
        """
        Train teacher 300 steps with normalized g targets.
        Gate: AUPRC on val > 5 × prevalence.
        If AUPRC << 0.2, consider multi-window g (vel_7 + vel_30 + amt_30).
        """
        # 1. Fixed g_std from train subset
        g_mean, g_std = compute_g_stats_from_data(
            self.x_num_tr, self.dt_bin_tr, self.x_cat_tr,
            self.tau, self.W, self.temp, self.Bbins, self.n_cat,
            valid_mask=self.mask_tr,
        )

        # 2. Pre-compute normalized g targets for all train rows (no gradient)
        with torch.no_grad():
            g_real_tr = compute_g_from_real(
                self.x_num_tr, self.dt_bin_tr, self.x_cat_tr,
                self.tau, self.W, self.temp, self.Bbins, self.n_cat,
                valid_mask=self.mask_tr,
            )
            g_target_tr = g_real_tr / g_std[None, None, :]   # (N, L, 4) normalized

        # 3. Mini-batch training for 300 steps
        teacher = BehaviorTeacher(self.d_num, self.Bbins, self.n_cat, d_hidden=64)
        optim = torch.optim.Adam(teacher.parameters(), lr=3e-3)
        N = self.x_num_tr.shape[0]
        batch_size = 256
        teacher.train()
        step_losses = []
        for step in range(300):
            idx = torch.randint(0, N, (batch_size,))
            g_pred, y_logit = teacher(
                self.x_num_tr[idx],
                self.dt_bin_tr[idx],
                self.x_cat_tr[idx],
                valid_mask=self.mask_tr[idx],
            )
            loss = (
                F.binary_cross_entropy_with_logits(
                    y_logit.float(), self.y_tr[idx].float())
                + 0.1 * F.mse_loss(g_pred, g_target_tr[idx])
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            step_losses.append(loss.item())

        print(f"\n[AMLSim Teacher] loss: {step_losses[0]:.4f} → {step_losses[-1]:.4f}")
        self.assertGreater(step_losses[0], step_losses[-1],
                           "Teacher loss did not decrease")

        # 4. Evaluate AUPRC on val set (masked positions only)
        teacher.eval()
        y_probs_all, y_trues_all = [], []
        with torch.no_grad():
            for start in range(0, self.x_num_va.shape[0], 512):
                end = min(start + 512, self.x_num_va.shape[0])
                _, y_logit = teacher(
                    self.x_num_va[start:end],
                    self.dt_bin_va[start:end],
                    self.x_cat_va[start:end],
                    valid_mask=self.mask_va[start:end],
                )
                m = self.mask_va[start:end]
                y_probs_all.append(torch.sigmoid(y_logit[m]).cpu().numpy())
                y_trues_all.append(self.y_va[start:end][m].cpu().numpy())

        y_prob = np.concatenate(y_probs_all)
        y_true = np.concatenate(y_trues_all)

        auprc     = _auprc(y_true, y_prob)
        gate      = max(0.01, 5.0 * self.prevalence)
        ratio     = auprc / (self.prevalence + 1e-10)

        print(f"[AMLSim AUPRC]  AUPRC={auprc:.4f} | "
              f"prevalence={self.prevalence:.4f} | "
              f"ratio={ratio:.1f}×  | gate={gate:.4f}")
        if auprc < 0.2:
            print("  ⚠  AUPRC < 0.2 — consider multi-window g extension:")
            print("     vel_7 + vel_30 + amt_sum_30 (call soft_g twice, concat)")

        self.assertGreater(
            auprc, gate,
            f"Teacher AUPRC {auprc:.4f} ≤ gate {gate:.4f} ({5}× prevalence). "
            "Extend g with multi-window features if this fails."
        )

    # ── Consistency on AMLSim ─────────────────────────────────────────────────

    def test_cof_seqgen_with_g_std_finite_loss(self):
        """CoFSeqGen with AMLSim g_std buffer produces finite loss."""
        _, g_std = compute_g_stats_from_data(
            self.x_num_tr, self.dt_bin_tr, self.x_cat_tr,
            self.tau, self.W, self.temp, self.Bbins, self.n_cat,
            valid_mask=self.mask_tr,
        )
        denoiser = SeqDenoiser(
            self.d_num, self.Bbins, [], d_model=64, n_heads=2, n_layers=2, L_max=64,
        )
        model = CoFSeqGen(
            denoiser=denoiser, tau=self.tau, W=self.W, temp=self.temp,
            coh_lambda=1.0, n_cat_classes=[], g_std=g_std,
        )
        B = 16
        loss, info = model.compute_loss(
            self.x_num_tr[:B], self.dt_bin_tr[:B], self.x_cat_tr[:B],
            self.y_tr[:B], self.mask_tr[:B], t_frac=0.5,
        )
        self.assertTrue(torch.isfinite(loss), f"Loss non-finite: {loss.item()}")
        self.assertEqual(set(info), {"L_diff", "L_coh", "L_label"})
        self.assertTrue(torch.allclose(model.g_std, g_std))
        print(f"\n[AMLSim CoFSeqGen] L_diff={info['L_diff']:.3f}  "
              f"L_coh={info['L_coh']:.3f}  g_std={[f'{v:.2f}' for v in model.g_std]}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
