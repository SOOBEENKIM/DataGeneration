import copy
import numpy as np
import torch
import torch.nn.functional as F
from tests.test_argn_state_first import fixture
from benchmarks.argn_state_adapter import FullHistoryCollator, model_class
from benchmarks.argn_past_state import STATE_COLUMN
from benchmarks.argn_event_weight_control import event_weighted_adapter
from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX
from mostlyai.engine._tabular.training import _calculate_sample_losses


def test_event_weighting_changes_only_transaction_loss_and_masks_padding(fixture):
    stats, codec, records, kwargs = fixture
    records = copy.deepcopy(records)
    for key in list(records[0]):
        if key.startswith("tgt:") or key == STATE_COLUMN:
            records[0][key] = records[0][key][:3]
    model = model_class(stats, False)(**kwargs).eval()
    batch = FullHistoryCollator(True, None, torch.device("cpu"))(records)
    native = _calculate_sample_losses(model, batch)
    output, _ = model(batch, mode="trn")
    length_loss = sum(F.cross_entropy(output[k][:, 0], batch[k][:, 0, 0], reduction="none")
                      for k in output if k.startswith(SLEN_SUB_COLUMN_PREFIX))
    with event_weighted_adapter(stats, mean_training_length=5.5) as train:
        # Native isinstance tests reference the configured subclass; load its
        # equivalent state dict so the same logits are compared under both losses.
        weighted_model = train.__globals__["SequentialModel"](**kwargs).eval()
        weighted_model.load_state_dict(model.state_dict())
        weighted = train.__globals__["_calculate_sample_losses"](weighted_model, batch)
        expected = length_loss + (native - length_loss) * torch.tensor([3., 8.]) / 5.5
        torch.testing.assert_close(weighted, expected, rtol=1e-5, atol=1e-5)
        weighted.mean().backward()
        assert weighted_model.state_projection.weight.grad.abs().sum() == 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in weighted_model.predictors.parameters())
