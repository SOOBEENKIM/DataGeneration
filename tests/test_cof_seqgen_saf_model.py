from __future__ import annotations

import numpy as np
import pytest
import torch

from models.cof_seqgen_saf import (
    MISSING_CODE,
    SAF_CANDIDATES,
    CoFSeqGenSAF,
    OrderedHazardGapDecoder,
    SAFModelConfig,
    fit_train_only_gap_support,
)


def _batch():
    gap = torch.tensor([[float("nan"), 0.0, 2.0, 5.0], [float("nan"), 1.0, 5.0, 0.0]])
    receiver = torch.tensor([[3, 4, 4, 5], [4, 3, 5, 5]])
    value = torch.tensor([[1.0, 2.0, 1.5, 3.0], [0.5, 1.0, 2.0, 4.0]])
    valid = torch.ones_like(gap, dtype=torch.bool)
    return gap, receiver, value, valid


def test_support_alignment_keeps_zero_and_observed_atoms() -> None:
    values = np.asarray([0, 0, 1, 2, 3, 5, 8, 13, 21, 34], dtype=float)
    state = fit_train_only_gap_support(values, max_positive_states=4)
    assert state.representatives[0] == 0.0
    assert set(state.representatives).issubset(set(values))
    encoded = state.encode_numpy(np.asarray([np.nan, 0.0, 100.0]))
    assert encoded[0] == MISSING_CODE
    assert encoded[1] == 3
    assert encoded[2] == state.n_states + 2


def test_support_alignment_does_not_hallucinate_unobserved_zero() -> None:
    state = fit_train_only_gap_support([1, 2, 3, 5])
    assert state.zero_is_explicit is False
    assert 0.0 not in state.representatives
    assert set(state.representatives) == {1.0, 2.0, 3.0, 5.0}


def test_five_candidates_are_finite_clean_contrasts() -> None:
    assert set(SAF_CANDIDATES) == {"SAF-C0", "SAF-U0", "SAF-O0", "SAF-U1", "SAF-O1"}
    assert SAF_CANDIDATES["SAF-U0"].route_gap_to_mark is False
    assert SAF_CANDIDATES["SAF-U1"].route_gap_to_mark is True
    assert SAF_CANDIDATES["SAF-O0"].ordered is True
    assert SAF_CANDIDATES["SAF-O1"].ordered is True


def test_all_candidates_use_parameter_matched_copy_new_mark_decoder() -> None:
    support = fit_train_only_gap_support([0, 1, 2, 5])
    assert {
        candidate.mark_decoder for candidate in SAF_CANDIDATES.values()
    } == {"copy_new"}
    models = {
        candidate_id: CoFSeqGenSAF(
            SAFModelConfig(
                candidate_id=candidate_id,
                receiver_vocab_size=8,
                hidden_dim=16,
            ),
            support,
        )
        for candidate_id in SAF_CANDIDATES
    }
    assert all(
        model.copy_base_logit_head is not None
        and model.copy_gap_gate_head is not None
        and model.copy_gap_delta_head is not None
        for model in models.values()
    )
    for control_id, routed_id in (("SAF-U0", "SAF-U1"), ("SAF-O0", "SAF-O1")):
        control = models[control_id]
        routed = models[routed_id]
        assert tuple(control.state_dict()) == tuple(routed.state_dict())
        assert [value.shape for value in control.state_dict().values()] == [
            value.shape for value in routed.state_dict().values()
        ]
        assert sum(parameter.numel() for parameter in control.parameters()) == sum(
            parameter.numel() for parameter in routed.parameters()
        )


def test_ordered_hazard_is_normalized_and_differentiable() -> None:
    decoder = OrderedHazardGapDecoder(8, 5)
    hidden = torch.randn(3, 4, 8, requires_grad=True)
    probabilities = decoder.probabilities(hidden)
    assert torch.allclose(probabilities.sum(-1), torch.ones(3, 4), atol=1e-6)
    loss = decoder.nll(hidden, torch.randint(0, 5, (3, 4))).mean()
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("candidate_id", tuple(SAF_CANDIDATES))
def test_all_ablation_losses_are_finite_and_backward(candidate_id: str) -> None:
    gap, receiver, value, valid = _batch()
    support = fit_train_only_gap_support([0, 1, 2, 5])
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id=candidate_id, receiver_vocab_size=8, hidden_dim=16),
        support,
    )
    losses = model.compute_loss(gap=gap, receiver=receiver, numeric_value=value, valid_mask=valid)
    assert all(torch.isfinite(item) for item in losses.values())
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_history_is_strictly_past_and_first_event_is_retained() -> None:
    torch.manual_seed(2)
    gap, receiver, value, valid = _batch()
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16),
        fit_train_only_gap_support([0, 1, 2, 5]),
    ).eval()
    original = model.encoder(gap, receiver, value, valid)
    changed = receiver.clone()
    changed[:, 2:] = 7
    changed_value = value.clone()
    changed_value[:, 2:] = 999
    modified = model.encoder(gap, changed, changed_value, valid)
    assert torch.allclose(original[:, :3], modified[:, :3])
    first_changed = receiver.clone()
    first_changed[:, 0] = 7
    first_modified = model.encoder(gap, first_changed, value, valid)
    assert not torch.allclose(original[:, 1], first_modified[:, 1])


def test_static_context_conditions_later_history_states_directly() -> None:
    gap, receiver, value, valid = _batch()
    model = CoFSeqGenSAF(
        SAFModelConfig(
            candidate_id="SAF-O1",
            receiver_vocab_size=8,
            static_dim=1,
            hidden_dim=16,
        ),
        fit_train_only_gap_support([0, 1, 2, 5]),
    ).eval()
    low = model.encoder(
        gap,
        receiver,
        value,
        valid,
        static=torch.zeros(2, 1),
    )
    high = model.encoder(
        gap,
        receiver,
        value,
        valid,
        static=torch.ones(2, 1),
    )
    assert not torch.allclose(low[:, 0], high[:, 0])
    assert not torch.allclose(low[:, -1], high[:, -1])


def test_explicit_gap_route_changes_copy_logits_only_for_routed_candidate() -> None:
    support = fit_train_only_gap_support([0, 1, 2, 5])
    hidden = torch.randn(2, 16)
    low = torch.tensor([0.0, 0.0])
    high = torch.tensor([5.0, 5.0])
    routed = CoFSeqGenSAF(SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16), support)
    unrouted = CoFSeqGenSAF(SAFModelConfig(candidate_id="SAF-O0", receiver_vocab_size=8, hidden_dim=16), support)
    with torch.no_grad():
        routed.copy_gap_delta_head.weight.fill_(1.0)
    assert not torch.allclose(
        routed._copy_logits(hidden, low),
        routed._copy_logits(hidden, high),
    )
    assert unrouted.copy_base_logit_head is not None
    assert torch.allclose(
        unrouted._copy_logits(hidden, low),
        unrouted._copy_logits(hidden, high),
    )


def test_gated_gap_effect_can_depend_on_history_with_matched_control() -> None:
    support = fit_train_only_gap_support([0, 1, 2, 5])
    routed = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-U1", receiver_vocab_size=8, hidden_dim=16),
        support,
    )
    control = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-U0", receiver_vocab_size=8, hidden_dim=16),
        support,
    )
    low = torch.tensor([0.0, 0.0])
    high = torch.tensor([5.0, 5.0])
    hidden = torch.zeros(2, 16)
    hidden[:, 0] = torch.tensor([-1.0, 1.0])

    with torch.no_grad():
        routed.copy_base_logit_head.weight.zero_()
        routed.copy_base_logit_head.bias.zero_()
        routed.copy_gap_gate_head.weight.zero_()
        routed.copy_gap_gate_head.weight[0, 0] = 8.0
        routed.copy_gap_gate_head.bias.zero_()
        routed.copy_gap_delta_head.weight.zero_()
        routed.copy_gap_delta_head.weight[0, 0] = 4.0
        routed.gap_route.weight.zero_()
        high_code = int(routed._support_code(high[:1]).item())
        routed.gap_route.weight[high_code, 0] = 1.0
        control.load_state_dict(routed.state_dict())

    routed_effect = routed._copy_logits(hidden, high) - routed._copy_logits(
        hidden, low
    )
    control_effect = control._copy_logits(hidden, high) - control._copy_logits(
        hidden, low
    )
    assert routed_effect[1] > 100.0 * routed_effect[0]
    assert torch.allclose(control_effect, torch.zeros_like(control_effect))


def test_matched_copy_head_gets_gap_gradient_only_when_route_is_enabled() -> None:
    gap, receiver, value, valid = _batch()
    support = fit_train_only_gap_support([0, 1, 2, 5])
    route_gradient = {}
    for candidate_id in ("SAF-U0", "SAF-U1"):
        model = CoFSeqGenSAF(
            SAFModelConfig(
                candidate_id=candidate_id,
                receiver_vocab_size=8,
                hidden_dim=16,
            ),
            support,
        )
        with torch.no_grad():
            model.copy_gap_delta_head.weight.fill_(0.1)
        model.compute_loss(
            gap=gap,
            receiver=receiver,
            numeric_value=value,
            valid_mask=valid,
        )["receiver_nll"].backward()
        assert model.copy_base_logit_head is not None
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in model.copy_base_logit_head.parameters()
        )
        route_gradient[candidate_id] = model.gap_route.weight.grad
    assert route_gradient["SAF-U0"] is None
    assert route_gradient["SAF-U1"] is not None
    assert float(route_gradient["SAF-U1"].abs().sum()) > 0.0


def test_copy_route_effect_targets_whichever_mark_is_previous() -> None:
    support = fit_train_only_gap_support([0, 1, 2, 5])
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16),
        support,
    )
    history = torch.zeros(2, 16)
    low = torch.zeros(2)
    high = torch.full((2,), 5.0)
    previous = torch.tensor([3, 4])
    has_previous = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        model.copy_gap_delta_head.weight.fill_(1.0)
    gap_effect = model._receiver_log_probabilities(
        history, high, previous, has_previous
    ) - model._receiver_log_probabilities(
        history, low, previous, has_previous
    )
    assert not torch.allclose(gap_effect[0], gap_effect[1])


def test_copy_new_distribution_is_normalized_and_can_force_copy() -> None:
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16),
        fit_train_only_gap_support([0, 1, 2, 5]),
    )
    assert model.copy_base_logit_head is not None
    with torch.no_grad():
        model.copy_base_logit_head.weight.zero_()
        model.copy_base_logit_head.bias.fill_(30.0)
        model.new_mark_head.weight.zero_()
        model.new_mark_head.bias.zero_()
    log_probabilities = model._receiver_log_probabilities(
        torch.zeros(2, 16),
        torch.tensor([0.0, 5.0]),
        torch.tensor([3, 4]),
        torch.ones(2, dtype=torch.bool),
    )
    probabilities = log_probabilities.exp()
    assert torch.allclose(probabilities.sum(-1), torch.ones(2), atol=1e-6)
    assert probabilities[0, 3] > 0.999
    assert probabilities[1, 4] > 0.999


def test_interactive_gap_route_receives_nonzero_mark_loss_gradient() -> None:
    gap, receiver, value, valid = _batch()
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16),
        fit_train_only_gap_support([0, 1, 2, 5]),
    )
    with torch.no_grad():
        model.copy_gap_delta_head.weight.fill_(0.1)
    receiver_loss = model.compute_loss(
        gap=gap,
        receiver=receiver,
        numeric_value=value,
        valid_mask=valid,
    )["receiver_nll"]
    receiver_loss.backward()
    assert model.copy_base_logit_head is not None
    gradients = [
        parameter.grad
        for module in (
            model.copy_base_logit_head,
            model.copy_gap_gate_head,
            model.copy_gap_delta_head,
        )
        for parameter in module.parameters()
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_sampling_uses_copy_new_decoder_after_first_event() -> None:
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16),
        fit_train_only_gap_support([0, 1, 2, 5]),
    )
    assert model.copy_base_logit_head is not None
    with torch.no_grad():
        model.copy_base_logit_head.weight.zero_()
        model.copy_base_logit_head.bias.fill_(80.0)
    sample = model.sample_fixed_lengths([5])
    marks = sample["receiver"][0]
    assert torch.equal(marks[1:], marks[:-1])


def test_sampling_obeys_support_missing_first_gap_and_length_plan() -> None:
    torch.manual_seed(4)
    support = fit_train_only_gap_support([0, 1, 2, 5])
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16, context_window=2),
        support,
    )
    sample = model.sample_fixed_lengths([1, 3, 5])
    valid = sample["valid_mask"]
    assert torch.isnan(sample["gap"][:, 0]).all()
    observed = sample["gap"][valid & torch.isfinite(sample["gap"])].tolist()
    assert set(observed).issubset(set(support.representatives))
    assert torch.equal(valid.sum(1), torch.tensor([1, 3, 5]))
    assert (sample["receiver"][valid] >= MISSING_CODE).all()


def test_architecture_contract_exposes_teacher_forcing_and_length_policy() -> None:
    model = CoFSeqGenSAF(
        SAFModelConfig(candidate_id="SAF-O1", receiver_vocab_size=8, hidden_dim=16),
        fit_train_only_gap_support([0, 1]),
    )
    contract = model.architecture_contract()
    assert contract["training_route"] == "observed_current_gap_and_mark_teacher_forcing"
    assert contract["length_policy"] == "shared_train_only_parent_length_plan"
    assert contract["mark_decoder_ablation_matched"] is True
