import math
from dataclasses import replace

import pytest
import torch

from models.cof_ccmtpp_v1 import (
    AMOUNT_CONTRACT_NAME,
    CausalEventDecoder,
    CoFCCMTPPV1,
    ConditionalAmountHead,
    FlatReceiverDecoder,
    SampledGap,
)
from models.cof_hcmttpp_v2 import (
    build_h1_checkpoint_bundle,
    CoFHCMTTPPV2H1,
    H1HurdleRQSGapDecoder,
    InvalidH1GapStateError,
    PERMANENT_V1_CHAIN_STATE,
    TrainOnlyH1TailState,
    fit_train_only_h1_tail_state,
    load_h1_checkpoint_bundle,
)


def _train_tail_fixture():
    positive_u = torch.linspace(0.01, 3.0, 300, dtype=torch.float64)
    gap = torch.zeros(2, 301, dtype=torch.float64)
    gap[:, :300] = torch.expm1(positive_u)
    gap[:, 300] = 1.0e12  # invalid padding must not enter fitted state
    valid = torch.zeros(2, 301, dtype=torch.bool)
    valid[:, :300] = True
    y = torch.tensor([0, 1], dtype=torch.long)
    provenance = {
        "train_manifest_sha256": "a" * 64,
        "transform_state_sha256": "b" * 64,
    }
    return gap, y, valid, positive_u, provenance


def test_tail_state_is_strict_exceedance_train_only_and_deterministic():
    gap, y, valid, positive_u, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap,
        y=y,
        valid_mask=valid,
        fit_split="train",
        provenance=provenance,
    )
    repeated = fit_train_only_h1_tail_state(
        gap=gap,
        y=y,
        valid_mask=valid,
        fit_split="train",
        provenance=provenance,
    )

    assert isinstance(state, TrainOnlyH1TailState)
    assert state == repeated
    assert state.n_positive == (300, 300)
    assert state.m_strict_exceedances == (128, 128)
    expected_threshold = float(positive_u[-129])
    expected_beta = float((positive_u[-128:] - positive_u[-129]).mean())
    assert state.u_tail == pytest.approx((expected_threshold, expected_threshold))
    assert state.p_tail_base == pytest.approx((128 / 300, 128 / 300))
    assert state.beta_base == pytest.approx((expected_beta, expected_beta))
    assert state.fit_split == "train"
    assert state.validation_rows_used == 0
    assert state.internal_test_rows_used == 0
    assert state.fraud_test_rows_used == 0
    assert len(state.state_sha256) == 64
    assert all(math.isfinite(value) and value > 0 for value in state.beta_base)

    with pytest.raises(ValueError, match="train-only"):
        fit_train_only_h1_tail_state(
            gap=gap,
            y=y,
            valid_mask=valid,
            fit_split="validation",
            provenance=provenance,
        )


def test_tail_gate_and_signed_scale_equal_baseline_then_move_both_directions():
    gap, y_fit, valid, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap,
        y=y_fit,
        valid_mask=valid,
        fit_split="train",
        provenance=provenance,
    )
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    hidden = torch.tensor([[[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    y = torch.tensor([0], dtype=torch.long)

    baseline = decoder(hidden, y=y)
    assert torch.equal(
        baseline.pi_tail,
        torch.full_like(baseline.pi_tail, state.p_tail_base[0]),
    )
    assert torch.equal(
        baseline.beta,
        torch.full_like(baseline.beta, state.beta_base[0]),
    )

    with torch.no_grad():
        decoder.tail_gate_residual.weight[0, 0] = 1.0
        decoder.tail_scale_residual.weight[0, 0] = 1.0
    conditional = decoder(hidden, y=y)
    assert conditional.pi_tail[0, 0] < state.p_tail_base[0]
    assert conditional.pi_tail[0, 1] > state.p_tail_base[0]
    assert conditional.beta[0, 0] < state.beta_base[0]
    assert conditional.beta[0, 1] > state.beta_base[0]
    assert torch.all(conditional.beta > 0)


def test_central_rqs_is_monotone_bijective_with_analytic_log_jacobian():
    gap, y_fit, valid, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap,
        y=y_fit,
        valid_mask=valid,
        fit_split="train",
        provenance=provenance,
    )
    torch.manual_seed(7)
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    q = torch.tensor([[0.01, 0.2, 0.5, 0.8, 0.99]])
    hidden = torch.zeros(1, q.shape[1], 4)
    parameters = decoder(hidden, y=torch.tensor([0]))

    forward = decoder.central_quantile(q, parameters)
    inverse = decoder.central_inverse(forward.value, parameters)

    assert torch.all(torch.diff(forward.value[0]) > 0)
    assert torch.all(forward.value > 0)
    assert torch.all(forward.value < parameters.u_tail)
    assert torch.allclose(
        inverse.value,
        q.to(dtype=inverse.value.dtype),
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.isfinite(forward.log_abs_det).all()
    assert torch.isfinite(inverse.log_abs_det).all()
    assert torch.allclose(
        forward.log_abs_det + inverse.log_abs_det,
        torch.zeros_like(inverse.log_abs_det),
        atol=3e-5,
        rtol=3e-5,
    )


@pytest.mark.parametrize("u_tail", [2.3978952727983707, 11.30861593474973])
def test_event_nll_is_finite_at_exact_tail_threshold_for_valid_initial_spline(
    u_tail,
):
    state = TrainOnlyH1TailState(
        n_positive=(1_000, 1_000),
        m_strict_exceedances=(128, 128),
        u_tail=(u_tail, u_tail),
        p_tail_base=(0.05, 0.05),
        beta_base=(0.5, 0.5),
        ordered_positive_u_sha256=("0" * 64, "1" * 64),
        fit_split="train",
        provenance=(("fixture", "attempt_001_threshold"),),
        state_sha256="2" * 64,
    )
    torch.manual_seed(0)
    decoder = H1HurdleRQSGapDecoder(d_model=128, tail_state=state)
    hidden = torch.randn(16, 32, 128, requires_grad=True)
    parameters = decoder(hidden, y=torch.zeros(16, dtype=torch.long))
    exact_threshold_gap = torch.expm1(parameters.u_tail)
    valid = torch.ones_like(exact_threshold_gap, dtype=torch.bool)

    loss = decoder.nll(exact_threshold_gap, parameters, valid_mask=valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.dtype == hidden.dtype
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("dominant_bin", [0, 15])
def test_extreme_valid_rqs_bin_has_stable_inverse_jacobian_and_gradient(
    dominant_bin,
):
    gap_fit, y_fit, valid_fit, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=provenance,
    )
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    q = torch.linspace(1.0e-7, 1.0 - 1.0e-7, 4_096).unsqueeze(0)
    q.requires_grad_()
    parameters = decoder(torch.zeros(1, q.shape[1], 4), y=torch.tensor([0]))
    minimum = decoder.minimum_height_fraction
    extreme_heights = torch.full_like(parameters.height_fractions, minimum)
    extreme_heights[..., dominant_bin] = 1.0 - minimum * (
        decoder.central_bins - 1
    )
    extreme = replace(
        parameters,
        height_fractions=extreme_heights,
        derivatives=torch.full_like(
            parameters.derivatives,
            decoder.minimum_derivative,
        ),
    )

    forward = decoder.central_quantile(q, extreme)
    inverse = decoder.central_inverse(forward.value, extreme)
    (forward.value.sum() + forward.log_abs_det.sum()).backward()

    assert torch.allclose(
        inverse.value,
        q.to(dtype=inverse.value.dtype),
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(
        forward.log_abs_det + inverse.log_abs_det,
        torch.zeros_like(inverse.log_abs_det),
        atol=2e-4,
        rtol=2e-4,
    )
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


def test_near_linear_rqs_has_exact_endpoint_geometry_and_finite_inverse():
    gap_fit, y_fit, valid_fit, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=provenance,
    )
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    q = torch.tensor([[1.0e-6, 1 / 16, 0.5, 15 / 16, 1 - 1.0e-6]])
    parameters = decoder(torch.zeros(1, q.shape[1], 4), y=torch.tensor([0]))
    linear = replace(
        parameters,
        height_fractions=torch.full_like(
            parameters.height_fractions,
            1.0 / decoder.central_bins,
        ),
        derivatives=parameters.u_tail.unsqueeze(-1).expand_as(
            parameters.derivatives
        ),
    )

    forward = decoder.central_quantile(q, linear)
    inverse = decoder.central_inverse(forward.value, linear)
    threshold_inverse = decoder.central_inverse(linear.u_tail, linear)

    assert torch.allclose(
        forward.value,
        (q * linear.u_tail).to(dtype=forward.value.dtype),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        inverse.value,
        q.to(dtype=inverse.value.dtype),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(
        threshold_inverse.value,
        torch.ones_like(threshold_inverse.value),
    )
    assert torch.isfinite(forward.log_abs_det).all()
    assert torch.isfinite(inverse.log_abs_det).all()


def test_central_inverse_still_fails_closed_for_invalid_state_and_domain():
    gap_fit, y_fit, valid_fit, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=provenance,
    )
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    parameters = decoder(torch.zeros(1, 2, 4), y=torch.tensor([0]))

    with pytest.raises(ValueError, match="central log gap"):
        decoder.central_inverse(parameters.u_tail + 1.0, parameters)

    invalid_heights = parameters.height_fractions.clone()
    invalid_heights[..., 0] = float("nan")
    invalid = replace(parameters, height_fractions=invalid_heights)
    with pytest.raises(InvalidH1GapStateError, match="non-finite"):
        decoder.central_inverse(parameters.u_tail * 0.5, invalid)


def test_hurdle_likelihood_is_finite_and_positive_cdf_is_normalized_continuous():
    gap_fit, y_fit, valid_fit, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=provenance,
    )
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    hidden = torch.zeros(1, 4, 4)
    parameters = decoder(hidden, y=torch.tensor([0]))
    threshold = parameters.u_tail[0, 0]
    beta = parameters.beta[0, 0]
    gap = torch.stack(
        (
            torch.zeros_like(threshold),
            torch.expm1(threshold * 0.25),
            torch.expm1(threshold),
            torch.expm1(threshold + beta),
        )
    ).unsqueeze(0)
    valid = torch.ones_like(gap, dtype=torch.bool)

    event_nll = decoder.event_nll(gap, parameters, valid_mask=valid)
    assert torch.isfinite(event_nll[valid]).all()
    assert float(event_nll[0, 0].detach()) == pytest.approx(
        float(torch.nn.functional.softplus(-parameters.zero_logits[0, 0]))
    )

    assert torch.equal(
        (1.0 - parameters.pi_tail) + parameters.pi_tail,
        torch.ones_like(parameters.pi_tail),
    )
    epsilon = threshold * 1.0e-6
    u_probe = torch.stack(
        (threshold - epsilon, threshold, threshold + epsilon, threshold + beta)
    ).unsqueeze(0)
    cdf_parameters = decoder(torch.zeros(1, 4, 4), y=torch.tensor([0]))
    cdf = decoder.positive_cdf_u(u_probe, cdf_parameters)
    expected_at_threshold = 1.0 - cdf_parameters.pi_tail[0, 1]
    assert float(cdf[0, 1].detach()) == pytest.approx(
        float(expected_at_threshold.detach()), abs=1e-6
    )
    assert abs(float(cdf[0, 0] - cdf[0, 1])) < 2e-5
    assert abs(float(cdf[0, 2] - cdf[0, 1])) < 2e-5
    assert torch.all((cdf >= 0) & (cdf <= 1))


def test_open_interval_sampling_has_exact_zero_unbounded_tail_and_no_clip():
    gap_fit, y_fit, valid_fit, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=provenance,
    )
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    with torch.no_grad():
        decoder.zero_head.weight.zero_()
        decoder.zero_head.bias.zero_()
    parameters = decoder(torch.zeros(1, 3, 4), y=torch.tensor([0]))
    zeros = decoder.sample_from_uniforms(
        parameters,
        zero_uniform=torch.full((1, 3), 0.25),
        tail_uniform=torch.full((1, 3), 0.25),
        value_uniform=torch.full((1, 3), 0.75),
    )
    assert torch.equal(zeros.gap, torch.zeros_like(zeros.gap))
    assert torch.equal(zeros.u, torch.zeros_like(zeros.u))

    epsilon = torch.finfo(parameters.beta.dtype).eps
    tail = decoder.sample_from_uniforms(
        parameters,
        zero_uniform=torch.full((1, 3), 0.75),
        tail_uniform=torch.full((1, 3), epsilon),
        value_uniform=torch.full((1, 3), 1.0 - epsilon),
    )
    assert torch.all(tail.u > parameters.u_tail)
    assert torch.all(tail.gap > torch.expm1(parameters.u_tail))
    assert tail.hard_upper_clip_count == 0
    assert tail.redraw_count == 0

    with torch.no_grad():
        decoder.tail_scale_residual.bias.fill_(1000.0)
    invalid = decoder(torch.zeros(1, 3, 4), y=torch.tensor([0]))
    assert invalid.invalid_mask.all()
    with pytest.raises(InvalidH1GapStateError, match="invalid"):
        decoder.sample_from_uniforms(
            invalid,
            zero_uniform=torch.full((1, 3), 0.75),
            tail_uniform=torch.full((1, 3), 0.25),
            value_uniform=torch.full((1, 3), 0.75),
        )


def test_high_precision_rqs_sampling_returns_model_dtype_for_amount_path():
    gap_fit, y_fit, valid_fit, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=provenance,
    )
    decoder = H1HurdleRQSGapDecoder(d_model=4, tail_state=state)
    amount_head = ConditionalAmountHead(receiver_classes=7, d_model=4)
    hidden = torch.zeros(1, 3, 4)
    parameters = decoder(hidden, y=torch.tensor([0]))

    sampled = decoder.sample_from_uniforms(
        parameters,
        zero_uniform=torch.full((1, 3), 0.75),
        tail_uniform=torch.full((1, 3), 0.75),
        value_uniform=torch.tensor([[0.25, 0.5, 0.75]]),
    )
    amount = amount_head(
        hidden=hidden,
        sampled_gap=SampledGap(
            gap=sampled.gap,
            u=sampled.u,
            source="cof_ccmtpp_v1_continuous_gap_head",
        ),
        receiver=torch.ones(1, 3, dtype=torch.long),
    )

    assert sampled.u.dtype == hidden.dtype
    assert sampled.gap.dtype == hidden.dtype
    assert amount.dtype == hidden.dtype
    assert torch.isfinite(amount).all()


def test_h1_replaces_only_gap_decoder_and_keeps_v1_chain_stopped():
    gap_fit, y_fit, valid_fit, _, provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=provenance,
    )
    model = CoFHCMTTPPV2H1(
        candidate="H1",
        receiver_classes=7,
        tail_state=state,
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_length=8,
        dropout=0.0,
    )
    c1 = CoFCCMTPPV1(
        candidate="C1",
        receiver_classes=7,
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_length=8,
        gap_components=5,
        gap_max=100.0,
        dropout=0.0,
    )

    assert type(model.event_decoder) is type(c1.event_decoder) is CausalEventDecoder
    assert isinstance(model.gap_decoder, H1HurdleRQSGapDecoder)
    assert not hasattr(model, "gap_head")
    assert type(model.receiver_decoder) is type(c1.receiver_decoder) is FlatReceiverDecoder
    assert type(model.amount_head) is type(c1.amount_head) is ConditionalAmountHead
    for h1_module, c1_module in (
        (model.event_decoder, c1.event_decoder),
        (model.receiver_decoder, c1.receiver_decoder),
        (model.amount_head, c1.amount_head),
    ):
        assert {
            name: tuple(value.shape) for name, value in h1_module.state_dict().items()
        } == {
            name: tuple(value.shape) for name, value in c1_module.state_dict().items()
        }
    config = model.model_config()
    assert config["candidate"] == "H1"
    assert config["changed_factors"] == ["gap_decoder"]
    assert config["receiver_path"] == "flat_no_copy"
    assert config["amount_contract"] == AMOUNT_CONTRACT_NAME
    assert config["y_balanced_likelihood"] is False
    assert config["structure_loss"] is None
    assert config["permanent_v1_chain_state"] == PERMANENT_V1_CHAIN_STATE
    assert config["tail_state_sha256"] == state.state_sha256

    with pytest.raises(ValueError, match="only H1"):
        CoFHCMTTPPV2H1(
            candidate="H2",
            receiver_classes=7,
            tail_state=state,
        )


def test_h1_checkpoint_round_trip_binds_tail_state_and_train_only_provenance():
    gap_fit, y_fit, valid_fit, _, tail_provenance = _train_tail_fixture()
    state = fit_train_only_h1_tail_state(
        gap=gap_fit,
        y=y_fit,
        valid_mask=valid_fit,
        fit_split="train",
        provenance=tail_provenance,
    )
    model = CoFHCMTTPPV2H1(
        candidate="H1",
        receiver_classes=7,
        tail_state=state,
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_length=8,
    )
    provenance = {
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "train_manifest_sha256": "3" * 64,
        "transform_state_sha256": "4" * 64,
        "sampling_plan_sha256": "5" * 64,
        "amount_contract_sha256": "6" * 64,
        "receiver_vocabulary_sha256": "7" * 64,
        "tail_state_sha256": state.state_sha256,
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    bundle = build_h1_checkpoint_bundle(model=model, provenance=provenance)
    restored = load_h1_checkpoint_bundle(
        bundle,
        expected_provenance=provenance,
    )
    assert restored.model_config() == model.model_config()
    assert restored.tail_state == state
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    forbidden = {**provenance, "validation_rows_used": 1}
    with pytest.raises(ValueError, match="train-only"):
        build_h1_checkpoint_bundle(model=model, provenance=forbidden)
    mismatched = {**provenance, "tail_state_sha256": "9" * 64}
    with pytest.raises(ValueError, match="provenance"):
        load_h1_checkpoint_bundle(bundle, expected_provenance=mismatched)
