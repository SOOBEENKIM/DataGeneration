import math

import pytest
import torch

from models.cof_ccmtpp_v1 import (
    AMOUNT_CONTRACT_NAME,
    CausalEventDecoder,
    CoFCCMTPPV1,
    FlatCopyReceiverDecoder,
    FlatReceiverDecoder,
    HierarchicalCopyReceiverDecoder,
    MixtureLogisticGapHead,
    ReceiverHierarchyState,
    SampledGap,
    balanced_event_mean,
    build_checkpoint_bundle,
    fit_train_receiver_hierarchy,
    load_checkpoint_bundle,
    receiver_nll_diagnostics,
)


def _causal_decoder() -> CausalEventDecoder:
    torch.manual_seed(17)
    decoder = CausalEventDecoder(
        receiver_classes=7,
        d_model=16,
        n_heads=4,
        n_layers=2,
        max_length=8,
        amount_dimensions=1,
        dropout=0.0,
    )
    decoder.eval()
    return decoder


def test_causal_decoder_is_prefix_invariant_and_padding_invariant():
    decoder = _causal_decoder()
    amount = torch.tensor(
        [[[0.1], [0.2], [0.3], [0.4], [9.0]],
         [[0.1], [0.2], [0.3], [-8.0], [-9.0]]]
    )
    gap = torch.tensor(
        [[0.0, 1.0, 2.0, 30.0, 100.0],
         [0.0, 1.0, 2.0, 999.0, 9999.0]]
    )
    receiver = torch.tensor(
        [[2, 3, 4, 5, 6], [2, 3, 4, 6, 5]], dtype=torch.long
    )
    valid = torch.tensor(
        [[True, True, True, True, False],
         [True, True, True, True, False]]
    )
    y = torch.tensor([1, 1], dtype=torch.long)
    lengths = valid.sum(dim=1)

    hidden = decoder(
        amount=amount,
        gap=gap,
        receiver=receiver,
        y=y,
        lengths=lengths,
        valid_mask=valid,
    )

    # Position t consumes only the prefix ending at t-1.  The first four
    # hidden states therefore cannot see the differing event at position 3.
    assert torch.allclose(hidden[0, :4], hidden[1, :4], atol=1e-6)
    assert torch.equal(hidden[0, 4], torch.zeros_like(hidden[0, 4]))
    assert torch.equal(hidden[1, 4], torch.zeros_like(hidden[1, 4]))

    altered = amount.clone()
    altered[0, 4] = -12345.0
    altered_hidden = decoder(
        amount=altered,
        gap=gap,
        receiver=receiver,
        y=y,
        lengths=lengths,
        valid_mask=valid,
    )
    assert torch.allclose(hidden, altered_hidden, atol=1e-6)


def test_causal_decoder_rejects_entity_identifier_and_invalid_mask():
    decoder = _causal_decoder()
    tensors = {
        "amount": torch.zeros(1, 2, 1),
        "gap": torch.zeros(1, 2),
        "receiver": torch.tensor([[2, 3]], dtype=torch.long),
        "y": torch.zeros(1, dtype=torch.long),
        "lengths": torch.tensor([2]),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    with pytest.raises(TypeError):
        decoder(**tensors, entity_id=torch.tensor([99]))
    with pytest.raises(ValueError, match="prefix"):
        decoder(
            **{
                **tensors,
                "lengths": torch.tensor([1]),
                "valid_mask": torch.tensor([[False, True]]),
            }
        )
    with pytest.raises(ValueError, match="nonnegative"):
        decoder(**{**tensors, "gap": torch.tensor([[0.0, -1.0]])})


def test_continuous_gap_mixture_has_finite_density_nll_and_inverse_sample():
    head = MixtureLogisticGapHead(
        d_model=4,
        components=3,
        min_gap=0.0,
        max_gap=1000.0,
        min_scale=1e-3,
    )
    hidden = torch.zeros(2, 3, 4)
    gap = torch.tensor([[0.0, 0.5, 3.0], [10.0, 99.0, 1000.0]])
    valid = torch.tensor(
        [[True, True, True], [True, True, False]]
    )

    parameters = head(hidden)
    log_density = head.log_prob_gap(gap, parameters)
    nll = head.nll(gap, parameters, valid_mask=valid)
    assert torch.isfinite(log_density[valid]).all()
    assert torch.isfinite(nll)

    generator = torch.Generator().manual_seed(123)
    sampled = head.sample(parameters, generator=generator)
    assert isinstance(sampled, SampledGap)
    assert sampled.source == "cof_ccmtpp_v1_continuous_gap_head"
    assert torch.all(sampled.gap >= 0.0)
    assert torch.all(sampled.gap <= 1000.0)
    assert torch.allclose(torch.expm1(sampled.u), sampled.gap, atol=1e-5)

    expected_u = torch.log1p(gap)
    assert torch.allclose(head.encode(gap), expected_u)
    assert torch.allclose(head.decode(expected_u), gap)
    assert math.isclose(head.u_max, math.log1p(1000.0))


def test_continuous_gap_head_rejects_bins_and_out_of_range_values():
    head = MixtureLogisticGapHead(d_model=4, components=2, max_gap=20.0)
    assert not hasattr(head, "gap_bin_head")
    with pytest.raises(ValueError, match="support"):
        head.encode(torch.tensor([-0.1]))
    with pytest.raises(ValueError, match="support"):
        head.encode(torch.tensor([20.1]))


def test_train_only_receiver_hierarchy_is_deterministic_and_pad_unk_safe():
    receiver = torch.tensor(
        [[2, 2, 3, 4, 5, 0], [2, 3, 6, 7, 8, 0]], dtype=torch.long
    )
    valid = torch.tensor(
        [[True, True, True, True, True, False],
         [True, True, True, True, True, False]]
    )
    provenance = {
        "train_manifest_sha256": "a" * 64,
        "receiver_vocabulary_sha256": "b" * 64,
    }
    state = fit_train_receiver_hierarchy(
        receiver=receiver,
        valid_mask=valid,
        receiver_classes=9,
        fit_split="train",
        provenance=provenance,
        head_min_count=2,
        max_head_categories=2,
        tail_cluster_count=2,
    )
    repeated = fit_train_receiver_hierarchy(
        receiver=receiver,
        valid_mask=valid,
        receiver_classes=9,
        fit_split="train",
        provenance=provenance,
        head_min_count=2,
        max_head_categories=2,
        tail_cluster_count=2,
    )
    assert isinstance(state, ReceiverHierarchyState)
    assert state == repeated
    assert state.pad_code == 0 and state.unk_code == 1
    assert state.head_codes == (2, 3)
    assert sorted(code for cluster in state.tail_clusters for code in cluster) == [
        4, 5, 6, 7, 8
    ]
    assert len(state.state_sha256) == 64
    assert state.fit_split == "train"
    assert state.validation_rows_used == 0
    assert state.test_rows_used == 0
    with pytest.raises(ValueError, match="train-only"):
        fit_train_receiver_hierarchy(
            receiver=receiver,
            valid_mask=valid,
            receiver_classes=9,
            fit_split="validation",
            provenance=provenance,
            head_min_count=2,
            max_head_categories=2,
            tail_cluster_count=2,
        )


def test_flat_copy_receiver_uses_sampled_gap_and_normalizes_causal_pointer():
    torch.manual_seed(3)
    decoder = FlatCopyReceiverDecoder(receiver_classes=7, d_model=8)
    hidden = torch.randn(1, 4, 8)
    receivers = torch.tensor([[2, 3, 2, 4]], dtype=torch.long)
    valid = torch.ones(1, 4, dtype=torch.bool)
    sampled_zero = SampledGap(gap=torch.zeros(1, 4), u=torch.zeros(1, 4))
    sampled_large = SampledGap(gap=torch.full((1, 4), 9.0), u=torch.full((1, 4), math.log(10.0)))

    zero = decoder(
        hidden=hidden,
        history_receiver=receivers,
        valid_mask=valid,
        sampled_gap=sampled_zero,
    )
    large = decoder(
        hidden=hidden,
        history_receiver=receivers,
        valid_mask=valid,
        sampled_gap=sampled_large,
    )
    assert torch.allclose(zero.probabilities.sum(dim=-1), torch.ones(1, 4))
    assert torch.equal(zero.pointer_weights[0, 0], torch.zeros(4))
    assert torch.equal(
        torch.triu(zero.pointer_weights[0], diagonal=0), torch.zeros(4, 4)
    )
    assert not torch.allclose(zero.probabilities, large.probabilities)
    with pytest.raises(ValueError, match="model-sampled gap"):
        decoder(
            hidden=hidden,
            history_receiver=receivers,
            valid_mask=valid,
            sampled_gap=torch.zeros(1, 4),
        )


def test_hierarchical_receiver_normalizes_without_flat_vocabulary_head():
    state = ReceiverHierarchyState(
        receiver_classes=8,
        pad_code=0,
        unk_code=1,
        head_codes=(2, 3),
        tail_clusters=((4, 6), (5, 7)),
        counts=(0, 0, 8, 6, 4, 3, 2, 1),
        fit_split="train",
        provenance=(("train_manifest_sha256", "a" * 64),),
        state_sha256="b" * 64,
    )
    decoder = HierarchicalCopyReceiverDecoder(hierarchy=state, d_model=8)
    assert not hasattr(decoder, "flat_receiver_head")
    hidden = torch.randn(2, 4, 8)
    history = torch.tensor([[2, 4, 2, 5], [3, 6, 7, 3]], dtype=torch.long)
    valid = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    sampled = SampledGap(
        gap=torch.ones(2, 4),
        u=torch.full((2, 4), math.log(2.0)),
    )
    distribution = decoder(
        hidden=hidden,
        history_receiver=history,
        valid_mask=valid,
        sampled_gap=sampled,
    )
    assert torch.allclose(
        distribution.total_probability_mass()[valid],
        torch.ones_like(distribution.total_probability_mass()[valid]),
        atol=1e-6,
    )
    assert torch.equal(
        distribution.target_probability(torch.zeros_like(history)),
        torch.zeros_like(sampled.gap),
    )
    for code in range(1, 8):
        probability = distribution.target_probability(
            torch.full_like(history, code)
        )
        assert torch.all(probability[valid] > 0)

    diagnostics = receiver_nll_diagnostics(
        distribution=distribution,
        target=history,
        valid_mask=valid,
        hierarchy=state,
    )
    assert set(diagnostics) == {
        "overall_nll", "overall_count", "head_nll", "head_count",
        "tail_nll", "tail_count", "unk_nll", "unk_count",
        "repeat_nll", "repeat_count", "new_nll", "new_count",
    }
    assert diagnostics["overall_count"] == int(valid.sum())


def test_c1_flat_receiver_has_no_copy_or_hierarchy_path():
    decoder = FlatReceiverDecoder(receiver_classes=6, d_model=8)
    probabilities = decoder(torch.zeros(2, 3, 8))
    assert probabilities.shape == (2, 3, 6)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3))
    assert torch.equal(probabilities[..., 0], torch.zeros(2, 3))
    assert not hasattr(decoder, "copy_gate")


def _hierarchy_for_model() -> ReceiverHierarchyState:
    return ReceiverHierarchyState(
        receiver_classes=7,
        pad_code=0,
        unk_code=1,
        head_codes=(2, 3),
        tail_clusters=((4, 6), (5,)),
        counts=(0, 0, 8, 6, 4, 3, 2),
        fit_split="train",
        provenance=(("train_manifest_sha256", "a" * 64),),
        state_sha256="c" * 64,
    )


def test_c1_to_c4_are_finite_nested_architectures_without_structure_loss():
    hierarchy = _hierarchy_for_model()
    models = {
        candidate: CoFCCMTPPV1(
            candidate=candidate,
            receiver_classes=7,
            hierarchy=hierarchy if candidate in {"C3", "C4"} else None,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_length=8,
            gap_components=3,
            gap_max=100.0,
        )
        for candidate in ("C1", "C2", "C3", "C4")
    }
    assert isinstance(models["C1"].receiver_decoder, FlatReceiverDecoder)
    assert isinstance(models["C2"].receiver_decoder, FlatCopyReceiverDecoder)
    assert isinstance(
        models["C3"].receiver_decoder, HierarchicalCopyReceiverDecoder
    )
    assert isinstance(
        models["C4"].receiver_decoder, HierarchicalCopyReceiverDecoder
    )
    assert models["C4"].y_balanced_likelihood is True
    assert all(model.structure_loss is None for model in models.values())
    assert all(model.amount_contract == AMOUNT_CONTRACT_NAME for model in models.values())
    with pytest.raises(ValueError, match="finite candidate"):
        CoFCCMTPPV1(candidate="C5", receiver_classes=7)


def test_y_balanced_likelihood_weights_classes_equally():
    event_loss = torch.tensor([[1.0, 1.0, 0.0], [9.0, 9.0, 9.0]])
    y = torch.tensor([0, 1])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    assert balanced_event_mean(
        event_loss, y=y, valid_mask=valid, y_balanced=False
    ).item() == pytest.approx(5.8)
    assert balanced_event_mean(
        event_loss, y=y, valid_mask=valid, y_balanced=True
    ).item() == pytest.approx(5.0)


def test_checkpoint_round_trip_preserves_model_and_train_only_provenance():
    model = CoFCCMTPPV1(
        candidate="C3",
        receiver_classes=7,
        hierarchy=_hierarchy_for_model(),
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_length=8,
        gap_components=3,
        gap_max=100.0,
    )
    provenance = {
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "train_manifest_sha256": "3" * 64,
        "transform_state_sha256": "4" * 64,
        "sampling_plan_sha256": "5" * 64,
        "amount_contract_sha256": "6" * 64,
        "receiver_hierarchy_sha256": "c" * 64,
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    bundle = build_checkpoint_bundle(model=model, provenance=provenance)
    restored = load_checkpoint_bundle(bundle, expected_provenance=provenance)
    assert restored.candidate == "C3"
    assert restored.amount_contract == AMOUNT_CONTRACT_NAME
    assert restored.receiver_decoder.hierarchy == _hierarchy_for_model()
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    tampered = {**provenance, "train_manifest_sha256": "9" * 64}
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint_bundle(bundle, expected_provenance=tampered)


def test_mark_sampling_uses_continuous_gap_and_structured_receiver_contract():
    model = CoFCCMTPPV1(
        candidate="C3",
        receiver_classes=7,
        hierarchy=_hierarchy_for_model(),
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_length=8,
        gap_components=3,
        gap_max=100.0,
    )
    hidden = torch.randn(2, 4, 16)
    history = torch.tensor([[2, 3, 2, 4], [3, 5, 6, 2]])
    valid = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    mark = model.sample_mark_from_hidden(
        hidden=hidden,
        history_receiver=history,
        valid_mask=valid,
        generator=torch.Generator().manual_seed(19),
    )
    assert mark.gap_source == "cof_ccmtpp_v1_continuous_gap_head"
    assert mark.amount_contract == AMOUNT_CONTRACT_NAME
    assert torch.all((mark.receiver[valid] >= 1) & (mark.receiver[valid] < 7))
    assert torch.all(mark.gap[valid] >= 0)
    assert torch.equal(mark.receiver[~valid], torch.zeros_like(mark.receiver[~valid]))
    assert torch.equal(mark.gap[~valid], torch.zeros_like(mark.gap[~valid]))
    assert torch.equal(
        mark.normalized_amount[~valid],
        torch.zeros_like(mark.normalized_amount[~valid]),
    )
