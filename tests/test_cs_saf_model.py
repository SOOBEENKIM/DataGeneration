import numpy as np
import pytest
import torch

from models.cs_saf import CSSAF, CANDIDATES
from models.cof_seqgen_saf import fit_train_only_gap_support
from experiments.cs_saf_pilot import decide_pilot, audit_model


def model(candidate="CS-B1"):
    return CSSAF(candidate, fit_train_only_gap_support([.1, .5, 1, 3, 8]),
                 receiver_vocab_size=7, static_categorical_vocab_sizes=(5,))


def inputs():
    return {"gap": torch.tensor([[float("nan"), .1, .5, 1], [float("nan"), .5, 3, float("nan")]]),
            "receiver": torch.tensor([[3, 3, 4, 4], [4, 5, 5, 0]]),
            "numeric_value": torch.tensor([[1., 2, 3, 1], [2., 1, 3, 0]]),
            "valid_mask": torch.tensor([[True]*4, [True]*3+[False]]),
            "static_categorical": (torch.tensor([3, 4]),)}


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_all_candidates_finite_losses_gradients_and_closed_vocabulary(candidate):
    torch.manual_seed(12)
    m = model(candidate)
    terms = m.loss_terms(**inputs())
    loss, _, _ = m.objective(terms, train_entities=2, transition_counts_by_code={3: 3, 4: 2})
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    sample = m.sample_fixed_lengths([4, 3], static_categorical=(torch.tensor([3, 4]),))
    assert (sample["receiver"][sample["valid_mask"]] >= 3).all()
    assert torch.isnan(sample["gap"][:, 0]).all()
    if candidate != "CS-C0":
        nonfirst = sample["valid_mask"].clone(); nonfirst[:, 0] = False
        assert torch.isin(sample["gap"][nonfirst], torch.tensor(m.support.representatives)).all()


def test_route_and_objective_controls_have_identical_parameters_and_initialization():
    states = []
    for candidate in ("CS-U0", "CS-U1", "CS-B0", "CS-B1"):
        torch.manual_seed(42)
        states.append(model(candidate).state_dict())
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        assert all(torch.equal(state[k], states[0][k]) for k in state)


def test_bilinear_route_gets_gradients_and_zero_gap_control_remains_invariant():
    torch.manual_seed(3)
    m = model()
    optimizer = torch.optim.AdamW(m.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss, _, _ = m.objective(m.loss_terms(**inputs()), train_entities=2, transition_counts_by_code={3: 3, 4: 2})
        loss.backward(); optimizer.step()
    assert m.interaction_weight.grad.abs().sum() > 0
    assert m.context_projection.weight.grad.abs().sum() > 0
    assert m.gap_projection.weight.grad.abs().sum() > 0
    x = inputs()
    hidden = m.encoder(**x)
    context = m.context(hidden, x["static_categorical"])[:, 1]
    curves, _ = m.response_curves(context, x["receiver"][:, 0])
    zero, _ = m.response_curves(context, x["receiver"][:, 0], zero_gap=True)
    assert (curves.max(1).values-curves.min(1).values).max() > 0
    assert (zero.max(1).values-zero.min(1).values).max() == 0


def test_no_current_mark_or_future_leakage_in_current_mark_distribution():
    m = model().eval(); x = inputs()
    hidden = m.encoder(**x)
    changed = dict(x, receiver=x["receiver"].clone(), numeric_value=x["numeric_value"].clone())
    changed["receiver"][0, 2:] = 6
    changed["numeric_value"][0, 2:] = 99
    other = m.encoder(**changed)
    assert torch.equal(hidden[:, :3], other[:, :3])
    assert not torch.equal(hidden[0, 3], other[0, 3])


def test_observable_repeat_includes_accidental_new_mark_match():
    m = model()
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    context = torch.zeros(1, 136)
    logmark, logr, lognr = m.mark_distribution(context, torch.tensor([.1]), torch.tensor([3]), torch.tensor([True]))
    # q=.5, four valid fresh categories => .5 + .5/4 = .625.
    assert logr.exp().item() == pytest.approx(.625)
    assert lognr.exp().item() == pytest.approx(.375)
    assert logmark.exp()[0, 3].item() == pytest.approx(.625)
    assert logmark.exp()[0, :3].sum() == 0


def test_fixed_global_auxiliary_denominator_survives_single_context_minibatches():
    m = model()
    x = inputs()
    whole = m.objective(m.loss_terms(**x), train_entities=2, transition_counts_by_code={3: 3, 4: 2})[2]
    pieces = []
    for i in (0, 1):
        one = {k: (tuple(t[i:i+1] for t in v) if isinstance(v, tuple) else v[i:i+1]) for k, v in x.items()}
        pieces.append(m.objective(m.loss_terms(**one), train_entities=2, transition_counts_by_code={3: 3, 4: 2})[2])
    assert torch.allclose(whole, torch.stack(pieces).mean(), atol=1e-6)


def test_pilot_requires_selective_observable_and_latent_response():
    gate = {"active_mean_range_min": .05, "noncausal_mean_range_max": .05,
            "selectivity_ratio_min": 2, "zero_gap_control_max_range": 1e-8}
    audits = {k: {"responses": {str(y): {f"mean_{m}_range": .2 if k*y else 0
                    for m in ("copy", "repeat")} for y in (0, 1)},
                  "zero_gap_control_max_range": 0, "gap_support_violations": 0,
                  "invalid_reserved_marks": 0, "finite_generated_values": True} for k in (0, 1)}
    assert decide_pilot(audits, gate)["decision"] == "PASS"
    audits[1]["responses"]["0"]["mean_repeat_range"] = .1
    assert decide_pilot(audits, gate)["decision"] == "FAIL"


def test_pilot_audit_adapter_generates_and_checks_both_response_definitions():
    m = model().eval()
    x = inputs()
    data = {k: v for k, v in x.items() if k != "static_categorical"}
    data.update(codes=x["static_categorical"][0], lengths=torch.tensor([4, 3]), entity_ids=["a", "b"])
    cfg = {"sampling_seed": 42, "generation_entities": 8, "generation_batch_size": 4,
           "audit_chunk_histories": 2}
    result = audit_model(m, {"train": data, "validation": data}, cfg, torch.device("cpu"))
    assert result["gap_support_violations"] == 0
    assert result["invalid_reserved_marks"] == 0
    assert result["finite_generated_values"]
    assert result["zero_gap_control_max_range"] == 0
    assert set(result["responses"]) == {"0", "1"}


def test_public_loss_and_architecture_describe_cs_saf_not_inherited_scalar_gate():
    m = model()
    contract = m.architecture_contract()
    assert contract["family"] == "CS-SAF"
    assert contract["gap_to_mark_route"] == "rank_32_bilinear"
    losses = m.compute_loss(**inputs(), train_entities=2, transition_counts_by_code={3: 3, 4: 2})
    assert torch.allclose(losses["loss"], losses["base_nll"]+losses["balanced_repeat_nll"])
