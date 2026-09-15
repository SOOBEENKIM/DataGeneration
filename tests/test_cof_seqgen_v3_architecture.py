import torch

from models.cof_seqgen_v3 import (
    CoFSeqGenV3,
    CoFSeqDenoiserV3,
    DirectJointDiscretePath,
    FactorizedJointDiscretePath,
    JointStateCodec,
    corrupt_joint_state,
)


def test_direct_joint_codec_and_corruption_are_paired_and_ignore_padding():
    codec = JointStateCodec(gap_bins=3, receiver_classes=4)
    gap = torch.arange(3).repeat_interleave(4).reshape(2, 6)
    receiver = torch.arange(4).repeat(3).reshape(2, 6)
    valid_mask = torch.tensor(
        [[True, True, True, True, False, False]] * 2
    )

    joint = codec.encode(gap, receiver)
    decoded_gap, decoded_receiver = codec.decode(joint)
    assert torch.equal(decoded_gap, gap)
    assert torch.equal(decoded_receiver, receiver)

    random_values = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.8, 0.0, 0.0],
            [0.9, 0.1, 0.8, 0.2, 0.0, 0.0],
        ]
    )
    corrupted, pair_mask = corrupt_joint_state(
        joint,
        valid_mask=valid_mask,
        mask_probability=0.5,
        mask_state=codec.mask_state,
        random_values=random_values,
    )

    expected_mask = (random_values < 0.5) & valid_mask
    assert torch.equal(pair_mask, expected_mask)
    assert torch.equal(
        corrupted[expected_mask],
        torch.full_like(corrupted[expected_mask], codec.mask_state),
    )
    assert torch.equal(corrupted[~expected_mask], joint[~expected_mask])
    assert not pair_mask[~valid_mask].any()


def test_direct_joint_head_normalizes_and_decodes_one_joint_selection():
    codec = JointStateCodec(gap_bins=2, receiver_classes=3)
    path = DirectJointDiscretePath(codec=codec, d_model=4)
    hidden = torch.zeros(1, 3, 4)
    with torch.no_grad():
        path.joint_head.weight.zero_()
        path.joint_head.bias.copy_(
            torch.tensor([-9.0, -8.0, -7.0, -6.0, 12.0, -5.0])
        )

    logits = path.logits(hidden)
    probabilities = path.probabilities(logits)
    assert logits.shape == (1, 3, codec.state_count)
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(1, 3),
    )

    valid_mask = torch.tensor([[True, True, False]])
    joint, gap, receiver = path.sample_and_decode(
        logits,
        valid_mask=valid_mask,
        temperature=0.0,
    )
    assert torch.equal(joint, torch.tensor([[4, 4, 0]]))
    assert torch.equal(gap, torch.tensor([[1, 1, 0]]))
    assert torch.equal(receiver, torch.tensor([[1, 1, 0]]))


def test_factorized_receiver_depends_on_gap_and_joint_matches_product():
    codec = JointStateCodec(gap_bins=2, receiver_classes=2)
    path = FactorizedJointDiscretePath(codec=codec, d_model=2)
    hidden = torch.zeros(1, 1, 2)
    with torch.no_grad():
        path.gap_head.weight.zero_()
        path.gap_head.bias.copy_(torch.tensor([-1.0, 2.0]))
        path.gap_condition_embedding.weight.copy_(
            torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        )
        first = path.receiver_head[0]
        last = path.receiver_head[2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 2] = 1.0
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 2.0
        last.weight[1, 0] = -2.0

    gap_zero = torch.zeros(1, 1, dtype=torch.long)
    gap_one = torch.ones(1, 1, dtype=torch.long)
    receiver_zero = path.receiver_probabilities(hidden, gap_zero)
    receiver_one = path.receiver_probabilities(hidden, gap_one)
    assert not torch.allclose(receiver_zero, receiver_one)

    gap_probabilities = path.gap_probabilities(path.gap_logits(hidden))
    joint = path.joint_probabilities(hidden)
    expected = torch.stack(
        [
            gap_probabilities[..., gap]
            .unsqueeze(-1)
            * path.receiver_probabilities(
                hidden,
                torch.full_like(gap_zero, gap),
            )
            for gap in range(codec.gap_bins)
        ],
        dim=-2,
    )
    assert torch.allclose(joint, expected)
    assert torch.allclose(joint.sum(dim=(-2, -1)), torch.ones(1, 1))

    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    sampled_gap, sampled_receiver = path.sample_and_decode(
        hidden,
        valid_mask=valid_mask,
        temperature=0.0,
    )
    assert sampled_gap.item() == 1
    assert sampled_receiver.item() == int(
        path.receiver_logits(hidden, sampled_gap).argmax(dim=-1).item()
    )


def test_v3_denoiser_retains_sequence_conditioning_without_legacy_heads():
    codec = JointStateCodec(gap_bins=2, receiver_classes=3)
    x_num = torch.zeros(2, 3, 1)
    joint = codec.encode(
        torch.tensor([[0, 1, 0], [1, 0, 0]]),
        torch.tensor([[1, 2, 0], [0, 1, 0]]),
    )
    time = torch.tensor([0.25, 0.75])
    labels = torch.tensor([0, 1])
    valid_mask = torch.tensor(
        [[True, True, False], [True, True, False]]
    )

    direct = CoFSeqDenoiserV3(
        d_num=1,
        codec=codec,
        candidate="direct_joint",
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_length=3,
    )
    assert hasattr(direct, "transformer")
    assert hasattr(direct, "position_embedding")
    assert hasattr(direct, "time_mlp")
    assert hasattr(direct, "label_embedding")
    assert hasattr(direct, "amount_head")
    assert not hasattr(direct, "bin_head")
    assert not hasattr(direct, "cat_heads")
    direct_output = direct(
        x_num,
        joint,
        time,
        valid_mask=valid_mask,
        y_cond=labels,
    )
    assert direct_output.amount_hat.shape == x_num.shape
    assert direct_output.joint_logits.shape == (
        2,
        3,
        codec.state_count,
    )
    assert direct_output.gap_logits is None
    assert direct_output.receiver_logits is None

    factorized = CoFSeqDenoiserV3(
        d_num=1,
        codec=codec,
        candidate="factorized_joint",
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_length=3,
    )
    factorized_output = factorized(
        x_num,
        joint,
        time,
        valid_mask=valid_mask,
        y_cond=labels,
        teacher_gap=torch.tensor([[0, 1, 0], [1, 0, 0]]),
    )
    assert factorized_output.joint_logits is None
    assert factorized_output.gap_logits.shape == (2, 3, codec.gap_bins)
    assert factorized_output.receiver_logits.shape == (
        2,
        3,
        codec.receiver_classes,
    )


def test_v3_loss_uses_paired_joint_targets_and_true_gap_conditioning():
    codec = JointStateCodec(gap_bins=2, receiver_classes=3)
    valid_mask = torch.tensor(
        [[True, True, False], [True, True, False]]
    )
    amount = torch.tensor(
        [[[0.1], [0.2], [0.0]], [[-0.2], [0.3], [0.0]]]
    )
    gap = torch.tensor([[0, 1, 0], [1, 0, 0]])
    receiver = torch.tensor([[1, 2, 0], [0, 1, 0]])
    y_position = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    random_values = torch.tensor(
        [[0.1, 0.9, 0.0], [0.2, 0.8, 0.0]]
    )

    for candidate in ("direct_joint", "factorized_joint"):
        denoiser = CoFSeqDenoiserV3(
            d_num=1,
            codec=codec,
            candidate=candidate,
            d_model=8,
            n_heads=2,
            n_layers=1,
            max_length=3,
        )
        model = CoFSeqGenV3(
            denoiser=denoiser,
            codec=codec,
            joint_support_mask=torch.ones(
                codec.state_count,
                dtype=torch.bool,
            ),
            joint_support_sha256="a" * 64,
            coherence_lambda=0.0,
        )
        loss, metrics = model.compute_loss(
            x_num=amount,
            gap=gap,
            receiver=receiver,
            y=y_position,
            valid_mask=valid_mask,
            time_fraction=0.5,
            corruption_random_values=random_values,
        )
        assert torch.isfinite(loss)
        assert metrics["paired_mask_positions"] == 2
        assert metrics["amount_loss"] >= 0
        assert metrics["discrete_loss"] >= 0
        assert metrics["coherence_loss"] == 0.0

    with torch.no_grad():
        factorized = model.denoiser.discrete_path
        assert isinstance(factorized, FactorizedJointDiscretePath)
        conditioned = factorized.receiver_logits(
            torch.zeros(2, 3, 8),
            gap,
        )
        assert conditioned.shape == (2, 3, codec.receiver_classes)
