from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .class_conditional_markov import (
    ClassConditionalMarkov,
    JointObservedMarkov,
)
from .cof_seqgen_adapter import CoFSeqGenAdapter
from .conditional_ctgan import ConditionalCTGAN
from .conditional_tvae import ConditionalTVAE
from .contracts_v2_5 import BaselineSpec, validate_adapter
from .empirical_conditional_block import EmpiricalConditionalBlock
from .empirical_conditional_iid import EmpiricalConditionalIID
from .joint_sequence_baseline import NeuralSequenceBaseline
from .plug_in_state_baselines_v2_5 import (
    ClassConditionalPlugInHMM,
    ClassConditionalPlugInHSMM,
)


BASELINE_DEFINITION_VERSION = "benchmark-v2.5"


def baseline_registry() -> "OrderedDict[str, BaselineSpec]":
    """Return the complete, ordered, preregistered v2.5 generator registry."""
    specs = [
        BaselineSpec(
            "empirical_iid",
            "label-conditional empirical i.i.d. row bootstrap",
            "structural_reference",
            EmpiricalConditionalIID,
            "cpu",
            True,
        ),
        *[
            BaselineSpec(
                f"block_{length}",
                f"contiguous block bootstrap (length {length})",
                "structural_reference",
                lambda length=length: EmpiricalConditionalBlock(length),
                "cpu",
                True,
            )
            for length in (2, 4, 8)
        ],
        BaselineSpec(
            "full_sequence_reference",
            "full-sequence block empirical reference",
            "oracle_reference",
            lambda: EmpiricalConditionalBlock("full"),
            "cpu",
            True,
            full_sequence_competitor=False,
        ),
        BaselineSpec(
            "independent_markov",
            "class-conditional independent Markov",
            "stochastic_baseline",
            ClassConditionalMarkov,
            "cpu",
            True,
        ),
        BaselineSpec(
            "joint_markov",
            "class-conditional joint observed Markov",
            "stochastic_baseline",
            JointObservedMarkov,
            "cpu",
            True,
        ),
        BaselineSpec(
            "plug_in_hmm",
            "class-conditional plug-in HMM",
            "stochastic_baseline",
            ClassConditionalPlugInHMM,
            "cpu",
            True,
        ),
        BaselineSpec(
            "plug_in_hsmm",
            "class-conditional explicit-duration HSMM",
            "stochastic_baseline",
            ClassConditionalPlugInHSMM,
            "cpu",
            True,
        ),
        BaselineSpec(
            "ctgan_separate_class",
            "separate-class-model CTGAN",
            "learned_baseline",
            ConditionalCTGAN,
            "gpu",
            True,
        ),
        BaselineSpec(
            "tvae_separate_class",
            "separate-class-model TVAE",
            "learned_baseline",
            ConditionalTVAE,
            "gpu",
            True,
        ),
        BaselineSpec(
            "neural_sequence",
            "class-conditional neural sequence baseline",
            "learned_baseline",
            NeuralSequenceBaseline,
            "gpu",
            True,
        ),
        BaselineSpec(
            "cof_seqgen",
            "CoF-SeqGen",
            "proposed",
            CoFSeqGenAdapter,
            "gpu",
            True,
        ),
    ]
    registry = OrderedDict((spec.generator_id, spec) for spec in specs)
    if len(registry) != len(specs):
        raise AssertionError("duplicate v2.5 generator id")
    for spec in registry.values():
        validate_adapter(spec.factory())
    return registry


def generator_for_v2_5(generator_id: str) -> Any:
    try:
        spec = baseline_registry()[generator_id]
    except KeyError as error:
        raise ValueError(f"unregistered v2.5 generator: {generator_id}") from error
    return spec.factory()
