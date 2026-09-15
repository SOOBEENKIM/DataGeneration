from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import numpy as np

from .binning import apply_binning, fit_binning
from .semi_markov import calibrated_duration_pmf, sample_equilibrium_alternating
from .types import DatasetBundle, SequenceBatch


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: str = "benchmark_v2.0"
    scenario: str = "markov_persistence_v2a"
    kappa: float = 0.0
    n_train: int = 31_951
    n_test: int = 7_989
    min_length: int = 16
    max_length: int = 32
    fraud_rate: float = 0.05
    n_gap_bins: int = 16
    n_receiver_categories: int = 64
    pi_burst: float = 0.30
    gap_normal_scale: float = 2.0
    gap_burst_scale: float = 0.5
    rho_max: float = 0.90
    q_low: float = 0.05
    q_high: float = 0.90
    amount_mean: float = 5.0
    amount_std: float = 1.0
    duration_burst_mean: float = 6.0
    duration_normal_mean: float = 14.0
    duration_sigma: float = 0.60
    duration_max: int = 128
    bin_edges: Optional[np.ndarray] = None
    randomness_nonce: int = 0

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], scenario: str, kappa: float
    ) -> "BenchmarkConfig":
        data, coupling = raw["data"], raw["coupling"]
        gap, joint = coupling["gap_regime"], coupling["joint_semimarkov"]
        amount = coupling["amount_log"]
        return cls(
            schema_version=str(raw.get("schema_version", "benchmark_v2.0")),
            scenario=scenario,
            kappa=float(kappa),
            n_train=int(data["n_train"]),
            n_test=int(data["n_test"]),
            min_length=int(data["sequence_length"]["min"]),
            max_length=int(data["sequence_length"]["max"]),
            fraud_rate=float(data["fraud_rate"]),
            n_gap_bins=int(data["n_gap_bins"]),
            n_receiver_categories=int(data["n_receiver_categories"]),
            pi_burst=float(gap["burst_stationary_probability"]),
            gap_normal_scale=float(gap["normal_gap_scale"]),
            gap_burst_scale=float(gap["burst_gap_scale"]),
            rho_max=float(gap["rho_fraud_max"]),
            q_low=float(joint["receiver_repeat_probability"]["low"]),
            q_high=float(joint["receiver_repeat_probability"]["high"]),
            amount_mean=float(amount["mean"]),
            amount_std=float(amount["std"]),
            duration_burst_mean=float(joint["duration"]["burst_target_mean"]),
            duration_normal_mean=float(joint["duration"]["normal_target_mean"]),
            duration_sigma=float(joint["duration"]["sigma"]),
            duration_max=int(joint["duration"]["max_duration"]),
        )


def transition_matrix(pi: float, rho: float) -> np.ndarray:
    matrix = np.array(
        [[1 - (1 - rho) * pi, (1 - rho) * pi],
         [(1 - rho) * (1 - pi), pi + (1 - pi) * rho]],
        dtype=float,
    )
    stationary = np.array([1 - pi, pi])
    if not np.allclose(matrix.sum(1), 1) or not np.allclose(stationary @ matrix, stationary):
        raise AssertionError("invalid stationary transition matrix")
    return matrix


def _markov_state(rng: np.random.Generator, length: int, pi: float, rho: float) -> np.ndarray:
    transition = transition_matrix(pi, rho)
    out = np.empty(length, dtype=np.int8)
    out[0] = rng.random() < pi
    for t in range(1, length):
        out[t] = rng.choice(2, p=transition[out[t - 1]])
    return out


def _sample_receiver_path(
    config: BenchmarkConfig,
    *,
    label: int,
    length: int,
    z_gap: np.ndarray,
    state_rng: np.random.Generator,
    cat_rng: np.random.Generator,
    burst_pmf: np.ndarray,
    normal_pmf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Production receiver path shared by the full DGP and receiver-only MC."""
    if config.scenario == "markov_persistence_v2a":
        z_receiver = z_gap.copy()
        q = np.full(length, config.rho_max * config.kappa * int(label))
    elif config.scenario == "joint_semimarkov_v2b":
        z_receiver = sample_equilibrium_alternating(
            state_rng, length, burst_pmf, normal_pmf, config.pi_burst
        )
        driver = (
            (1 - config.kappa * int(label)) * z_receiver
            + config.kappa * int(label) * z_gap
        )
        q = config.q_low + (config.q_high - config.q_low) * driver
    else:
        raise ValueError(f"unknown scenario {config.scenario}")
    categories = np.zeros(length, dtype=np.int64)
    repeats = np.zeros(length, dtype=np.int8)
    categories[0] = cat_rng.integers(config.n_receiver_categories)
    for position in range(1, length):
        repeat = cat_rng.random() < q[position]
        categories[position] = (
            categories[position - 1]
            if repeat
            else cat_rng.integers(config.n_receiver_categories)
        )
        repeats[position] = categories[position] == categories[position - 1]
    return z_receiver, categories, repeats


def _split(config: BenchmarkConfig, n: int, seed: int, split_id: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # label/length streams intentionally exclude scenario and kappa.
    shared = np.random.SeedSequence([seed, split_id])
    label_ss, length_ss = shared.spawn(2)
    labels = (np.random.default_rng(label_ss).random(n) < config.fraud_rate).astype(np.int64)
    lengths = np.random.default_rng(length_ss).integers(
        config.min_length, config.max_length + 1, n, dtype=np.int64
    )
    root = np.random.SeedSequence(
        [
            seed, split_id, 1 if config.scenario.endswith("v2b") else 0,
            int(round(100 * config.kappa)), config.randomness_nonce,
        ]
    )
    state_ss, emit_ss, cat_ss, amount_ss = root.spawn(4)
    state_rng, emit_rng = np.random.default_rng(state_ss), np.random.default_rng(emit_ss)
    cat_rng, amount_rng = np.random.default_rng(cat_ss), np.random.default_rng(amount_ss)
    max_l = config.max_length
    gaps = np.zeros((n, max_l), dtype=np.float32)
    cats = np.zeros((n, max_l, 1), dtype=np.int64)
    amounts = np.zeros((n, max_l, 1), dtype=np.float32)
    z_gap = np.zeros((n, max_l), dtype=np.int8)
    z_receiver = np.zeros((n, max_l), dtype=np.int8)
    repeats = np.zeros((n, max_l), dtype=np.int8)
    valid = np.arange(max_l)[None, :] < lengths[:, None]
    mu_b, pmf_b = calibrated_duration_pmf(
        config.duration_burst_mean, config.duration_sigma, config.duration_max
    )
    mu_n, pmf_n = calibrated_duration_pmf(
        config.duration_normal_mean, config.duration_sigma, config.duration_max
    )
    for i, (label, length) in enumerate(zip(labels, lengths)):
        length = int(length)
        if config.scenario == "markov_persistence_v2a":
            rho = config.rho_max * config.kappa * int(label)
            zg = _markov_state(state_rng, length, config.pi_burst, rho)
        elif config.scenario == "joint_semimarkov_v2b":
            zg = sample_equilibrium_alternating(
                state_rng, length, pmf_b, pmf_n, config.pi_burst
            )
        else:
            raise ValueError(f"unknown scenario {config.scenario}")
        zr, receiver_path, receiver_repeat = _sample_receiver_path(
            config, label=int(label), length=length, z_gap=zg,
            state_rng=state_rng, cat_rng=cat_rng,
            burst_pmf=pmf_b, normal_pmf=pmf_n,
        )
        z_gap[i, :length], z_receiver[i, :length] = zg, zr
        scales = np.where(zg, config.gap_burst_scale, config.gap_normal_scale)
        gaps[i, :length] = emit_rng.exponential(scales).astype(np.float32)
        cats[i, :length, 0] = receiver_path
        repeats[i, :length] = receiver_repeat
        amounts[i, :length, 0] = amount_rng.normal(
            config.amount_mean, config.amount_std, length
        ).astype(np.float32)
    latent = {
        "raw_gap": gaps,
        "gap_state": z_gap,
        "receiver_state": z_receiver,
        "receiver_repeat": repeats,
    }
    values = {
        "x_num": amounts,
        "x_cat": cats,
        "valid_mask": valid.astype(bool),
        "y_entity": labels,
        "lengths": lengths,
    }
    metadata = {
        "duration_mu_burst": mu_b,
        "duration_mu_normal": mu_n,
        "duration_pmf_burst": pmf_b.tolist(),
        "duration_pmf_normal": pmf_n.tolist(),
        "seed_spawn_keys": {
            "label": label_ss.spawn_key,
            "length": length_ss.spawn_key,
            "state": state_ss.spawn_key,
            "emission": emit_ss.spawn_key,
            "category": cat_ss.spawn_key,
            "amount": amount_ss.spawn_key,
        },
    }
    return values, latent | {"metadata": metadata}


def generate_receiver_only(
    config: BenchmarkConfig,
    *,
    n_entities: int,
    seed: int,
    split_id: int = 1,
) -> dict[str, np.ndarray]:
    """Lightweight MC using the exact production receiver/state code path."""
    shared = np.random.SeedSequence([seed, split_id])
    label_ss, length_ss = shared.spawn(2)
    labels = (
        np.random.default_rng(label_ss).random(n_entities) < config.fraud_rate
    ).astype(np.int64)
    lengths = np.random.default_rng(length_ss).integers(
        config.min_length, config.max_length + 1, n_entities, dtype=np.int64
    )
    root = np.random.SeedSequence(
        [
            seed, split_id, 1 if config.scenario.endswith("v2b") else 0,
            int(round(100 * config.kappa)), config.randomness_nonce,
        ]
    )
    state_ss, _, cat_ss, _ = root.spawn(4)
    state_rng, cat_rng = np.random.default_rng(state_ss), np.random.default_rng(cat_ss)
    _, pmf_b = calibrated_duration_pmf(
        config.duration_burst_mean, config.duration_sigma, config.duration_max
    )
    _, pmf_n = calibrated_duration_pmf(
        config.duration_normal_mean, config.duration_sigma, config.duration_max
    )
    categories = np.zeros((n_entities, config.max_length), dtype=np.int64)
    receiver_state = np.zeros((n_entities, config.max_length), dtype=np.int8)
    gap_state = np.zeros((n_entities, config.max_length), dtype=np.int8)
    repeats = np.zeros((n_entities, config.max_length), dtype=np.int8)
    for index, (label, length_value) in enumerate(zip(labels, lengths)):
        length = int(length_value)
        if config.scenario == "markov_persistence_v2a":
            rho = config.rho_max * config.kappa * int(label)
            z_gap = _markov_state(state_rng, length, config.pi_burst, rho)
        elif config.scenario == "joint_semimarkov_v2b":
            z_gap = sample_equilibrium_alternating(
                state_rng, length, pmf_b, pmf_n, config.pi_burst
            )
        else:
            raise ValueError(f"unknown scenario {config.scenario}")
        state, path, repeat = _sample_receiver_path(
            config, label=int(label), length=length, z_gap=z_gap,
            state_rng=state_rng, cat_rng=cat_rng,
            burst_pmf=pmf_b, normal_pmf=pmf_n,
        )
        gap_state[index, :length] = z_gap
        receiver_state[index, :length] = state
        categories[index, :length] = path
        repeats[index, :length] = repeat
    return {
        "categories": categories,
        "gap_state": gap_state,
        "receiver_state": receiver_state,
        "receiver_repeat": repeats,
        "labels": labels,
        "lengths": lengths,
        "valid_mask": np.arange(config.max_length)[None, :] < lengths[:, None],
    }


def generate_single_row_audit(
    config: BenchmarkConfig,
    *,
    n_entities: int,
    seed: int,
    split_id: int,
    position_seed: int,
) -> dict[str, np.ndarray]:
    """One production-path row per entity for cluster-independent AUROC audit."""
    if config.bin_edges is None:
        raise ValueError("fixed bin_edges are required")
    receiver = generate_receiver_only(
        config, n_entities=n_entities, seed=seed, split_id=split_id
    )
    root = np.random.SeedSequence(
        [
            seed,
            split_id,
            1 if config.scenario.endswith("v2b") else 0,
            int(round(100 * config.kappa)),
            config.randomness_nonce,
        ]
    )
    _, emit_ss, _, amount_ss = root.spawn(4)
    emit_rng = np.random.default_rng(emit_ss)
    amount_rng = np.random.default_rng(amount_ss)
    position_rng = np.random.default_rng(position_seed)
    features = np.empty((n_entities, 3), dtype=float)
    for index, length_value in enumerate(receiver["lengths"]):
        length = int(length_value)
        position = int(position_rng.integers(0, length))
        state = receiver["gap_state"][index, :length]
        scales = np.where(
            state, config.gap_burst_scale, config.gap_normal_scale
        )
        gaps = emit_rng.exponential(scales)
        amounts = amount_rng.normal(
            config.amount_mean, config.amount_std, length
        )
        features[index] = (
            amounts[position],
            apply_binning(
                np.asarray([gaps[position]]), np.asarray(config.bin_edges)
            )[0],
            receiver["categories"][index, position],
        )
    return {
        "features": features,
        "labels": receiver["labels"],
        "lengths": receiver["lengths"],
    }


def generate_benchmark(config: BenchmarkConfig, seed: int) -> DatasetBundle:
    train, train_latent = _split(config, config.n_train, seed, 0)
    test, test_latent = _split(config, config.n_test, seed + 100_000, 1)
    if config.bin_edges is None:
        edges, tau = fit_binning(
            train_latent["raw_gap"][train["valid_mask"]], config.n_gap_bins
        )
    else:
        edges = np.asarray(config.bin_edges)
        bins = apply_binning(
            train_latent["raw_gap"][train["valid_mask"]], edges
        )
        tau = np.array([
            np.median(train_latent["raw_gap"][train["valid_mask"]][bins == i])
            for i in range(len(edges) - 1)
        ])
    def batch(values: dict[str, np.ndarray], latent: dict[str, np.ndarray], prefix: str) -> SequenceBatch:
        dt = np.zeros_like(values["valid_mask"], dtype=np.int64)
        dt[values["valid_mask"]] = apply_binning(
            latent["raw_gap"][values["valid_mask"]], edges
        )
        return SequenceBatch(
            x_num=values["x_num"], dt_bin=dt, x_cat=values["x_cat"],
            valid_mask=values["valid_mask"], y_entity=values["y_entity"],
            lengths=values["lengths"],
            entity_ids=np.array([f"{prefix}-{i}" for i in range(len(values["lengths"]))]),
        )
    return DatasetBundle(
        train=batch(train, train_latent, "train"),
        test=batch(test, test_latent, "test"),
        metadata={
            "schema_version": config.schema_version,
            "scenario": config.scenario,
            "kappa": config.kappa,
            "bin_edges": edges.tolist(),
            "tau": tau.tolist(),
            "train_latent": train_latent,
            "test_latent": test_latent,
        },
    )


def generate_fixed_binning_split(
    config: BenchmarkConfig,
    *,
    n_entities: int,
    seed: int,
    split_id: int,
    prefix: str = "oracle",
) -> tuple[SequenceBatch, dict[str, np.ndarray]]:
    """Generate one fresh split without regenerating the full benchmark."""
    if config.bin_edges is None:
        raise ValueError("fixed bin_edges are required for an oracle split")
    values, latent = _split(config, n_entities, seed, split_id)
    edges = np.asarray(config.bin_edges)
    dt_bin = np.zeros_like(values["valid_mask"], dtype=np.int64)
    dt_bin[values["valid_mask"]] = apply_binning(
        latent["raw_gap"][values["valid_mask"]], edges
    )
    batch = SequenceBatch(
        x_num=values["x_num"],
        dt_bin=dt_bin,
        x_cat=values["x_cat"],
        valid_mask=values["valid_mask"],
        y_entity=values["y_entity"],
        lengths=values["lengths"],
        entity_ids=np.array([f"{prefix}-{index}" for index in range(n_entities)]),
    )
    return batch, latent
