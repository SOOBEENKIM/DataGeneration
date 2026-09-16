"""Paired prevalence views of frozen controlled data, preserving split identity.

The new observed labels use the production label random stream. When a former
null entity becomes active, only kappa=1 marks are resampled using its original
gap regime. Realized regimes are used here by the DGP, never by the model.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def production_label_uniforms(entity_ids, *, generation_seed=42):
    indices = np.array([int(str(x).removeprefix("controlled-")) for x in entity_ids])
    if len(set(indices)) != len(indices) or np.any(indices < 0):
        raise ValueError("invalid production entity identity")
    label_seed = np.random.SeedSequence([generation_seed, 0]).spawn(2)[0]
    rng = np.random.default_rng(label_seed)
    result = np.empty(len(indices))
    previous = -1
    for position in np.argsort(indices):
        index = int(indices[position])
        # Do not even reconstruct draws for sealed entity IDs.
        rng.bit_generator.advance(index-previous-1)
        result[position] = rng.random()
        previous = index
    return result


def active_mark_path(entity_id, gap_state, *, seed, category_count, q_low, q_high):
    digest = hashlib.sha256(f"cs-saf-prevalence-v1:{seed}:{entity_id}".encode()).digest()
    rng = np.random.default_rng(np.frombuffer(digest, dtype="<u4"))
    coins = rng.random(len(gap_state))
    fresh = rng.integers(category_count, size=len(gap_state))
    result = fresh.copy()
    for t in range(1, len(result)):
        q = q_low+(q_high-q_low)*int(gap_state[t])
        if coins[t] < q:
            result[t] = result[t-1]
    return result


def prevalence_view(static, events, oracle, *, prevalence, kappa,
                    generation_seed=42, mark_seed=20260930, category_count=64,
                    q_low=.05, q_high=.90):
    if prevalence not in (.05, .10, .25, .50) or kappa not in (0, 1):
        raise ValueError("unregistered prevalence or kappa")
    static = static.sort_values("entity_id").reset_index(drop=True).copy()
    events = events.sort_values(["entity_id", "event_index"]).reset_index(drop=True).copy()
    oracle = oracle.sort_values(["entity_id", "event_index"]).reset_index(drop=True)
    identity = ["entity_id", "event_id", "event_index"]
    if not events[identity].equals(oracle[identity]):
        raise ValueError("oracle and observed identities differ")
    if not oracle.gap_state.isin([0, 1]).all():
        raise ValueError("invalid DGP regime")
    uniforms = production_label_uniforms(static.entity_id, generation_seed=generation_seed)
    original = static.entity_label.to_numpy(dtype=int)
    if not np.array_equal(original, (uniforms < .05).astype(int)):
        raise ValueError("original label stream does not reproduce the frozen data")
    new_labels = (uniforms < prevalence).astype(int)
    promoted = set(static.loc[(new_labels == 1) & (original == 0), "entity_id"])
    if kappa == 1 and promoted:
        for entity, indices in events.groupby("entity_id", sort=False).indices.items():
            if entity in promoted:
                indices = np.asarray(indices)
                path = active_mark_path(entity, oracle.gap_state.to_numpy()[indices],
                                        seed=mark_seed, category_count=category_count,
                                        q_low=q_low, q_high=q_high)
                events.loc[indices, "receiver_or_mark"] = [f"receiver-{x}" for x in path]
    static["entity_label"] = new_labels
    return static, events, {"promoted_entities": len(promoted),
                            "regenerated_mark_entities": len(promoted) if kappa else 0}
