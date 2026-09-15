from __future__ import annotations

from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


METADATA_FLOAT_ENCODING_SCHEMA = (
    "benchmark-v2.5-provenance-float-sequence-v1"
)
FLOAT_TAG = "__benchmark_v2_5_float__"
PROVENANCE_FIELDS = ("bin_edges", "tau")


def _encode_float_sequence(values: Sequence[Any]) -> list[Any]:
    encoded: list[Any] = []
    for value in values:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError("invalid v2.5 provenance float value")
        number = float(value)
        if np.isnan(number):
            raise ValueError("NaN is forbidden in v2.5 frozen metadata")
        if np.isneginf(number):
            encoded.append({FLOAT_TAG: "-Infinity"})
        elif np.isposinf(number):
            encoded.append({FLOAT_TAG: "+Infinity"})
        else:
            encoded.append(number)
    return encoded


def _decode_float_sequence(
    values: Sequence[Any],
    *,
    tagged_encoding_declared: bool,
) -> list[float]:
    decoded: list[float] = []
    for value in values:
        if isinstance(value, Mapping):
            if not tagged_encoding_declared:
                raise ValueError(
                    "tagged float encoding schema is required"
                )
            if set(value) != {FLOAT_TAG}:
                raise ValueError("invalid v2.5 tagged float representation")
            token = value[FLOAT_TAG]
            if token == "-Infinity":
                decoded.append(float("-inf"))
            elif token == "+Infinity":
                decoded.append(float("inf"))
            else:
                raise ValueError("unknown v2.5 tagged float token")
        elif isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
            if np.isnan(number):
                raise ValueError("NaN is forbidden in v2.5 frozen metadata")
            if not np.isfinite(number):
                raise ValueError(
                    "non-finite metadata must use the explicit tagged encoding"
                )
            decoded.append(number)
        else:
            raise ValueError("invalid v2.5 provenance float value")
    return decoded


def encode_frozen_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Encode only provenance float sequences for strict JSON output."""
    missing = [field for field in PROVENANCE_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"frozen metadata is missing {missing}")
    encoded = dict(metadata)
    for field in PROVENANCE_FIELDS:
        encoded[field] = _encode_float_sequence(metadata[field])
    encoded["provenance_float_encoding"] = {
        "schema_version": METADATA_FLOAT_ENCODING_SCHEMA,
        "fields": list(PROVENANCE_FIELDS),
        "representation": {
            "negative_infinity": {FLOAT_TAG: "-Infinity"},
            "positive_infinity": {FLOAT_TAG: "+Infinity"},
        },
    }
    return encoded


def decode_frozen_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Decode tagged provenance floats while rejecting NaN and bare infinity."""
    encoding = metadata.get("provenance_float_encoding")
    if encoding is not None:
        if (
            not isinstance(encoding, Mapping)
            or encoding.get("schema_version")
            != METADATA_FLOAT_ENCODING_SCHEMA
            or encoding.get("fields") != list(PROVENANCE_FIELDS)
        ):
            raise ValueError("unsupported v2.5 provenance float encoding")
    decoded = dict(metadata)
    for field in PROVENANCE_FIELDS:
        if field in metadata:
            decoded[field] = _decode_float_sequence(
                metadata[field],
                tagged_encoding_declared=encoding is not None,
            )
    return decoded
