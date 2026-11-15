from __future__ import annotations

import math


EFFECT_DIMENSIONS = (
    "energy",
    "stress",
    "irreversible",
    "external_scope",
)


class InvalidEffectVectorError(
    ValueError
):
    pass


def validate_effect_vector(
    vector: dict[str, float],
) -> None:
    """
    Validate a typed effect vector.

    Every known dimension must:
    - exist
    - be finite
    - be non-negative
    """

    for dimension in EFFECT_DIMENSIONS:

        if dimension not in vector:
            raise InvalidEffectVectorError(
                f"Missing effect dimension: "
                f"{dimension}"
            )

        value = float(
            vector[dimension]
        )

        if not math.isfinite(value):
            raise InvalidEffectVectorError(
                f"{dimension} must be finite"
            )

        if value < 0:
            raise InvalidEffectVectorError(
                f"{dimension} cannot "
                f"be negative"
            )


def vector_leq(
    left: dict[str, float],
    right: dict[str, float],
    epsilon: float = 1e-12,
) -> bool:
    """
    Component-wise comparison:

        left <= right
    """

    validate_effect_vector(left)
    validate_effect_vector(right)

    return all(
        left[d]
        <= right[d] + epsilon

        for d in EFFECT_DIMENSIONS
    )


def vector_add(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:

    validate_effect_vector(left)
    validate_effect_vector(right)

    return {
        d: (
            left[d]
            + right[d]
        )
        for d in EFFECT_DIMENSIONS
    }


def vector_subtract(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """
    Return left - right.

    Raises if the result would become
    meaningfully negative.
    """

    validate_effect_vector(left)
    validate_effect_vector(right)

    result = {}

    for d in EFFECT_DIMENSIONS:

        value = (
            left[d]
            - right[d]
        )

        if value < -1e-12:
            raise InvalidEffectVectorError(
                f"Negative result for {d}: "
                f"{value}"
            )

        # Avoid tiny floating point negatives.
        result[d] = max(
            0.0,
            value,
        )

    return result


def zero_vector() -> dict[str, float]:

    return {
        d: 0.0
        for d in EFFECT_DIMENSIONS
    }