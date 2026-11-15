from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.budget import (
    EFFECT_DIMENSIONS,
    vector_leq,
    vector_subtract,
)

from valor.models import (
    Action,
    ToolSpec,
)

from valor.normal_policy import (
    is_normally_authorized,
)


# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class CumulativeExhaustionCase:
    """
    Construction-guaranteed cumulative-exhaustion case.

    Every individual action is admissible against the full
    initial reserve B0.

    However, after a feasible prefix has consumed reserve,
    the final action no longer fits the remaining reserve.

    Therefore rejection is caused only by cumulative prior
    exceptional spend.
    """

    case_id: str

    actions: tuple[
        Action,
        ...
    ]

    expected_reject_index: int

    binding_dimension: str

    repeat_capacity: int


# ============================================================
# Exceptional parameter
# ============================================================

def _exceptional_parameter(
    tool: ToolSpec,
    fraction: float,
) -> float:
    """
    Produce a parameter strictly above normal_max while
    remaining inside exceptional_max.

    fraction:
        0 < fraction <= 1
    """

    if not (
        0.0
        < fraction
        <= 1.0
    ):
        raise ValueError(
            "fraction must be in (0, 1]"
        )

    gap = (
        tool.exceptional_max
        - tool.normal_max
    )

    if gap <= 0:

        raise RuntimeError(
            f"Tool {tool.name} has no "
            "exceptional parameter region."
        )

    return (
        tool.normal_max
        + gap
        * fraction
    )


# ============================================================
# How many repetitions fit?
# ============================================================

def _repeat_capacity(
    *,
    effect: dict[str, float],
    reserve: dict[str, float],
) -> tuple[
    int,
    str,
]:
    """
    Calculate the maximum number n of identical actions that
    can fit cumulatively inside B0.

    For every non-zero dimension:

        n_d = floor(B0[d] / effect[d])

    Overall capacity:

        n = min_d n_d

    Then:

        n copies fit
        n + 1 copies do not fit

    provided the action itself is individually feasible.
    """

    capacities = []

    for dimension in (
        EFFECT_DIMENSIONS
    ):

        value = effect[
            dimension
        ]

        budget = reserve[
            dimension
        ]

        if value <= 0:
            continue

        capacity = math.floor(
            (
                budget
                + 1e-12
            )
            / value
        )

        capacities.append(
            (
                capacity,
                dimension,
            )
        )

    if not capacities:

        raise RuntimeError(
            "Action has zero effect in all dimensions."
        )

    capacity, dimension = min(
        capacities,
        key=lambda item:
            item[0],
    )

    return (
        int(
            capacity
        ),
        dimension,
    )


# ============================================================
# Candidate validation
# ============================================================

def _candidate_is_valid(
    *,
    action: Action,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
) -> tuple[
    bool,
    dict[str, float] | None,
    int | None,
    str | None,
]:
    """
    Validate that this exact action can be used to construct:

        feasible prefix
        +
        one cumulatively infeasible final action
    """

    # Must require exceptional authorization.
    if is_normally_authorized(
        action,
        tools,
    ):
        return (
            False,
            None,
            None,
            None,
        )

    contract = (
        adapter
        .contract(
            action
        )
    )

    effect = (
        contract
        .effect_vector()
    )

    # Individual action must fit the entire B0.
    if not vector_leq(
        effect,
        reserve,
    ):
        return (
            False,
            None,
            None,
            None,
        )

    (
        capacity,
        binding_dimension,
    ) = _repeat_capacity(
        effect=effect,
        reserve=reserve,
    )

    # At least one action must fit first.
    if capacity < 1:

        return (
            False,
            None,
            None,
            None,
        )

    # --------------------------------------------------------
    # Verify construction sequentially.
    #
    # First capacity actions must fit.
    # The next identical action must fail.
    # --------------------------------------------------------

    remaining = dict(
        reserve
    )

    for _ in range(
        capacity
    ):

        if not vector_leq(
            effect,
            remaining,
        ):
            return (
                False,
                None,
                None,
                None,
            )

        remaining = (
            vector_subtract(
                remaining,
                effect,
            )
        )

    # One more must NOT fit.
    if vector_leq(
        effect,
        remaining,
    ):
        return (
            False,
            None,
            None,
            None,
        )

    return (
        True,
        effect,
        capacity,
        binding_dimension,
    )


# ============================================================
# Candidate universe
# ============================================================

def build_cumulative_candidates(
    *,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
) -> list[
    tuple[
        Action,
        dict[str, float],
        int,
        str,
    ]
]:
    """
    Build a reusable pool of construction-guaranteed
    cumulative-exhaustion actions.

    Multiple exceptional magnitudes are considered for every
    registered Dmax tool.
    """

    candidates = []

    fractions = (
        0.25,
        0.40,
        0.55,
        0.70,
        0.85,
        1.00,
    )

    for tool in tools.values():

        if not (
            tool.registered
            and tool.in_dmax
        ):
            continue

        for fraction in fractions:

            action = Action(
                tool=tool.name,

                parameter=(
                    _exceptional_parameter(
                        tool,
                        fraction,
                    )
                ),
            )

            (
                valid,
                effect,
                capacity,
                dimension,
            ) = (
                _candidate_is_valid(
                    action=action,

                    tools=tools,

                    adapter=adapter,

                    reserve=reserve,
                )
            )

            if not valid:
                continue

            candidates.append(
                (
                    action,
                    effect,
                    capacity,
                    dimension,
                )
            )

    if not candidates:

        raise RuntimeError(
            "No valid cumulative-exhaustion "
            "candidates were found."
        )

    return candidates


# ============================================================
# Case generation
# ============================================================

def generate_cumulative_exhaustion_cases(
    *,
    seed: int,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
    num_cases: int = 300,
) -> list[
    CumulativeExhaustionCase
]:
    """
    Generate deterministic cumulative-exhaustion cases.

    For a candidate with repeat capacity n:

        actions = (a, a, ..., a)
                    n + 1 times

    Expected:

        first n actions accepted

        action at zero-based index n rejected

    while a itself remains individually admissible against B0.
    """

    if num_cases <= 0:

        raise ValueError(
            "num_cases must be positive"
        )

    candidates = (
        build_cumulative_candidates(
            tools=tools,

            adapter=adapter,

            reserve=reserve,
        )
    )

    rng = np.random.default_rng(
        seed
        + 3_000_071
    )

    cases = []

    for case_index in range(
        num_cases
    ):

        candidate_index = int(
            rng.integers(
                0,
                len(
                    candidates
                ),
            )
        )

        (
            action,
            _,
            capacity,
            binding_dimension,
        ) = candidates[
            candidate_index
        ]

        actions = tuple(
            action
            for _ in range(
                capacity
                + 1
            )
        )

        cases.append(
            CumulativeExhaustionCase(
                case_id=(
                    f"seed_{seed}_"
                    f"cumex_"
                    f"{case_index:04d}"
                ),

                actions=actions,

                # Zero-based index.
                expected_reject_index=(
                    capacity
                ),

                binding_dimension=(
                    binding_dimension
                ),

                repeat_capacity=(
                    capacity
                ),
            )
        )

    return cases