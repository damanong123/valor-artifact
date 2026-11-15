from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from workload.trace_generator import (
    generate_feasible_traces,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.models import (
    Action,
    ToolSpec,
)

from valor.normal_policy import (
    is_normally_authorized,
)


# ============================================================
# Stateful trace used by RQ5
# ============================================================

@dataclass(frozen=True)
class NegativeControlCase:
    """
    One RQ5 admission case.

    states[i] is the trusted synthetic state used when
    evaluating actions[i].

    mode:
        feasible
        outside_dmax
        exceeds_initial_reserve
    """

    case_id: str

    actions: tuple[
        Action,
        ...
    ]

    states: tuple[
        dict | None,
        ...
    ]

    mode: str

    expected_admit: bool

    target_index: int | None = None

    target_dimension: str | None = None

    target_multiplier: float | None = None


# ============================================================
# Parameter helpers
# ============================================================

def _normal_parameter(
    tool: ToolSpec,
) -> float:
    """
    Deterministic normal parameter used for filler actions.
    """

    return (
        tool.normal_min
        +
        (
            tool.normal_max
            - tool.normal_min
        )
        * 0.5
    )


def _exceptional_parameter(
    tool: ToolSpec,
) -> float:
    """
    Deterministic exceptional parameter strictly above
    normal_max but below exceptional_max.
    """

    return (
        tool.normal_max
        +
        (
            tool.exceptional_max
            - tool.normal_max
        )
        * 0.5
    )


# ============================================================
# Feasible class
# ============================================================

def generate_feasible_cases(
    *,
    seed: int,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
    num_cases: int = 600,
) -> list[NegativeControlCase]:
    """
    Reuse the same feasible workload construction used by
    RQ1/RQ2.

    Every action is evaluated under the default synthetic
    state.
    """

    traces = generate_feasible_traces(
        seed=seed,
        tools=tools,
        adapter=adapter,
        reserve=reserve,
        num_traces=num_cases,
        min_actions=2,
        max_actions=5,
    )

    cases = []

    for index, trace in enumerate(
        traces
    ):

        cases.append(
            NegativeControlCase(
                case_id=(
                    f"seed_{seed}_"
                    f"feasible_{index:04d}"
                ),

                actions=trace.actions,

                states=tuple(
                    None
                    for _ in trace.actions
                ),

                mode="feasible",

                expected_admit=True,
            )
        )

    return cases


# ============================================================
# Outside-Dmax negative control
# ============================================================

def _generate_outside_dmax_case(
    *,
    rng: np.random.Generator,
    seed: int,
    case_index: int,
    tools: dict[str, ToolSpec],
) -> NegativeControlCase:
    """
    Construct a trace containing exactly one exceptional
    action through the permanently excluded tool.

    Other actions are normal allowed actions so the intended
    rejection reason is isolated to Dmax.
    """

    excluded_tools = [
        tool
        for tool in tools.values()
        if (
            tool.registered
            and not tool.in_dmax
        )
    ]

    if not excluded_tools:
        raise RuntimeError(
            "RQ5 requires at least one "
            "registered tool outside Dmax."
        )

    excluded_tool = (
        excluded_tools[0]
    )

    allowed_tools = [
        tool
        for tool in tools.values()
        if (
            tool.registered
            and tool.in_dmax
        )
    ]

    num_actions = int(
        rng.integers(
            2,
            6,
        )
    )

    target_index = int(
        rng.integers(
            0,
            num_actions,
        )
    )

    filler_count = (
        num_actions
        - 1
    )

    filler_indices = (
        rng.choice(
            len(allowed_tools),
            size=filler_count,
            replace=False,
        )
    )

    fillers = [
        allowed_tools[
            int(i)
        ]
        for i in filler_indices
    ]

    actions = []

    states = []

    filler_position = 0

    for position in range(
        num_actions
    ):

        if position == target_index:

            action = Action(
                tool=excluded_tool.name,

                parameter=(
                    _exceptional_parameter(
                        excluded_tool
                    )
                ),
            )

        else:

            tool = fillers[
                filler_position
            ]

            filler_position += 1

            action = Action(
                tool=tool.name,

                parameter=(
                    _normal_parameter(
                        tool
                    )
                ),
            )

        actions.append(
            action
        )

        states.append(
            None
        )

    return NegativeControlCase(
        case_id=(
            f"seed_{seed}_"
            f"outside_dmax_"
            f"{case_index:04d}"
        ),

        actions=tuple(
            actions
        ),

        states=tuple(
            states
        ),

        mode="outside_dmax",

        expected_admit=False,

        target_index=target_index,

        target_dimension=None,

        target_multiplier=None,
    )


# ============================================================
# Reserve-exceeding negative control
# ============================================================

def _generate_reserve_exceeding_case(
    *,
    rng: np.random.Generator,
    seed: int,
    case_index: int,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
) -> NegativeControlCase:
    """
    Construct an action whose DECLARED effect exceeds the
    ENTIRE initial reserve B0 in at least one dimension.

    We keep the exact action parameter inside the adapter's
    certified action range.

    Because the effect contract is state-dependent in the
    formal model b(a, x), this reproduction analytically
    chooses a synthetic execution-state multiplier M such
    that:

        b_k(a, x) > B0[k]

    for at least one protected dimension k.

    This makes the action infeasible regardless of prior
    reserve consumption.
    """

    allowed_tools = [
        tool
        for tool in tools.values()
        if (
            tool.registered
            and tool.in_dmax
        )
    ]

    # Randomized deterministic ordering.
    candidate_indices = (
        rng.permutation(
            len(allowed_tools)
        )
    )

    selected_tool = None
    selected_action = None
    selected_dimension = None
    selected_multiplier = None

    for raw_index in (
        candidate_indices
    ):

        tool = allowed_tools[
            int(raw_index)
        ]

        action = Action(
            tool=tool.name,

            parameter=(
                _exceptional_parameter(
                    tool
                )
            ),
        )

        base_effect = (
            adapter
            .contract(
                action
            )
            .effect_vector()
        )

        thresholds = []

        for dimension, budget in (
            reserve.items()
        ):

            effect = (
                base_effect[
                    dimension
                ]
            )

            if effect <= 0:
                continue

            threshold = (
                budget
                / effect
            )

            thresholds.append(
                (
                    threshold,
                    dimension,
                )
            )

        if not thresholds:
            continue

        # Smallest multiplier needed to exceed any one
        # initial reserve dimension.
        (
            threshold,
            dimension,
        ) = min(
            thresholds,
            key=lambda item:
                item[0],
        )

        # 5% beyond the analytical boundary.
        multiplier = (
            threshold
            * 1.05
        )

        selected_tool = tool
        selected_action = action
        selected_dimension = (
            dimension
        )
        selected_multiplier = (
            multiplier
        )

        break

    if selected_action is None:
        raise RuntimeError(
            "Unable to construct a reserve-"
            "exceeding RQ5 action."
        )

    num_actions = int(
        rng.integers(
            2,
            6,
        )
    )

    target_index = int(
        rng.integers(
            0,
            num_actions,
        )
    )

    filler_pool = [
        tool
        for tool in allowed_tools
        if tool.name
        != selected_tool.name
    ]

    filler_indices = (
        rng.choice(
            len(filler_pool),
            size=(
                num_actions
                - 1
            ),
            replace=False,
        )
    )

    fillers = [
        filler_pool[
            int(i)
        ]
        for i in filler_indices
    ]

    actions = []

    states = []

    filler_position = 0

    for position in range(
        num_actions
    ):

        if position == target_index:

            actions.append(
                selected_action
            )

            states.append(
                {
                    "effect_multiplier":
                        selected_multiplier
                }
            )

        else:

            tool = fillers[
                filler_position
            ]

            filler_position += 1

            actions.append(
                Action(
                    tool=tool.name,

                    parameter=(
                        _normal_parameter(
                            tool
                        )
                    ),
                )
            )

            states.append(
                None
            )

    return NegativeControlCase(
        case_id=(
            f"seed_{seed}_"
            f"exceeds_b0_"
            f"{case_index:04d}"
        ),

        actions=tuple(
            actions
        ),

        states=tuple(
            states
        ),

        mode=(
            "exceeds_initial_reserve"
        ),

        expected_admit=False,

        target_index=(
            target_index
        ),

        target_dimension=(
            selected_dimension
        ),

        target_multiplier=float(
            selected_multiplier
        ),
    )


# ============================================================
# Complete infeasible class
# ============================================================

def generate_infeasible_cases(
    *,
    seed: int,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
    num_cases: int = 600,
) -> list[NegativeControlCase]:
    """
    Generate construction-guaranteed infeasible cases.

    Approximately half:
        outside_dmax

    Approximately half:
        exceeds_initial_reserve
    """

    if num_cases <= 0:
        raise ValueError(
            "num_cases must be positive"
        )

    rng = np.random.default_rng(
        seed
        + 2_000_033
    )

    outside_count = (
        num_cases
        // 2
    )

    reserve_count = (
        num_cases
        - outside_count
    )

    cases = []

    for index in range(
        outside_count
    ):

        cases.append(
            _generate_outside_dmax_case(
                rng=rng,

                seed=seed,

                case_index=index,

                tools=tools,
            )
        )

    for index in range(
        reserve_count
    ):

        cases.append(
            _generate_reserve_exceeding_case(
                rng=rng,

                seed=seed,

                case_index=index,

                tools=tools,

                adapter=adapter,

                reserve=reserve,
            )
        )

    # Shuffle deterministically so the two negative modes
    # are not stored in two contiguous blocks.
    permutation = (
        rng.permutation(
            len(cases)
        )
    )

    return [
        cases[
            int(i)
        ]
        for i in permutation
    ]