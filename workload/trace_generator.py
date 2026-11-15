from __future__ import annotations

from typing import Iterable

import numpy as np

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.budget import (
    vector_add,
    vector_leq,
    zero_vector,
)

from valor.models import (
    Action,
    ToolSpec,
    Trace,
)

from valor.normal_policy import (
    is_normally_authorized,
)


# ============================================================
# Errors
# ============================================================

class TraceGenerationError(
    RuntimeError
):
    """
    Raised when a feasible synthetic trace cannot be
    generated within the configured attempt budget.
    """


# ============================================================
# Effect accounting helpers
# ============================================================

def cumulative_effect(
    actions: Iterable[Action],
    adapter: SyntheticEffectAdapter,
) -> dict[str, float]:
    """
    Calculate cumulative declared effect:

        sum_i b(a_i)

    for a sequence of exact actions.
    """

    total = zero_vector()

    for action in actions:

        contract = adapter.contract(
            action
        )

        total = vector_add(
            total,
            contract.effect_vector(),
        )

    return total

def cumulative_exceptional_effect(
    actions: Iterable[Action],
    *,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
) -> dict[str, float]:
    """
    Sum only effects of actions that require
    exceptional authorization.

    Normally authorized actions do not consume
    VALOR's exceptional reserve.
    """

    total = zero_vector()

    for action in actions:

        if is_normally_authorized(
            action,
            tools,
        ):
            continue

        contract = adapter.contract(
            action
        )

        total = vector_add(
            total,
            contract.effect_vector(),
        )

    return total


def trace_is_effect_feasible(
    trace: Trace,
    *,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
) -> bool:
    """
    Check whether the exceptional portion of the
    trace fits VALOR's typed reserve B0.
    """

    total = cumulative_exceptional_effect(
        trace.actions,
        tools=tools,
        adapter=adapter,
    )

    return vector_leq(
        total,
        reserve,
    )

def trace_has_exceptional_action(
    trace: Trace,
    *,
    tools: dict[str, ToolSpec],
) -> bool:
    """
    Return True iff at least one action is not
    normally authorized.
    """

    return any(
        not is_normally_authorized(
            action,
            tools,
        )
        for action in trace.actions
    )


# ============================================================
# Parameter sampling
# ============================================================

def _sample_normal_parameter(
    rng: np.random.Generator,
    tool: ToolSpec,
) -> float:
    """
    Sample an action inside normal authorization.
    """

    return float(
        rng.uniform(
            tool.normal_min,
            tool.normal_max,
        )
    )


def _sample_exceptional_parameter(
    rng: np.random.Generator,
    tool: ToolSpec,
) -> float:
    """
    Sample strictly above the normal range while remaining
    inside the adapter's exceptional contract range.

    This gives:

        N(a, x) = 0

    while the action can still be evaluated by VALOR.
    """

    gap = (
        tool.exceptional_max
        - tool.normal_max
    )

    if gap <= 0:
        raise TraceGenerationError(
            f"Tool {tool.name} has no exceptional range."
        )

    # Avoid sampling numerically too close to normal_max.
    lower = (
        tool.normal_max
        + 0.05 * gap
    )

    return float(
        rng.uniform(
            lower,
            tool.exceptional_max,
        )
    )


# ============================================================
# Single trace generation
# ============================================================

def generate_one_feasible_trace(
    *,
    rng: np.random.Generator,
    trace_id: str,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
    min_actions: int = 2,
    max_actions: int = 5,
    exceptional_probability: float = 0.45,
    max_attempts: int = 10_000,
) -> Trace:
    """
    Generate one feasible synthetic emergency trace.

    Requirements:

    1. Number of actions is between min_actions and
       max_actions.

    2. Action instances are distinct.
       For simplicity and stronger separation, this
       implementation uses distinct tools within a trace.

    3. At least one action is outside normal permission.

    4. Every action uses a registered tool inside Dmax.

    5. Every action remains inside the tool adapter's
       certified exceptional range.

    6. Cumulative typed effect fits B0.
    """

    if min_actions < 1:
        raise ValueError(
            "min_actions must be >= 1"
        )

    if max_actions < min_actions:
        raise ValueError(
            "max_actions must be >= min_actions"
        )

    if not (
        0.0
        <= exceptional_probability
        <= 1.0
    ):
        raise ValueError(
            "exceptional_probability must be in [0, 1]"
        )

    # VALOR-feasible workload only uses tools whose
    # protected domains belong to Dmax.
    allowed_tools = [
        tool
        for tool in tools.values()
        if (
            tool.registered
            and tool.in_dmax
        )
    ]

    if len(allowed_tools) < max_actions:
        raise TraceGenerationError(
            "Not enough allowed tools to construct "
            "distinct-action traces."
        )

    for _ in range(max_attempts):

        num_actions = int(
            rng.integers(
                min_actions,
                max_actions + 1,
            )
        )

        # Distinct tools -> distinct action instances.
        selected_indices = (
            rng.choice(
                len(allowed_tools),
                size=num_actions,
                replace=False,
            )
        )

        selected_tools = [
            allowed_tools[int(i)]
            for i in selected_indices
        ]

        # One action is forced to be exceptional.
        forced_exceptional_position = int(
            rng.integers(
                0,
                num_actions,
            )
        )

        actions: list[Action] = []

        for position, tool in enumerate(
            selected_tools
        ):

            if (
                position
                == forced_exceptional_position
            ):
                parameter = (
                    _sample_exceptional_parameter(
                        rng,
                        tool,
                    )
                )

            else:

                make_exceptional = (
                    rng.random()
                    < exceptional_probability
                )

                if make_exceptional:

                    parameter = (
                        _sample_exceptional_parameter(
                            rng,
                            tool,
                        )
                    )

                else:

                    parameter = (
                        _sample_normal_parameter(
                            rng,
                            tool,
                        )
                    )

            actions.append(
                Action(
                    tool=tool.name,
                    parameter=parameter,
                )
            )

        trace = Trace(
            trace_id=trace_id,
            actions=tuple(actions),
        )

        # ----------------------------------------------------
        # Sanity:
        # at least one exceptional action
        # ----------------------------------------------------

        if not trace_has_exceptional_action(
            trace,
            tools=tools,
        ):
            continue

        # ----------------------------------------------------
        # Complete trace must fit B0
        # ----------------------------------------------------

        if not trace_is_effect_feasible(
            trace,
            tools=tools,
            adapter=adapter,
            reserve=reserve,
        ):
            continue

        return trace

    raise TraceGenerationError(
        f"Unable to generate feasible trace "
        f"{trace_id} after {max_attempts} attempts."
    )


# ============================================================
# Complete workload generation
# ============================================================

def generate_feasible_traces(
    *,
    seed: int,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
    num_traces: int = 1200,
    min_actions: int = 2,
    max_actions: int = 5,
    exceptional_probability: float = 0.45,
) -> list[Trace]:
    """
    Generate a deterministic feasible workload for one seed.

    Same seed + same tool universe + same configuration
    produces the same trace workload.
    """

    if num_traces <= 0:
        raise ValueError(
            "num_traces must be positive"
        )

    # Use a deterministic derived seed so the trace RNG
    # stream is distinct from the tool-generator RNG stream.
    trace_seed = (
        seed
        + 1_000_003
    )

    rng = np.random.default_rng(
        trace_seed
    )

    traces: list[Trace] = []

    for trace_index in range(
        num_traces
    ):

        trace_id = (
            f"seed_{seed}_"
            f"trace_{trace_index:04d}"
        )

        trace = (
            generate_one_feasible_trace(
                rng=rng,

                trace_id=trace_id,

                tools=tools,

                adapter=adapter,

                reserve=reserve,

                min_actions=min_actions,

                max_actions=max_actions,

                exceptional_probability=(
                    exceptional_probability
                ),
            )
        )

        traces.append(
            trace
        )

    return traces