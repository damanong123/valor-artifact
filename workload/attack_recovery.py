from __future__ import annotations

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.budget import (
    vector_add,
    vector_leq,
    vector_subtract,
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

from workload.adversary import (
    normalized_exposure,
)


# ============================================================
# Attack candidate space
# ============================================================

def build_attack_candidates(
    tools: dict[str, ToolSpec],
) -> list[Action]:
    """
    Build the full VALOR exceptional-envelope attack space
    used by the reserve-exhaustion experiment.

    One maximally parameterized exceptional action is created
    for every registered tool inside Dmax.

    This is intentionally stronger than sampling arbitrary
    planner requests: the attacker chooses from the complete
    registered exceptional envelope.
    """

    return [
        Action(
            tool=tool.name,
            parameter=(
                tool.exceptional_max
            ),
        )
        for tool in tools.values()
        if (
            tool.registered
            and tool.in_dmax
        )
    ]


# ============================================================
# False-trigger reserve-exhaustion attack
# ============================================================

def apply_greedy_attack(
    *,
    requested_steps: int,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
) -> dict[str, object]:
    """
    Spend exceptional reserve using repeated false-trigger
    requests.

    At each step the adversary chooses, among requests that
    still fit the remaining typed reserve, the action with
    the largest normalized effect relative to B0.

    Each attack action is charged at its full declared effect.

    This corresponds to a conservative/worst-case settlement:
    no unused reservation is refunded.

    Important:
        The attacker may EXHAUST authority but can never
        manufacture authority beyond B0.
    """

    if requested_steps < 0:
        raise ValueError(
            "requested_steps cannot be negative"
        )

    candidates = (
        build_attack_candidates(
            tools
        )
    )

    remaining = dict(
        reserve
    )

    total_effect = (
        zero_vector()
    )

    accepted_actions = []

    for _ in range(
        requested_steps
    ):

        feasible = []

        for action in candidates:

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

            if not vector_leq(
                effect,
                remaining,
            ):
                continue

            score = (
                normalized_exposure(
                    effect,
                    reserve,
                )
            )

            feasible.append(
                (
                    score,
                    action,
                    effect,
                )
            )

        # No further exceptional request fits.
        if not feasible:
            break

        (
            _,
            selected_action,
            selected_effect,
        ) = max(
            feasible,
            key=lambda item:
                item[0],
        )

        remaining = (
            vector_subtract(
                remaining,
                selected_effect,
            )
        )

        total_effect = (
            vector_add(
                total_effect,
                selected_effect,
            )
        )

        accepted_actions.append(
            selected_action
        )

    exposure_x = (
        normalized_exposure(
            total_effect,
            reserve,
        )
    )

    return {
        "requested_steps":
            requested_steps,

        "accepted_steps":
            len(
                accepted_actions
            ),

        "accepted_actions":
            tuple(
                accepted_actions
            ),

        "attack_effect":
            total_effect,

        "remaining_reserve":
            remaining,

        "attack_exposure_x":
            exposure_x,
    }


# ============================================================
# Legitimate recovery admission
# ============================================================

def authorize_recovery_trace(
    trace: Trace,
    *,
    tools: dict[str, ToolSpec],
    adapter: SyntheticEffectAdapter,
    dmax: set[str],
    remaining_reserve:
        dict[str, float],
) -> tuple[
    bool,
    str,
]:
    """
    Evaluate a legitimate recovery trace against the reserve
    remaining AFTER the attack.

    Normal actions do not consume exceptional reserve.

    Exceptional actions require:
        D subset Dmax
        hard guard
        effect <= current remaining reserve

    Returns:
        (admitted, reason)
    """

    remaining = dict(
        remaining_reserve
    )

    for action in trace.actions:

        # ----------------------------------------------------
        # Normal path
        # ----------------------------------------------------

        if is_normally_authorized(
            action,
            tools,
        ):
            continue

        # ----------------------------------------------------
        # Exceptional path
        # ----------------------------------------------------

        contract = (
            adapter
            .contract(
                action
            )
        )

        if not (
            contract
            .hard_guard_ok
        ):
            return (
                False,
                "hard_guard",
            )

        if not (
            contract
            .domains
            .issubset(
                dmax
            )
        ):
            return (
                False,
                "outside_dmax",
            )

        effect = (
            contract
            .effect_vector()
        )

        if not vector_leq(
            effect,
            remaining,
        ):
            return (
                False,
                "insufficient_remaining_reserve",
            )

        remaining = (
            vector_subtract(
                remaining,
                effect,
            )
        )

    return (
        True,
        "admit",
    )