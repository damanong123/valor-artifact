from __future__ import annotations

from valor.adapter import SyntheticEffectAdapter
from valor.budget import (
    EFFECT_DIMENSIONS,
    vector_add,
    vector_leq,
    vector_subtract,
    zero_vector,
)
from valor.models import Action


def normalized_exposure(
    effect: dict[str, float],
    reserve: dict[str, float],
) -> float:
    """
    Report exposure relative to the full VALOR one-session
    reserve.

    We use the maximum normalized effect dimension:

        max_d effect[d] / B0[d]

    Thus:
        1.0 means at least one protected dimension has
        consumed the entire VALOR session budget.
    """

    values = []

    for dimension in EFFECT_DIMENSIONS:

        budget = reserve[
            dimension
        ]

        used = effect[
            dimension
        ]

        if budget == 0:
            if used > 0:
                return float("inf")

            continue

        values.append(
            used / budget
        )

    if not values:
        return 0.0

    return max(values)


def action_score(
    action: Action,
    *,
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
) -> float:
    """
    How much of the normalized one-session reserve would this
    action consume?
    """

    effect = (
        adapter
        .contract(action)
        .effect_vector()
    )

    return normalized_exposure(
        effect,
        reserve,
    )


def union_false_trigger_exposure(
    *,
    candidate_actions: list[Action],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
    num_requests: int,
) -> tuple[
    dict[str, float],
    float,
    list[Action],
]:
    """
    Repeated false-trigger attack against union break-glass.

    There is no cumulative reserve.

    The adversary repeatedly submits the currently strongest
    exposed request.
    """

    if not candidate_actions:
        return (
            zero_vector(),
            0.0,
            [],
        )

    strongest = max(
        candidate_actions,
        key=lambda action:
            action_score(
                action,
                adapter=adapter,
                reserve=reserve,
            ),
    )

    total = zero_vector()

    requests: list[Action] = []

    effect = (
        adapter
        .contract(strongest)
        .effect_vector()
    )

    for _ in range(
        num_requests
    ):
        total = vector_add(
            total,
            effect,
        )

        requests.append(
            strongest
        )

    ratio = normalized_exposure(
        total,
        reserve,
    )

    return (
        total,
        ratio,
        requests,
    )


def valor_false_trigger_exposure(
    *,
    candidate_actions: list[Action],
    adapter: SyntheticEffectAdapter,
    reserve: dict[str, float],
    num_requests: int,
) -> tuple[
    dict[str, float],
    float,
    list[Action],
]:
    """
    Repeated false-trigger attack against VALOR.

    Every accepted request consumes typed reserve first.

    The adversary greedily chooses the feasible request with
    the largest normalized effect.
    """

    remaining = dict(
        reserve
    )

    total = zero_vector()

    accepted: list[Action] = []

    for _ in range(
        num_requests
    ):

        feasible: list[
            tuple[
                float,
                Action,
                dict[str, float],
            ]
        ] = []

        for action in candidate_actions:

            effect = (
                adapter
                .contract(action)
                .effect_vector()
            )

            if not vector_leq(
                effect,
                remaining,
            ):
                continue

            score = normalized_exposure(
                effect,
                reserve,
            )

            feasible.append(
                (
                    score,
                    action,
                    effect,
                )
            )

        if not feasible:
            break

        _, action, effect = max(
            feasible,
            key=lambda item:
                item[0],
        )

        remaining = vector_subtract(
            remaining,
            effect,
        )

        total = vector_add(
            total,
            effect,
        )

        accepted.append(
            action
        )

    ratio = normalized_exposure(
        total,
        reserve,
    )

    return (
        total,
        ratio,
        accepted,
    )