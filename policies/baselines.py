from __future__ import annotations

from dataclasses import dataclass

from valor.adapter import SyntheticEffectAdapter
from valor.budget import (
    EFFECT_DIMENSIONS,
    vector_add,
    vector_leq,
    vector_subtract,
    zero_vector,
)
from valor.models import Action, ToolSpec, Trace
from valor.normal_policy import is_normally_authorized


# ============================================================
# Exact catalog
# ============================================================

class ExactCatalogPolicy:
    """
    Known contingency profiles authorize only the exact
    exceptional action instances that appeared in the catalog.
    """

    def __init__(
        self,
        *,
        tools: dict[str, ToolSpec],
        known_profiles: list[Trace],
    ):
        self.tools = tools

        self.allowed_exceptional_actions: set[
            Action
        ] = set()

        for trace in known_profiles:

            for action in trace.actions:

                if not is_normally_authorized(
                    action,
                    tools,
                ):
                    self.allowed_exceptional_actions.add(
                        action
                    )

    def authorize_action(
        self,
        action: Action,
    ) -> bool:

        if is_normally_authorized(
            action,
            self.tools,
        ):
            return True

        return (
            action
            in self.allowed_exceptional_actions
        )

    def authorize_trace(
        self,
        trace: Trace,
    ) -> bool:

        return all(
            self.authorize_action(
                action
            )
            for action in trace.actions
        )


# ============================================================
# Union break-glass
# ============================================================

class UnionBreakGlassPolicy:
    """
    Catalog widening baseline.

    For each tool that appeared exceptionally in a known
    contingency profile, the policy accumulates an allowed
    exceptional parameter range.

    Since our synthetic parameter is one-dimensional and
    positive, this reproduction stores:

        max exceptional parameter observed for each tool.

    An exceptional request is authorized when:

        parameter <= maximum known emergency parameter

    for that tool.

    This is intentionally a standing emergency scope and
    contains no cumulative reserve.
    """

    def __init__(
        self,
        *,
        tools: dict[str, ToolSpec],
        known_profiles: list[Trace],
    ):
        self.tools = tools

        self.allowed_max: dict[
            str,
            float,
        ] = {}

        for trace in known_profiles:

            for action in trace.actions:

                if is_normally_authorized(
                    action,
                    tools,
                ):
                    continue

                previous = self.allowed_max.get(
                    action.tool
                )

                if (
                    previous is None
                    or action.parameter > previous
                ):
                    self.allowed_max[
                        action.tool
                    ] = action.parameter

    def authorize_action(
        self,
        action: Action,
    ) -> bool:

        # Normal policy remains available.
        if is_normally_authorized(
            action,
            self.tools,
        ):
            return True

        if action.tool not in self.allowed_max:
            return False

        return (
            action.parameter
            <= self.allowed_max[
                action.tool
            ]
        )

    def authorize_trace(
        self,
        trace: Trace,
    ) -> bool:

        return all(
            self.authorize_action(
                action
            )
            for action in trace.actions
        )

    def exceptional_attack_candidates(
        self,
    ) -> list[Action]:
        """
        Return strongest currently exposed action for each
        catalog-covered tool.
        """

        return [
            Action(
                tool=tool_name,
                parameter=max_parameter,
            )
            for (
                tool_name,
                max_parameter,
            ) in self.allowed_max.items()
        ]


# ============================================================
# Catalog + Reserve
# ============================================================

class CatalogReservePolicy:
    """
    RQ2 ablation:

        Union catalog scope
        +
        same typed cumulative reserve as VALOR.

    The reserve bounds cumulative spend, but the policy still
    cannot instantiate an exceptional action outside the
    catalog-derived scope.
    """

    def __init__(
        self,
        *,
        tools: dict[str, ToolSpec],
        known_profiles: list[Trace],
        adapter: SyntheticEffectAdapter,
        reserve: dict[str, float],
    ):
        self.tools = tools
        self.adapter = adapter

        self.union_policy = UnionBreakGlassPolicy(
            tools=tools,
            known_profiles=known_profiles,
        )

        self.initial_reserve = dict(
            reserve
        )

    def authorize_trace(
        self,
        trace: Trace,
    ) -> bool:

        remaining = dict(
            self.initial_reserve
        )

        for action in trace.actions:

            if is_normally_authorized(
                action,
                self.tools,
            ):
                continue

            if not self.union_policy.authorize_action(
                action
            ):
                return False

            effect = (
                self.adapter
                .contract(action)
                .effect_vector()
            )

            if not vector_leq(
                effect,
                remaining,
            ):
                return False

            remaining = vector_subtract(
                remaining,
                effect,
            )

        return True


# ============================================================
# Envelope only
# ============================================================

class EnvelopeOnlyPolicy:
    """
    RQ2 ablation:

    An exceptional action may be instantiated whenever it is
    inside Dmax and its hard guard passes.

    There is deliberately NO cumulative reserve state.
    """

    def __init__(
        self,
        *,
        tools: dict[str, ToolSpec],
        adapter: SyntheticEffectAdapter,
        dmax: set[str],
    ):
        self.tools = tools
        self.adapter = adapter
        self.dmax = set(dmax)

    def authorize_action(
        self,
        action: Action,
    ) -> bool:

        if is_normally_authorized(
            action,
            self.tools,
        ):
            return True

        contract = self.adapter.contract(
            action
        )

        return (
            contract.hard_guard_ok
            and
            contract.domains.issubset(
                self.dmax
            )
        )

    def authorize_trace(
        self,
        trace: Trace,
    ) -> bool:

        return all(
            self.authorize_action(
                action
            )
            for action in trace.actions
        )


# ============================================================
# VALOR trace admission helper
# ============================================================

class ValorTracePolicy:
    """
    Lightweight trace-level realization of VALOR admission
    for RQ1/RQ2 coverage experiments.

    Normal actions do not consume exceptional reserve.

    Exceptional actions require:
        D subset Dmax
        hard guard
        cumulative typed reserve
    """

    def __init__(
        self,
        *,
        tools: dict[str, ToolSpec],
        adapter: SyntheticEffectAdapter,
        dmax: set[str],
        reserve: dict[str, float],
    ):
        self.tools = tools
        self.adapter = adapter
        self.dmax = set(dmax)

        self.initial_reserve = dict(
            reserve
        )

    def authorize_trace(
        self,
        trace: Trace,
    ) -> bool:

        remaining = dict(
            self.initial_reserve
        )

        for action in trace.actions:

            if is_normally_authorized(
                action,
                self.tools,
            ):
                continue

            contract = self.adapter.contract(
                action
            )

            if not contract.hard_guard_ok:
                return False

            if not contract.domains.issubset(
                self.dmax
            ):
                return False

            effect = (
                contract.effect_vector()
            )

            if not vector_leq(
                effect,
                remaining,
            ):
                return False

            remaining = vector_subtract(
                remaining,
                effect,
            )

        return True