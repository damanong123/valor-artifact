from __future__ import annotations

import math

from valor.models import (
    Action,
    EffectContract,
    ToolSpec,
)


# ============================================================
# Adapter errors
# ============================================================

class AdapterError(Exception):
    """
    Base exception for effect-adapter failures.
    """


class UnknownToolError(AdapterError):
    pass


class UnregisteredToolError(AdapterError):
    pass


class ParameterOutOfContractError(AdapterError):
    pass


class InvalidParameterError(AdapterError):
    pass


class InvalidStateError(AdapterError):
    pass


# ============================================================
# Synthetic effect adapter
# ============================================================

class SyntheticEffectAdapter:
    """
    Trusted synthetic effect adapter.

    Maps:

        a = (tool, parameter)

    and local state x into:

        χ(a, x) = <D, b, q, rho>

    Synthetic state currently supports:

        effect_multiplier

    Example:

        state = {
            "effect_multiplier": 1.5
        }

    means that the same action has 1.5x effect
    at the current state.

    This lets us reproduce execution-time state drift
    without requiring a physical simulator.
    """

    def __init__(
        self,
        tools: dict[str, ToolSpec],
    ):
        self.tools = tools

    # --------------------------------------------------------
    # State helper
    # --------------------------------------------------------

    @staticmethod
    def _effect_multiplier(
        state: dict | None,
    ) -> float:

        if state is None:
            return 1.0

        value = float(
            state.get(
                "effect_multiplier",
                1.0,
            )
        )

        if not math.isfinite(value):
            raise InvalidStateError(
                "effect_multiplier must be finite"
            )

        if value < 0:
            raise InvalidStateError(
                "effect_multiplier cannot be negative"
            )

        return value

    # --------------------------------------------------------
    # Effect contract
    # --------------------------------------------------------

    def contract(
        self,
        action: Action,
        state: dict | None = None,
    ) -> EffectContract:

        # --------------------------------------------------
        # 1. Validate action parameter
        # --------------------------------------------------

        if not math.isfinite(
            action.parameter
        ):
            raise InvalidParameterError(
                "Action parameter must be finite"
            )

        # --------------------------------------------------
        # 2. Tool must exist
        # --------------------------------------------------

        if action.tool not in self.tools:
            raise UnknownToolError(
                f"Unknown tool: {action.tool}"
            )

        tool = self.tools[
            action.tool
        ]

        # --------------------------------------------------
        # 3. Tool must be registered
        # --------------------------------------------------

        if not tool.registered:
            raise UnregisteredToolError(
                f"Tool is not registered: "
                f"{action.tool}"
            )

        # --------------------------------------------------
        # 4. Action must remain inside certified
        #    adapter operating domain
        # --------------------------------------------------

        if not (
            tool.exceptional_min
            <= action.parameter
            <= tool.exceptional_max
        ):
            raise ParameterOutOfContractError(
                f"Parameter "
                f"{action.parameter} "
                f"is outside certified range "
                f"[{tool.exceptional_min}, "
                f"{tool.exceptional_max}]"
            )

        # --------------------------------------------------
        # 5. State-dependent synthetic effect
        # --------------------------------------------------

        multiplier = (
            self._effect_multiplier(
                state
            )
        )

        magnitude = abs(
            action.parameter
        )

        energy = (
            tool.energy_coeff
            * magnitude
            * multiplier
        )

        stress = (
            tool.stress_coeff
            * magnitude
            * multiplier
        )

        irreversible = (
            tool.irreversible_coeff
            * magnitude
            * multiplier
        )

        external_scope = (
            tool.external_scope_coeff
            * magnitude
            * multiplier
        )

        # --------------------------------------------------
        # 6. Hard state predicate q(a, x)
        # --------------------------------------------------

        hard_guard_ok = True

        if state is not None:

            if state.get(
                "force_hard_guard_failure",
                False,
            ):
                hard_guard_ok = False

        # --------------------------------------------------
        # 7. Protected domain
        # --------------------------------------------------

        domains = frozenset(
            {
                tool.domain
            }
        )

        return EffectContract(
            domains=domains,

            energy=float(
                energy
            ),

            stress=float(
                stress
            ),

            irreversible=float(
                irreversible
            ),

            external_scope=float(
                external_scope
            ),

            hard_guard_ok=(
                hard_guard_ok
            ),

            receipt_class=(
                "synthetic"
            ),
        )