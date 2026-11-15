from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from valor.models import Action


# ============================================================
# Errors
# ============================================================

class AgentEnvironmentError(RuntimeError):
    """
    Base exception for the synthetic real-Agent environment.
    """


class EnvironmentToolExecutionError(
    AgentEnvironmentError
):
    """
    A registered tool failed during execution.
    """


# ============================================================
# Environment state
# ============================================================

@dataclass
class AgentEnvironmentState:
    """
    Mutable synthetic environment state.

    The LLM may observe selected values from this state,
    but it does not directly modify the state.

    State changes occur only through registered tools.
    """

    power_level: float = 3.0

    actuator_position: float = 0.0

    network_scope: float = 0.0

    service_restart_intensity: float = 0.0

    service_status: str = "degraded"

    actuator_status: str = "underpowered"

    temperature: float = 45.0

    sensor_health: float = 0.80

    # --------------------------------------------------------
    # Trusted execution-time state used by the adapter.
    # --------------------------------------------------------

    effect_multiplier: float = 1.0

    force_hard_guard_failure: bool = False


# ============================================================
# Synthetic tool environment
# ============================================================

class AgentToolEnvironment:
    """
    Deterministic local environment for the real-LLM
    Agent integration experiment.

    Important security property:

        The LLM never receives a direct reference to these
        execution methods.

    The only intended path is:

        LLM
          ->
        ToolProposal
          ->
        ToolGateway
          ->
        VALOR / normal policy
          ->
        environment.execute()

    This makes ToolGateway the execution boundary.
    """

    def __init__(
        self,
        state: AgentEnvironmentState | None = None,
    ):
        if state is None:
            state = AgentEnvironmentState()

        self.state = state

    # ========================================================
    # Trusted state for VALOR adapter
    # ========================================================

    def adapter_state(
        self,
    ) -> dict[str, Any]:
        """
        Return trusted execution-time state consumed by
        SyntheticEffectAdapter.

        The Agent cannot supply these values.
        """

        return {
            "effect_multiplier":
                self.state.effect_multiplier,

            "force_hard_guard_failure":
                self.state.force_hard_guard_failure,
        }

    # ========================================================
    # Observation
    # ========================================================

    def observation(
        self,
    ) -> dict[str, Any]:
        """
        Return the current Agent-visible system state.

        Later Task classes can expose only selected portions
        of this observation if needed.
        """

        return {
            "power_level":
                self.state.power_level,

            "actuator_position":
                self.state.actuator_position,

            "network_scope":
                self.state.network_scope,

            "service_status":
                self.state.service_status,

            "actuator_status":
                self.state.actuator_status,

            "temperature":
                self.state.temperature,

            "sensor_health":
                self.state.sensor_health,
        }

    # ========================================================
    # State snapshot
    # ========================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:

        return deepcopy(
            self.observation()
        )

    # ========================================================
    # Tool execution
    # ========================================================

    def execute(
        self,
        action: Action,
    ) -> dict[str, Any]:
        """
        Execute one already-authorized exact Action.

        IMPORTANT:
            This method performs NO authorization itself.

            ToolGateway must authorize the Action before
            calling this method.
        """

        tool = action.tool
        value = float(
            action.parameter
        )

        # ----------------------------------------------------
        # READ SENSOR
        # ----------------------------------------------------

        if tool == "read_sensor":

            sensor_id = int(
                round(
                    value
                )
            )

            sensor_map = {
                0: {
                    "sensor":
                        "power_level",

                    "value":
                        self.state.power_level,
                },

                1: {
                    "sensor":
                        "temperature",

                    "value":
                        self.state.temperature,
                },

                2: {
                    "sensor":
                        "actuator_position",

                    "value":
                        self.state.actuator_position,
                },

                3: {
                    "sensor":
                        "sensor_health",

                    "value":
                        self.state.sensor_health,
                },
            }

            if sensor_id not in sensor_map:
                raise EnvironmentToolExecutionError(
                    f"Unknown sensor_id: "
                    f"{sensor_id}"
                )

            return {
                "tool":
                    tool,

                "status":
                    "ok",

                **sensor_map[
                    sensor_id
                ],
            }

        # ----------------------------------------------------
        # DIAGNOSE COMPONENT
        # ----------------------------------------------------

        if tool == "diagnose_component":

            depth = int(
                round(
                    value
                )
            )

            issues = []

            if (
                self.state.power_level
                < 4.0
            ):
                issues.append(
                    "insufficient_power"
                )

            if (
                self.state.service_status
                != "healthy"
            ):
                issues.append(
                    "service_degraded"
                )

            if (
                self.state.actuator_status
                != "healthy"
            ):
                issues.append(
                    self.state
                    .actuator_status
                )

            return {
                "tool":
                    tool,

                "status":
                    "ok",

                "diagnostic_depth":
                    depth,

                "issues":
                    issues,
            }

        # ----------------------------------------------------
        # RESTART SERVICE
        # ----------------------------------------------------

        if tool == "restart_service":

            self.state.service_restart_intensity = (
                value
            )

            if value >= 1.5:

                self.state.service_status = (
                    "healthy"
                )

            else:

                self.state.service_status = (
                    "partially_recovered"
                )

            return {
                "tool":
                    tool,

                "status":
                    "executed",

                "restart_intensity":
                    value,

                "service_status":
                    self.state
                    .service_status,
            }

        # ----------------------------------------------------
        # INCREASE POWER
        # ----------------------------------------------------

        if tool == "increase_power":

            self.state.power_level = (
                value
            )

            if value >= 5.0:

                self.state.actuator_status = (
                    "powered"
                )

            elif value >= 4.0:

                self.state.actuator_status = (
                    "marginal"
                )

            else:

                self.state.actuator_status = (
                    "underpowered"
                )

            return {
                "tool":
                    tool,

                "status":
                    "executed",

                "power_level":
                    self.state
                    .power_level,

                "actuator_status":
                    self.state
                    .actuator_status,
            }

        # ----------------------------------------------------
        # SWITCH NETWORK
        # ----------------------------------------------------

        if tool == "switch_network":

            self.state.network_scope = (
                value
            )

            return {
                "tool":
                    tool,

                "status":
                    "executed",

                "network_scope":
                    self.state
                    .network_scope,
            }

        # ----------------------------------------------------
        # MOVE ACTUATOR
        # ----------------------------------------------------

        if tool == "move_actuator":

            # Simple deterministic synthetic dynamics.
            self.state.actuator_position += (
                value
                * 0.10
            )

            if value >= 5.0:

                self.state.actuator_status = (
                    "released"
                )

            else:

                self.state.actuator_status = (
                    "moved"
                )

            return {
                "tool":
                    tool,

                "status":
                    "executed",

                "force":
                    value,

                "actuator_position":
                    self.state
                    .actuator_position,

                "actuator_status":
                    self.state
                    .actuator_status,
            }

        # ----------------------------------------------------
        # Unknown execution target
        # ----------------------------------------------------

        raise EnvironmentToolExecutionError(
            f"No environment executor is "
            f"registered for tool: {tool}"
        )


# ============================================================
# Smoke test
# ============================================================

def main() -> None:

    environment = (
        AgentToolEnvironment()
    )

    print(
        "Initial state:"
    )

    print(
        environment.observation()
    )

    result = (
        environment.execute(
            Action(
                tool="increase_power",
                parameter=6.2,
            )
        )
    )

    print()

    print(
        "Tool result:"
    )

    print(
        result
    )

    print()

    print(
        "Final state:"
    )

    print(
        environment.observation()
    )


if __name__ == "__main__":
    main()
