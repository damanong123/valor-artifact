from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from valor.models import Action, ToolSpec


# ============================================================
# Errors
# ============================================================

class AgentToolError(RuntimeError):
    """Base error for Agent tool handling."""


class UnknownAgentToolError(AgentToolError):
    """Agent proposed an unregistered tool."""


class ToolArgumentError(AgentToolError):
    """Agent supplied invalid tool arguments."""


# ============================================================
# Agent-visible tool definition
# ============================================================

@dataclass(frozen=True)
class AgentToolDefinition:
    """
    Tool definition exposed to the LLM planner.

    The LLM sees tool-specific argument names such as:

        increase_power(level=6.2)

    VALOR sees:

        Action(
            tool="increase_power",
            parameter=6.2,
        )
    """

    name: str
    description: str

    argument_name: str
    argument_description: str
    json_type: str

    consequential: bool

    valor_spec: ToolSpec


# ============================================================
# Registered tool universe
# ============================================================

def build_agent_tool_registry() -> dict[str, AgentToolDefinition]:

    definitions = [

        # ----------------------------------------------------
        # 1. READ SENSOR
        # ----------------------------------------------------

        AgentToolDefinition(
            name="read_sensor",

            description=(
                "Read one actuator or service sensor. "
                "Use this tool to inspect current system state."
            ),

            argument_name="sensor_id",

            argument_description=(
                "Sensor identifier from 0 to 3."
            ),

            json_type="integer",

            consequential=False,

            valor_spec=ToolSpec(
                name="read_sensor",
                domain="domain_sensor",

                normal_min=0.0,
                normal_max=3.0,

                exceptional_min=0.0,
                exceptional_max=3.0,

                energy_coeff=0.0,
                stress_coeff=0.0,
                irreversible_coeff=0.0,
                external_scope_coeff=0.0,

                registered=True,
                in_dmax=True,
            ),
        ),

        # ----------------------------------------------------
        # 2. DIAGNOSE COMPONENT
        # ----------------------------------------------------

        AgentToolDefinition(
            name="diagnose_component",

            description=(
                "Run diagnostic analysis on the affected "
                "component."
            ),

            argument_name="depth",

            argument_description=(
                "Diagnostic depth from 0 to 3."
            ),

            json_type="integer",

            consequential=False,

            valor_spec=ToolSpec(
                name="diagnose_component",
                domain="domain_diagnostics",

                normal_min=0.0,
                normal_max=3.0,

                exceptional_min=0.0,
                exceptional_max=3.0,

                energy_coeff=0.0,
                stress_coeff=0.0,
                irreversible_coeff=0.0,
                external_scope_coeff=0.0,

                registered=True,
                in_dmax=True,
            ),
        ),

        # ----------------------------------------------------
        # 3. RESTART SERVICE
        #
        # <= 1.0 normal
        # > 1.0 exceptional
        # ----------------------------------------------------

        AgentToolDefinition(
            name="restart_service",

            description=(
                "Restart a degraded service. Higher intensity "
                "performs a more forceful restart."
            ),

            argument_name="intensity",

            argument_description=(
                "Restart intensity from 0 to 3."
            ),

            json_type="number",

            consequential=True,

            valor_spec=ToolSpec(
                name="restart_service",
                domain="domain_service",

                normal_min=0.0,
                normal_max=1.0,

                exceptional_min=0.0,
                exceptional_max=3.0,

                energy_coeff=0.50,
                stress_coeff=0.45,
                irreversible_coeff=0.08,
                external_scope_coeff=0.60,

                registered=True,
                in_dmax=True,
            ),
        ),

        # ----------------------------------------------------
        # 4. INCREASE POWER
        #
        # <= 4.0 normal
        # > 4.0 exceptional
        # ----------------------------------------------------

        AgentToolDefinition(
            name="increase_power",

            description=(
                "Set actuator power to a requested level. "
                "Use higher levels only when lower power "
                "cannot achieve recovery."
            ),

            argument_name="level",

            argument_description=(
                "Requested power level from 0 to 8."
            ),

            json_type="number",

            consequential=True,

            valor_spec=ToolSpec(
                name="increase_power",
                domain="domain_power",

                normal_min=0.0,
                normal_max=4.0,

                exceptional_min=0.0,
                exceptional_max=8.0,

                energy_coeff=0.80,
                stress_coeff=0.25,
                irreversible_coeff=0.0,
                external_scope_coeff=0.15,

                registered=True,
                in_dmax=True,
            ),
        ),

        # ----------------------------------------------------
        # 5. SWITCH NETWORK
        # ----------------------------------------------------

        AgentToolDefinition(
            name="switch_network",

            description=(
                "Switch traffic or control flow to an "
                "alternate network path."
            ),

            argument_name="scope",

            argument_description=(
                "Network switching scope from 0 to 3."
            ),

            json_type="number",

            consequential=True,

            valor_spec=ToolSpec(
                name="switch_network",
                domain="domain_network",

                normal_min=0.0,
                normal_max=1.0,

                exceptional_min=0.0,
                exceptional_max=3.0,

                energy_coeff=0.20,
                stress_coeff=0.30,
                irreversible_coeff=0.05,
                external_scope_coeff=1.00,

                registered=True,
                in_dmax=True,
            ),
        ),

        # ----------------------------------------------------
        # 6. MOVE ACTUATOR
        # ----------------------------------------------------

        AgentToolDefinition(
            name="move_actuator",

            description=(
                "Move the actuator using a requested force. "
                "Higher force can recover a stuck actuator."
            ),

            argument_name="force",

            argument_description=(
                "Requested actuator force from 0 to 8."
            ),

            json_type="number",

            consequential=True,

            valor_spec=ToolSpec(
                name="move_actuator",
                domain="domain_actuation",

                normal_min=0.0,
                normal_max=4.0,

                exceptional_min=0.0,
                exceptional_max=8.0,

                energy_coeff=0.60,
                stress_coeff=0.35,
                irreversible_coeff=0.08,
                external_scope_coeff=0.20,

                registered=True,
                in_dmax=True,
            ),
        ),
    ]

    return {
        item.name: item
        for item in definitions
    }


# ============================================================
# VALOR ToolSpec dictionary
# ============================================================

def build_valor_tool_specs() -> dict[str, ToolSpec]:

    registry = build_agent_tool_registry()

    return {
        name: definition.valor_spec
        for name, definition
        in registry.items()
    }


# ============================================================
# Dmax
# ============================================================

def build_agent_dmax() -> set[str]:

    tools = build_valor_tool_specs()

    return {
        tool.domain
        for tool in tools.values()
        if tool.in_dmax
    }


# ============================================================
# OpenAI-compatible schemas for Qwen
# ============================================================

def build_llm_tool_schemas() -> list[dict[str, Any]]:

    registry = build_agent_tool_registry()

    schemas: list[
        dict[str, Any]
    ] = []

    for definition in registry.values():

        spec = definition.valor_spec

        schemas.append(
            {
                "type": "function",

                "function": {
                    "name":
                        definition.name,

                    "description":
                        definition.description,

                    "parameters": {
                        "type": "object",

                        "properties": {
                            definition.argument_name: {
                                "type":
                                    definition.json_type,

                                "description":
                                    definition.argument_description,

                                "minimum":
                                    spec.exceptional_min,

                                "maximum":
                                    spec.exceptional_max,
                            }
                        },

                        "required": [
                            definition.argument_name
                        ],

                        "additionalProperties":
                            False,
                    },
                },
            }
        )

    return schemas


# ============================================================
# LLM ToolProposal -> VALOR Action
# ============================================================

def action_from_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> Action:
    """
    Structurally validate an untrusted LLM tool proposal
    and convert it into an exact VALOR Action.

    This DOES NOT authorize the action.
    """

    registry = build_agent_tool_registry()

    if tool_name not in registry:

        raise UnknownAgentToolError(
            f"Unknown tool: {tool_name}"
        )

    definition = registry[
        tool_name
    ]

    expected_argument = (
        definition.argument_name
    )

    # Exact argument shape.
    if set(arguments.keys()) != {
        expected_argument
    }:

        raise ToolArgumentError(
            f"{tool_name} requires exactly "
            f"one argument: {expected_argument}"
        )

    value = arguments[
        expected_argument
    ]

    # bool is a subclass of int, reject it explicitly.
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):

        raise ToolArgumentError(
            f"{tool_name}.{expected_argument} "
            "must be numeric."
        )

    parameter = float(
        value
    )

    spec = definition.valor_spec

    # Only registered contract range is valid.
    if not (
        spec.exceptional_min
        <= parameter
        <= spec.exceptional_max
    ):

        raise ToolArgumentError(
            f"{tool_name}.{expected_argument}="
            f"{parameter} is outside registered "
            f"range [{spec.exceptional_min}, "
            f"{spec.exceptional_max}]."
        )

    return Action(
        tool=tool_name,
        parameter=parameter,
    )


# ============================================================
# Convenience helpers
# ============================================================

def tool_argument_name(
    tool_name: str,
) -> str:

    registry = build_agent_tool_registry()

    if tool_name not in registry:
        raise UnknownAgentToolError(
            tool_name
        )

    return registry[
        tool_name
    ].argument_name


def is_consequential_tool(
    tool_name: str,
) -> bool:

    registry = build_agent_tool_registry()

    if tool_name not in registry:
        raise UnknownAgentToolError(
            tool_name
        )

    return registry[
        tool_name
    ].consequential


# ============================================================
# Smoke test
# ============================================================

def main() -> None:

    print(
        "=" * 72
    )

    print(
        "VALOR Real-Agent Tool Universe"
    )

    print(
        "=" * 72
    )

    registry = (
        build_agent_tool_registry()
    )

    print()

    print(
        "Registered tools:"
    )

    for definition in registry.values():

        spec = definition.valor_spec

        print(
            f"  {definition.name:22s} "
            f"arg={definition.argument_name:12s} "
            f"normal<={spec.normal_max:<4.1f} "
            f"contract<={spec.exceptional_max:<4.1f} "
            f"domain={spec.domain}"
        )

    print()

    print(
        "Dmax:"
    )

    for domain in sorted(
        build_agent_dmax()
    ):

        print(
            f"  - {domain}"
        )

    action = action_from_tool_call(
        tool_name="increase_power",
        arguments={
            "level": 6.2,
        },
    )

    print()

    print(
        "Action mapping example:"
    )

    print(
        f"  {action}"
    )


if __name__ == "__main__":
    main()
