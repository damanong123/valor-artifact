from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# ============================================================
# Data models
# ============================================================

@dataclass(frozen=True)
class ToolProposal:
    """
    Exact tool call proposed by the LLM.

    This is only a proposal.
    It is NOT an authorization.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentDecision:
    """
    Result of one LLM planning turn.

    assistant_message:
        Exact OpenAI-compatible assistant message that can be
        appended to the conversation before tool results.
    """

    content: str | None

    tool_proposals: tuple[
        ToolProposal,
        ...
    ]

    finish_reason: str | None

    assistant_message: dict[str, Any]


# ============================================================
# Errors
# ============================================================

class AgentError(RuntimeError):
    pass


class AgentConnectionError(
    AgentError
):
    pass


class AgentToolParseError(
    AgentError
):
    pass


# ============================================================
# Local LLM Agent
# ============================================================

class LocalLLMAgent:
    """
    Real LLM planner backed by local vLLM.

    Security boundary:

        LLM proposes
              ↓
        ToolProposal

    The Agent does NOT:

        - execute tools;
        - modify reserve;
        - choose Dmax;
        - compute trusted effects;
        - mint capabilities;
        - decide authorization.
    """

    def __init__(
        self,
        *,
        base_url: str = (
            "http://127.0.0.1:8000/v1"
        ),
        model: str = "qwen3-8b",
        api_key: str = "local-vllm",
        temperature: float = 0.7,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        timeout_seconds: float = 120.0,
    ):
        if not base_url:
            raise ValueError(
                "base_url cannot be empty"
            )

        if not model:
            raise ValueError(
                "model cannot be empty"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be positive"
            )

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = (
            enable_thinking
        )

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    # ========================================================
    # Health check
    # ========================================================

    def health_check(
        self,
    ) -> list[str]:

        try:
            response = (
                self.client
                .models
                .list()
            )

        except Exception as exc:
            raise AgentConnectionError(
                "Unable to reach local "
                f"LLM server at {self.base_url}: "
                f"{exc}"
            ) from exc

        return [
            model.id
            for model in response.data
        ]

    # ========================================================
    # Planning
    # ========================================================

    def propose(
        self,
        *,
        messages: list[
            dict[str, Any]
        ],
        tools: list[
            dict[str, Any]
        ],
        tool_choice: str = "auto",
    ) -> AgentDecision:
        """
        Ask the LLM for its next step.

        The returned ToolProposal objects remain untrusted
        until ToolGateway evaluates them.
        """

        if not messages:
            raise ValueError(
                "messages cannot be empty"
            )

        try:
            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model,

                    messages=messages,

                    tools=tools,

                    tool_choice=(
                        tool_choice
                    ),

                    temperature=(
                        self.temperature
                    ),

                    max_tokens=(
                        self.max_tokens
                    ),

                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking":
                                self.enable_thinking
                        }
                    },
                )
            )

        except Exception as exc:
            raise AgentConnectionError(
                "Local LLM inference failed: "
                f"{exc}"
            ) from exc

        if not response.choices:
            raise AgentError(
                "LLM returned no choices."
            )

        choice = response.choices[
            0
        ]

        message = choice.message

        proposals: list[
            ToolProposal
        ] = []

        # ----------------------------------------------------
        # Parse structured tool calls
        # ----------------------------------------------------

        for tool_call in (
            message.tool_calls
            or []
        ):
            function = (
                tool_call.function
            )

            raw_arguments = (
                function.arguments
            )

            try:
                arguments = json.loads(
                    raw_arguments
                )

            except json.JSONDecodeError as exc:
                raise AgentToolParseError(
                    "Malformed arguments for "
                    f"{function.name}: "
                    f"{raw_arguments}"
                ) from exc

            if not isinstance(
                arguments,
                dict,
            ):
                raise AgentToolParseError(
                    "Tool arguments must be "
                    "a JSON object."
                )

            proposals.append(
                ToolProposal(
                    call_id=tool_call.id,
                    tool_name=function.name,
                    arguments=arguments,
                )
            )

        # ----------------------------------------------------
        # Preserve assistant message exactly enough for
        # subsequent tool-result turns.
        # ----------------------------------------------------

        assistant_message = (
            message.model_dump(
                exclude_none=True
            )
        )

        return AgentDecision(
            content=message.content,

            tool_proposals=tuple(
                proposals
            ),

            finish_reason=(
                choice.finish_reason
            ),

            assistant_message=(
                assistant_message
            ),
        )


# ============================================================
# Smoke test
# ============================================================

def main() -> None:

    agent = LocalLLMAgent()

    print(
        "Models:"
    )

    print(
        agent.health_check()
    )


if __name__ == "__main__":
    main()
