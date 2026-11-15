from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent_integration.agent import ToolProposal
from agent_integration.catalog import (
    FrozenEmergencyCatalog,
    build_frozen_pilot_catalog,
)
from agent_integration.environment import AgentToolEnvironment
from agent_integration.tools import (
    action_from_tool_call,
    build_agent_dmax,
    build_valor_tool_specs,
)

from valor.adapter import SyntheticEffectAdapter
from valor.kernel import ValorKernel
from valor.ledger import InMemoryReserveLedger
from valor.models import Action
from valor.normal_policy import is_normally_authorized


# ============================================================
# Policy modes
# ============================================================

POLICY_NORMAL_ONLY = "normal_only"
POLICY_CATALOG = "catalog"
POLICY_VALOR = "valor"

VALID_POLICIES = {
    POLICY_NORMAL_ONLY,
    POLICY_CATALOG,
    POLICY_VALOR,
}


# ============================================================
# Default constitution
# ============================================================

DEFAULT_INITIAL_RESERVE = {
    "energy": 40.0,
    "stress": 25.0,
    "irreversible": 3.0,
    "external_scope": 18.0,
}


# ============================================================
# Gateway result
# ============================================================

@dataclass(frozen=True)
class GatewayResult:
    """
    Result of one LLM-generated tool proposal.

    status:
        executed
        denied
        execution_error

    authorization_path:
        normal
        catalog
        valor
        denied
    """

    policy: str

    status: str

    authorization_path: str

    reason: str | None

    action: Action | None

    normal_authorized: bool | None

    catalog_authorized: bool | None

    tool_result: dict[str, Any] | None

    reserved_effect: dict[str, float] | None

    realized_effect: dict[str, float] | None

    reserve_before: dict[str, float]

    reserve_after: dict[str, float]

    capability_nonce: str | None


# ============================================================
# Tool Gateway
# ============================================================

class ToolGateway:
    """
    Trusted enforcement layer between the untrusted LLM
    planner and registered tools.

    Three experiment policies are supported:

        normal_only
            normal policy only

        catalog
            normal policy
            +
            frozen break-glass catalog

        valor
            normal policy
            +
            VALOR exceptional authorization

    Important:

        The Agent never chooses the policy.

        The experiment harness creates a Gateway with a
        fixed policy before the run starts.
    """

    def __init__(
        self,
        *,
        environment: AgentToolEnvironment,
        policy: str,
        initial_reserve: dict[str, float],
        capability_key: bytes,
        catalog: FrozenEmergencyCatalog | None = None,
        epoch: int = 1,
        capability_ttl_seconds: float = 60.0,
    ):
        if policy not in VALID_POLICIES:
            raise ValueError(
                f"Unknown policy: {policy}"
            )

        self.environment = environment

        self.policy = policy

        self.tools = build_valor_tool_specs()

        self.dmax = build_agent_dmax()

        self.catalog = (
            catalog
            if catalog is not None
            else build_frozen_pilot_catalog()
        )

        self.adapter = SyntheticEffectAdapter(
            self.tools
        )

        self.ledger = InMemoryReserveLedger(
            initial_reserve
        )

        self.kernel = ValorKernel(
            adapter=self.adapter,
            ledger=self.ledger,
            dmax=self.dmax,
            capability_key=capability_key,
            epoch=epoch,
            capability_ttl_seconds=(
                capability_ttl_seconds
            ),
        )

        self.initial_reserve = deepcopy(
            initial_reserve
        )

    # ========================================================
    # Accounting
    # ========================================================

    def remaining_reserve(
        self,
    ) -> dict[str, float]:

        return (
            self.ledger
            .remaining_reserve()
        )

    def committed_spend(
        self,
    ) -> dict[str, float]:

        return (
            self.ledger
            .committed_spend()
        )

    def conservation_holds(
        self,
    ) -> bool:

        return (
            self.ledger
            .conservation_holds()
        )

    # ========================================================
    # Direct execution path
    # ========================================================

    def _execute_direct(
        self,
        *,
        action: Action,
        authorization_path: str,
        reserve_before: dict[str, float],
        normal_authorized: bool,
        catalog_authorized: bool | None,
    ) -> GatewayResult:
        """
        Execute an action that was authorized outside the
        VALOR exceptional path.

        Used for:
            normal policy
            frozen catalog baseline

        Neither path consumes VALOR exceptional reserve.
        """

        try:

            tool_result = (
                self.environment
                .execute(
                    action
                )
            )

        except Exception as exc:

            return GatewayResult(
                policy=self.policy,

                status="execution_error",

                authorization_path=(
                    authorization_path
                ),

                reason=(
                    "tool_execution_failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                action=action,

                normal_authorized=(
                    normal_authorized
                ),

                catalog_authorized=(
                    catalog_authorized
                ),

                tool_result=None,

                reserved_effect=None,

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=None,
            )

        return GatewayResult(
            policy=self.policy,

            status="executed",

            authorization_path=(
                authorization_path
            ),

            reason=None,

            action=action,

            normal_authorized=(
                normal_authorized
            ),

            catalog_authorized=(
                catalog_authorized
            ),

            tool_result=tool_result,

            reserved_effect=None,

            realized_effect=None,

            reserve_before=(
                reserve_before
            ),

            reserve_after=(
                self.remaining_reserve()
            ),

            capability_nonce=None,
        )

    # ========================================================
    # Public Agent entrypoint
    # ========================================================

    def call(
        self,
        proposal: ToolProposal,
    ) -> GatewayResult:
        """
        Process one untrusted LLM tool proposal.

        The same proposal can be evaluated under:

            normal_only
            catalog
            valor

        depending only on the fixed Gateway policy.
        """

        reserve_before = (
            self.remaining_reserve()
        )

        # ----------------------------------------------------
        # 1. Structural validation
        # ----------------------------------------------------

        try:

            action = action_from_tool_call(
                tool_name=(
                    proposal.tool_name
                ),

                arguments=(
                    proposal.arguments
                ),
            )

        except Exception as exc:

            return GatewayResult(
                policy=self.policy,

                status="denied",

                authorization_path="denied",

                reason=(
                    "invalid_tool_proposal: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                action=None,

                normal_authorized=None,

                catalog_authorized=None,

                tool_result=None,

                reserved_effect=None,

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=None,
            )

        # ----------------------------------------------------
        # 2. Normal policy
        #
        # All three policy groups share exactly the same
        # normal authority.
        # ----------------------------------------------------

        normal_authorized = (
            is_normally_authorized(
                action,
                self.tools,
            )
        )

        if normal_authorized:

            return self._execute_direct(
                action=action,

                authorization_path=(
                    "normal"
                ),

                reserve_before=(
                    reserve_before
                ),

                normal_authorized=True,

                catalog_authorized=None,
            )

        # ====================================================
        # POLICY 1:
        # NORMAL ONLY
        # ====================================================

        if (
            self.policy
            == POLICY_NORMAL_ONLY
        ):

            return GatewayResult(
                policy=self.policy,

                status="denied",

                authorization_path=(
                    "denied"
                ),

                reason=(
                    "normal_policy_denied"
                ),

                action=action,

                normal_authorized=False,

                catalog_authorized=None,

                tool_result=None,

                reserved_effect=None,

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=None,
            )

        # ====================================================
        # POLICY 2:
        # FROZEN CATALOG BREAK-GLASS
        # ====================================================

        if (
            self.policy
            == POLICY_CATALOG
        ):

            catalog_authorized = (
                self.catalog
                .authorizes(
                    action
                )
            )

            if not catalog_authorized:

                return GatewayResult(
                    policy=self.policy,

                    status="denied",

                    authorization_path=(
                        "denied"
                    ),

                    reason=(
                        "catalog_scope_denied"
                    ),

                    action=action,

                    normal_authorized=False,

                    catalog_authorized=False,

                    tool_result=None,

                    reserved_effect=None,

                    realized_effect=None,

                    reserve_before=(
                        reserve_before
                    ),

                    reserve_after=(
                        self.remaining_reserve()
                    ),

                    capability_nonce=None,
                )

            # ------------------------------------------------
            # Standing catalog authority.
            #
            # No typed cumulative reserve is consumed.
            #
            # This is deliberate:
            # it matches the break-glass baseline semantics.
            # ------------------------------------------------

            return self._execute_direct(
                action=action,

                authorization_path=(
                    "catalog"
                ),

                reserve_before=(
                    reserve_before
                ),

                normal_authorized=False,

                catalog_authorized=True,
            )

        # ====================================================
        # POLICY 3:
        # VALOR
        # ====================================================

        assert (
            self.policy
            == POLICY_VALOR
        )

        issuance_state = (
            self.environment
            .adapter_state()
        )

        # ----------------------------------------------------
        # 3A. Propose
        # ----------------------------------------------------

        try:

            capability = (
                self.kernel
                .propose(
                    action,

                    state=(
                        issuance_state
                    ),
                )
            )

        except Exception as exc:

            return GatewayResult(
                policy=self.policy,

                status="denied",

                authorization_path=(
                    "denied"
                ),

                reason=(
                    "valor_propose_rejected: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                action=action,

                normal_authorized=False,

                catalog_authorized=None,

                tool_result=None,

                reserved_effect=None,

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=None,
            )

        reserved_effect = (
            capability
            .reserved_effect_vector()
        )

        # ----------------------------------------------------
        # 3B. Fresh execution-time revalidation
        # ----------------------------------------------------

        execution_state = (
            self.environment
            .adapter_state()
        )

        try:

            permitted = (
                self.kernel
                .execute(
                    capability,

                    action,

                    state=(
                        execution_state
                    ),
                )
            )

        except Exception as exc:

            return GatewayResult(
                policy=self.policy,

                status="denied",

                authorization_path="valor",

                reason=(
                    "valor_execute_rejected: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                action=action,

                normal_authorized=False,

                catalog_authorized=None,

                tool_result=None,

                reserved_effect=(
                    reserved_effect
                ),

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=(
                    capability.nonce
                ),
            )

        if not permitted:

            return GatewayResult(
                policy=self.policy,

                status="denied",

                authorization_path="valor",

                reason=(
                    "valor_execute_returned_false"
                ),

                action=action,

                normal_authorized=False,

                catalog_authorized=None,

                tool_result=None,

                reserved_effect=(
                    reserved_effect
                ),

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=(
                    capability.nonce
                ),
            )

        # ----------------------------------------------------
        # 3C. Tool execution boundary
        # ----------------------------------------------------

        try:

            tool_result = (
                self.environment
                .execute(
                    action
                )
            )

        except Exception as exc:

            # ------------------------------------------------
            # Tool outcome may be indeterminate.
            # Conservatively settle full reservation.
            # ------------------------------------------------

            try:

                self.kernel.commit(
                    capability,

                    realized_effect=None,
                )

            except Exception:
                pass

            return GatewayResult(
                policy=self.policy,

                status=(
                    "execution_error"
                ),

                authorization_path=(
                    "valor"
                ),

                reason=(
                    "tool_execution_failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                action=action,

                normal_authorized=False,

                catalog_authorized=None,

                tool_result=None,

                reserved_effect=(
                    reserved_effect
                ),

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=(
                    capability.nonce
                ),
            )

        # ----------------------------------------------------
        # 3D. Trusted realized effect
        # ----------------------------------------------------

        try:

            realized_effect = (
                self.adapter
                .contract(
                    action,
                    execution_state,
                )
                .effect_vector()
            )

        except Exception as exc:

            try:

                self.kernel.commit(
                    capability,

                    realized_effect=None,
                )

            except Exception:
                pass

            return GatewayResult(
                policy=self.policy,

                status=(
                    "execution_error"
                ),

                authorization_path=(
                    "valor"
                ),

                reason=(
                    "receipt_generation_failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                action=action,

                normal_authorized=False,

                catalog_authorized=None,

                tool_result=(
                    tool_result
                ),

                reserved_effect=(
                    reserved_effect
                ),

                realized_effect=None,

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=(
                    capability.nonce
                ),
            )

        # ----------------------------------------------------
        # 3E. Commit
        # ----------------------------------------------------

        try:

            self.kernel.commit(
                capability,

                realized_effect=(
                    realized_effect
                ),
            )

        except Exception as exc:

            return GatewayResult(
                policy=self.policy,

                status=(
                    "execution_error"
                ),

                authorization_path=(
                    "valor"
                ),

                reason=(
                    "valor_commit_failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

                action=action,

                normal_authorized=False,

                catalog_authorized=None,

                tool_result=(
                    tool_result
                ),

                reserved_effect=(
                    reserved_effect
                ),

                realized_effect=(
                    realized_effect
                ),

                reserve_before=(
                    reserve_before
                ),

                reserve_after=(
                    self.remaining_reserve()
                ),

                capability_nonce=(
                    capability.nonce
                ),
            )

        return GatewayResult(
            policy=self.policy,

            status="executed",

            authorization_path="valor",

            reason=None,

            action=action,

            normal_authorized=False,

            catalog_authorized=None,

            tool_result=(
                tool_result
            ),

            reserved_effect=(
                reserved_effect
            ),

            realized_effect=(
                realized_effect
            ),

            reserve_before=(
                reserve_before
            ),

            reserve_after=(
                self.remaining_reserve()
            ),

            capability_nonce=(
                capability.nonce
            ),
        )


# ============================================================
# Gateway factory
# ============================================================

def build_gateway(
    *,
    policy: str,
    environment:
        AgentToolEnvironment
        | None = None,
) -> ToolGateway:

    if environment is None:
        environment = (
            AgentToolEnvironment()
        )

    return ToolGateway(
        environment=environment,

        policy=policy,

        initial_reserve=(
            DEFAULT_INITIAL_RESERVE
        ),

        capability_key=(
            b"VALOR_AGENT_DEVELOPMENT_KEY"
        ),

        catalog=(
            build_frozen_pilot_catalog()
        ),
    )


# ============================================================
# Backward-compatible helper
# ============================================================

def build_default_gateway(
) -> ToolGateway:
    """
    Existing smoke tests call this helper.

    Default remains VALOR.
    """

    return build_gateway(
        policy=POLICY_VALOR
    )


# ============================================================
# Smoke-test helpers
# ============================================================

def print_result(
    title: str,
    result: GatewayResult,
) -> None:

    print()

    print(
        title
    )

    print(
        "-" * len(
            title
        )
    )

    print(
        f"policy             = "
        f"{result.policy}"
    )

    print(
        f"status             = "
        f"{result.status}"
    )

    print(
        f"authorization_path = "
        f"{result.authorization_path}"
    )

    print(
        f"reason             = "
        f"{result.reason}"
    )

    print(
        f"action             = "
        f"{result.action}"
    )

    print(
        f"normal_authorized  = "
        f"{result.normal_authorized}"
    )

    print(
        f"catalog_authorized = "
        f"{result.catalog_authorized}"
    )

    print(
        f"tool_result        = "
        f"{result.tool_result}"
    )

    print(
        f"reserve_before     = "
        f"{result.reserve_before}"
    )

    print(
        f"reserve_after      = "
        f"{result.reserve_after}"
    )


# ============================================================
# Policy smoke test
# ============================================================

def main() -> None:

    print(
        "=" * 88
    )

    print(
        "VALOR Policy-Controlled Tool Gateway"
    )

    print(
        "=" * 88
    )

    # ========================================================
    # Case 1:
    # NORMAL ACTION
    #
    # All policies must allow.
    # ========================================================

    normal_proposal = ToolProposal(
        call_id="normal-001",

        tool_name="increase_power",

        arguments={
            "level": 3.8
        },
    )

    print()

    print(
        "CASE 1 — NORMAL ACTION "
        "increase_power(3.8)"
    )

    for policy in (
        POLICY_NORMAL_ONLY,
        POLICY_CATALOG,
        POLICY_VALOR,
    ):

        gateway = build_gateway(
            policy=policy
        )

        result = gateway.call(
            normal_proposal
        )

        print_result(
            policy,
            result,
        )

        assert (
            result.status
            == "executed"
        )

        assert (
            result.authorization_path
            == "normal"
        )

    # ========================================================
    # Case 2:
    # KNOWN EMERGENCY
    #
    # 5.5 is in frozen catalog.
    # ========================================================

    known_proposal = ToolProposal(
        call_id="known-001",

        tool_name="increase_power",

        arguments={
            "level": 5.5
        },
    )

    print()

    print(
        "=" * 88
    )

    print(
        "CASE 2 — KNOWN EMERGENCY "
        "increase_power(5.5)"
    )

    normal_gateway = build_gateway(
        policy=(
            POLICY_NORMAL_ONLY
        )
    )

    catalog_gateway = build_gateway(
        policy=(
            POLICY_CATALOG
        )
    )

    valor_gateway = build_gateway(
        policy=(
            POLICY_VALOR
        )
    )

    normal_result = (
        normal_gateway.call(
            known_proposal
        )
    )

    catalog_result = (
        catalog_gateway.call(
            known_proposal
        )
    )

    valor_result = (
        valor_gateway.call(
            known_proposal
        )
    )

    print_result(
        "normal_only",
        normal_result,
    )

    print_result(
        "catalog",
        catalog_result,
    )

    print_result(
        "valor",
        valor_result,
    )

    assert (
        normal_result.status
        == "denied"
    )

    assert (
        catalog_result.status
        == "executed"
    )

    assert (
        catalog_result
        .authorization_path
        == "catalog"
    )

    assert (
        valor_result.status
        == "executed"
    )

    assert (
        valor_result
        .authorization_path
        == "valor"
    )

    # ========================================================
    # Case 3:
    # NOVEL FEASIBLE
    #
    # 6.2 is NOT in catalog,
    # but is valid for VALOR.
    # ========================================================

    novel_proposal = ToolProposal(
        call_id="novel-001",

        tool_name="increase_power",

        arguments={
            "level": 6.2
        },
    )

    print()

    print(
        "=" * 88
    )

    print(
        "CASE 3 — NOVEL FEASIBLE "
        "increase_power(6.2)"
    )

    normal_gateway = build_gateway(
        policy=(
            POLICY_NORMAL_ONLY
        )
    )

    catalog_gateway = build_gateway(
        policy=(
            POLICY_CATALOG
        )
    )

    valor_gateway = build_gateway(
        policy=(
            POLICY_VALOR
        )
    )

    normal_result = (
        normal_gateway.call(
            novel_proposal
        )
    )

    catalog_result = (
        catalog_gateway.call(
            novel_proposal
        )
    )

    valor_result = (
        valor_gateway.call(
            novel_proposal
        )
    )

    print_result(
        "normal_only",
        normal_result,
    )

    print_result(
        "catalog",
        catalog_result,
    )

    print_result(
        "valor",
        valor_result,
    )

    assert (
        normal_result.status
        == "denied"
    )

    assert (
        catalog_result.status
        == "denied"
    )

    assert (
        catalog_result.reason
        == "catalog_scope_denied"
    )

    assert (
        valor_result.status
        == "executed"
    )

    assert (
        valor_result
        .authorization_path
        == "valor"
    )

    assert (
        valor_gateway
        .conservation_holds()
    )

    print()

    print(
        "=" * 88
    )

    print(
        "POLICY-CONTROLLED GATEWAY: PASS"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
