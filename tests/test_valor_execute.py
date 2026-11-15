from dataclasses import replace

import pytest

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.kernel import (
    ValorKernel,
    ActionMismatchError,
    CapabilityExpiredError,
    CapabilityVerificationError,
    ExecutionRevalidationError,
    ReplayDetectedError,
)

from valor.ledger import (
    InMemoryReserveLedger,
)

from valor.models import (
    Action,
)


INITIAL_RESERVE = {
    "energy": 40.0,
    "stress": 25.0,
    "irreversible": 3.0,
    "external_scope": 18.0,
}


def build_system():

    tools = generate_tools(
        seed=20260903
    )

    adapter = (
        SyntheticEffectAdapter(
            tools
        )
    )

    dmax = {
        tool.domain
        for tool in tools.values()
        if tool.in_dmax
    }

    ledger = (
        InMemoryReserveLedger(
            INITIAL_RESERVE
        )
    )

    counter = iter(
        [
            "nonce-001",
            "nonce-002",
            "nonce-003",
            "nonce-004",
            "nonce-005",
        ]
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,

        capability_key=(
            b"K" * 32
        ),

        epoch=1,

        capability_ttl_seconds=5,

        nonce_factory=lambda:
            next(counter),
    )

    return (
        tools,
        adapter,
        ledger,
        kernel,
    )


def get_allowed_tool(
    tools
):

    return next(
        tool
        for tool in tools.values()
        if tool.in_dmax
    )


def make_exceptional_action(
    tool
):

    parameter = (
        tool.normal_max
        +
        (
            tool.exceptional_max
            - tool.normal_max
        )
        * 0.5
    )

    return Action(
        tool=tool.name,
        parameter=parameter,
    )


# ============================================================
# Successful execution
# ============================================================

def test_execute_success():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    permitted = kernel.execute(
        capability,
        action,

        state={
            "effect_multiplier": 1.0
        },

        now_ns=2000,
    )

    assert permitted is True

    assert ledger.is_resolved(
        capability.nonce
    )

    assert ledger.is_in_flight(
        capability.nonce
    )

    # Reservation must remain in C
    # until Commit.
    assert ledger.has_reservation(
        capability.nonce
    )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Lower execution-time effect still fits
# ============================================================

def test_smaller_execution_effect_is_allowed():

    (
        tools,
        _,
        _,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    assert kernel.execute(
        capability,
        action,

        state={
            "effect_multiplier": 0.8
        },

        now_ns=2000,
    )


# ============================================================
# State drift makes effect too large
# ============================================================

def test_state_drift_aborts_and_refunds():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    with pytest.raises(
        ExecutionRevalidationError
    ):
        kernel.execute(
            capability,
            action,

            state={
                "effect_multiplier": 2.0
            },

            now_ns=2000,
        )

    # Aborted execution resolves nonce.
    assert ledger.is_resolved(
        capability.nonce
    )

    # It never became in-flight.
    assert not ledger.is_in_flight(
        capability.nonce
    )

    # Reservation was refunded.
    assert not ledger.has_reservation(
        capability.nonce
    )

    assert (
        ledger.remaining_reserve()
        == pytest.approx(
            INITIAL_RESERVE
        )
    )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Hard guard changes between Propose and Execute
# ============================================================

def test_execution_hard_guard_failure_refunds():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    with pytest.raises(
        ExecutionRevalidationError
    ):
        kernel.execute(
            capability,
            action,

            state={
                "force_hard_guard_failure":
                    True
            },

            now_ns=2000,
        )

    assert ledger.is_resolved(
        capability.nonce
    )

    assert (
        ledger.remaining_reserve()
        == pytest.approx(
            INITIAL_RESERVE
        )
    )


# ============================================================
# Replay after successful execution
# ============================================================

def test_replay_rejected_after_success():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    kernel.execute(
        capability,
        action,
        now_ns=2000,
    )

    with pytest.raises(
        ReplayDetectedError
    ):
        kernel.execute(
            capability,
            action,
            now_ns=3000,
        )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Replay after aborted execution
# ============================================================

def test_replay_rejected_after_abort():

    (
        tools,
        _,
        _,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    with pytest.raises(
        ExecutionRevalidationError
    ):
        kernel.execute(
            capability,
            action,

            state={
                "effect_multiplier": 2.0
            },

            now_ns=2000,
        )

    with pytest.raises(
        ReplayDetectedError
    ):
        kernel.execute(
            capability,
            action,
            now_ns=3000,
        )


# ============================================================
# Exact action mismatch does NOT resolve nonce
# ============================================================

def test_action_mismatch_does_not_consume_nonce():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    wrong_action = Action(
        tool=action.tool,
        parameter=(
            action.parameter
            + 0.01
        ),
    )

    with pytest.raises(
        ActionMismatchError
    ):
        kernel.execute(
            capability,
            wrong_action,
            now_ns=2000,
        )

    assert not ledger.is_resolved(
        capability.nonce
    )

    assert ledger.has_reservation(
        capability.nonce
    )


# ============================================================
# Expired capability does NOT resolve nonce
# ============================================================

def test_expired_capability_does_not_consume_nonce():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    with pytest.raises(
        CapabilityExpiredError
    ):
        kernel.execute(
            capability,
            action,

            now_ns=(
                capability.expiry_ns
                + 1
            ),
        )

    assert not ledger.is_resolved(
        capability.nonce
    )

    assert ledger.has_reservation(
        capability.nonce
    )


# ============================================================
# Tampered capability rejected
# ============================================================

def test_tampered_capability_rejected_without_state_change():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = (
        make_exceptional_action(
            tool
        )
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    tampered = replace(
        capability,
        mac_hex="00" * 32,
    )

    with pytest.raises(
        CapabilityVerificationError
    ):
        kernel.execute(
            tampered,
            action,
            now_ns=2000,
        )

    assert not ledger.is_resolved(
        capability.nonce
    )

    assert ledger.has_reservation(
        capability.nonce
    )