import pytest

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.capability import (
    verify_capability,
)

from valor.kernel import (
    ValorKernel,
    DomainNotAllowedError,
    HardGuardFailedError,
    InsufficientReserveError,
)

from valor.ledger import (
    InMemoryReserveLedger,
)

from valor.models import Action


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

    adapter = SyntheticEffectAdapter(
        tools
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

    key = b"K" * 32

    nonce_counter = iter(
        [
            "nonce-001",
            "nonce-002",
            "nonce-003",
            "nonce-004",
        ]
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=key,
        epoch=1,
        capability_ttl_seconds=5,

        nonce_factory=lambda: next(
            nonce_counter
        ),
    )

    return (
        tools,
        adapter,
        ledger,
        kernel,
        key,
    )


def get_allowed_tool(tools):

    return next(
        tool
        for tool in tools.values()
        if tool.in_dmax
    )


def get_excluded_tool(tools):

    return next(
        tool
        for tool in tools.values()
        if not tool.in_dmax
    )


def exceptional_action(tool):

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


def test_successful_proposal_reduces_reserve():

    (
        tools,
        adapter,
        ledger,
        kernel,
        _,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = exceptional_action(
        tool
    )

    contract = adapter.contract(
        action
    )

    before = (
        ledger.remaining_reserve()
    )

    kernel.propose(
        action,
        now_ns=1000,
    )

    after = (
        ledger.remaining_reserve()
    )

    effect = (
        contract.effect_vector()
    )

    for dimension in before:

        assert after[
            dimension
        ] == pytest.approx(
            before[dimension]
            - effect[dimension]
        )


def test_capability_is_valid():

    (
        tools,
        _,
        _,
        kernel,
        key,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = exceptional_action(
        tool
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    assert verify_capability(
        key=key,
        capability=capability,
    )


def test_capability_binds_exact_action():

    (
        tools,
        _,
        _,
        kernel,
        _,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = exceptional_action(
        tool
    )

    capability = kernel.propose(
        action,
        now_ns=1000,
    )

    assert (
        capability.action
        == action
    )


def test_conservation_holds_after_propose():

    (
        tools,
        _,
        ledger,
        kernel,
        _,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = exceptional_action(
        tool
    )

    kernel.propose(
        action,
        now_ns=1000,
    )

    assert (
        ledger.conservation_holds()
    )


def test_outside_dmax_rejected_without_spend():

    (
        tools,
        _,
        ledger,
        kernel,
        _,
    ) = build_system()

    tool = get_excluded_tool(
        tools
    )

    action = exceptional_action(
        tool
    )

    before = (
        ledger.remaining_reserve()
    )

    with pytest.raises(
        DomainNotAllowedError
    ):
        kernel.propose(
            action,
            now_ns=1000,
        )

    after = (
        ledger.remaining_reserve()
    )

    assert after == before


def test_hard_guard_failure_does_not_spend():

    (
        tools,
        _,
        ledger,
        kernel,
        _,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = exceptional_action(
        tool
    )

    before = (
        ledger.remaining_reserve()
    )

    with pytest.raises(
        HardGuardFailedError
    ):
        kernel.propose(
            action,

            state={
                "force_hard_guard_failure":
                    True
            },

            now_ns=1000,
        )

    after = (
        ledger.remaining_reserve()
    )

    assert after == before


def test_insufficient_reserve_rejected():

    tools = generate_tools(
        seed=20260903
    )

    adapter = SyntheticEffectAdapter(
        tools
    )

    dmax = {
        tool.domain
        for tool in tools.values()
        if tool.in_dmax
    }

    tiny_reserve = {
        "energy": 0.0001,
        "stress": 0.0001,
        "irreversible": 0.0001,
        "external_scope": 0.0001,
    }

    ledger = (
        InMemoryReserveLedger(
            tiny_reserve
        )
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=b"K" * 32,
        nonce_factory=lambda:
            "tiny-reserve-nonce",
    )

    tool = get_allowed_tool(
        tools
    )

    action = exceptional_action(
        tool
    )

    before = (
        ledger.remaining_reserve()
    )

    with pytest.raises(
        InsufficientReserveError
    ):
        kernel.propose(
            action,
            now_ns=1000,
        )

    assert (
        ledger.remaining_reserve()
        == before
    )