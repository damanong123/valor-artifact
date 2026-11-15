import pytest

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.kernel import (
    ValorKernel,
    CapabilityNotInFlightCommitError,
)

from valor.ledger import (
    InMemoryReserveLedger,
    EffectReceiptExceedsReservationError,
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
            "commit-nonce-001",
            "commit-nonce-002",
            "commit-nonce-003",
            "commit-nonce-004",
            "commit-nonce-005",
        ]
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=b"K" * 32,
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


def make_action(
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
# Partial realized effect
# ============================================================

def test_partial_commit_refunds_unused_reserve():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = make_action(
        tool
    )

    cap = kernel.propose(
        action,
        now_ns=1000,
    )

    kernel.execute(
        cap,
        action,
        now_ns=2000,
    )

    reserved = (
        cap
        .reserved_effect_vector()
    )

    realized = {
        dimension:
            value * 0.5

        for dimension, value
        in reserved.items()
    }

    result = kernel.commit(
        cap,
        realized_effect=realized,
    )

    assert result[
        "committed"
    ] == pytest.approx(
        realized
    )

    assert not ledger.has_reservation(
        cap.nonce
    )

    assert not ledger.is_in_flight(
        cap.nonce
    )

    assert ledger.is_resolved(
        cap.nonce
    )

    assert ledger.committed_spend() == (
        pytest.approx(
            realized
        )
    )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Indeterminate outcome
# ============================================================

def test_indeterminate_commit_charges_full_reservation():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = make_action(
        tool
    )

    cap = kernel.propose(
        action,
        now_ns=1000,
    )

    kernel.execute(
        cap,
        action,
        now_ns=2000,
    )

    reserved = (
        cap
        .reserved_effect_vector()
    )

    before_commit = (
        ledger.remaining_reserve()
    )

    result = kernel.commit(
        cap,
        realized_effect=None,
    )

    assert result[
        "committed"
    ] == pytest.approx(
        reserved
    )

    assert result[
        "refunded"
    ] == pytest.approx(
        {
            k: 0.0
            for k in reserved
        }
    )

    # No refund occurs.
    assert (
        ledger.remaining_reserve()
        == pytest.approx(
            before_commit
        )
    )

    assert (
        ledger.committed_spend()
        == pytest.approx(
            reserved
        )
    )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Receipt cannot exceed reservation
# ============================================================

def test_commit_rejects_effect_above_reservation():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = make_action(
        tool
    )

    cap = kernel.propose(
        action,
        now_ns=1000,
    )

    kernel.execute(
        cap,
        action,
        now_ns=2000,
    )

    reserved = (
        cap
        .reserved_effect_vector()
    )

    invalid = dict(
        reserved
    )

    invalid[
        "energy"
    ] += 1.0

    before_remaining = (
        ledger.remaining_reserve()
    )

    before_committed = (
        ledger.committed_spend()
    )

    with pytest.raises(
        EffectReceiptExceedsReservationError
    ):
        kernel.commit(
            cap,
            realized_effect=invalid,
        )

    # State must remain unsettled.
    assert ledger.is_in_flight(
        cap.nonce
    )

    assert ledger.has_reservation(
        cap.nonce
    )

    assert (
        ledger.remaining_reserve()
        == before_remaining
    )

    assert (
        ledger.committed_spend()
        == before_committed
    )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Cannot Commit before Execute
# ============================================================

def test_commit_before_execute_rejected():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = make_action(
        tool
    )

    cap = kernel.propose(
        action,
        now_ns=1000,
    )

    with pytest.raises(
        CapabilityNotInFlightCommitError
    ):
        kernel.commit(
            cap,
            realized_effect=None,
        )

    assert ledger.has_reservation(
        cap.nonce
    )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Cannot Commit twice
# ============================================================

def test_double_commit_rejected():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = make_action(
        tool
    )

    cap = kernel.propose(
        action,
        now_ns=1000,
    )

    kernel.execute(
        cap,
        action,
        now_ns=2000,
    )

    kernel.commit(
        cap,
        realized_effect=None,
    )

    with pytest.raises(
        CapabilityNotInFlightCommitError
    ):
        kernel.commit(
            cap,
            realized_effect=None,
        )

    assert (
        ledger.conservation_holds()
    )


# ============================================================
# Zero realized effect => full refund
# ============================================================

def test_zero_effect_commit_refunds_all():

    (
        tools,
        _,
        ledger,
        kernel,
    ) = build_system()

    tool = get_allowed_tool(
        tools
    )

    action = make_action(
        tool
    )

    cap = kernel.propose(
        action,
        now_ns=1000,
    )

    kernel.execute(
        cap,
        action,
        now_ns=2000,
    )

    zero = {
        "energy": 0.0,
        "stress": 0.0,
        "irreversible": 0.0,
        "external_scope": 0.0,
    }

    kernel.commit(
        cap,
        realized_effect=zero,
    )

    assert (
        ledger.remaining_reserve()
        == pytest.approx(
            INITIAL_RESERVE
        )
    )

    assert (
        ledger.committed_spend()
        == pytest.approx(
            zero
        )
    )

    assert (
        ledger.conservation_holds()
    )