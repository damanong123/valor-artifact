from pathlib import Path

import pytest

from valor.sqlite_ledger import (
    SQLiteReserveLedger,
)


RESERVE = {
    "energy": 40.0,
    "stress": 25.0,
    "irreversible": 3.0,
    "external_scope": 18.0,
}


def make_ledger(
    tmp_path: Path,
):

    return SQLiteReserveLedger(
        tmp_path / "valor_test.sqlite",
        RESERVE,
    )


def test_sqlite_initial_state(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    assert (
        ledger.remaining_reserve()
        == pytest.approx(
            RESERVE
        )
    )

    assert (
        ledger.conservation_holds()
    )


def test_sqlite_reserve(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    bound = {
        "energy": 4.0,
        "stress": 2.0,
        "irreversible": 1.0,
        "external_scope": 3.0,
    }

    assert ledger.try_reserve(
        nonce="n1",
        bound=bound,
    )

    assert ledger.has_reservation(
        "n1"
    )

    assert (
        ledger.conservation_holds()
    )


def test_sqlite_execute_and_commit(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    bound = {
        "energy": 4.0,
        "stress": 2.0,
        "irreversible": 1.0,
        "external_scope": 3.0,
    }

    ledger.try_reserve(
        nonce="n1",
        bound=bound,
    )

    assert (
        ledger.try_resolve_for_execution(
            "n1"
        )
    )

    assert ledger.is_resolved(
        "n1"
    )

    assert ledger.is_in_flight(
        "n1"
    )

    realized = {
        "energy": 3.0,
        "stress": 1.0,
        "irreversible": 0.5,
        "external_scope": 2.0,
    }

    ledger.settle_commit(
        nonce="n1",
        realized_effect=realized,
    )

    assert not ledger.has_reservation(
        "n1"
    )

    assert not ledger.is_in_flight(
        "n1"
    )

    assert (
        ledger.committed_spend()
        == pytest.approx(
            realized
        )
    )

    assert (
        ledger.conservation_holds()
    )


def test_sqlite_abort_refund(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    bound = {
        "energy": 4.0,
        "stress": 2.0,
        "irreversible": 1.0,
        "external_scope": 3.0,
    }

    ledger.try_reserve(
        nonce="n1",
        bound=bound,
    )

    assert (
        ledger.resolve_abort_and_refund(
            "n1"
        )
    )

    assert ledger.is_resolved(
        "n1"
    )

    assert (
        ledger.remaining_reserve()
        == pytest.approx(
            RESERVE
        )
    )

    assert (
        ledger.conservation_holds()
    )


def test_sqlite_reset_for_benchmark(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    bound = {
        "energy": 4.0,
        "stress": 2.0,
        "irreversible": 1.0,
        "external_scope": 3.0,
    }

    ledger.try_reserve(
        nonce="n1",
        bound=bound,
    )

    ledger.reset_for_benchmark()

    assert (
        ledger.remaining_reserve()
        == pytest.approx(
            RESERVE
        )
    )

    assert (
        ledger.outstanding_total()
        == pytest.approx(
            {
                "energy": 0.0,
                "stress": 0.0,
                "irreversible": 0.0,
                "external_scope": 0.0,
            }
        )
    )

    assert (
        ledger.conservation_holds()
    )