import pytest

from valor.sqlite_steady_ledger import (
    SQLiteSteadyStateReserveLedger,
)


RESERVE = {
    "energy": 40.0,
    "stress": 25.0,
    "irreversible": 3.0,
    "external_scope": 18.0,
}


def make_ledger(
    tmp_path,
):

    return (
        SQLiteSteadyStateReserveLedger(
            tmp_path
            / "steady.sqlite",

            RESERVE,
        )
    )


def test_steady_ledger_configuration(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    try:

        settings = (
            ledger.sqlite_settings()
        )

        assert (
            str(
                settings[
                    "journal_mode"
                ]
            ).lower()
            == "wal"
        )

        # SQLite FULL = 2
        assert (
            settings[
                "synchronous_numeric"
            ]
            == 2
        )

    finally:

        ledger.close()


def test_steady_reservation(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    try:

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

        remaining = (
            ledger
            .remaining_reserve()
        )

        assert remaining == pytest.approx(
            {
                "energy": 36.0,
                "stress": 23.0,
                "irreversible": 2.0,
                "external_scope": 15.0,
            }
        )

        assert (
            ledger.conservation_holds()
        )

    finally:

        ledger.close()


def test_steady_reset(
    tmp_path,
):

    ledger = make_ledger(
        tmp_path
    )

    try:

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

    finally:

        ledger.close()