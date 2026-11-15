from experiments.concurrency_replay import (
    run_replay_and_commit_stress,
    run_reservation_stress,
)


FULL_RESERVE = {
    "energy": 40.0,
    "stress": 25.0,
    "irreversible": 3.0,
    "external_scope": 18.0,
}


def test_concurrent_reservations_do_not_overspend(
    tmp_path,
):

    (
        _,
        summary,
    ) = run_reservation_stress(
        db_path=(
            tmp_path
            / "reserve.sqlite"
        ),

        reserve_units=40.0,

        reservation_units=7.0,

        requests_per_round=24,

        worker_threads=16,

        rounds=3,
    )

    assert (
        summary[
            "expected_successes_per_round"
        ]
        == 5
    )

    assert (
        summary[
            "total_successes"
        ]
        == 15
    )

    assert (
        summary[
            "exact_success_rounds"
        ]
        == 3
    )

    assert (
        summary[
            "negative_reserve_violations"
        ]
        == 0
    )

    assert (
        summary[
            "overspend_violations"
        ]
        == 0
    )

    assert (
        summary[
            "conservation_violations"
        ]
        == 0
    )

    assert (
        summary[
            "unexpected_errors"
        ]
        == 0
    )


def test_concurrent_replay_and_commit_are_one_shot(
    tmp_path,
):

    summary = (
        run_replay_and_commit_stress(
            db_path=(
                tmp_path
                / "replay.sqlite"
            ),

            seed=20260903,

            reserve=FULL_RESERVE,

            requests=24,

            worker_threads=16,
        )
    )

    assert (
        summary[
            "execute_successes"
        ]
        == 1
    )

    assert (
        summary[
            "execute_replay_rejections"
        ]
        == 23
    )

    assert (
        summary[
            "execute_unexpected"
        ]
        == 0
    )

    assert (
        summary[
            "commit_successes"
        ]
        == 1
    )

    assert (
        summary[
            "commit_rejections"
        ]
        == 23
    )

    assert (
        summary[
            "commit_unexpected"
        ]
        == 0
    )

    assert (
        summary[
            "conservation_ok"
        ]
    )