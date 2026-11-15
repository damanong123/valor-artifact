from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
)

from pathlib import Path

import math
import random
import threading

import pandas as pd
import yaml

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.kernel import (
    ValorKernel,
    ReplayDetectedError,
    CapabilityNotInFlightCommitError,
    ReservationNotFoundError,
)

from valor.ledger import (
    DuplicateNonceError,
    CapabilityNotInFlightError,
)

from valor.models import (
    Action,
)

from valor.sqlite_ledger import (
    SQLiteReserveLedger,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# SQLite cleanup
# ============================================================

def remove_sqlite_files(
    path: Path,
) -> None:

    candidates = [
        path,

        Path(
            str(path)
            + "-wal"
        ),

        Path(
            str(path)
            + "-shm"
        ),
    ]

    for candidate in candidates:

        if candidate.exists():

            candidate.unlink()


# ============================================================
# Effect-vector helpers
# ============================================================

def concurrency_initial_reserve(
    energy: float,
) -> dict[str, float]:
    """
    Paper-style concurrency workload:

        energy reserve = 40

    Other effect dimensions are unused in this isolated
    transaction stress experiment.
    """

    return {
        "energy":
            float(
                energy
            ),

        "stress":
            0.0,

        "irreversible":
            0.0,

        "external_scope":
            0.0,
    }


def concurrency_bound(
    energy: float,
) -> dict[str, float]:

    return {
        "energy":
            float(
                energy
            ),

        "stress":
            0.0,

        "irreversible":
            0.0,

        "external_scope":
            0.0,
    }


# ============================================================
# Part A:
# Concurrent reserve stress
# ============================================================

def run_reservation_stress(
    *,
    db_path: Path,
    reserve_units: float,
    reservation_units: float,
    requests_per_round: int,
    worker_threads: int,
    rounds: int,
) -> tuple[
    pd.DataFrame,
    dict[str, float | int | bool],
]:
    """
    Run paper-style concurrent reserve stress.

    Important implementation detail
    -------------------------------
    Each worker thread gets its own SQLiteReserveLedger
    instance pointing at the SAME SQLite database.

    Therefore, Python's per-object RLock does not serialize
    all workers globally.

    Correctness depends on SQLite:

        BEGIN IMMEDIATE

    and durable transaction state.
    """

    remove_sqlite_files(
        db_path
    )

    initial = (
        concurrency_initial_reserve(
            reserve_units
        )
    )

    bound = (
        concurrency_bound(
            reservation_units
        )
    )

    # Main-thread handle used only for reset / inspection.
    controller = (
        SQLiteReserveLedger(
            db_path,
            initial,
        )
    )

    expected_successes = int(
        math.floor(
            reserve_units
            / reservation_units
        )
    )

    # --------------------------------------------------------
    # Thread-local ledger handle
    #
    # Each worker creates its own ledger object.
    # They all point at the same database.
    # --------------------------------------------------------

    thread_local = (
        threading.local()
    )

    def worker_ledger():

        if not hasattr(
            thread_local,
            "ledger",
        ):

            thread_local.ledger = (
                SQLiteReserveLedger(
                    db_path,
                    initial,
                )
            )

        return (
            thread_local.ledger
        )

    # --------------------------------------------------------
    # Reservation worker
    # --------------------------------------------------------

    def reserve_worker(
        nonce: str,
    ) -> tuple[
        str,
        str,
    ]:

        ledger = (
            worker_ledger()
        )

        try:

            success = (
                ledger.try_reserve(
                    nonce=nonce,
                    bound=bound,
                )
            )

            if success:

                return (
                    "success",
                    "",
                )

            return (
                "insufficient",
                "",
            )

        except DuplicateNonceError:

            return (
                "duplicate",
                "",
            )

        except Exception as exc:

            return (
                "unexpected_error",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    # --------------------------------------------------------
    # Run rounds
    # --------------------------------------------------------

    raw_rows = []

    total_attempts = 0

    total_successes = 0

    total_duplicates = 0

    total_insufficient = 0

    total_unexpected = 0

    exact_success_rounds = 0

    negative_reserve_violations = 0

    conservation_violations = 0

    overspend_violations = 0

    # Keep executor alive for the full experiment so the same
    # worker threads and thread-local ledger handles are reused.
    with ThreadPoolExecutor(
        max_workers=worker_threads
    ) as executor:

        for round_index in range(
            rounds
        ):

            # ----------------------------------------------
            # Fresh independent trial.
            #
            # This reset is outside concurrent request
            # execution and is not a VALOR runtime action.
            # ----------------------------------------------

            controller.reset_for_benchmark()

            # ----------------------------------------------
            # Generate enough unique nonces that reserve
            # capacity, not nonce scarcity, determines the
            # maximum number of successes.
            #
            # For paper configuration:
            #
            #   16 unique
            #   + 8 duplicated requests
            #   = 24 requests
            # ----------------------------------------------

            unique_count = min(
                worker_threads,
                requests_per_round,
            )

            unique_count = max(
                unique_count,
                expected_successes
                + 1,
            )

            unique_count = min(
                unique_count,
                requests_per_round,
            )

            unique_nonces = [
                (
                    f"round-{round_index:04d}-"
                    f"nonce-{index:03d}"
                )
                for index in range(
                    unique_count
                )
            ]

            request_nonces = list(
                unique_nonces
            )

            duplicate_index = 0

            while len(
                request_nonces
            ) < requests_per_round:

                request_nonces.append(
                    unique_nonces[
                        duplicate_index
                        % len(
                            unique_nonces
                        )
                    ]
                )

                duplicate_index += 1

            # Deterministic shuffle:
            # concurrency scheduling is nondeterministic,
            # request multiset is deterministic.
            rng = random.Random(
                20260903
                + round_index
            )

            rng.shuffle(
                request_nonces
            )

            # ----------------------------------------------
            # Concurrent reservation
            # ----------------------------------------------

            results = list(
                executor.map(
                    reserve_worker,
                    request_nonces,
                )
            )

            statuses = [
                status
                for (
                    status,
                    _
                ) in results
            ]

            error_messages = [
                message
                for (
                    status,
                    message
                ) in results
                if status
                == "unexpected_error"
            ]

            successes = (
                statuses.count(
                    "success"
                )
            )

            duplicates = (
                statuses.count(
                    "duplicate"
                )
            )

            insufficient = (
                statuses.count(
                    "insufficient"
                )
            )

            unexpected = (
                statuses.count(
                    "unexpected_error"
                )
            )

            # ----------------------------------------------
            # Inspect durable state
            # ----------------------------------------------

            remaining = (
                controller
                .remaining_reserve()
            )

            outstanding = (
                controller
                .outstanding_total()
            )

            committed = (
                controller
                .committed_spend()
            )

            conservation_ok = (
                controller
                .conservation_holds()
            )

            negative_reserve = any(
                value < -1e-9

                for value in (
                    remaining.values()
                )
            )

            reserved_energy = (
                outstanding[
                    "energy"
                ]
            )

            spend_or_reserve = (
                reserve_units
                - remaining[
                    "energy"
                ]
            )

            overspend = (
                spend_or_reserve
                > reserve_units
                + 1e-9
            )

            exact_success = (
                successes
                == expected_successes
            )

            # ----------------------------------------------
            # Aggregate counters
            # ----------------------------------------------

            total_attempts += (
                len(
                    request_nonces
                )
            )

            total_successes += (
                successes
            )

            total_duplicates += (
                duplicates
            )

            total_insufficient += (
                insufficient
            )

            total_unexpected += (
                unexpected
            )

            exact_success_rounds += (
                int(
                    exact_success
                )
            )

            negative_reserve_violations += (
                int(
                    negative_reserve
                )
            )

            conservation_violations += (
                int(
                    not conservation_ok
                )
            )

            overspend_violations += (
                int(
                    overspend
                )
            )

            raw_rows.append(
                {
                    "round":
                        round_index,

                    "attempts":
                        len(
                            request_nonces
                        ),

                    "expected_successes":
                        expected_successes,

                    "successes":
                        successes,

                    "duplicates":
                        duplicates,

                    "insufficient":
                        insufficient,

                    "unexpected_errors":
                        unexpected,

                    "remaining_energy":
                        remaining[
                            "energy"
                        ],

                    "reserved_energy":
                        reserved_energy,

                    "committed_energy":
                        committed[
                            "energy"
                        ],

                    "spend_or_reserved_energy":
                        spend_or_reserve,

                    "exact_success_count":
                        exact_success,

                    "negative_reserve":
                        negative_reserve,

                    "overspend":
                        overspend,

                    "conservation_ok":
                        conservation_ok,

                    "error_messages":
                        " | ".join(
                            error_messages
                        ),
                }
            )

    raw_df = pd.DataFrame(
        raw_rows
    )

    summary = {
        "rounds":
            rounds,

        "requests_per_round":
            requests_per_round,

        "worker_threads":
            worker_threads,

        "reserve_units":
            reserve_units,

        "reservation_units":
            reservation_units,

        "expected_successes_per_round":
            expected_successes,

        "total_attempts":
            total_attempts,

        "total_successes":
            total_successes,

        "expected_total_successes":
            (
                expected_successes
                * rounds
            ),

        "exact_success_rounds":
            exact_success_rounds,

        "duplicate_nonce_rejections":
            total_duplicates,

        "insufficient_reserve_rejections":
            total_insufficient,

        "unexpected_errors":
            total_unexpected,

        "negative_reserve_violations":
            negative_reserve_violations,

        "overspend_violations":
            overspend_violations,

        "conservation_violations":
            conservation_violations,

        "all_rounds_exact":
            (
                exact_success_rounds
                == rounds
            ),
    }

    return (
        raw_df,
        summary,
    )


# ============================================================
# Part B:
# Concurrent replay and double Commit
# ============================================================

def run_replay_and_commit_stress(
    *,
    db_path: Path,
    seed: int,
    reserve: dict[str, float],
    requests: int,
    worker_threads: int,
) -> dict[str, int | bool | float]:
    """
    Concurrently present the SAME capability many times.

    Expected:
        exactly one successful Execute.

    Then concurrently present Commit many times.

    Expected:
        exactly one successful Commit.
    """

    remove_sqlite_files(
        db_path
    )

    tools = generate_tools(
        seed=seed
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
        SQLiteReserveLedger(
            db_path,
            reserve,
        )
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=(
            b"K" * 32
        ),
        epoch=1,
        capability_ttl_seconds=60.0,
    )

    tool = next(
        tool
        for tool in tools.values()
        if tool.in_dmax
    )

    parameter = (
        tool.normal_max
        +
        (
            tool.exceptional_max
            - tool.normal_max
        )
        * 0.5
    )

    action = Action(
        tool=tool.name,
        parameter=parameter,
    )

    capability = (
        kernel.propose(
            action,
            now_ns=1_000,
        )
    )

    # --------------------------------------------------------
    # Concurrent Execute replay
    # --------------------------------------------------------

    def execute_worker(
        _,
    ) -> tuple[
        str,
        str,
    ]:

        try:

            permitted = (
                kernel.execute(
                    capability,
                    action,
                    now_ns=2_000,
                )
            )

            if permitted:

                return (
                    "success",
                    "",
                )

            return (
                "unexpected_false",
                "",
            )

        except ReplayDetectedError:

            return (
                "replay_rejected",
                "",
            )

        except Exception as exc:

            return (
                "unexpected_error",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    with ThreadPoolExecutor(
        max_workers=worker_threads
    ) as executor:

        execute_results = list(
            executor.map(
                execute_worker,
                range(
                    requests
                ),
            )
        )

    execute_statuses = [
        status
        for (
            status,
            _
        ) in execute_results
    ]

    execute_successes = (
        execute_statuses.count(
            "success"
        )
    )

    execute_rejections = (
        execute_statuses.count(
            "replay_rejected"
        )
    )

    execute_unexpected = (
        len(
            execute_statuses
        )
        - execute_successes
        - execute_rejections
    )

    # --------------------------------------------------------
    # Concurrent Commit replay
    # --------------------------------------------------------

    realized_effect = (
        capability
        .reserved_effect_vector()
    )

    def commit_worker(
        _,
    ) -> tuple[
        str,
        str,
    ]:

        try:

            kernel.commit(
                capability,
                realized_effect=(
                    realized_effect
                ),
            )

            return (
                "success",
                "",
            )

        except (
            CapabilityNotInFlightCommitError,
            CapabilityNotInFlightError,
            ReservationNotFoundError,
        ):

            return (
                "already_settled",
                "",
            )

        except Exception as exc:

            return (
                "unexpected_error",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    with ThreadPoolExecutor(
        max_workers=worker_threads
    ) as executor:

        commit_results = list(
            executor.map(
                commit_worker,
                range(
                    requests
                ),
            )
        )

    commit_statuses = [
        status
        for (
            status,
            _
        ) in commit_results
    ]

    commit_successes = (
        commit_statuses.count(
            "success"
        )
    )

    commit_rejections = (
        commit_statuses.count(
            "already_settled"
        )
    )

    commit_unexpected = (
        len(
            commit_statuses
        )
        - commit_successes
        - commit_rejections
    )

    # --------------------------------------------------------
    # Final accounting
    # --------------------------------------------------------

    conservation_ok = (
        ledger
        .conservation_holds()
    )

    remaining = (
        ledger
        .remaining_reserve()
    )

    outstanding = (
        ledger
        .outstanding_total()
    )

    committed = (
        ledger
        .committed_spend()
    )

    return {
        "execute_attempts":
            requests,

        "execute_successes":
            execute_successes,

        "execute_replay_rejections":
            execute_rejections,

        "execute_unexpected":
            execute_unexpected,

        "commit_attempts":
            requests,

        "commit_successes":
            commit_successes,

        "commit_rejections":
            commit_rejections,

        "commit_unexpected":
            commit_unexpected,

        "final_remaining_energy":
            remaining[
                "energy"
            ],

        "final_outstanding_energy":
            outstanding[
                "energy"
            ],

        "final_committed_energy":
            committed[
                "energy"
            ],

        "conservation_ok":
            conservation_ok,

        "exactly_one_execute":
            (
                execute_successes
                == 1
            ),

        "exactly_one_commit":
            (
                commit_successes
                == 1
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    with open(
        ROOT
        / "config"
        / "experiment.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(
            f
        )

    concurrency_cfg = (
        cfg[
            "concurrency"
        ]
    )

    reserve_units = float(
        concurrency_cfg[
            "reserve"
        ]
    )

    reservation_units = float(
        concurrency_cfg[
            "reservation_per_request"
        ]
    )

    requests_per_round = int(
        concurrency_cfg[
            "requests_per_round"
        ]
    )

    worker_threads = int(
        concurrency_cfg[
            "worker_threads"
        ]
    )

    rounds = int(
        concurrency_cfg[
            "rounds"
        ]
    )

    seed = (
        cfg[
            "reproducibility"
        ]["seeds"][0]
    )

    full_reserve = (
        cfg[
            "constitution"
        ]["initial_reserve"]
    )

    reservation_db = (
        ROOT
        / "data"
        / "concurrency_reserve.sqlite"
    )

    replay_db = (
        ROOT
        / "data"
        / "concurrency_replay.sqlite"
    )

    # ========================================================
    # A. Reservation concurrency
    # ========================================================

    print(
        "=" * 84
    )

    print(
        "Concurrency / Replay Stress"
    )

    print(
        "=" * 84
    )

    print(
        "[1/2] Concurrent reserve stress..."
    )

    (
        round_df,
        reservation_summary,
    ) = (
        run_reservation_stress(
            db_path=reservation_db,

            reserve_units=(
                reserve_units
            ),

            reservation_units=(
                reservation_units
            ),

            requests_per_round=(
                requests_per_round
            ),

            worker_threads=(
                worker_threads
            ),

            rounds=rounds,
        )
    )

    # ========================================================
    # B. Replay / Commit concurrency
    # ========================================================

    print(
        "[2/2] Concurrent replay and Commit stress..."
    )

    replay_summary = (
        run_replay_and_commit_stress(
            db_path=replay_db,

            seed=seed,

            reserve=full_reserve,

            requests=(
                requests_per_round
            ),

            worker_threads=(
                worker_threads
            ),
        )
    )

    # ========================================================
    # Save raw reservation rounds
    # ========================================================

    raw_path = (
        ROOT
        / "results"
        / "raw"
        / "concurrency_raw.csv"
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    round_df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # Summary table
    # ========================================================

    summary_rows = [
        {
            "experiment":
                "Concurrent reservation",

            "attempts":
                reservation_summary[
                    "total_attempts"
                ],

            "successes":
                reservation_summary[
                    "total_successes"
                ],

            "expected_successes":
                reservation_summary[
                    "expected_total_successes"
                ],

            "rejections_or_other":
                (
                    reservation_summary[
                        "total_attempts"
                    ]
                    - reservation_summary[
                        "total_successes"
                    ]
                ),

            "unexpected_errors":
                reservation_summary[
                    "unexpected_errors"
                ],

            "violations":
                (
                    reservation_summary[
                        "negative_reserve_violations"
                    ]
                    + reservation_summary[
                        "overspend_violations"
                    ]
                    + reservation_summary[
                        "conservation_violations"
                    ]
                ),

            "pass":
                (
                    reservation_summary[
                        "all_rounds_exact"
                    ]
                    and reservation_summary[
                        "unexpected_errors"
                    ] == 0
                    and reservation_summary[
                        "negative_reserve_violations"
                    ] == 0
                    and reservation_summary[
                        "overspend_violations"
                    ] == 0
                    and reservation_summary[
                        "conservation_violations"
                    ] == 0
                ),
        },

        {
            "experiment":
                "Concurrent replay Execute",

            "attempts":
                replay_summary[
                    "execute_attempts"
                ],

            "successes":
                replay_summary[
                    "execute_successes"
                ],

            "expected_successes":
                1,

            "rejections_or_other":
                replay_summary[
                    "execute_replay_rejections"
                ],

            "unexpected_errors":
                replay_summary[
                    "execute_unexpected"
                ],

            "violations":
                int(
                    not replay_summary[
                        "exactly_one_execute"
                    ]
                ),

            "pass":
                (
                    replay_summary[
                        "exactly_one_execute"
                    ]
                    and replay_summary[
                        "execute_unexpected"
                    ] == 0
                ),
        },

        {
            "experiment":
                "Concurrent double Commit",

            "attempts":
                replay_summary[
                    "commit_attempts"
                ],

            "successes":
                replay_summary[
                    "commit_successes"
                ],

            "expected_successes":
                1,

            "rejections_or_other":
                replay_summary[
                    "commit_rejections"
                ],

            "unexpected_errors":
                replay_summary[
                    "commit_unexpected"
                ],

            "violations":
                int(
                    not replay_summary[
                        "exactly_one_commit"
                    ]
                ),

            "pass":
                (
                    replay_summary[
                        "exactly_one_commit"
                    ]
                    and replay_summary[
                        "commit_unexpected"
                    ] == 0
                    and replay_summary[
                        "conservation_ok"
                    ]
                ),
        },
    ]

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    summary_path = (
        ROOT
        / "results"
        / "tables"
        / "concurrency_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Terminal
    # ========================================================

    print()

    print(
        "=" * 84
    )

    print(
        "RESERVATION STRESS SUMMARY"
    )

    print(
        "=" * 84
    )

    for key, value in (
        reservation_summary.items()
    ):

        print(
            f"{key:34s}: "
            f"{value}"
        )

    print()

    print(
        "=" * 84
    )

    print(
        "REPLAY / COMMIT SUMMARY"
    )

    print(
        "=" * 84
    )

    for key, value in (
        replay_summary.items()
    ):

        print(
            f"{key:34s}: "
            f"{value}"
        )

    print()

    print(
        "=" * 84
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 84
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"[OK] raw -> "
        f"{raw_path}"
    )

    print(
        f"[OK] summary -> "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()