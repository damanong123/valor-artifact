from __future__ import annotations

from pathlib import Path

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
    ExecutionRevalidationError,
    ReplayDetectedError,
)

from valor.ledger import (
    InMemoryReserveLedger,
)

from valor.models import (
    Action,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# Helpers
# ============================================================

def make_exceptional_action(
    tool,
    fraction: float,
) -> Action:
    """
    Construct an exact action strictly outside the
    normal range but inside the adapter-certified
    exceptional range.
    """

    gap = (
        tool.exceptional_max
        - tool.normal_max
    )

    parameter = (
        tool.normal_max
        + gap
        * fraction
    )

    return Action(
        tool=tool.name,
        parameter=parameter,
    )


def vector_equal(
    left,
    right,
    epsilon=1e-9,
) -> bool:

    return all(
        abs(
            left[key]
            - right[key]
        )
        <= epsilon

        for key in left
    )


# ============================================================
# One trial
# ============================================================

def run_trial(
    *,
    seed: int,
    trial_index: int,
    condition: str,
    tools,
    adapter,
    dmax,
    reserve,
    lower_multiplier: float,
    higher_multiplier: float,
) -> dict[str, object]:

    # --------------------------------------------------------
    # Deterministically choose one allowed tool.
    # --------------------------------------------------------

    allowed_tools = [
        tool
        for tool in tools.values()
        if (
            tool.registered
            and tool.in_dmax
        )
    ]

    tool = allowed_tools[
        trial_index
        % len(
            allowed_tools
        )
    ]

    # Vary the exact exceptional parameter slightly.
    fraction = (
        0.20
        + (
            trial_index
            % 7
        )
        * 0.10
    )

    fraction = min(
        fraction,
        0.90,
    )

    action = make_exceptional_action(
        tool,
        fraction,
    )

    # --------------------------------------------------------
    # Fresh independent VALOR state per trial.
    # --------------------------------------------------------

    ledger = (
        InMemoryReserveLedger(
            reserve
        )
    )

    nonce = (
        f"e7-{seed}-"
        f"{condition}-"
        f"{trial_index:04d}"
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=b"K" * 32,
        epoch=1,
        capability_ttl_seconds=60.0,
        nonce_factory=lambda:
            nonce,
    )

    initial = (
        ledger.initial_reserve()
    )

    # --------------------------------------------------------
    # PROPOSE always occurs at baseline state xi.
    # --------------------------------------------------------

    propose_state = {
        "effect_multiplier":
            1.0
    }

    capability = (
        kernel.propose(
            action,

            state=(
                propose_state
            ),

            now_ns=1_000,
        )
    )

    reserved = (
        capability
        .reserved_effect_vector()
    )

    remaining_after_propose = (
        ledger
        .remaining_reserve()
    )

    # --------------------------------------------------------
    # Select execution-time state xe.
    # --------------------------------------------------------

    if condition == "stable":

        execute_state = {
            "effect_multiplier":
                1.0
        }

        expected_permit = True

    elif condition == "lower_effect":

        execute_state = {
            "effect_multiplier":
                lower_multiplier
        }

        expected_permit = True

    elif condition == "higher_effect":

        execute_state = {
            "effect_multiplier":
                higher_multiplier
        }

        expected_permit = False

    elif condition == "hard_guard_failure":

        execute_state = {
            "effect_multiplier":
                1.0,

            "force_hard_guard_failure":
                True,
        }

        expected_permit = False

    else:

        raise ValueError(
            f"Unknown E7 condition: "
            f"{condition}"
        )

    # --------------------------------------------------------
    # Fresh execution-time contract for reporting.
    # --------------------------------------------------------

    execution_contract = (
        adapter.contract(
            action,
            execute_state,
        )
    )

    execution_effect = (
        execution_contract
        .effect_vector()
    )

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    permitted = False

    aborted = False

    reason = ""

    replay_rejected = None

    try:

        permitted = kernel.execute(
            capability,
            action,

            state=(
                execute_state
            ),

            now_ns=2_000,
        )

    except ExecutionRevalidationError as exc:

        aborted = True

        reason = str(
            exc
        )

    # --------------------------------------------------------
    # State checks after Execute attempt
    # --------------------------------------------------------

    nonce_resolved = (
        ledger.is_resolved(
            capability.nonce
        )
    )

    in_flight = (
        ledger.is_in_flight(
            capability.nonce
        )
    )

    reservation_exists = (
        ledger.has_reservation(
            capability.nonce
        )
    )

    remaining_after_execute = (
        ledger
        .remaining_reserve()
    )

    conservation_ok = (
        ledger
        .conservation_holds()
    )

    # --------------------------------------------------------
    # Abort-specific checks
    # --------------------------------------------------------

    refund_ok = None

    if not expected_permit:

        refund_ok = (
            vector_equal(
                remaining_after_execute,
                initial,
            )
        )

        # Old aborted capability must be replay-resistant.
        try:

            kernel.execute(
                capability,
                action,

                state={
                    "effect_multiplier":
                        1.0
                },

                now_ns=3_000,
            )

            replay_rejected = False

        except ReplayDetectedError:

            replay_rejected = True

    # --------------------------------------------------------
    # Success-specific checks
    # --------------------------------------------------------

    if expected_permit:

        lifecycle_ok = (
            permitted
            and nonce_resolved
            and in_flight
            and reservation_exists
            and conservation_ok
        )

    else:

        lifecycle_ok = (
            aborted
            and nonce_resolved
            and not in_flight
            and not reservation_exists
            and refund_ok
            and replay_rejected
            and conservation_ok
        )

    outcome_correct = (
        permitted
        == expected_permit
    )

    correct = (
        outcome_correct
        and lifecycle_ok
    )

    return {
        "seed":
            seed,

        "trial_index":
            trial_index,

        "condition":
            condition,

        "tool":
            action.tool,

        "parameter":
            action.parameter,

        "expected_permit":
            expected_permit,

        "observed_permit":
            permitted,

        "aborted":
            aborted,

        "reason":
            reason,

        "reserved_energy":
            reserved[
                "energy"
            ],

        "execution_energy":
            execution_effect[
                "energy"
            ],

        "reserved_stress":
            reserved[
                "stress"
            ],

        "execution_stress":
            execution_effect[
                "stress"
            ],

        "hard_guard_ok":
            execution_contract
            .hard_guard_ok,

        "nonce_resolved":
            nonce_resolved,

        "in_flight":
            in_flight,

        "reservation_exists":
            reservation_exists,

        "refund_ok":
            refund_ok,

        "replay_rejected":
            replay_rejected,

        "conservation_ok":
            conservation_ok,

        "outcome_correct":
            outcome_correct,

        "lifecycle_ok":
            lifecycle_ok,

        "correct":
            correct,

        "remaining_energy_after_propose":
            remaining_after_propose[
                "energy"
            ],

        "remaining_energy_after_execute":
            remaining_after_execute[
                "energy"
            ],
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

    seeds = (
        cfg[
            "reproducibility"
        ]["seeds"]
    )

    reserve = (
        cfg[
            "constitution"
        ]["initial_reserve"]
    )

    supplemental = (
        cfg.get(
            "supplemental_experiments",
            {},
        )
    )

    e7_cfg = (
        supplemental.get(
            "state_drift",
            {},
        )
    )

    cases_per_condition = int(
        e7_cfg.get(
            "cases_per_condition_per_seed",
            100,
        )
    )

    lower_multiplier = float(
        e7_cfg.get(
            "lower_effect_multiplier",
            0.8,
        )
    )

    higher_multiplier = float(
        e7_cfg.get(
            "higher_effect_multiplier",
            1.5,
        )
    )

    conditions = [
        "stable",
        "lower_effect",
        "higher_effect",
        "hard_guard_failure",
    ]

    rows = []

    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    for seed in seeds:

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

        for condition in conditions:

            correct_count = 0

            for trial_index in range(
                cases_per_condition
            ):

                result = run_trial(
                    seed=seed,

                    trial_index=(
                        trial_index
                    ),

                    condition=(
                        condition
                    ),

                    tools=tools,

                    adapter=adapter,

                    dmax=dmax,

                    reserve=reserve,

                    lower_multiplier=(
                        lower_multiplier
                    ),

                    higher_multiplier=(
                        higher_multiplier
                    ),
                )

                rows.append(
                    result
                )

                if result[
                    "correct"
                ]:

                    correct_count += 1

            print(
                f"seed={seed} | "
                f"condition="
                f"{condition:20s} | "
                f"correct="
                f"{correct_count}/"
                f"{cases_per_condition}"
            )

    # ========================================================
    # Raw
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    raw_path = (
        ROOT
        / "results"
        / "raw"
        / "e7_state_drift_raw.csv"
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # Condition summary
    # ========================================================

    condition_summary = (
        df
        .groupby(
            "condition",
            sort=False,
        )
        .agg(
            trials=(
                "trial_index",
                "count",
            ),

            expected_permit_rate=(
                "expected_permit",
                "mean",
            ),

            observed_permit_rate=(
                "observed_permit",
                "mean",
            ),

            abort_rate=(
                "aborted",
                "mean",
            ),

            nonce_resolved_rate=(
                "nonce_resolved",
                "mean",
            ),

            conservation_rate=(
                "conservation_ok",
                "mean",
            ),

            lifecycle_correct_rate=(
                "lifecycle_ok",
                "mean",
            ),

            correct_rate=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        ROOT
        / "results"
        / "tables"
        / "e7_state_drift_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    condition_summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Abort-only summary
    # ========================================================

    abort_df = (
        df[
            ~df[
                "expected_permit"
            ]
        ]
    )

    abort_summary = pd.DataFrame(
        [
            {
                "abort_trials":
                    len(
                        abort_df
                    ),

                "refund_success_rate":
                    abort_df[
                        "refund_ok"
                    ].mean(),

                "replay_rejection_rate":
                    abort_df[
                        "replay_rejected"
                    ].mean(),

                "conservation_rate":
                    abort_df[
                        "conservation_ok"
                    ].mean(),
            }
        ]
    )

    abort_path = (
        ROOT
        / "results"
        / "tables"
        / "e7_abort_properties.csv"
    )

    abort_summary.to_csv(
        abort_path,
        index=False,
    )

    # ========================================================
    # Per-seed summary
    # ========================================================

    seed_summary = (
        df
        .groupby(
            "seed"
        )
        .agg(
            trials=(
                "trial_index",
                "count",
            ),

            correct_rate=(
                "correct",
                "mean",
            ),

            conservation_rate=(
                "conservation_ok",
                "mean",
            ),
        )
        .reset_index()
    )

    seed_path = (
        ROOT
        / "results"
        / "tables"
        / "e7_state_drift_by_seed.csv"
    )

    seed_summary.to_csv(
        seed_path,
        index=False,
    )

    # ========================================================
    # Terminal
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "E7 EXECUTION-TIME STATE DRIFT SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        condition_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Abort properties:"
    )

    print(
        abort_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "By seed:"
    )

    print(
        seed_summary.to_string(
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

    print(
        f"[OK] abort properties -> "
        f"{abort_path}"
    )

    print(
        f"[OK] seed summary -> "
        f"{seed_path}"
    )


if __name__ == "__main__":
    main()