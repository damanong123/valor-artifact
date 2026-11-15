from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RAW_PATH = (
    ROOT
    / "agent_integration"
    / "results"
    / "raw"
    / "main_runs_main_v1.jsonl"
)

OUT_PATH = (
    ROOT
    / "agent_integration"
    / "results"
    / "tables"
    / "main_mismatches_refined_main_v1.csv"
)

BREAKDOWN_PATH = (
    ROOT
    / "agent_integration"
    / "results"
    / "tables"
    / "main_mismatch_breakdown_refined_main_v1.csv"
)


def load_records():

    records = []

    with open(
        RAW_PATH,
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def classify(record):

    expected = record.get(
        "expected_behavior"
    )

    expected_met = record.get(
        "expected_behavior_met"
    )

    success = bool(
        record.get(
            "mission_success"
        )
    )

    denied = int(
        record.get(
            "denied_calls",
            0,
        )
    )

    termination = record.get(
        "termination_reason"
    )

    if expected_met is not False:
        return None

    # ------------------------------------------
    # Expected success, but task failed.
    # ------------------------------------------

    if expected == "success":

        if termination == "max_steps":

            return (
                "planner_loop_or_nonconvergence"
            )

        if denied > 0:

            return (
                "planner_failed_after_authorization_denial"
            )

        return (
            "planner_premature_finish_or_wrong_action"
        )

    # ------------------------------------------
    # Expected authorization failure, but no
    # authorization denial was ever exercised.
    # ------------------------------------------

    if expected in {
        "authorization_failure",
        "authorization_deadlock",
    }:

        if denied == 0:

            if termination == "max_steps":

                return (
                    "planner_never_reached_authorization_"
                    "boundary_max_steps"
                )

            return (
                "planner_never_reached_authorization_"
                "boundary"
            )

        return (
            "authorization_failure_observed_"
            "but_expected_flag_mismatch"
        )

    # ------------------------------------------
    # Expected VALOR security rejection.
    # ------------------------------------------

    if expected == "security_rejection":

        if denied == 0:

            return (
                "planner_never_reached_security_boundary"
            )

        if success:

            return (
                "unexpected_success_after_security_denial"
            )

        return (
            "security_rejection_observed_"
            "but_expected_flag_mismatch"
        )

    return "other"


def main():

    rows = []

    for record in load_records():

        category = classify(
            record
        )

        if category is None:
            continue

        rows.append(
            {
                "run_id":
                    record["run_id"],

                "task_id":
                    record["task_id"],

                "task_class":
                    record["task_class"],

                "policy":
                    record["policy"],

                "repetition":
                    record["repetition"],

                "expected_behavior":
                    record[
                        "expected_behavior"
                    ],

                "mission_success":
                    record[
                        "mission_success"
                    ],

                "termination_reason":
                    record[
                        "termination_reason"
                    ],

                "total_tool_calls":
                    record[
                        "total_tool_calls"
                    ],

                "denied_calls":
                    record[
                        "denied_calls"
                    ],

                "classification":
                    category,

                "called_tools":
                    json.dumps(
                        record.get(
                            "called_tools",
                            []
                        )
                    ),

                "final_answer":
                    record.get(
                        "final_answer"
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUT_PATH,
        index=False,
    )

    breakdown = (
        df[
            "classification"
        ]
        .value_counts()
        .rename_axis(
            "classification"
        )
        .reset_index(
            name="count"
        )
    )

    breakdown[
        "fraction"
    ] = (
        breakdown[
            "count"
        ]
        / len(df)
    )

    breakdown.to_csv(
        BREAKDOWN_PATH,
        index=False,
    )

    print(
        "=" * 90
    )

    print(
        "REFINED MAIN MISMATCH ANALYSIS"
    )

    print(
        "=" * 90
    )

    print()

    print(
        breakdown.to_string(
            index=False
        )
    )

    print()

    print(
        "By task class / policy:"
    )

    print()

    print(
        df.groupby(
            [
                "task_class",
                "policy",
                "classification",
            ]
        )
        .size()
        .to_string()
    )

    print()

    print(
        f"Total mismatches: "
        f"{len(df)}"
    )

    print()

    print(
        f"[OK] refined mismatches -> "
        f"{OUT_PATH}"
    )

    print(
        f"[OK] refined breakdown -> "
        f"{BREAKDOWN_PATH}"
    )


if __name__ == "__main__":
    main()
