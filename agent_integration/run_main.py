from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from agent_integration.agent import (
    LocalLLMAgent,
)

from agent_integration.main_tasks import (
    DEFAULT_INSTANCES_PER_CLASS,
    build_main_tasks,
    main_task_manifest_sha256,
    validate_main_tasks,
)

from agent_integration.run_pilot import (
    run_task,
)

from agent_integration.tool_gateway import (
    POLICY_CATALOG,
    POLICY_NORMAL_ONLY,
    POLICY_VALOR,
)


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULT_ROOT = (
    ROOT
    / "agent_integration"
    / "results"
)

RAW_DIR = (
    RESULT_ROOT
    / "raw"
)

TABLE_DIR = (
    RESULT_ROOT
    / "tables"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


POLICIES = (
    POLICY_NORMAL_ONLY,
    POLICY_CATALOG,
    POLICY_VALOR,
)


# ============================================================
# Run-label
# ============================================================

def sanitize_run_label(
    value: str,
) -> str:

    value = value.strip()

    if not value:
        raise ValueError(
            "run_label cannot be empty"
        )

    value = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    )

    value = value.strip("_")

    if not value:
        raise ValueError(
            "invalid run_label"
        )

    return value


# ============================================================
# Existing run recovery
# ============================================================

def load_completed_run_ids(
    path: Path,
) -> set[str]:

    if not path.exists():
        return set()

    completed = set()

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    "Malformed JSONL at "
                    f"{path}:{line_number}"
                ) from exc

            run_id = record.get(
                "run_id"
            )

            if run_id:
                completed.add(
                    str(run_id)
                )

    return completed


def load_existing_records(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        return []

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                records.append(
                    json.loads(
                        line
                    )
                )

    return records


# ============================================================
# Append one completed run immediately
# ============================================================

def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )

        f.write(
            "\n"
        )

        # Important for long-running experiment.
        f.flush()


# ============================================================
# Summary writer
# ============================================================

def save_summary_csv(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    fields = [
        "run_id",
        "task_id",
        "task_class",
        "policy",
        "repetition",
        "mission_success",
        "expected_behavior",
        "expected_behavior_met",
        "termination_reason",
        "total_steps",
        "total_tool_calls",
        "denied_calls",
        "normal_calls",
        "catalog_calls",
        "valor_calls",
        "novel_exact_proposals",
        "novel_exceptional_proposals",
        "novel_exceptional_admitted",
        "max_exceptional_exposure_x",
        "authority_conservation",
    ]

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:

            writer.writerow(
                {
                    field:
                        record.get(field)
                    for field in fields
                }
            )


# ============================================================
# Main-specific run ID
# ============================================================

def make_run_id(
    *,
    task_id: str,
    policy: str,
    repetition: int,
) -> str:

    return (
        f"{task_id}"
        f"__{policy}"
        f"__r{repetition}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run VALOR real-Agent Main Experiment."
        )
    )

    parser.add_argument(
        "--run-label",
        type=str,
        default="main_v1",
    )

    parser.add_argument(
        "--instances-per-class",
        type=int,
        default=(
            DEFAULT_INSTANCES_PER_CLASS
        ),
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--policy",
        type=str,
        choices=POLICIES,
        default=None,
    )

    parser.add_argument(
        "--task-class",
        type=str,
        default=None,
        choices=(
            "normal",
            "known_emergency",
            "novel_feasible",
            "novel_infeasible",
            "adversarial",
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip run IDs already present "
            "in the JSONL result file."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete an existing result set "
            "before starting."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    run_label = (
        sanitize_run_label(
            args.run_label
        )
    )

    # --------------------------------------------------------
    # Frozen Main tasks
    # --------------------------------------------------------

    tasks = (
        build_main_tasks(
            instances_per_class=(
                args.instances_per_class
            )
        )
    )

    validate_main_tasks(
        tasks
    )

    if args.task_class is not None:

        tasks = [
            task
            for task in tasks
            if (
                task.task_class
                == args.task_class
            )
        ]

    # --------------------------------------------------------
    # Fixed policy list
    # --------------------------------------------------------

    policies = list(
        POLICIES
    )

    if args.policy is not None:

        policies = [
            args.policy
        ]

    # --------------------------------------------------------
    # Result files
    # --------------------------------------------------------

    raw_path = (
        RAW_DIR
        / f"main_runs_{run_label}.jsonl"
    )

    summary_path = (
        TABLE_DIR
        / f"main_summary_{run_label}.csv"
    )

    manifest_path = (
        TABLE_DIR
        / f"main_manifest_{run_label}.json"
    )

    if (
        args.overwrite
        and raw_path.exists()
    ):
        raw_path.unlink()

    if (
        args.overwrite
        and summary_path.exists()
    ):
        summary_path.unlink()

    if (
        raw_path.exists()
        and not args.resume
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{raw_path} already exists.\n"
            "Use --resume to continue or "
            "--overwrite to restart."
        )

    # --------------------------------------------------------
    # Frozen task manifest
    # --------------------------------------------------------

    manifest_sha = (
        main_task_manifest_sha256(
            instances_per_class=(
                args.instances_per_class
            )
        )
    )

    manifest_document = {
        "run_label":
            run_label,

        "instances_per_class":
            args.instances_per_class,

        "repetitions":
            args.repetitions,

        "max_steps":
            args.max_steps,

        "task_manifest_sha256":
            manifest_sha,

        "policies":
            policies,

        "task_class_filter":
            args.task_class,

        "model":
            "qwen3-8b",

        "temperature":
            0.7,

        "enable_thinking":
            False,
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest_document,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Existing completed runs
    # --------------------------------------------------------

    completed_run_ids = (
        load_completed_run_ids(
            raw_path
        )
        if args.resume
        else set()
    )

    # --------------------------------------------------------
    # Real LLM planner
    # --------------------------------------------------------

    agent = LocalLLMAgent(
        base_url=(
            "http://127.0.0.1:8000/v1"
        ),

        model="qwen3-8b",

        temperature=0.7,

        max_tokens=512,

        enable_thinking=False,
    )

    # --------------------------------------------------------
    # Schedule size
    # --------------------------------------------------------

    total_scheduled = (
        len(tasks)
        * len(policies)
        * args.repetitions
    )

    skipped = 0
    executed = 0

    print(
        "=" * 100
    )

    print(
        "VALOR REAL-AGENT MAIN EXPERIMENT"
    )

    print(
        "=" * 100
    )

    print(
        f"Run label: "
        f"{run_label}"
    )

    print(
        f"Task manifest SHA256: "
        f"{manifest_sha}"
    )

    print(
        f"Tasks selected: "
        f"{len(tasks)}"
    )

    print(
        f"Policies: "
        f"{policies}"
    )

    print(
        f"Repetitions: "
        f"{args.repetitions}"
    )

    print(
        f"Runs scheduled: "
        f"{total_scheduled}"
    )

    print(
        f"Already completed: "
        f"{len(completed_run_ids)}"
    )

    # ========================================================
    # Experiment
    # ========================================================

    schedule_index = 0

    for task in tasks:

        for policy in policies:

            for repetition in range(
                1,
                args.repetitions + 1,
            ):

                schedule_index += 1

                run_id = make_run_id(
                    task_id=(
                        task.task_id
                    ),
                    policy=policy,
                    repetition=repetition,
                )

                if (
                    run_id
                    in completed_run_ids
                ):

                    skipped += 1

                    print(
                        f"[{schedule_index}/"
                        f"{total_scheduled}] "
                        f"SKIP {run_id}"
                    )

                    continue

                print()

                print(
                    f"[{schedule_index}/"
                    f"{total_scheduled}] "
                    f"{task.task_id} | "
                    f"{policy} | "
                    f"rep={repetition}"
                )

                result = (
                    run_task(
                        task=task,
                        policy=policy,
                        repetition=repetition,
                        agent=agent,
                        max_steps=(
                            args.max_steps
                        ),
                    )
                )

                # run_task already creates exactly
                # the same run_id structure, but fix it here
                # explicitly for Main Experiment consistency.
                result[
                    "run_id"
                ] = run_id

                result[
                    "main_task_manifest_sha256"
                ] = manifest_sha

                result[
                    "experiment_stage"
                ] = "main"

                append_jsonl(
                    raw_path,
                    result,
                )

                executed += 1

                print(
                    f"  success="
                    f"{result['mission_success']} | "
                    f"expected="
                    f"{result['expected_behavior']} | "
                    f"expected_met="
                    f"{result['expected_behavior_met']} | "
                    f"calls="
                    f"{result['total_tool_calls']} | "
                    f"denied="
                    f"{result['denied_calls']} | "
                    f"novel_exc="
                    f"{result['novel_exceptional_proposals']} | "
                    f"exposure="
                    f"{result['max_exceptional_exposure_x']:.3f}x | "
                    f"termination="
                    f"{result['termination_reason']}"
                )

    # ========================================================
    # Consolidate current JSONL into CSV
    # ========================================================

    records = (
        load_existing_records(
            raw_path
        )
    )

    save_summary_csv(
        path=summary_path,
        records=records,
    )

    # ========================================================
    # Final checks
    # ========================================================

    valor_records = [
        record
        for record in records
        if (
            record[
                "policy"
            ]
            == POLICY_VALOR
        )
    ]

    all_valor_conservation = all(
        record[
            "authority_conservation"
        ]
        for record in valor_records
    )

    expected_records = [
        record
        for record in records
        if (
            record[
                "expected_behavior_met"
            ]
            is not None
        )
    ]

    expected_behavior_rate = (
        sum(
            int(
                bool(
                    record[
                        "expected_behavior_met"
                    ]
                )
            )
            for record in (
                expected_records
            )
        )
        / len(
            expected_records
        )
        if expected_records
        else 0.0
    )

    print()

    print(
        "=" * 100
    )

    print(
        "MAIN EXPERIMENT CURRENT STATE"
    )

    print(
        "=" * 100
    )

    print(
        f"Runs in raw log: "
        f"{len(records)}"
    )

    print(
        f"Executed this invocation: "
        f"{executed}"
    )

    print(
        f"Skipped as already complete: "
        f"{skipped}"
    )

    print(
        f"Expected-behavior rate: "
        f"{expected_behavior_rate * 100:.2f}%"
    )

    print(
        f"VALOR authority conservation: "
        f"{all_valor_conservation}"
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
        f"[OK] manifest -> "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
