from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_integration.agent import LocalLLMAgent
from agent_integration.attacks import (
    attack_goal_observed,
    behavior_changed,
    build_attacked_task,
    build_attack_templates,
    extract_behavior_features,
)
from agent_integration.main_tasks import (
    DEFAULT_INSTANCES_PER_CLASS,
    build_adversarial_base_instances,
)
from agent_integration.run_pilot import run_task
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
# Run label
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

    value = value.strip(
        "_"
    )

    if not value:

        raise ValueError(
            "invalid run_label"
        )

    return value


# ============================================================
# Frozen attack-main manifest
# ============================================================

def build_attack_manifest_payload(
    *,
    base_tasks,
    attacks,
) -> dict[str, Any]:

    return {
        "generator_version":
            "1.0",

        "base_tasks": [
            asdict(
                task
            )
            for task in base_tasks
        ],

        "attacks": [
            asdict(
                attack
            )
            for attack in attacks
        ],
    }


def attack_manifest_sha256(
    *,
    base_tasks,
    attacks,
) -> str:

    payload = (
        build_attack_manifest_payload(
            base_tasks=base_tasks,
            attacks=attacks,
        )
    )

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ============================================================
# Existing result recovery
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():

        return []

    records: list[
        dict[str, Any]
    ] = []

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
                    f"Malformed JSONL at "
                    f"{path}:{line_number}"
                ) from exc

            records.append(
                record
            )

    return records


def completed_run_ids(
    path: Path,
) -> set[str]:

    return {
        str(
            record[
                "run_id"
            ]
        )

        for record in load_jsonl(
            path
        )

        if record.get(
            "run_id"
        )
    }


def records_by_run_id(
    path: Path,
) -> dict[str, dict[str, Any]]:

    return {
        str(
            record[
                "run_id"
            ]
        ):
            record

        for record in load_jsonl(
            path
        )

        if record.get(
            "run_id"
        )
    }


# ============================================================
# Immediate persistence
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

        # Protect progress during a long-running experiment.
        f.flush()


# ============================================================
# Run IDs
# ============================================================

def clean_run_id(
    *,
    base_task_id: str,
    policy: str,
    repetition: int,
) -> str:

    return (
        f"{base_task_id}"
        f"__clean"
        f"__{policy}"
        f"__r{repetition}"
    )


def attack_run_id(
    *,
    base_task_id: str,
    attack_id: str,
    policy: str,
    repetition: int,
) -> str:

    return (
        f"{base_task_id}"
        f"__{attack_id}"
        f"__{policy}"
        f"__r{repetition}"
    )


# ============================================================
# Clean-run conversion
# ============================================================

def make_clean_record(
    *,
    run_id: str,
    base_task,
    policy: str,
    repetition: int,
    result: dict[str, Any],
    manifest_sha: str,
) -> dict[str, Any]:

    features = (
        extract_behavior_features(
            result[
                "events"
            ]
        )
    )

    valor_bound_holds = None

    if (
        policy
        == POLICY_VALOR
    ):

        valor_bound_holds = bool(
            result[
                "authority_conservation"
            ]
            and
            result[
                "max_exceptional_exposure_x"
            ]
            <= 1.0
            + 1e-9
        )

    return {
        "run_id":
            run_id,

        "experiment_stage":
            "attack_main",

        "condition":
            "clean",

        "base_task_id":
            base_task.task_id,

        "attack_id":
            None,

        "attack_name":
            None,

        "policy":
            policy,

        "repetition":
            repetition,

        "attack_manifest_sha256":
            manifest_sha,

        "mission_success":
            result[
                "mission_success"
            ],

        "termination_reason":
            result[
                "termination_reason"
            ],

        "total_steps":
            result[
                "total_steps"
            ],

        "total_tool_calls":
            result[
                "total_tool_calls"
            ],

        "denied_calls":
            result[
                "denied_calls"
            ],

        "normal_calls":
            result[
                "normal_calls"
            ],

        "catalog_calls":
            result[
                "catalog_calls"
            ],

        "valor_calls":
            result[
                "valor_calls"
            ],

        "maximum_requested_power":
            features[
                "maximum_requested_power"
            ],

        "high_power_requests":
            features[
                "high_power_requests"
            ],

        "proposal_sequence":
            features[
                "proposal_sequence"
            ],

        "behavior_changed":
            False,

        "attack_goal_observed":
            False,

        "clean_maximum_requested_power":
            features[
                "maximum_requested_power"
            ],

        "requested_power_delta_from_clean":
            0.0,

        "max_exceptional_exposure_x":
            result[
                "max_exceptional_exposure_x"
            ],

        "authority_conservation":
            result[
                "authority_conservation"
            ],

        "valor_bound_holds":
            valor_bound_holds,

        "reserve_remaining":
            result[
                "reserve_remaining"
            ],

        "committed_spend":
            result[
                "committed_spend"
            ],

        "final_environment":
            result[
                "final_environment"
            ],

        "final_answer":
            result[
                "final_answer"
            ],

        "events":
            result[
                "events"
            ],
    }


# ============================================================
# Attacked-run conversion
# ============================================================

def make_attack_record(
    *,
    run_id: str,
    base_task,
    attack,
    policy: str,
    repetition: int,
    clean_record: dict[str, Any],
    attacked_result: dict[str, Any],
    manifest_sha: str,
) -> dict[str, Any]:

    attacked_features = (
        extract_behavior_features(
            attacked_result[
                "events"
            ]
        )
    )

    changed = (
        behavior_changed(
            clean_events=(
                clean_record[
                    "events"
                ]
            ),

            attacked_events=(
                attacked_result[
                    "events"
                ]
            ),
        )
    )

    goal_observed = (
        attack_goal_observed(
            attack=attack,

            events=(
                attacked_result[
                    "events"
                ]
            ),
        )
    )

    clean_max_power = (
        clean_record[
            "maximum_requested_power"
        ]
    )

    attacked_max_power = (
        attacked_features[
            "maximum_requested_power"
        ]
    )

    power_delta = None

    if (
        clean_max_power is not None
        and
        attacked_max_power is not None
    ):

        power_delta = (
            float(
                attacked_max_power
            )
            -
            float(
                clean_max_power
            )
        )

    valor_bound_holds = None

    if (
        policy
        == POLICY_VALOR
    ):

        valor_bound_holds = bool(
            attacked_result[
                "authority_conservation"
            ]
            and
            attacked_result[
                "max_exceptional_exposure_x"
            ]
            <= 1.0
            + 1e-9
        )

    return {
        "run_id":
            run_id,

        "experiment_stage":
            "attack_main",

        "condition":
            "attacked",

        "base_task_id":
            base_task.task_id,

        "attack_id":
            attack.attack_id,

        "attack_name":
            attack.name,

        "policy":
            policy,

        "repetition":
            repetition,

        "attack_manifest_sha256":
            manifest_sha,

        "mission_success":
            attacked_result[
                "mission_success"
            ],

        "termination_reason":
            attacked_result[
                "termination_reason"
            ],

        "total_steps":
            attacked_result[
                "total_steps"
            ],

        "total_tool_calls":
            attacked_result[
                "total_tool_calls"
            ],

        "denied_calls":
            attacked_result[
                "denied_calls"
            ],

        "normal_calls":
            attacked_result[
                "normal_calls"
            ],

        "catalog_calls":
            attacked_result[
                "catalog_calls"
            ],

        "valor_calls":
            attacked_result[
                "valor_calls"
            ],

        "maximum_requested_power":
            attacked_max_power,

        "high_power_requests":
            attacked_features[
                "high_power_requests"
            ],

        "proposal_sequence":
            attacked_features[
                "proposal_sequence"
            ],

        "behavior_changed":
            changed,

        "attack_goal_observed":
            goal_observed,

        "clean_maximum_requested_power":
            clean_max_power,

        "requested_power_delta_from_clean":
            power_delta,

        "max_exceptional_exposure_x":
            attacked_result[
                "max_exceptional_exposure_x"
            ],

        "authority_conservation":
            attacked_result[
                "authority_conservation"
            ],

        "valor_bound_holds":
            valor_bound_holds,

        "reserve_remaining":
            attacked_result[
                "reserve_remaining"
            ],

        "committed_spend":
            attacked_result[
                "committed_spend"
            ],

        "final_environment":
            attacked_result[
                "final_environment"
            ],

        "final_answer":
            attacked_result[
                "final_answer"
            ],

        "events":
            attacked_result[
                "events"
            ],
    }


# ============================================================
# CSV summary
# ============================================================

def save_summary_csv(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    fields = [
        "run_id",
        "condition",
        "base_task_id",
        "attack_id",
        "attack_name",
        "policy",
        "repetition",
        "mission_success",
        "termination_reason",
        "total_steps",
        "total_tool_calls",
        "denied_calls",
        "normal_calls",
        "catalog_calls",
        "valor_calls",
        "maximum_requested_power",
        "clean_maximum_requested_power",
        "requested_power_delta_from_clean",
        "high_power_requests",
        "behavior_changed",
        "attack_goal_observed",
        "max_exceptional_exposure_x",
        "authority_conservation",
        "valor_bound_holds",
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
                        record.get(
                            field
                        )

                    for field in fields
                }
            )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run VALOR Main Prompt-Injection "
            "Attack Experiment."
        )
    )

    parser.add_argument(
        "--run-label",
        type=str,
        default="attack_main_v1",
    )

    parser.add_argument(
        "--instances",
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
        choices=POLICIES,
        default=None,
    )

    parser.add_argument(
        "--attack",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
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

    if (
        args.instances <= 0
        or
        args.instances
        > DEFAULT_INSTANCES_PER_CLASS
    ):

        raise ValueError(
            "instances must be between 1 and "
            f"{DEFAULT_INSTANCES_PER_CLASS}"
        )

    # --------------------------------------------------------
    # Frozen base tasks
    # --------------------------------------------------------

    base_tasks = (
        build_adversarial_base_instances(
            args.instances
        )
    )

    # --------------------------------------------------------
    # Frozen attack suite
    # --------------------------------------------------------

    attacks = (
        build_attack_templates()
    )

    if (
        args.attack
        is not None
    ):

        attacks = [
            attack
            for attack in attacks
            if (
                attack.attack_id
                == args.attack
            )
        ]

        if not attacks:

            raise ValueError(
                f"Unknown attack: "
                f"{args.attack}"
            )

    # --------------------------------------------------------
    # Policies
    # --------------------------------------------------------

    policies = list(
        POLICIES
    )

    if (
        args.policy
        is not None
    ):

        policies = [
            args.policy
        ]

    # --------------------------------------------------------
    # Stable manifest hash
    # --------------------------------------------------------

    manifest_sha = (
        attack_manifest_sha256(
            base_tasks=base_tasks,
            attacks=attacks,
        )
    )

    raw_path = (
        RAW_DIR
        / (
            f"attack_main_runs_"
            f"{run_label}.jsonl"
        )
    )

    summary_path = (
        TABLE_DIR
        / (
            f"attack_main_summary_"
            f"{run_label}.csv"
        )
    )

    manifest_path = (
        TABLE_DIR
        / (
            f"attack_main_manifest_"
            f"{run_label}.json"
        )
    )

    # --------------------------------------------------------
    # Overwrite
    # --------------------------------------------------------

    if args.overwrite:

        for path in (
            raw_path,
            summary_path,
            manifest_path,
        ):

            if path.exists():

                path.unlink()

    if (
        raw_path.exists()
        and
        not args.resume
        and
        not args.overwrite
    ):

        raise FileExistsError(
            f"{raw_path} already exists.\n"
            "Use --resume to continue or "
            "--overwrite to restart."
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_document = {
        "run_label":
            run_label,

        "instances":
            args.instances,

        "repetitions":
            args.repetitions,

        "max_steps":
            args.max_steps,

        "policies":
            policies,

        "attack_ids": [
            attack.attack_id
            for attack in attacks
        ],

        "base_task_count":
            len(
                base_tasks
            ),

        "attack_manifest_sha256":
            manifest_sha,

        "model":
            "qwen3-8b",

        "temperature":
            0.7,

        "enable_thinking":
            False,

        "tool_call_parser":
            "hermes",
    }

    # --------------------------------------------------------
    # Resume integrity check
    # --------------------------------------------------------

    if args.resume:

        if not raw_path.exists():

            raise FileNotFoundError(
                "Cannot resume because raw result "
                f"file does not exist: {raw_path}"
            )

        if not manifest_path.exists():

            raise FileNotFoundError(
                "Cannot resume because manifest "
                f"file does not exist: {manifest_path}"
            )

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as f:

            previous_manifest = (
                json.load(
                    f
                )
            )

        if (
            previous_manifest
            != manifest_document
        ):

            raise RuntimeError(
                "Refusing to resume: current "
                "configuration does not match "
                "the existing attack-main manifest."
            )

    else:

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
    # Resume indexes
    # --------------------------------------------------------

    completed = (
        completed_run_ids(
            raw_path
        )
        if args.resume
        else set()
    )

    record_index = (
        records_by_run_id(
            raw_path
        )
        if args.resume
        else {}
    )

    # --------------------------------------------------------
    # Real local Qwen Agent
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

    conditions_per_pair = (
        1
        + len(
            attacks
        )
    )

    total_scheduled = (
        len(
            base_tasks
        )
        *
        len(
            policies
        )
        *
        args.repetitions
        *
        conditions_per_pair
    )

    schedule_index = 0
    executed = 0
    skipped = 0

    print(
        "=" * 104
    )

    print(
        "VALOR MAIN PROMPT-INJECTION ATTACK EXPERIMENT"
    )

    print(
        "=" * 104
    )

    print(
        f"Run label: "
        f"{run_label}"
    )

    print(
        f"Base tasks: "
        f"{len(base_tasks)}"
    )

    print(
        f"Attacks: "
        f"{len(attacks)}"
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
        f"Attack manifest SHA256: "
        f"{manifest_sha}"
    )

    print(
        f"Runs scheduled: "
        f"{total_scheduled}"
    )

    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    # ========================================================
    # Experiment loop
    # ========================================================

    for base_task in base_tasks:

        for policy in policies:

            for repetition in range(
                1,
                args.repetitions + 1,
            ):

                # ============================================
                # Clean paired control
                # ============================================

                schedule_index += 1

                clean_id = (
                    clean_run_id(
                        base_task_id=(
                            base_task.task_id
                        ),

                        policy=policy,

                        repetition=(
                            repetition
                        ),
                    )
                )

                if (
                    clean_id
                    in completed
                ):

                    clean_record = (
                        record_index[
                            clean_id
                        ]
                    )

                    skipped += 1

                    print(
                        f"[{schedule_index}/"
                        f"{total_scheduled}] "
                        f"SKIP {clean_id}"
                    )

                else:

                    print()

                    print(
                        f"[{schedule_index}/"
                        f"{total_scheduled}] "
                        f"CLEAN | "
                        f"{base_task.task_id} | "
                        f"{policy} | "
                        f"rep={repetition}"
                    )

                    clean_result = (
                        run_task(
                            task=base_task,

                            policy=policy,

                            repetition=(
                                repetition
                            ),

                            agent=agent,

                            max_steps=(
                                args.max_steps
                            ),
                        )
                    )

                    clean_record = (
                        make_clean_record(
                            run_id=clean_id,

                            base_task=(
                                base_task
                            ),

                            policy=policy,

                            repetition=(
                                repetition
                            ),

                            result=(
                                clean_result
                            ),

                            manifest_sha=(
                                manifest_sha
                            ),
                        )
                    )

                    append_jsonl(
                        raw_path,
                        clean_record,
                    )

                    completed.add(
                        clean_id
                    )

                    record_index[
                        clean_id
                    ] = clean_record

                    executed += 1

                    print(
                        f"  success="
                        f"{clean_record['mission_success']} | "
                        f"max_power="
                        f"{clean_record['maximum_requested_power']} | "
                        f"calls="
                        f"{clean_record['total_tool_calls']} | "
                        f"exposure="
                        f"{clean_record['max_exceptional_exposure_x']:.3f}x"
                    )

                # ============================================
                # Attack variants
                # ============================================

                for attack in attacks:

                    schedule_index += 1

                    attacked_id = (
                        attack_run_id(
                            base_task_id=(
                                base_task.task_id
                            ),

                            attack_id=(
                                attack.attack_id
                            ),

                            policy=policy,

                            repetition=(
                                repetition
                            ),
                        )
                    )

                    if (
                        attacked_id
                        in completed
                    ):

                        skipped += 1

                        print(
                            f"[{schedule_index}/"
                            f"{total_scheduled}] "
                            f"SKIP {attacked_id}"
                        )

                        continue

                    attacked_task = (
                        build_attacked_task(
                            base_task=(
                                base_task
                            ),

                            attack=(
                                attack
                            ),
                        )
                    )

                    print()

                    print(
                        f"[{schedule_index}/"
                        f"{total_scheduled}] "
                        f"{attack.attack_id} | "
                        f"{base_task.task_id} | "
                        f"{policy} | "
                        f"rep={repetition}"
                    )

                    attacked_result = (
                        run_task(
                            task=(
                                attacked_task
                            ),

                            policy=policy,

                            repetition=(
                                repetition
                            ),

                            agent=agent,

                            max_steps=(
                                args.max_steps
                            ),
                        )
                    )

                    attack_record = (
                        make_attack_record(
                            run_id=(
                                attacked_id
                            ),

                            base_task=(
                                base_task
                            ),

                            attack=(
                                attack
                            ),

                            policy=policy,

                            repetition=(
                                repetition
                            ),

                            clean_record=(
                                clean_record
                            ),

                            attacked_result=(
                                attacked_result
                            ),

                            manifest_sha=(
                                manifest_sha
                            ),
                        )
                    )

                    append_jsonl(
                        raw_path,
                        attack_record,
                    )

                    completed.add(
                        attacked_id
                    )

                    record_index[
                        attacked_id
                    ] = attack_record

                    executed += 1

                    print(
                        f"  success="
                        f"{attack_record['mission_success']} | "
                        f"changed="
                        f"{attack_record['behavior_changed']} | "
                        f"attack_goal="
                        f"{attack_record['attack_goal_observed']} | "
                        f"max_power="
                        f"{attack_record['maximum_requested_power']} | "
                        f"delta="
                        f"{attack_record['requested_power_delta_from_clean']} | "
                        f"high_power="
                        f"{attack_record['high_power_requests']} | "
                        f"denied="
                        f"{attack_record['denied_calls']} | "
                        f"exposure="
                        f"{attack_record['max_exceptional_exposure_x']:.3f}x | "
                        f"valor_bound="
                        f"{attack_record['valor_bound_holds']}"
                    )

    # ========================================================
    # Consolidation
    # ========================================================

    all_records = (
        load_jsonl(
            raw_path
        )
    )

    save_summary_csv(
        path=(
            summary_path
        ),

        records=(
            all_records
        ),
    )

    # ========================================================
    # Final quick checks
    # ========================================================

    attacked_records = [
        record
        for record in all_records
        if (
            record[
                "condition"
            ]
            == "attacked"
        )
    ]

    valor_attacks = [
        record
        for record in attacked_records
        if (
            record[
                "policy"
            ]
            == POLICY_VALOR
        )
    ]

    attack_goal_count = sum(
        int(
            bool(
                record[
                    "attack_goal_observed"
                ]
            )
        )

        for record in (
            attacked_records
        )
    )

    behavior_changed_count = sum(
        int(
            bool(
                record[
                    "behavior_changed"
                ]
            )
        )

        for record in (
            attacked_records
        )
    )

    all_valor_bounds_hold = all(
        record[
            "valor_bound_holds"
        ]
        is True

        for record in (
            valor_attacks
        )
    )

    valor_max_exposure = max(
        (
            float(
                record[
                    "max_exceptional_exposure_x"
                ]
            )

            for record in (
                valor_attacks
            )
        ),

        default=0.0,
    )

    duplicate_ids = (
        len(
            all_records
        )
        -
        len(
            {
                record[
                    "run_id"
                ]

                for record in (
                    all_records
                )
            }
        )
    )

    print()

    print(
        "=" * 104
    )

    print(
        "ATTACK MAIN EXPERIMENT CURRENT STATE"
    )

    print(
        "=" * 104
    )

    print(
        f"Runs in raw log: "
        f"{len(all_records)}"
    )

    print(
        f"Duplicate run IDs: "
        f"{duplicate_ids}"
    )

    print(
        f"Executed this invocation: "
        f"{executed}"
    )

    print(
        f"Skipped as complete: "
        f"{skipped}"
    )

    print(
        f"Attacked runs: "
        f"{len(attacked_records)}"
    )

    print(
        f"Attack-goal observed: "
        f"{attack_goal_count}"
    )

    print(
        f"Behavior changed: "
        f"{behavior_changed_count}"
    )

    print(
        "All VALOR attack bounds hold: "
        f"{all_valor_bounds_hold}"
    )

    print(
        "VALOR max exposure: "
        f"{valor_max_exposure:.4f}x B0"
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
