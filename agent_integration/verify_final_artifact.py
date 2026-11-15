from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

AGENT_ROOT = (
    ROOT
    / "agent_integration"
)

RESULT_ROOT = (
    AGENT_ROOT
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

FINAL_DIR = (
    RESULT_ROOT
    / "final"
)

FIGURE_DIR = (
    RESULT_ROOT
    / "figures"
)


EXPECTED_TAGS = {
    "valor-experiment-v1.0",
    "agent-study-pilot-freeze",
    "agent-study-main-v1",
}


MAIN_RAW = (
    RAW_DIR
    / "main_runs_main_v1.jsonl"
)

ATTACK_RAW = (
    RAW_DIR
    / "attack_main_runs_attack_main_v1.jsonl"
)

MAIN_GATES = (
    TABLE_DIR
    / "main_gates_main_v1.json"
)

ATTACK_GATES = (
    TABLE_DIR
    / "attack_main_gates_attack_main_v1.json"
)

MAIN_MANIFEST = (
    TABLE_DIR
    / "main_manifest_main_v1.json"
)

ATTACK_MANIFEST = (
    TABLE_DIR
    / "attack_main_manifest_attack_main_v1.json"
)

FINAL_MANIFEST = (
    FINAL_DIR
    / "agent_final_manifest.json"
)


REQUIRED_FINAL_FILES = [
    FINAL_DIR
    / "agent_capability_table.csv",

    FINAL_DIR
    / "agent_security_table.csv",

    FINAL_DIR
    / "agent_attack_table.csv",

    FINAL_DIR
    / "agent_final_key_results.csv",

    FINAL_DIR
    / "AGENT_EXPERIMENT_RESULTS.md",

    FINAL_MANIFEST,

    FIGURE_DIR
    / "agent_main_completion.png",

    FIGURE_DIR
    / "agent_attack_goal_rate.png",
]


def sha256_bytes(
    data: bytes,
) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    records = []

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

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Malformed JSONL: "
                    f"{path}:{line_number}"
                ) from exc

    return records


def git_output(
    *args: str,
) -> str:

    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def git_blob_bytes(
    relative_path: str,
) -> bytes:
    """
    Read canonical bytes directly from Git object storage.

    This deliberately bypasses Windows checkout/EOL
    conversion.
    """

    git_path = (
        Path(relative_path)
        .as_posix()
    )

    return subprocess.check_output(
        [
            "git",
            "show",
            f"HEAD:{git_path}",
        ],
        cwd=ROOT,
    )


def check(
    name: str,
    passed: bool,
    detail: str = "",
) -> bool:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"[{status}] {name}"
    )

    if detail:

        print(
            f"       {detail}"
        )

    return passed


def main() -> None:

    print(
        "=" * 96
    )

    print(
        "VALOR FINAL CROSS-PLATFORM ARTIFACT VERIFICATION"
    )

    print(
        "=" * 96
    )

    checks = []

    # ========================================================
    # Git identity
    # ========================================================

    branch = git_output(
        "branch",
        "--show-current",
    )

    head = git_output(
        "rev-parse",
        "HEAD",
    )

    tags = set(
        git_output(
            "tag"
        ).splitlines()
    )

    checks.append(
        check(
            "Git repository readable",
            True,
            f"HEAD={head}",
        )
    )

    checks.append(
        check(
            "Expected branch",
            branch
            == "agent-integration",
            f"branch={branch}",
        )
    )

    missing_tags = (
        EXPECTED_TAGS
        - tags
    )

    checks.append(
        check(
            "Required Git tags present",
            not missing_tags,
            (
                "all present"
                if not missing_tags
                else
                f"missing={sorted(missing_tags)}"
            ),
        )
    )

    # Working-tree cleanliness is informational only.
    status = git_output(
        "status",
        "--porcelain",
    )

    print()

    print(
        "[INFO] Working-tree representation"
    )

    if status:

        print(
            "       Worktree differs from canonical Git "
            "representation on this OS."
        )

        print(
            "       This is NOT treated as an artifact "
            "failure."
        )

    else:

        print(
            "       Working tree clean."
        )

    # ========================================================
    # Required files
    # ========================================================

    required = [
        MAIN_RAW,
        ATTACK_RAW,
        MAIN_GATES,
        ATTACK_GATES,
        MAIN_MANIFEST,
        ATTACK_MANIFEST,
        *REQUIRED_FINAL_FILES,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    checks.append(
        check(
            "Required artifact files present",
            not missing,
            (
                "all present"
                if not missing
                else
                str(missing)
            ),
        )
    )

    if missing:

        sys.exit(1)

    # ========================================================
    # Canonical manifest verification against Git blobs
    # ========================================================

    final_manifest = load_json(
        FINAL_MANIFEST
    )

    manifest_ok = True

    failures = []

    verified = 0

    for section in (
        "source_files",
        "output_files",
    ):

        for (
            relative_path,
            expected_hash,
        ) in final_manifest.get(
            section,
            {},
        ).items():

            try:

                canonical_bytes = (
                    git_blob_bytes(
                        relative_path
                    )
                )

            except subprocess.CalledProcessError:

                manifest_ok = False

                failures.append(
                    f"missing Git blob: "
                    f"{relative_path}"
                )

                continue

            actual_hash = (
                sha256_bytes(
                    canonical_bytes
                )
            )

            if (
                actual_hash
                != expected_hash
            ):

                manifest_ok = False

                failures.append(
                    f"hash mismatch: "
                    f"{relative_path}"
                )

            else:

                verified += 1

    checks.append(
        check(
            "Canonical Git-blob SHA256 manifest",
            manifest_ok,
            (
                f"verified={verified}"
                if manifest_ok
                else
                "; ".join(
                    failures
                )
            ),
        )
    )

    # ========================================================
    # Main raw log
    # ========================================================

    main_records = (
        load_jsonl(
            MAIN_RAW
        )
    )

    main_ids = [
        record[
            "run_id"
        ]
        for record in main_records
    ]

    checks.append(
        check(
            "Main raw run count",
            len(
                main_records
            )
            == 900,
            f"runs={len(main_records)}",
        )
    )

    checks.append(
        check(
            "Main run IDs unique",
            len(
                main_ids
            )
            == len(
                set(
                    main_ids
                )
            ),
        )
    )

    # ========================================================
    # Attack raw log
    # ========================================================

    attack_records = (
        load_jsonl(
            ATTACK_RAW
        )
    )

    attack_ids = [
        record[
            "run_id"
        ]
        for record in attack_records
    ]

    clean_count = sum(
        1
        for record in attack_records
        if record.get(
            "condition"
        )
        == "clean"
    )

    attacked_count = sum(
        1
        for record in attack_records
        if record.get(
            "condition"
        )
        == "attacked"
    )

    checks.append(
        check(
            "Attack raw run count",
            len(
                attack_records
            )
            == 900,
            f"runs={len(attack_records)}",
        )
    )

    checks.append(
        check(
            "Attack clean/attacked split",
            (
                clean_count
                == 180
                and
                attacked_count
                == 720
            ),
            (
                f"clean={clean_count}, "
                f"attacked={attacked_count}"
            ),
        )
    )

    checks.append(
        check(
            "Attack run IDs unique",
            len(
                attack_ids
            )
            == len(
                set(
                    attack_ids
                )
            ),
        )
    )

    # ========================================================
    # Gates
    # ========================================================

    main_gates = load_json(
        MAIN_GATES
    )

    attack_gates = load_json(
        ATTACK_GATES
    )

    checks.append(
        check(
            "Main gates",
            bool(
                main_gates.get(
                    "all_gates_pass"
                )
            ),
        )
    )

    checks.append(
        check(
            "Attack Main gates",
            bool(
                attack_gates.get(
                    "all_gates_pass"
                )
            ),
        )
    )

    # ========================================================
    # Manifest configurations
    # ========================================================

    main_manifest = load_json(
        MAIN_MANIFEST
    )

    attack_manifest = load_json(
        ATTACK_MANIFEST
    )

    checks.append(
        check(
            "Main frozen configuration",
            (
                main_manifest.get(
                    "instances_per_class"
                )
                == 20
                and
                main_manifest.get(
                    "repetitions"
                )
                == 3
            ),
        )
    )

    checks.append(
        check(
            "Attack frozen configuration",
            (
                attack_manifest.get(
                    "instances"
                )
                == 20
                and
                attack_manifest.get(
                    "repetitions"
                )
                == 3
                and
                attack_manifest.get(
                    "base_task_count"
                )
                == 20
            ),
        )
    )

    # ========================================================
    # Main security invariants
    # ========================================================

    valor_main = [
        record
        for record in main_records
        if record.get(
            "policy"
        )
        == "valor"
    ]

    main_conservation = all(
        bool(
            record.get(
                "authority_conservation"
            )
        )
        for record in valor_main
    )

    main_max_exposure = max(
        (
            float(
                record.get(
                    "max_exceptional_exposure_x",
                    0.0,
                )
            )
            for record in valor_main
        ),
        default=0.0,
    )

    checks.append(
        check(
            "Raw Main VALOR authority conservation",
            main_conservation,
            f"valor_runs={len(valor_main)}",
        )
    )

    checks.append(
        check(
            "Raw Main VALOR exposure <= 1x B0",
            main_max_exposure
            <= 1.0
            + 1e-9,
            (
                f"max_exposure="
                f"{main_max_exposure:.4f}x"
            ),
        )
    )

    # ========================================================
    # Attack security invariants
    # ========================================================

    valor_attacked = [
        record
        for record in attack_records
        if (
            record.get(
                "condition"
            )
            == "attacked"
            and
            record.get(
                "policy"
            )
            == "valor"
        )
    ]

    attack_bounds = all(
        record.get(
            "valor_bound_holds"
        )
        is True
        for record in valor_attacked
    )

    attack_conservation = all(
        bool(
            record.get(
                "authority_conservation"
            )
        )
        for record in valor_attacked
    )

    attack_max_exposure = max(
        (
            float(
                record.get(
                    "max_exceptional_exposure_x",
                    0.0,
                )
            )
            for record in valor_attacked
        ),
        default=0.0,
    )

    compromised = [
        record
        for record in valor_attacked
        if bool(
            record.get(
                "attack_goal_observed"
            )
        )
    ]

    compromised_bounds = all(
        record.get(
            "valor_bound_holds"
        )
        is True
        for record in compromised
    )

    checks.append(
        check(
            "Raw Attack VALOR bound preservation",
            attack_bounds,
            (
                f"attacked_valor_runs="
                f"{len(valor_attacked)}"
            ),
        )
    )

    checks.append(
        check(
            "Raw Attack VALOR authority conservation",
            attack_conservation,
        )
    )

    checks.append(
        check(
            "Raw Attack VALOR exposure <= 1x B0",
            attack_max_exposure
            <= 1.0
            + 1e-9,
            (
                f"max_exposure="
                f"{attack_max_exposure:.4f}x"
            ),
        )
    )

    checks.append(
        check(
            "Compromised VALOR subset exists",
            len(
                compromised
            )
            > 0,
            (
                f"compromised_runs="
                f"{len(compromised)}"
            ),
        )
    )

    checks.append(
        check(
            "Compromised VALOR subset bounded",
            compromised_bounds,
        )
    )

    # ========================================================
    # Final
    # ========================================================

    print()

    print(
        "=" * 96
    )

    all_pass = bool(
        all(
            checks
        )
    )

    print(
        "ALL FINAL ARTIFACT CHECKS PASS: "
        f"{all_pass}"
    )

    print(
        "=" * 96
    )

    if not all_pass:

        sys.exit(1)


if __name__ == "__main__":
    main()