from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

DOCS = ROOT / "docs"

FINAL_DIR = (
    RESULTS
    / "final"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOCS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Expected result files
# ============================================================

FILES = {
    "rq1":
        TABLES
        / "rq1_summary.csv",

    "rq2":
        TABLES
        / "rq2_ablation_summary.csv",

    "rq3":
        TABLES
        / "rq3_latency_summary.csv",

    "rq3_env":
        RAW
        / "rq3_environment.json",

    "rq4":
        TABLES
        / "rq4_summary.csv",

    "rq5":
        TABLES
        / "rq5_summary.csv",

    "rq5_mode":
        TABLES
        / "rq5_by_mode.csv",

    "concurrency":
        TABLES
        / "concurrency_summary.csv",

    "e6":
        TABLES
        / "e6_cumulative_exhaustion_summary.csv",

    "e6_dimension":
        TABLES
        / "e6_cumulative_exhaustion_by_dimension.csv",

    "e7":
        TABLES
        / "e7_state_drift_summary.csv",

    "e7_abort":
        TABLES
        / "e7_abort_properties.csv",

    "e8":
        TABLES
        / "e8_attack_then_emergency_summary.csv",
}


# ============================================================
# Utility
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
    )


def git_commit() -> str | None:

    try:

        value = (
            subprocess
            .check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=ROOT,
                text=True,
                stderr=(
                    subprocess.DEVNULL
                ),
            )
            .strip()
        )

        return value

    except Exception:

        return None


def git_dirty() -> bool | None:

    try:

        output = (
            subprocess
            .check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                ],
                cwd=ROOT,
                text=True,
                stderr=(
                    subprocess.DEVNULL
                ),
            )
        )

        return bool(
            output.strip()
        )

    except Exception:

        return None


def require_files() -> None:

    missing = [
        path
        for path in (
            FILES.values()
        )
        if not path.exists()
    ]

    if missing:

        print(
            "Missing required "
            "experiment outputs:"
        )

        for path in missing:

            print(
                f"  - {path}"
            )

        raise SystemExit(
            1
        )


def read_csv(
    key: str,
) -> pd.DataFrame:

    return pd.read_csv(
        FILES[key]
    )


def format_value(
    value: Any,
) -> str:

    if pd.isna(
        value
    ):
        return ""

    if isinstance(
        value,
        bool,
    ):
        return (
            "True"
            if value
            else "False"
        )

    if isinstance(
        value,
        int,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        float,
    ):

        # Large count-like numbers
        if (
            abs(
                value
                - round(
                    value
                )
            )
            < 1e-10
            and abs(value) >= 100
        ):

            return str(
                int(
                    round(
                        value
                    )
                )
            )

        return (
            f"{value:.4f}"
        )

    return str(
        value
    )


def markdown_table(
    df: pd.DataFrame,
    *,
    columns: list[str],
    labels:
        dict[str, str]
        | None = None,
    decimals:
        dict[str, int]
        | None = None,
) -> str:

    labels = (
        labels
        or {}
    )

    decimals = (
        decimals
        or {}
    )

    headers = [
        labels.get(
            column,
            column,
        )
        for column in columns
    ]

    lines = []

    lines.append(
        "| "
        + " | ".join(
            headers
        )
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |"
    )

    for _, row in (
        df.iterrows()
    ):

        values = []

        for column in columns:

            value = row[
                column
            ]

            if pd.isna(
                value
            ):

                text = ""

            elif (
                column
                in decimals
            ):

               text = (
    f"{float(value):.{decimals[column]}f}"
)

            elif isinstance(
                value,
                bool,
            ):

                text = (
                    "True"
                    if value
                    else "False"
                )

            else:

                text = (
                    format_value(
                        value
                    )
                )

            values.append(
                text
            )

        lines.append(
            "| "
            + " | ".join(
                values
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def add_key_result(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    setting: str,
    metric: str,
    value: Any,
    unit: str,
) -> None:

    rows.append(
        {
            "experiment":
                experiment,

            "setting":
                setting,

            "metric":
                metric,

            "value":
                value,

            "unit":
                unit,
        }
    )


# ============================================================
# Validation checks
# ============================================================

def run_final_checks(
    rq1: pd.DataFrame,
    rq2: pd.DataFrame,
    rq5: pd.DataFrame,
    concurrency: pd.DataFrame,
    e6: pd.DataFrame,
    e7: pd.DataFrame,
    e8: pd.DataFrame,
) -> dict[str, bool]:

    checks = {}

    # --------------------------------------------------------
    # RQ1
    # --------------------------------------------------------

    checks[
        "rq1_valor_coverage_100"
    ] = bool(
        (
            rq1[
                "valor_coverage_mean"
            ]
            >= 99.999
        ).all()
    )

    checks[
        "rq1_valor_exposure_bounded"
    ] = bool(
        (
            rq1[
                "valor_exposure_mean"
            ]
            <= 1.0
            + 1e-9
        ).all()
    )

    # --------------------------------------------------------
    # RQ2
    # --------------------------------------------------------

    rq2_indexed = (
        rq2
        .set_index(
            "method"
        )
    )

    required_methods = {
        "Union Catalog",
        "Catalog + Reserve",
        "Envelope Only",
        "Full VALOR",
    }

    checks[
        "rq2_methods_present"
    ] = (
        required_methods
        .issubset(
            set(
                rq2_indexed.index
            )
        )
    )

    if checks[
        "rq2_methods_present"
    ]:

        checks[
            "rq2_full_valor_coverage_100"
        ] = (
            float(
                rq2_indexed.loc[
                    "Full VALOR",
                    "coverage_mean",
                ]
            )
            >= 99.999
        )

        checks[
            "rq2_full_valor_exposure_bounded"
        ] = (
            float(
                rq2_indexed.loc[
                    "Full VALOR",
                    "exposure_mean",
                ]
            )
            <= 1.0
            + 1e-9
        )

    # --------------------------------------------------------
    # RQ5
    # --------------------------------------------------------

    checks[
        "rq5_all_correct"
    ] = bool(
        (
            rq5[
                "correctly_handled_pct"
            ]
            >= 99.999
        ).all()
    )

    # --------------------------------------------------------
    # Concurrency
    # --------------------------------------------------------

    checks[
        "concurrency_all_pass"
    ] = bool(
        concurrency[
            "pass"
        ].all()
    )

    checks[
        "concurrency_no_violations"
    ] = bool(
        (
            concurrency[
                "violations"
            ]
            == 0
        ).all()
    )

    # --------------------------------------------------------
    # E6
    # --------------------------------------------------------

    checks[
        "e6_cumulative_rejection_correct"
    ] = (
        float(
            e6.iloc[0][
                "correct_rejection_rate"
            ]
        )
        >= 0.999999
    )

    # --------------------------------------------------------
    # E7
    # --------------------------------------------------------

    checks[
        "e7_all_lifecycle_correct"
    ] = bool(
        (
            e7[
                "lifecycle_correct_rate"
            ]
            >= 0.999999
        ).all()
    )

    checks[
        "e7_all_conservation"
    ] = bool(
        (
            e7[
                "conservation_rate"
            ]
            >= 0.999999
        ).all()
    )

    # --------------------------------------------------------
    # E8
    # --------------------------------------------------------

    ordered = (
        e8
        .sort_values(
            "requested_attack_steps"
        )
    )

    baseline = (
        ordered[
            ordered[
                "requested_attack_steps"
            ]
            == 0
        ]
    )

    checks[
        "e8_baseline_100"
    ] = (
        not baseline.empty
        and
        float(
            baseline.iloc[0][
                "recovery_admission_mean"
            ]
        )
        >= 99.999
    )

    rates = (
        ordered[
            "recovery_admission_mean"
        ]
        .tolist()
    )

    checks[
        "e8_recovery_monotone"
    ] = all(
        rates[index + 1]
        <= rates[index]
        + 1e-9

        for index in range(
            len(rates)
            - 1
        )
    )

    return checks


# ============================================================
# Main
# ============================================================

def main():

    require_files()

    # ========================================================
    # Load result tables
    # ========================================================

    rq1 = read_csv(
        "rq1"
    )

    rq2 = read_csv(
        "rq2"
    )

    rq3 = read_csv(
        "rq3"
    )

    rq4 = read_csv(
        "rq4"
    )

    rq5 = read_csv(
        "rq5"
    )

    rq5_mode = read_csv(
        "rq5_mode"
    )

    concurrency = read_csv(
        "concurrency"
    )

    e6 = read_csv(
        "e6"
    )

    e6_dimension = read_csv(
        "e6_dimension"
    )

    e7 = read_csv(
        "e7"
    )

    e7_abort = read_csv(
        "e7_abort"
    )

    e8 = read_csv(
        "e8"
    )

    with open(
        FILES[
            "rq3_env"
        ],
        "r",
        encoding="utf-8",
    ) as f:

        rq3_env = json.load(
            f
        )

    # ========================================================
    # Final consistency checks
    # ========================================================

    checks = run_final_checks(
        rq1=rq1,
        rq2=rq2,
        rq5=rq5,
        concurrency=concurrency,
        e6=e6,
        e7=e7,
        e8=e8,
    )

    all_checks_pass = all(
        checks.values()
    )

    # ========================================================
    # Key result long-form CSV
    # ========================================================

    key_rows = []

    # --------------------------------------------------------
    # RQ1
    # --------------------------------------------------------

    for _, row in (
        rq1.iterrows()
    ):

        setting = (
            f"K={int(row['K'])}"
        )

        add_key_result(
            key_rows,
            experiment="RQ1",
            setting=setting,
            metric="union_coverage",
            value=row[
                "union_coverage_mean"
            ],
            unit="percent",
        )

        add_key_result(
            key_rows,
            experiment="RQ1",
            setting=setting,
            metric="valor_coverage",
            value=row[
                "valor_coverage_mean"
            ],
            unit="percent",
        )

        add_key_result(
            key_rows,
            experiment="RQ1",
            setting=setting,
            metric="union_exposure",
            value=row[
                "union_exposure_mean"
            ],
            unit="x_reserve",
        )

        add_key_result(
            key_rows,
            experiment="RQ1",
            setting=setting,
            metric="valor_exposure",
            value=row[
                "valor_exposure_mean"
            ],
            unit="x_reserve",
        )

    # --------------------------------------------------------
    # RQ2
    # --------------------------------------------------------

    for _, row in (
        rq2.iterrows()
    ):

        add_key_result(
            key_rows,
            experiment="RQ2",
            setting=str(
                row[
                    "method"
                ]
            ),
            metric="coverage",
            value=row[
                "coverage_mean"
            ],
            unit="percent",
        )

        add_key_result(
            key_rows,
            experiment="RQ2",
            setting=str(
                row[
                    "method"
                ]
            ),
            metric="exposure",
            value=row[
                "exposure_mean"
            ],
            unit="x_reserve",
        )

    # --------------------------------------------------------
    # RQ3
    # --------------------------------------------------------

    for _, row in (
        rq3.iterrows()
    ):

        operation = str(
            row[
                "operation"
            ]
        )

        for metric in [
            "median_us",
            "p95_us",
            "p99_us",
        ]:

            add_key_result(
                key_rows,
                experiment="RQ3",
                setting=operation,
                metric=metric,
                value=row[
                    metric
                ],
                unit="microseconds",
            )

    # --------------------------------------------------------
    # RQ4
    # --------------------------------------------------------

    rq4_row = (
        rq4.iloc[0]
    )

    for metric in [
        "declared_exposure_mean",
        "true_exposure_mean",
        "declared_energy_x_mean",
        "true_energy_x_mean",
        "declared_safe_rate",
        "true_safe_rate",
        "physical_violation_rate",
    ]:

        if metric in (
            rq4.columns
        ):

            add_key_result(
                key_rows,
                experiment="RQ4",
                setting="30% energy underestimation",
                metric=metric,
                value=rq4_row[
                    metric
                ],
                unit=(
                    "rate"
                    if metric.endswith(
                        "_rate"
                    )
                    else "x_reserve"
                ),
            )

    # --------------------------------------------------------
    # RQ5
    # --------------------------------------------------------

    for _, row in (
        rq5.iterrows()
    ):

        add_key_result(
            key_rows,
            experiment="RQ5",
            setting=str(
                row[
                    "trace_class"
                ]
            ),
            metric="correctly_handled",
            value=row[
                "correctly_handled_pct"
            ],
            unit="percent",
        )

    # --------------------------------------------------------
    # Concurrency
    # --------------------------------------------------------

    for _, row in (
        concurrency.iterrows()
    ):

        add_key_result(
            key_rows,
            experiment="Concurrency",
            setting=str(
                row[
                    "experiment"
                ]
            ),
            metric="successes",
            value=row[
                "successes"
            ],
            unit="count",
        )

        add_key_result(
            key_rows,
            experiment="Concurrency",
            setting=str(
                row[
                    "experiment"
                ]
            ),
            metric="violations",
            value=row[
                "violations"
            ],
            unit="count",
        )

    # --------------------------------------------------------
    # E6
    # --------------------------------------------------------

    e6_row = (
        e6.iloc[0]
    )

    for metric in [
        "correct_rejection_rate",
        "individually_admissible_rate",
        "inside_envelope_rate",
        "cumulative_rejection_rate",
    ]:

        add_key_result(
            key_rows,
            experiment="E6",
            setting="Cumulative exhaustion",
            metric=metric,
            value=e6_row[
                metric
            ],
            unit="rate",
        )

    # --------------------------------------------------------
    # E7
    # --------------------------------------------------------

    for _, row in (
        e7.iterrows()
    ):

        add_key_result(
            key_rows,
            experiment="E7",
            setting=str(
                row[
                    "condition"
                ]
            ),
            metric="correct_rate",
            value=row[
                "correct_rate"
            ],
            unit="rate",
        )

    # --------------------------------------------------------
    # E8
    # --------------------------------------------------------

    for _, row in (
        e8.iterrows()
    ):

        setting = (
            "requested_attack_steps="
            f"{int(row['requested_attack_steps'])}"
        )

        add_key_result(
            key_rows,
            experiment="E8",
            setting=setting,
            metric="attack_exposure",
            value=row[
                "attack_exposure_mean"
            ],
            unit="x_reserve",
        )

        add_key_result(
            key_rows,
            experiment="E8",
            setting=setting,
            metric="recovery_admission",
            value=row[
                "recovery_admission_mean"
            ],
            unit="percent",
        )

    key_df = (
        pd.DataFrame(
            key_rows
        )
    )

    key_path = (
        FINAL_DIR
        / "final_key_results.csv"
    )

    key_df.to_csv(
        key_path,
        index=False,
    )

    # ========================================================
    # Git / reproducibility metadata
    # ========================================================

    commit = (
        git_commit()
    )

    dirty = (
        git_dirty()
    )

    # ========================================================
    # Markdown report
    # ========================================================

    lines = []

    lines.append(
        "# VALOR Final Experimental Results"
    )

    lines.append(
        ""
    )

    lines.append(
        "> Automatically generated from the experiment CSV outputs. "
        "Do not manually edit reported numerical results."
    )

    lines.append(
        ""
    )

    lines.append(
        "## 1. Reproducibility metadata"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- Generated UTC: "
        f"`{datetime.now(timezone.utc).isoformat()}`"
    )

    lines.append(
        f"- Git commit: "
        f"`{commit or 'unavailable'}`"
    )

    lines.append(
        f"- Git working tree dirty: "
        f"`{dirty}`"
    )

    lines.append(
        f"- Python: "
        f"`{rq3_env.get('python', 'unknown')}`"
    )

    lines.append(
        f"- SQLite: "
        f"`{rq3_env.get('sqlite', 'unknown')}`"
    )

    lines.append(
        f"- Platform: "
        f"`{rq3_env.get('platform', 'unknown')}`"
    )

    lines.append(
        ""
    )

    # ========================================================
    # Final checks
    # ========================================================

    lines.append(
        "## 2. Final consistency checks"
    )

    lines.append(
        ""
    )

    check_df = pd.DataFrame(
        [
            {
                "check":
                    name,

                "pass":
                    value,
            }

            for name, value in (
                checks.items()
            )
        ]
    )

    lines.append(
        markdown_table(
            check_df,
            columns=[
                "check",
                "pass",
            ],
            labels={
                "check":
                    "Check",

                "pass":
                    "Pass",
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        f"**All automated checks passed:** "
        f"`{all_checks_pass}`"
    )

    lines.append(
        ""
    )

    # ========================================================
    # RQ1
    # ========================================================

    lines.append(
        "## 3. RQ1 — Coverage vs. false-trigger exposure"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            rq1,
            columns=[
                "K",
                "union_coverage_mean",
                "union_coverage_sd",
                "valor_coverage_mean",
                "union_exposure_mean",
                "valor_exposure_mean",
                "union_catalog_tools_mean",
            ],
            labels={
                "K":
                    "K",

                "union_coverage_mean":
                    "Union coverage (%)",

                "union_coverage_sd":
                    "Coverage SD",

                "valor_coverage_mean":
                    "VALOR coverage (%)",

                "union_exposure_mean":
                    "Union exposure (×)",

                "valor_exposure_mean":
                    "VALOR exposure (×)",

                "union_catalog_tools_mean":
                    "Catalog tools",
            },
            decimals={
                "union_coverage_mean":
                    2,

                "union_coverage_sd":
                    2,

                "valor_coverage_mean":
                    2,

                "union_exposure_mean":
                    3,

                "valor_exposure_mean":
                    3,

                "union_catalog_tools_mean":
                    1,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "Interpretation: catalog widening increases unseen-trace "
        "coverage while accumulating standing emergency scope. "
        "VALOR admission in this construction-guaranteed feasible "
        "workload does not depend on prior contingency membership, "
        "while cumulative exceptional exposure remains bounded by "
        "the precommitted reserve."
    )

    lines.append(
        ""
    )

    # ========================================================
    # RQ2
    # ========================================================

    lines.append(
        "## 4. RQ2 — Ablation"
    )

    lines.append(
        ""
    )

    rq2_columns = [
        "method",
        "coverage_mean",
        "coverage_sd",
        "exposure_mean",
        "exposure_sd",
        "attack_requests_mean",
    ]

    lines.append(
        markdown_table(
            rq2,
            columns=rq2_columns,
            labels={
                "method":
                    "Method",

                "coverage_mean":
                    "Coverage (%)",

                "coverage_sd":
                    "Coverage SD",

                "exposure_mean":
                    "Exposure (×)",

                "exposure_sd":
                    "Exposure SD",

                "attack_requests_mean":
                    "Attack requests accepted",
            },
            decimals={
                "coverage_mean":
                    2,

                "coverage_sd":
                    2,

                "exposure_mean":
                    3,

                "exposure_sd":
                    3,

                "attack_requests_mean":
                    2,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "Interpretation: adding only a reserve constrains cumulative "
        "spend but does not create scope for unseen actions; adding "
        "only an effect envelope removes the catalog-membership "
        "bottleneck but does not bound repeated cumulative use. "
        "Full VALOR combines both properties."
    )

    lines.append(
        ""
    )

    # ========================================================
    # RQ3
    # ========================================================

    lines.append(
        "## 5. RQ3 — Local enforcement overhead"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            rq3,
            columns=[
                "operation",
                "n",
                "median_us",
                "p95_us",
                "p99_us",
            ],
            labels={
                "operation":
                    "Operation",

                "n":
                    "N",

                "median_us":
                    "Median (µs)",

                "p95_us":
                    "p95 (µs)",

                "p99_us":
                    "p99 (µs)",
            },
            decimals={
                "median_us":
                    2,

                "p95_us":
                    2,

                "p99_us":
                    2,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "Primary durable per-action result: use **SQLite steady-state "
        "issuance** in the main paper. The cold-connect path is a "
        "supplementary measurement that includes repeated connection "
        "and PRAGMA setup."
    )

    lines.append(
        ""
    )

    # ========================================================
    # RQ4
    # ========================================================

    lines.append(
        "## 6. RQ4 — Adapter-underestimation failure boundary"
    )

    lines.append(
        ""
    )

    rq4_columns = [
        column
        for column in [
            "target_dimension",
            "underestimate_fraction",
            "num_seeds",
            "declared_exposure_mean",
            "true_exposure_mean",
            "declared_energy_x_mean",
            "true_energy_x_mean",
            "declared_safe_rate",
            "true_safe_rate",
            "physical_violation_rate",
        ]
        if column in rq4.columns
    ]

    lines.append(
        markdown_table(
            rq4,
            columns=rq4_columns,
            decimals={
                "underestimate_fraction":
                    2,

                "declared_exposure_mean":
                    3,

                "true_exposure_mean":
                    3,

                "declared_energy_x_mean":
                    3,

                "true_energy_x_mean":
                    3,

                "declared_safe_rate":
                    3,

                "true_safe_rate":
                    3,

                "physical_violation_rate":
                    3,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "Interpretation: declared ledger accounting can remain inside "
        "the signed reserve even when synthetic ground-truth effects "
        "cross the intended boundary. The experiment therefore "
        "illustrates the load-bearing nature of adapter "
        "conservativeness rather than contradicting the accounting "
        "invariant."
    )

    lines.append(
        ""
    )

    # ========================================================
    # RQ5
    # ========================================================

    lines.append(
        "## 7. RQ5 — Construction-guaranteed negative control"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            rq5,
            columns=[
                "trace_class",
                "count",
                "correctly_handled_pct",
                "observed_admit_pct",
            ],
            labels={
                "trace_class":
                    "Trace class",

                "count":
                    "Count",

                "correctly_handled_pct":
                    "Correctly handled (%)",

                "observed_admit_pct":
                    "Observed admit (%)",
            },
            decimals={
                "correctly_handled_pct":
                    1,

                "observed_admit_pct":
                    1,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "### RQ5 by construction mode"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            rq5_mode,
            columns=[
                "mode",
                "count",
                "expected_admit_rate",
                "observed_admit_rate",
                "correct_rate",
            ],
            decimals={
                "expected_admit_rate":
                    3,

                "observed_admit_rate":
                    3,

                "correct_rate":
                    3,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "These are deterministic construction-guaranteed classes. "
        "The result is an implementation correctness check, not a "
        "classification-accuracy estimate."
    )

    lines.append(
        ""
    )

    # ========================================================
    # Concurrency
    # ========================================================

    lines.append(
        "## 8. Concurrency / replay stress"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            concurrency,
            columns=[
                "experiment",
                "attempts",
                "successes",
                "expected_successes",
                "rejections_or_other",
                "unexpected_errors",
                "violations",
                "pass",
            ],
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "The concurrent reservation experiment checks that atomic "
        "reserve updates prevent overspend. The replay and Commit "
        "tests additionally check one-shot execution and one-shot "
        "settlement under concurrent presentation."
    )

    lines.append(
        ""
    )

    # ========================================================
    # E6
    # ========================================================

    lines.append(
        "## 9. E6 — Cumulative exhaustion"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            e6,
            columns=[
                "num_seeds",
                "total_cases",
                "correct_rejection_rate",
                "individually_admissible_rate",
                "inside_envelope_rate",
                "cumulative_rejection_rate",
                "mean_sequence_length",
                "min_sequence_length",
                "max_sequence_length",
            ],
            decimals={
                "correct_rejection_rate":
                    3,

                "individually_admissible_rate":
                    3,

                "inside_envelope_rate":
                    3,

                "cumulative_rejection_rate":
                    3,

                "mean_sequence_length":
                    2,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "### Binding effect dimensions"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            e6_dimension,
            columns=[
                "binding_dimension",
                "cases",
                "mean_sequence_length",
                "correct_rate",
            ],
            decimals={
                "mean_sequence_length":
                    2,

                "correct_rate":
                    3,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "E6 isolates order-dependent cumulative infeasibility: every "
        "rejected action is individually admissible against a fresh "
        "B0, but prior exceptional spend leaves insufficient "
        "remaining reserve."
    )

    lines.append(
        ""
    )

    # ========================================================
    # E7
    # ========================================================

    lines.append(
        "## 10. E7 — Execution-time state drift"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            e7,
            columns=[
                "condition",
                "trials",
                "expected_permit_rate",
                "observed_permit_rate",
                "abort_rate",
                "nonce_resolved_rate",
                "conservation_rate",
                "lifecycle_correct_rate",
                "correct_rate",
            ],
            decimals={
                "expected_permit_rate":
                    3,

                "observed_permit_rate":
                    3,

                "abort_rate":
                    3,

                "nonce_resolved_rate":
                    3,

                "conservation_rate":
                    3,

                "lifecycle_correct_rate":
                    3,

                "correct_rate":
                    3,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "### Abort properties"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            e7_abort,
            columns=[
                "abort_trials",
                "refund_success_rate",
                "replay_rejection_rate",
                "conservation_rate",
            ],
            decimals={
                "refund_success_rate":
                    3,

                "replay_rejection_rate":
                    3,

                "conservation_rate":
                    3,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "E7 makes execution-time revalidation explicit: a capability "
        "that was valid at issuance is permitted only when the fresh "
        "execution-time contract still fits its reservation and hard "
        "guards."
    )

    lines.append(
        ""
    )

    # ========================================================
    # E8
    # ========================================================

    lines.append(
        "## 11. E8 — False-trigger attack followed by legitimate emergency"
    )

    lines.append(
        ""
    )

    lines.append(
        markdown_table(
            e8,
            columns=[
                "requested_attack_steps",
                "accepted_attack_steps_mean",
                "attack_exposure_mean",
                "attack_exposure_sd",
                "recovery_admission_mean",
                "recovery_admission_sd",
                "insufficient_rejections_mean",
            ],
            labels={
                "requested_attack_steps":
                    "Requested attack steps",

                "accepted_attack_steps_mean":
                    "Accepted attack steps",

                "attack_exposure_mean":
                    "Attack exposure (×)",

                "attack_exposure_sd":
                    "Exposure SD",

                "recovery_admission_mean":
                    "Recovery admission (%)",

                "recovery_admission_sd":
                    "Recovery SD",

                "insufficient_rejections_mean":
                    "Insufficient-reserve rejects",
            },
            decimals={
                "accepted_attack_steps_mean":
                    2,

                "attack_exposure_mean":
                    3,

                "attack_exposure_sd":
                    3,

                "recovery_admission_mean":
                    2,

                "recovery_admission_sd":
                    2,

                "insufficient_rejections_mean":
                    1,
            },
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "E8 quantifies the availability tradeoff: false-trigger "
        "requests remain bounded by the precommitted reserve, but "
        "spending that finite authority reduces the set of later "
        "legitimate recovery traces that remain affordable."
    )

    lines.append(
        ""
    )

    # ========================================================
    # Interpretation boundaries
    # ========================================================

    lines.append(
        "## 12. Interpretation boundaries"
    )

    lines.append(
        ""
    )

    lines.append(
        "- The experiments are mechanism-level synthetic evaluations, "
        "not mission-risk estimates."
    )

    lines.append(
        "- VALOR's 100% coverage results are conditioned on workloads "
        "constructed to be effect-feasible under a fresh signed "
        "envelope and reserve."
    )

    lines.append(
        "- The RQ5 100%/100% result is a deterministic implementation "
        "correctness check over construction-guaranteed classes, not "
        "a learned-classifier accuracy result."
    )

    lines.append(
        "- The physical/consequence containment interpretation depends "
        "on conservative effect adapters; RQ4 deliberately violates "
        "this assumption."
    )

    lines.append(
        "- Reserve exhaustion remains an availability risk. E8 "
        "quantifies this tradeoff."
    )

    lines.append(
        "- The current artifact uses registered synthetic tools rather "
        "than a real LLM Agent or physical robot."
    )

    lines.append(
        "- 'Open-world' refers to previously uninstantiated action "
        "instances over registered interfaces, not arbitrary new "
        "actuators or tools."
    )

    lines.append(
        "- Universal security claims come from the formal model under "
        "its stated assumptions; finite experiments check the "
        "implementation and empirical boundary conditions."
    )

    lines.append(
        ""
    )

    # ========================================================
    # Result files
    # ========================================================

    lines.append(
        "## 13. Generated result artifacts"
    )

    lines.append(
        ""
    )

    for key, path in (
        FILES.items()
    ):

        lines.append(
            f"- `{key}`: "
            f"`{path.relative_to(ROOT)}`"
        )

    lines.append(
        ""
    )

    lines.append(
        "Final consolidation artifacts:"
    )

    lines.append(
        ""
    )

    lines.append(
        "- `results/final/final_key_results.csv`"
    )

    lines.append(
        "- `results/final/final_manifest.json`"
    )

    lines.append(
        "- `docs/EXPERIMENT_RESULTS.md`"
    )

    report = "\n".join(
        lines
    )

    report_path = (
        DOCS
        / "EXPERIMENT_RESULTS.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # Manifest / hashes
    # ========================================================

    manifest_files = {}

    source_paths = list(
        FILES.values()
    )

    # Include generated figures as well.
    for figure in sorted(
        FIGURES.glob(
            "*.png"
        )
    ):

        source_paths.append(
            figure
        )

    for path in source_paths:

        if not path.exists():
            continue

        relative = str(
            path.relative_to(
                ROOT
            )
        )

        manifest_files[
            relative
        ] = {
            "sha256":
                sha256_file(
                    path
                ),

            "size_bytes":
                path.stat()
                .st_size,
        }

    config_path = (
        ROOT
        / "config"
        / "experiment.yaml"
    )

    manifest = {
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            commit,

        "git_dirty":
            dirty,

        "all_checks_pass":
            all_checks_pass,

        "checks":
            checks,

        "config": {
            "path":
                str(
                    config_path
                    .relative_to(
                        ROOT
                    )
                ),

            "sha256":
                (
                    sha256_file(
                        config_path
                    )
                    if config_path.exists()
                    else None
                ),
        },

        "environment":
            rq3_env,

        "result_files":
            manifest_files,
    }

    manifest_path = (
        FINAL_DIR
        / "final_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ========================================================
    # Terminal
    # ========================================================

    print(
        "=" * 88
    )

    print(
        "FINAL EXPERIMENT CONSOLIDATION"
    )

    print(
        "=" * 88
    )

    print()

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name:42s}: "
            f"{passed}"
        )

    print()

    print(
        f"ALL CHECKS PASS: "
        f"{all_checks_pass}"
    )

    print()

    print(
        f"[OK] paper-ready results -> "
        f"{report_path}"
    )

    print(
        f"[OK] key results -> "
        f"{key_path}"
    )

    print(
        f"[OK] manifest -> "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()