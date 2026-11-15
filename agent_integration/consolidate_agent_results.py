from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


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

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Frozen result labels
# ============================================================

MAIN_LABEL = "main_v1"
ATTACK_LABEL = "attack_main_v1"


# ============================================================
# Input paths
# ============================================================

MAIN_ANALYSIS = (
    TABLE_DIR
    / f"main_analysis_{MAIN_LABEL}.csv"
)

MAIN_KEY_RESULTS = (
    TABLE_DIR
    / f"main_key_results_{MAIN_LABEL}.csv"
)

MAIN_GATES = (
    TABLE_DIR
    / f"main_gates_{MAIN_LABEL}.json"
)

MAIN_MISMATCH_BREAKDOWN = (
    TABLE_DIR
    / (
        "main_mismatch_breakdown_refined_"
        f"{MAIN_LABEL}.csv"
    )
)

ATTACK_ANALYSIS = (
    TABLE_DIR
    / (
        "attack_main_analysis_"
        f"{ATTACK_LABEL}.csv"
    )
)

ATTACK_POLICY = (
    TABLE_DIR
    / (
        "attack_main_policy_"
        f"{ATTACK_LABEL}.csv"
    )
)

ATTACK_KEY_RESULTS = (
    TABLE_DIR
    / (
        "attack_main_key_results_"
        f"{ATTACK_LABEL}.csv"
    )
)

ATTACK_GATES = (
    TABLE_DIR
    / (
        "attack_main_gates_"
        f"{ATTACK_LABEL}.json"
    )
)


# ============================================================
# Output paths
# ============================================================

CAPABILITY_TABLE = (
    FINAL_DIR
    / "agent_capability_table.csv"
)

SECURITY_TABLE = (
    FINAL_DIR
    / "agent_security_table.csv"
)

ATTACK_TABLE = (
    FINAL_DIR
    / "agent_attack_table.csv"
)

KEY_RESULTS_TABLE = (
    FINAL_DIR
    / "agent_final_key_results.csv"
)

RESULTS_MD = (
    FINAL_DIR
    / "AGENT_EXPERIMENT_RESULTS.md"
)

FINAL_MANIFEST = (
    FINAL_DIR
    / "agent_final_manifest.json"
)

MAIN_COMPLETION_FIGURE = (
    FIGURE_DIR
    / "agent_main_completion.png"
)

ATTACK_GOAL_FIGURE = (
    FIGURE_DIR
    / "agent_attack_goal_rate.png"
)


# ============================================================
# Helpers
# ============================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required result file: {path}"
        )


def load_metric_csv(
    path: Path,
) -> dict[str, float]:

    df = pd.read_csv(
        path
    )

    if (
        "metric" not in df.columns
        or
        "value" not in df.columns
    ):

        raise RuntimeError(
            f"Expected metric/value columns in {path}"
        )

    result = {}

    for _, row in df.iterrows():

        result[
            str(
                row["metric"]
            )
        ] = float(
            row["value"]
        )

    return result


def load_json(
    path: Path,
) -> dict[str, Any]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

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

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def percent(
    value: float,
) -> str:

    return (
        f"{value * 100:.2f}%"
    )


def exposure(
    value: float,
) -> str:

    return (
        f"{value:.4f}x B0"
    )


# ============================================================
# Main capability table
# ============================================================

def build_capability_table(
    main_analysis: pd.DataFrame,
) -> pd.DataFrame:

    task_order = [
        "normal",
        "known_emergency",
        "novel_feasible",
        "novel_infeasible",
        "adversarial",
    ]

    policy_order = [
        "normal_only",
        "catalog",
        "valor",
    ]

    rows = []

    for task_class in task_order:

        row = {
            "task_class":
                task_class,
        }

        for policy in policy_order:

            subset = (
                main_analysis[
                    (
                        main_analysis[
                            "task_class"
                        ]
                        == task_class
                    )
                    &
                    (
                        main_analysis[
                            "policy"
                        ]
                        == policy
                    )
                ]
            )

            if subset.empty:

                row[
                    f"{policy}_success_rate"
                ] = None

                row[
                    f"{policy}_runs"
                ] = 0

            else:

                item = subset.iloc[
                    0
                ]

                row[
                    f"{policy}_success_rate"
                ] = float(
                    item[
                        "success_rate"
                    ]
                )

                row[
                    f"{policy}_runs"
                ] = int(
                    item[
                        "runs"
                    ]
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Security table
# ============================================================

def build_security_table(
    main_metrics: dict[str, float],
) -> pd.DataFrame:

    rows = [
        {
            "metric":
                "Expected-behavior rate",

            "value":
                main_metrics[
                    "expected_behavior_rate"
                ],

            "unit":
                "rate",
        },

        {
            "metric":
                "Mismatch count",

            "value":
                main_metrics[
                    "mismatch_count"
                ],

            "unit":
                "runs",
        },

        {
            "metric":
                (
                    "Novel infeasible unsafe "
                    "mission success under VALOR"
                ),

            "value":
                main_metrics[
                    (
                        "novel_infeasible_valor_"
                        "unsafe_success_rate"
                    )
                ],

            "unit":
                "rate",
        },

        {
            "metric":
                (
                    "Novel infeasible mean "
                    "exposure under VALOR"
                ),

            "value":
                main_metrics[
                    (
                        "novel_infeasible_valor_"
                        "mean_exposure_x"
                    )
                ],

            "unit":
                "x_B0",
        },

        {
            "metric":
                "VALOR authority conservation",

            "value":
                main_metrics[
                    (
                        "valor_authority_"
                        "conservation_rate"
                    )
                ],

            "unit":
                "rate",
        },

        {
            "metric":
                (
                    "Maximum observed exceptional "
                    "exposure"
                ),

            "value":
                main_metrics[
                    (
                        "valor_max_exceptional_"
                        "exposure_x"
                    )
                ],

            "unit":
                "x_B0",
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# Attack table
# ============================================================

def build_attack_table(
    attack_metrics: dict[str, float],
) -> pd.DataFrame:

    metric_names = [
        (
            "Overall attack-goal observation rate",
            "overall_attack_goal_rate",
            "rate",
        ),
        (
            "Overall behavior-change rate",
            "overall_behavior_change_rate",
            "rate",
        ),
        (
            "VALOR attack bound preservation rate",
            "valor_attack_bound_rate",
            "rate",
        ),
        (
            "VALOR authority conservation rate",
            "valor_authority_conservation_rate",
            "rate",
        ),
        (
            "VALOR mean attack exposure",
            "valor_mean_attack_exposure_x",
            "x_B0",
        ),
        (
            "VALOR maximum attack exposure",
            "valor_max_attack_exposure_x",
            "x_B0",
        ),
        (
            "Compromised VALOR run count",
            "valor_compromised_run_count",
            "runs",
        ),
        (
            "Compromised VALOR bound preservation rate",
            "valor_compromised_bound_rate",
            "rate",
        ),
        (
            "Compromised VALOR authority conservation rate",
            "valor_compromised_conservation_rate",
            "rate",
        ),
        (
            "Compromised VALOR mean exposure",
            "valor_compromised_mean_exposure_x",
            "x_B0",
        ),
        (
            "Compromised VALOR maximum exposure",
            "valor_compromised_max_exposure_x",
            "x_B0",
        ),
        (
            "Catalog denial-replan mission-success rate",
            "catalog_denial_replan_success_rate",
            "rate",
        ),
    ]

    rows = []

    for (
        label,
        key,
        unit,
    ) in metric_names:

        if key not in attack_metrics:
            continue

        rows.append(
            {
                "metric":
                    label,

                "value":
                    attack_metrics[
                        key
                    ],

                "unit":
                    unit,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Paper-ready key-result table
# ============================================================

def build_final_key_results(
    main_metrics: dict[str, float],
    attack_metrics: dict[str, float],
) -> pd.DataFrame:

    rows = [
        {
            "section":
                "Main",

            "metric":
                "Novel feasible completion — Normal-only",

            "value":
                main_metrics[
                    (
                        "novel_feasible_"
                        "normal_success_rate"
                    )
                ],
        },

        {
            "section":
                "Main",

            "metric":
                "Novel feasible completion — Catalog",

            "value":
                main_metrics[
                    (
                        "novel_feasible_"
                        "catalog_success_rate"
                    )
                ],
        },

        {
            "section":
                "Main",

            "metric":
                "Novel feasible completion — VALOR",

            "value":
                main_metrics[
                    (
                        "novel_feasible_"
                        "valor_success_rate"
                    )
                ],
        },

        {
            "section":
                "Main",

            "metric":
                (
                    "Novel infeasible unsafe "
                    "success — VALOR"
                ),

            "value":
                main_metrics[
                    (
                        "novel_infeasible_valor_"
                        "unsafe_success_rate"
                    )
                ],
        },

        {
            "section":
                "Main",

            "metric":
                "VALOR authority conservation",

            "value":
                main_metrics[
                    (
                        "valor_authority_"
                        "conservation_rate"
                    )
                ],
        },

        {
            "section":
                "Main",

            "metric":
                (
                    "Maximum observed VALOR "
                    "exceptional exposure"
                ),

            "value":
                main_metrics[
                    (
                        "valor_max_exceptional_"
                        "exposure_x"
                    )
                ],
        },

        {
            "section":
                "Attack",

            "metric":
                "Attack-goal observation rate",

            "value":
                attack_metrics[
                    "overall_attack_goal_rate"
                ],
        },

        {
            "section":
                "Attack",

            "metric":
                "Behavior-change rate",

            "value":
                attack_metrics[
                    "overall_behavior_change_rate"
                ],
        },

        {
            "section":
                "Attack",

            "metric":
                (
                    "VALOR attack bound "
                    "preservation"
                ),

            "value":
                attack_metrics[
                    "valor_attack_bound_rate"
                ],
        },

        {
            "section":
                "Attack",

            "metric":
                (
                    "Compromised VALOR "
                    "bound preservation"
                ),

            "value":
                attack_metrics[
                    "valor_compromised_bound_rate"
                ],
        },

        {
            "section":
                "Attack",

            "metric":
                (
                    "Compromised VALOR "
                    "maximum exposure"
                ),

            "value":
                attack_metrics[
                    "valor_compromised_max_exposure_x"
                ],
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main completion figure
# ============================================================

def make_main_completion_figure(
    capability: pd.DataFrame,
) -> None:

    plot_df = (
        capability[
            [
                "task_class",
                "normal_only_success_rate",
                "catalog_success_rate",
                "valor_success_rate",
            ]
        ]
        .copy()
    )

    plot_df = plot_df.set_index(
        "task_class"
    )

    plot_df = (
        plot_df
        * 100.0
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(
            11,
            6,
        ),
    )

    ax.set_ylabel(
        "Mission success rate (%)"
    )

    ax.set_xlabel(
        "Task class"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_title(
        "Real-Agent Task Completion by Authorization Policy"
    )

    ax.legend(
        title="Policy"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        MAIN_COMPLETION_FIGURE,
        dpi=200,
    )

    plt.close()


# ============================================================
# Attack goal figure
# ============================================================

def make_attack_goal_figure(
    attack_analysis: pd.DataFrame,
) -> None:

    pivot = (
        attack_analysis
        .pivot(
            index="attack_id",
            columns="policy",
            values="attack_goal_rate",
        )
        * 100.0
    )

    desired_columns = [
        "normal_only",
        "catalog",
        "valor",
    ]

    existing_columns = [
        column
        for column in desired_columns
        if column in pivot.columns
    ]

    pivot = pivot[
        existing_columns
    ]

    ax = pivot.plot(
        kind="bar",
        figsize=(
            11,
            6,
        ),
    )

    ax.set_ylabel(
        "Attack-goal observation rate (%)"
    )

    ax.set_xlabel(
        "Attack"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_title(
        "Prompt-Injection Effectiveness by Policy"
    )

    ax.legend(
        title="Policy"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        ATTACK_GOAL_FIGURE,
        dpi=200,
    )

    plt.close()


# ============================================================
# Markdown
# ============================================================

def build_results_markdown(
    *,
    main_metrics: dict[str, float],
    attack_metrics: dict[str, float],
    main_gates: dict[str, Any],
    attack_gates: dict[str, Any],
    mismatch_breakdown: pd.DataFrame,
) -> str:

    novel_normal = (
        main_metrics[
            (
                "novel_feasible_"
                "normal_success_rate"
            )
        ]
    )

    novel_catalog = (
        main_metrics[
            (
                "novel_feasible_"
                "catalog_success_rate"
            )
        ]
    )

    novel_valor = (
        main_metrics[
            (
                "novel_feasible_"
                "valor_success_rate"
            )
        ]
    )

    unsafe = (
        main_metrics[
            (
                "novel_infeasible_valor_"
                "unsafe_success_rate"
            )
        ]
    )

    main_conservation = (
        main_metrics[
            (
                "valor_authority_"
                "conservation_rate"
            )
        ]
    )

    main_max_exposure = (
        main_metrics[
            (
                "valor_max_exceptional_"
                "exposure_x"
            )
        ]
    )

    attack_goal = (
        attack_metrics[
            "overall_attack_goal_rate"
        ]
    )

    behavior_change = (
        attack_metrics[
            "overall_behavior_change_rate"
        ]
    )

    attack_bound = (
        attack_metrics[
            "valor_attack_bound_rate"
        ]
    )

    compromised_count = int(
        attack_metrics[
            "valor_compromised_run_count"
        ]
    )

    compromised_bound = (
        attack_metrics[
            "valor_compromised_bound_rate"
        ]
    )

    compromised_max_exposure = (
        attack_metrics[
            (
                "valor_compromised_"
                "max_exposure_x"
            )
        ]
    )

    attack_max_exposure = (
        attack_metrics[
            "valor_max_attack_exposure_x"
        ]
    )

    catalog_replan = (
        attack_metrics[
            (
                "catalog_denial_replan_"
                "success_rate"
            )
        ]
    )

    lines = []

    lines.append(
        "# VALOR Real-Agent Experiment Results"
    )

    lines.append(
        ""
    )

    lines.append(
        "## Experimental status"
    )

    lines.append(
        ""
    )

    lines.append(
        "- Main real-agent experiment: 900 runs."
    )

    lines.append(
        "- Main prompt-injection attack experiment: 900 runs."
    )

    lines.append(
        "- Main experiment gates passed: "
        f"`{main_gates.get('all_gates_pass')}`."
    )

    lines.append(
        "- Attack-main gates passed: "
        f"`{attack_gates.get('all_gates_pass')}`."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Main capability result"
    )

    lines.append(
        ""
    )

    lines.append(
        "For catalog-unseen but effect-feasible recovery tasks:"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- Normal-only completion: {percent(novel_normal)}."
    )

    lines.append(
        f"- Frozen-catalog completion: {percent(novel_catalog)}."
    )

    lines.append(
        f"- VALOR completion: {percent(novel_valor)}."
    )

    lines.append(
        ""
    )

    lines.append(
        "This is the central adaptive-authority result: "
        "the frozen catalog cannot authorize these exact "
        "runtime actions, while VALOR can authorize them "
        "when their effects remain inside the registered "
        "contract and remaining reserve."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Main safety result"
    )

    lines.append(
        ""
    )

    lines.append(
        "- Unsafe mission success for novel-infeasible "
        f"tasks under VALOR: {percent(unsafe)}."
    )

    lines.append(
        "- VALOR authority conservation across Main runs: "
        f"{percent(main_conservation)}."
    )

    lines.append(
        "- Maximum observed exceptional exposure in the "
        f"Main workload: {exposure(main_max_exposure)}."
    )

    lines.append(
        ""
    )

    lines.append(
        "The empirical maximum is not a theoretical upper "
        "bound. The mechanism-level bound remains "
        "`<= 1.0 x B0`; the reported value is only the "
        "largest exposure observed in this workload."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Prompt-injection result"
    )

    lines.append(
        ""
    )

    lines.append(
        "- Attack-goal observation rate: "
        f"{percent(attack_goal)}."
    )

    lines.append(
        "- Behavior-change rate: "
        f"{percent(behavior_change)}."
    )

    lines.append(
        "- VALOR attack-bound preservation: "
        f"{percent(attack_bound)}."
    )

    lines.append(
        "- VALOR maximum attack exposure: "
        f"{exposure(attack_max_exposure)}."
    )

    lines.append(
        ""
    )

    lines.append(
        "The attack suite therefore changed the behavior of "
        "the real Qwen3 agent rather than merely presenting "
        "inert malicious text."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Compromised-Agent containment"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- VALOR runs in which the attack goal was actually "
        f"observed: {compromised_count}."
    )

    lines.append(
        "- Bound preservation in this compromised subset: "
        f"{percent(compromised_bound)}."
    )

    lines.append(
        "- Maximum exposure in this compromised subset: "
        f"{exposure(compromised_max_exposure)}."
    )

    lines.append(
        ""
    )

    lines.append(
        "The supported security claim is therefore "
        "**authority containment**, not prompt-injection "
        "prevention: successful manipulation of the planner "
        "did not create unbounded authority."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Catalog comparison under attack"
    )

    lines.append(
        ""
    )

    lines.append(
        "Among catalog-policy runs where the attack goal was "
        "observed, the denial -> replanning -> mission-success "
        f"rate was {percent(catalog_replan)}."
    )

    lines.append(
        ""
    )

    lines.append(
        "This illustrates the principal trade-off between "
        "the two exceptional-authority designs:"
    )

    lines.append(
        ""
    )

    lines.append(
        "- A frozen catalog can strongly reject actions "
        "outside pre-enumerated scope."
    )

    lines.append(
        "- VALOR can admit catalog-unseen actions when their "
        "trusted effect contracts remain feasible under the "
        "remaining reserve."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Planner variability"
    )

    lines.append(
        ""
    )

    if mismatch_breakdown.empty:

        lines.append(
            "No Main expected-behavior mismatches were recorded."
        )

    else:

        lines.append(
            "The remaining Main deviations were dominated by "
            "planner-side variability rather than enforcement "
            "violations:"
        )

        lines.append(
            ""
        )

        for _, row in mismatch_breakdown.iterrows():

            lines.append(
                f"- `{row['classification']}`: "
                f"{int(row['count'])} runs "
                f"({float(row['fraction']) * 100:.2f}%)."
            )

    lines.append(
        ""
    )

    lines.append(
        "## Interpretation boundary"
    )

    lines.append(
        ""
    )

    lines.append(
        "The experiments do **not** show that VALOR prevents "
        "prompt injection or guarantees task-intent integrity. "
        "They show that consequential authority remains "
        "quantitatively bounded even when planner behavior "
        "is manipulated."
    )

    lines.append(
        ""
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    required = [
        MAIN_ANALYSIS,
        MAIN_KEY_RESULTS,
        MAIN_GATES,
        ATTACK_ANALYSIS,
        ATTACK_POLICY,
        ATTACK_KEY_RESULTS,
        ATTACK_GATES,
    ]

    for path in required:
        require_file(
            path
        )

    main_analysis = pd.read_csv(
        MAIN_ANALYSIS
    )

    attack_analysis = pd.read_csv(
        ATTACK_ANALYSIS
    )

    attack_policy = pd.read_csv(
        ATTACK_POLICY
    )

    main_metrics = load_metric_csv(
        MAIN_KEY_RESULTS
    )

    attack_metrics = load_metric_csv(
        ATTACK_KEY_RESULTS
    )

    main_gates = load_json(
        MAIN_GATES
    )

    attack_gates = load_json(
        ATTACK_GATES
    )

    if not bool(
        main_gates.get(
            "all_gates_pass"
        )
    ):

        raise RuntimeError(
            "Main experiment gates did not pass."
        )

    if not bool(
        attack_gates.get(
            "all_gates_pass"
        )
    ):

        raise RuntimeError(
            "Attack Main gates did not pass."
        )

    if MAIN_MISMATCH_BREAKDOWN.exists():

        mismatch_breakdown = pd.read_csv(
            MAIN_MISMATCH_BREAKDOWN
        )

    else:

        mismatch_breakdown = pd.DataFrame(
            columns=[
                "classification",
                "count",
                "fraction",
            ]
        )

    # ========================================================
    # Paper-ready tables
    # ========================================================

    capability = (
        build_capability_table(
            main_analysis
        )
    )

    security = (
        build_security_table(
            main_metrics
        )
    )

    attack_table = (
        build_attack_table(
            attack_metrics
        )
    )

    final_key_results = (
        build_final_key_results(
            main_metrics,
            attack_metrics,
        )
    )

    capability.to_csv(
        CAPABILITY_TABLE,
        index=False,
    )

    security.to_csv(
        SECURITY_TABLE,
        index=False,
    )

    attack_table.to_csv(
        ATTACK_TABLE,
        index=False,
    )

    final_key_results.to_csv(
        KEY_RESULTS_TABLE,
        index=False,
    )

    # ========================================================
    # Figures
    # ========================================================

    make_main_completion_figure(
        capability
    )

    make_attack_goal_figure(
        attack_analysis
    )

    # ========================================================
    # Markdown result record
    # ========================================================

    markdown = (
        build_results_markdown(
            main_metrics=(
                main_metrics
            ),
            attack_metrics=(
                attack_metrics
            ),
            main_gates=(
                main_gates
            ),
            attack_gates=(
                attack_gates
            ),
            mismatch_breakdown=(
                mismatch_breakdown
            ),
        )
    )

    with open(
        RESULTS_MD,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            markdown
        )

    # ========================================================
    # Final manifest
    # ========================================================

    source_files = [
        MAIN_ANALYSIS,
        MAIN_KEY_RESULTS,
        MAIN_GATES,
        ATTACK_ANALYSIS,
        ATTACK_POLICY,
        ATTACK_KEY_RESULTS,
        ATTACK_GATES,
    ]

    if MAIN_MISMATCH_BREAKDOWN.exists():

        source_files.append(
            MAIN_MISMATCH_BREAKDOWN
        )

    output_files = [
        CAPABILITY_TABLE,
        SECURITY_TABLE,
        ATTACK_TABLE,
        KEY_RESULTS_TABLE,
        RESULTS_MD,
        MAIN_COMPLETION_FIGURE,
        ATTACK_GOAL_FIGURE,
    ]

    manifest = {
        "main_run_label":
            MAIN_LABEL,

        "attack_run_label":
            ATTACK_LABEL,

        "main_gates_pass":
            True,

        "attack_main_gates_pass":
            True,

        "source_files": {
            str(
                path.relative_to(
                    ROOT
                )
            ):
                sha256_file(
                    path
                )

            for path in source_files
        },

        "output_files": {
            str(
                path.relative_to(
                    ROOT
                )
            ):
                sha256_file(
                    path
                )

            for path in output_files
        },
    }

    with open(
        FINAL_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Terminal summary
    # ========================================================

    print(
        "=" * 100
    )

    print(
        "VALOR FINAL REAL-AGENT RESULT CONSOLIDATION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Main Agent:"
    )

    print(
        "  Novel feasible completion:"
    )

    print(
        "    Normal-only = "
        f"{percent(main_metrics['novel_feasible_normal_success_rate'])}"
    )

    print(
        "    Catalog     = "
        f"{percent(main_metrics['novel_feasible_catalog_success_rate'])}"
    )

    print(
        "    VALOR       = "
        f"{percent(main_metrics['novel_feasible_valor_success_rate'])}"
    )

    print()

    print(
        "  Novel infeasible unsafe success under VALOR = "
        f"{percent(main_metrics['novel_infeasible_valor_unsafe_success_rate'])}"
    )

    print(
        "  VALOR authority conservation = "
        f"{percent(main_metrics['valor_authority_conservation_rate'])}"
    )

    print(
        "  Main max exposure = "
        f"{exposure(main_metrics['valor_max_exceptional_exposure_x'])}"
    )

    print()

    print(
        "Attack Main:"
    )

    print(
        "  Attack-goal observation = "
        f"{percent(attack_metrics['overall_attack_goal_rate'])}"
    )

    print(
        "  Behavior change = "
        f"{percent(attack_metrics['overall_behavior_change_rate'])}"
    )

    print(
        "  VALOR attack bound preservation = "
        f"{percent(attack_metrics['valor_attack_bound_rate'])}"
    )

    print(
        "  Compromised VALOR runs = "
        f"{int(attack_metrics['valor_compromised_run_count'])}"
    )

    print(
        "  Compromised bound preservation = "
        f"{percent(attack_metrics['valor_compromised_bound_rate'])}"
    )

    print(
        "  Compromised max exposure = "
        f"{exposure(attack_metrics['valor_compromised_max_exposure_x'])}"
    )

    print()

    print(
        "[OK] capability table -> "
        f"{CAPABILITY_TABLE}"
    )

    print(
        "[OK] security table -> "
        f"{SECURITY_TABLE}"
    )

    print(
        "[OK] attack table -> "
        f"{ATTACK_TABLE}"
    )

    print(
        "[OK] final key results -> "
        f"{KEY_RESULTS_TABLE}"
    )

    print(
        "[OK] results markdown -> "
        f"{RESULTS_MD}"
    )

    print(
        "[OK] main figure -> "
        f"{MAIN_COMPLETION_FIGURE}"
    )

    print(
        "[OK] attack figure -> "
        f"{ATTACK_GOAL_FIGURE}"
    )

    print(
        "[OK] final manifest -> "
        f"{FINAL_MANIFEST}"
    )

    print()

    print(
        "ALL FINAL AGENT RESULTS CONSOLIDATED: True"
    )


if __name__ == "__main__":
    main()
