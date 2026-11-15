from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from workload.negative_control import (
    NegativeControlCase,
    generate_feasible_cases,
    generate_infeasible_cases,
)

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    AdapterError,
    SyntheticEffectAdapter,
)

from valor.budget import (
    vector_leq,
    vector_subtract,
)

from valor.normal_policy import (
    is_normally_authorized,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# Admission checker
# ============================================================

def authorize_case(
    case: NegativeControlCase,
    *,
    tools,
    adapter,
    dmax,
    reserve,
) -> tuple[
    bool,
    str,
]:
    """
    Apply VALOR's effect-admissibility rule to one stateful
    synthetic trace.

    Normal actions bypass exceptional reserve.

    Exceptional actions require:

        D subset Dmax
        hard guard
        effect <= remaining reserve
    """

    remaining = dict(
        reserve
    )

    for (
        action,
        state,
    ) in zip(
        case.actions,
        case.states,
    ):

        # ----------------------------------------------------
        # Normal path
        # ----------------------------------------------------

        if is_normally_authorized(
            action,
            tools,
        ):
            continue

        # ----------------------------------------------------
        # Exceptional contract
        # ----------------------------------------------------

        try:

            contract = (
                adapter.contract(
                    action,
                    state,
                )
            )

        except AdapterError:

            return (
                False,
                "adapter_reject",
            )

        # ----------------------------------------------------
        # Dmax
        # ----------------------------------------------------

        if not (
            contract.domains
            .issubset(
                dmax
            )
        ):

            return (
                False,
                "outside_dmax",
            )

        # ----------------------------------------------------
        # Hard state predicate
        # ----------------------------------------------------

        if not (
            contract.hard_guard_ok
        ):

            return (
                False,
                "hard_guard",
            )

        effect = (
            contract
            .effect_vector()
        )

        # ----------------------------------------------------
        # Typed reserve
        # ----------------------------------------------------

        if not vector_leq(
            effect,
            remaining,
        ):

            return (
                False,
                "insufficient_reserve",
            )

        remaining = (
            vector_subtract(
                remaining,
                effect,
            )
        )

    return (
        True,
        "admit",
    )


# ============================================================
# Main experiment
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

    feasible_per_seed = (
        cfg[
            "negative_control"
        ]["feasible_traces_per_seed"]
    )

    infeasible_per_seed = (
        cfg[
            "negative_control"
        ]["infeasible_traces_per_seed"]
    )

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

        feasible_cases = (
            generate_feasible_cases(
                seed=seed,

                tools=tools,

                adapter=adapter,

                reserve=reserve,

                num_cases=(
                    feasible_per_seed
                ),
            )
        )

        infeasible_cases = (
            generate_infeasible_cases(
                seed=seed,

                tools=tools,

                adapter=adapter,

                reserve=reserve,

                num_cases=(
                    infeasible_per_seed
                ),
            )
        )

        all_cases = (
            feasible_cases
            + infeasible_cases
        )

        seed_correct = 0

        for case in all_cases:

            (
                admitted,
                decision_reason,
            ) = authorize_case(
                case,

                tools=tools,

                adapter=adapter,

                dmax=dmax,

                reserve=reserve,
            )

            correct = (
                admitted
                == case.expected_admit
            )

            if correct:
                seed_correct += 1

            rows.append(
                {
                    "seed":
                        seed,

                    "case_id":
                        case.case_id,

                    "mode":
                        case.mode,

                    "expected_admit":
                        case.expected_admit,

                    "observed_admit":
                        admitted,

                    "correct":
                        correct,

                    "decision_reason":
                        decision_reason,

                    "num_actions":
                        len(
                            case.actions
                        ),

                    "target_index":
                        case.target_index,

                    "target_dimension":
                        case.target_dimension,

                    "target_multiplier":
                        case.target_multiplier,
                }
            )

        print(
            f"seed={seed} | "
            f"feasible={len(feasible_cases)} | "
            f"infeasible={len(infeasible_cases)} | "
            f"correct={seed_correct}/"
            f"{len(all_cases)}"
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
        / "rq5_negative_control_raw.csv"
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
    # Class-level summary
    # ========================================================

    class_rows = []

    feasible_df = (
        df[
            df["mode"]
            == "feasible"
        ]
    )

    infeasible_df = (
        df[
            df["mode"]
            != "feasible"
        ]
    )

    class_rows.append(
        {
            "trace_class":
                "Feasible (admit)",

            "count":
                len(
                    feasible_df
                ),

            "correctly_handled_pct":
                (
                    feasible_df[
                        "correct"
                    ].mean()
                    * 100.0
                ),

            "observed_admit_pct":
                (
                    feasible_df[
                        "observed_admit"
                    ].mean()
                    * 100.0
                ),
        }
    )

    class_rows.append(
        {
            "trace_class":
                "Ground-truth infeasible (reject)",

            "count":
                len(
                    infeasible_df
                ),

            "correctly_handled_pct":
                (
                    infeasible_df[
                        "correct"
                    ].mean()
                    * 100.0
                ),

            "observed_admit_pct":
                (
                    infeasible_df[
                        "observed_admit"
                    ].mean()
                    * 100.0
                ),
        }
    )

    class_summary = (
        pd.DataFrame(
            class_rows
        )
    )

    class_path = (
        ROOT
        / "results"
        / "tables"
        / "rq5_summary.csv"
    )

    class_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_summary.to_csv(
        class_path,
        index=False,
    )

    # ========================================================
    # Per-negative-mode summary
    # ========================================================

    mode_summary = (
        df
        .groupby(
            "mode",
            sort=False,
        )
        .agg(
            count=(
                "case_id",
                "count",
            ),

            expected_admit_rate=(
                "expected_admit",
                "mean",
            ),

            observed_admit_rate=(
                "observed_admit",
                "mean",
            ),

            correct_rate=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    mode_path = (
        ROOT
        / "results"
        / "tables"
        / "rq5_by_mode.csv"
    )

    mode_summary.to_csv(
        mode_path,
        index=False,
    )

    # ========================================================
    # Per-seed correctness
    # ========================================================

    seed_summary = (
        df
        .groupby(
            "seed"
        )
        .agg(
            total=(
                "case_id",
                "count",
            ),

            correct=(
                "correct",
                "sum",
            ),

            correct_rate=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    seed_path = (
        ROOT
        / "results"
        / "tables"
        / "rq5_by_seed.csv"
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
        "=" * 90
    )

    print(
        "RQ5 NEGATIVE CONTROL SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "By construction mode:"
    )

    print(
        mode_summary.to_string(
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
        f"{class_path}"
    )

    print(
        f"[OK] modes -> "
        f"{mode_path}"
    )

    print(
        f"[OK] seeds -> "
        f"{seed_path}"
    )


if __name__ == "__main__":
    main()