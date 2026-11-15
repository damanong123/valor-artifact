from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from workload.attack_recovery import (
    apply_greedy_attack,
    authorize_recovery_trace,
)

from workload.tool_generator import (
    generate_tools,
)

from workload.trace_generator import (
    generate_feasible_traces,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Configuration
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

    e8_cfg = (
        supplemental.get(
            "attack_then_emergency",
            {},
        )
    )

    legitimate_traces_per_seed = int(
        e8_cfg.get(
            "legitimate_traces_per_seed",
            300,
        )
    )

    attack_steps = (
        e8_cfg.get(
            "attack_steps",
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
        )
    )

    attack_steps = [
        int(
            value
        )
        for value in attack_steps
    ]

    raw_rows = []

    scenario_rows = []

    # ========================================================
    # Seeds
    # ========================================================

    for seed in seeds:

        print(
            "=" * 88
        )

        print(
            f"E8 seed={seed}"
        )

        # ----------------------------------------------------
        # World
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Legitimate emergency workload.
        #
        # These traces are all feasible under a fresh B0.
        # ----------------------------------------------------

        legitimate_traces = (
            generate_feasible_traces(
                seed=seed,

                tools=tools,

                adapter=adapter,

                reserve=reserve,

                num_traces=(
                    legitimate_traces_per_seed
                ),

                min_actions=2,

                max_actions=5,
            )
        )

        # ----------------------------------------------------
        # Attack intensities
        # ----------------------------------------------------

        for requested_steps in (
            attack_steps
        ):

            attack = (
                apply_greedy_attack(
                    requested_steps=(
                        requested_steps
                    ),

                    tools=tools,

                    adapter=adapter,

                    reserve=reserve,
                )
            )

            remaining = (
                attack[
                    "remaining_reserve"
                ]
            )

            admitted_count = 0

            insufficient_count = 0

            other_reject_count = 0

            # ----------------------------------------------
            # Evaluate the SAME legitimate traces against
            # this post-attack reserve state.
            # ----------------------------------------------

            for trace in (
                legitimate_traces
            ):

                (
                    admitted,
                    reason,
                ) = (
                    authorize_recovery_trace(
                        trace,

                        tools=tools,

                        adapter=adapter,

                        dmax=dmax,

                        remaining_reserve=(
                            remaining
                        ),
                    )
                )

                if admitted:

                    admitted_count += 1

                elif reason == (
                    "insufficient_remaining_reserve"
                ):

                    insufficient_count += 1

                else:

                    other_reject_count += 1

                raw_rows.append(
                    {
                        "seed":
                            seed,

                        "requested_attack_steps":
                            requested_steps,

                        "accepted_attack_steps":
                            attack[
                                "accepted_steps"
                            ],

                        "attack_exposure_x":
                            attack[
                                "attack_exposure_x"
                            ],

                        "trace_id":
                            trace.trace_id,

                        "recovery_admitted":
                            admitted,

                        "recovery_reason":
                            reason,

                        "remaining_energy":
                            remaining[
                                "energy"
                            ],

                        "remaining_stress":
                            remaining[
                                "stress"
                            ],

                        "remaining_irreversible":
                            remaining[
                                "irreversible"
                            ],

                        "remaining_external_scope":
                            remaining[
                                "external_scope"
                            ],
                    }
                )

            recovery_rate = (
                100.0
                * admitted_count
                / len(
                    legitimate_traces
                )
            )

            scenario_rows.append(
                {
                    "seed":
                        seed,

                    "requested_attack_steps":
                        requested_steps,

                    "accepted_attack_steps":
                        attack[
                            "accepted_steps"
                        ],

                    "attack_exposure_x":
                        attack[
                            "attack_exposure_x"
                        ],

                    "legitimate_traces":
                        len(
                            legitimate_traces
                        ),

                    "recovery_admitted":
                        admitted_count,

                    "recovery_admission_pct":
                        recovery_rate,

                    "insufficient_reserve_rejections":
                        insufficient_count,

                    "other_rejections":
                        other_reject_count,

                    "remaining_energy":
                        remaining[
                            "energy"
                        ],

                    "remaining_stress":
                        remaining[
                            "stress"
                        ],

                    "remaining_irreversible":
                        remaining[
                            "irreversible"
                        ],

                    "remaining_external_scope":
                        remaining[
                            "external_scope"
                        ],
                }
            )

            print(
                f"attack requested="
                f"{requested_steps:2d} | "
                f"accepted="
                f"{attack['accepted_steps']:2d} | "
                f"exposure="
                f"{attack['attack_exposure_x']:.3f}x | "
                f"recovery="
                f"{recovery_rate:6.2f}% | "
                f"insufficient="
                f"{insufficient_count:3d}"
            )

    # ========================================================
    # Raw trace-level results
    # ========================================================

    raw_df = pd.DataFrame(
        raw_rows
    )

    raw_path = (
        ROOT
        / "results"
        / "raw"
        / "e8_attack_then_emergency_raw.csv"
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # Seed/scenario results
    # ========================================================

    scenario_df = (
        pd.DataFrame(
            scenario_rows
        )
    )

    scenario_path = (
        ROOT
        / "results"
        / "raw"
        / "e8_attack_then_emergency_by_seed.csv"
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
    )

    # ========================================================
    # Aggregate by requested attack steps
    # ========================================================

    summary = (
        scenario_df
        .groupby(
            "requested_attack_steps"
        )
        .agg(
            accepted_attack_steps_mean=(
                "accepted_attack_steps",
                "mean",
            ),

            accepted_attack_steps_sd=(
                "accepted_attack_steps",
                "std",
            ),

            attack_exposure_mean=(
                "attack_exposure_x",
                "mean",
            ),

            attack_exposure_sd=(
                "attack_exposure_x",
                "std",
            ),

            recovery_admission_mean=(
                "recovery_admission_pct",
                "mean",
            ),

            recovery_admission_sd=(
                "recovery_admission_pct",
                "std",
            ),

            insufficient_rejections_mean=(
                "insufficient_reserve_rejections",
                "mean",
            ),

            remaining_energy_mean=(
                "remaining_energy",
                "mean",
            ),

            remaining_stress_mean=(
                "remaining_stress",
                "mean",
            ),

            remaining_irreversible_mean=(
                "remaining_irreversible",
                "mean",
            ),

            remaining_external_scope_mean=(
                "remaining_external_scope",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        ROOT
        / "results"
        / "tables"
        / "e8_attack_then_emergency_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Sanity checks
    # ========================================================

    # No-attack baseline should admit all construction-
    # guaranteed feasible legitimate traces.
    baseline = (
        summary[
            summary[
                "requested_attack_steps"
            ]
            == 0
        ]
    )

    if not baseline.empty:

        baseline_rate = float(
            baseline.iloc[0][
                "recovery_admission_mean"
            ]
        )

        if abs(
            baseline_rate
            - 100.0
        ) > 1e-9:

            raise RuntimeError(
                "E8 baseline recovery "
                "must be 100%."
            )

    # Recovery rate should be monotone non-increasing as
    # requested attack steps increase.
    ordered = (
        summary
        .sort_values(
            "requested_attack_steps"
        )
    )

    rates = (
        ordered[
            "recovery_admission_mean"
        ]
        .tolist()
    )

    monotone = all(
        rates[index + 1]
        <= rates[index]
        + 1e-9

        for index in range(
            len(rates) - 1
        )
    )

    # ========================================================
    # Figure:
    # attack exposure vs recovery admission
    # ========================================================

    figure_path = (
        ROOT
        / "results"
        / "figures"
        / "e8_attack_then_emergency.png"
    )

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            7.5,
            5.0,
        )
    )

    plt.errorbar(
        summary[
            "attack_exposure_mean"
        ],

        summary[
            "recovery_admission_mean"
        ],

        xerr=summary[
            "attack_exposure_sd"
        ],

        yerr=summary[
            "recovery_admission_sd"
        ],

        marker="o",

        capsize=4,
    )

    plt.xlabel(
        "False-trigger exceptional spend (× B0)"
    )

    plt.ylabel(
        "Legitimate recovery admission (%)"
    )

    plt.ylim(
        0,
        105,
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Terminal
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "E8 ATTACK → LEGITIMATE EMERGENCY SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Baseline recovery admission:"
    )

    if not baseline.empty:

        print(
            f"  {baseline_rate:.2f}%"
        )

    print(
        "Recovery admission monotone "
        f"non-increasing: {monotone}"
    )

    print()

    print(
        f"[OK] raw -> "
        f"{raw_path}"
    )

    print(
        f"[OK] by seed -> "
        f"{scenario_path}"
    )

    print(
        f"[OK] summary -> "
        f"{summary_path}"
    )

    print(
        f"[OK] figure -> "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()