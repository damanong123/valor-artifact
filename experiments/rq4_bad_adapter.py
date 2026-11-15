from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.bad_adapter import (
    UnderestimatingEffectAdapter,
)

from valor.budget import (
    vector_add,
    vector_leq,
    zero_vector,
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

def normalized_ratio(
    effect: dict[str, float],
    reserve: dict[str, float],
) -> float:
    """
    Maximum normalized effect across all protected dimensions.

        max_d effect[d] / B0[d]

    This is the same multidimensional exposure notion used
    elsewhere in the synthetic experiments.
    """

    ratios = []

    for dimension in reserve:

        budget = reserve[
            dimension
        ]

        value = effect[
            dimension
        ]

        if budget == 0:

            if value > 0:
                return float("inf")

            continue

        ratios.append(
            value / budget
        )

    if not ratios:
        return 0.0

    return max(
        ratios
    )


# ============================================================
# RQ4 sequence construction
# ============================================================

def generate_repeated_action_sequence(
    *,
    tools,
    true_adapter:
        SyntheticEffectAdapter,
    bad_adapter:
        UnderestimatingEffectAdapter,
    reserve:
        dict[str, float],
) -> tuple[
    list[Action],
    dict[str, float],
    dict[str, float],
]:
    """
    Repeatedly execute one exceptional action according to
    the BAD adapter's declared effect until another instance
    no longer fits the declared remaining reserve.

    Returns
    -------
    accepted_actions:
        Sequence admitted according to the bad adapter.

    declared_total:
        Cumulative effect seen by VALOR.

    true_total:
        Synthetic ground-truth cumulative effect.

    Important
    ---------
    The purpose is to show that:

        declared accounting can remain <= B0

    while:

        ground-truth consequence can exceed B0

    when the conservative-adapter assumption is violated.
    """

    candidates = []

    # --------------------------------------------------------
    # Build candidate actions
    # --------------------------------------------------------

    for tool in tools.values():

        if not (
            tool.registered
            and tool.in_dmax
        ):
            continue

        action = Action(
            tool=tool.name,
            parameter=(
                tool.exceptional_max
            ),
        )

        declared = (
            bad_adapter
            .contract(action)
            .effect_vector()
        )

        true_effect = (
            true_adapter
            .contract(action)
            .effect_vector()
        )

        # Candidate must at least fit once according to
        # the declared contract.
        if vector_leq(
            declared,
            reserve,
        ):

            candidates.append(
                (
                    action,
                    declared,
                    true_effect,
                )
            )

    if not candidates:

        raise RuntimeError(
            "No effect-feasible candidate "
            "available for RQ4."
        )

    # --------------------------------------------------------
    # Preserve the original RQ4 selection rule.
    #
    # We deliberately corrupt the energy dimension.
    #
    # Select according to true/declaration discrepancy in
    # that dimension.
    # --------------------------------------------------------

    def gap_score(
        item,
    ) -> float:

        (
            _,
            declared,
            true_effect,
        ) = item

        declared_energy = (
            declared[
                "energy"
            ]
        )

        if declared_energy <= 0:
            return 0.0

        return (
            true_effect[
                "energy"
            ]
            / declared_energy
        )

    (
        selected_action,
        selected_declared,
        selected_true,
    ) = max(
        candidates,
        key=gap_score,
    )

    # --------------------------------------------------------
    # Spend according to declared effects
    # --------------------------------------------------------

    remaining = dict(
        reserve
    )

    declared_total = (
        zero_vector()
    )

    true_total = (
        zero_vector()
    )

    accepted = []

    # Defensive upper bound to avoid accidental infinite loop.
    for _ in range(
        10_000
    ):

        # VALOR sees only selected_declared.
        if not vector_leq(
            selected_declared,
            remaining,
        ):
            break

        # Debit the DECLARED amount.
        remaining = {
            dimension:
                remaining[
                    dimension
                ]
                - selected_declared[
                    dimension
                ]

            for dimension in remaining
        }

        # Ledger-visible effect.
        declared_total = (
            vector_add(
                declared_total,
                selected_declared,
            )
        )

        # Synthetic ground truth.
        true_total = (
            vector_add(
                true_total,
                selected_true,
            )
        )

        accepted.append(
            selected_action
        )

    return (
        accepted,
        declared_total,
        true_total,
    )


# ============================================================
# Main experiment
# ============================================================

def main():

    # ========================================================
    # Configuration
    # ========================================================

    config_path = (
        ROOT
        / "config"
        / "experiment.yaml"
    )

    with open(
        config_path,
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

    underestimate_fraction = (
        cfg[
            "bad_adapter"
        ]["underestimate_fraction"]
    )

    # We deliberately violate conservativeness
    # only in the energy dimension.
    target_dimension = (
        "energy"
    )

    rows = []

    # ========================================================
    # Run each seed
    # ========================================================

    for seed in seeds:

        # ----------------------------------------------------
        # Synthetic world
        # ----------------------------------------------------

        tools = generate_tools(
            seed=seed
        )

        # ----------------------------------------------------
        # Ground-truth effect model
        # ----------------------------------------------------

        true_adapter = (
            SyntheticEffectAdapter(
                tools
            )
        )

        # ----------------------------------------------------
        # Deliberately incorrect declared adapter
        # ----------------------------------------------------

        bad_adapter = (
            UnderestimatingEffectAdapter(
                tools,

                dimension=(
                    target_dimension
                ),

                underestimate_fraction=(
                    underestimate_fraction
                ),
            )
        )

        # ----------------------------------------------------
        # Generate repeated admitted sequence
        # ----------------------------------------------------

        (
            accepted,
            declared_total,
            true_total,
        ) = (
            generate_repeated_action_sequence(
                tools=tools,

                true_adapter=(
                    true_adapter
                ),

                bad_adapter=(
                    bad_adapter
                ),

                reserve=reserve,
            )
        )

        # ----------------------------------------------------
        # Overall multidimensional exposure
        # ----------------------------------------------------

        declared_exposure = (
            normalized_ratio(
                declared_total,
                reserve,
            )
        )

        true_exposure = (
            normalized_ratio(
                true_total,
                reserve,
            )
        )

        # ----------------------------------------------------
        # Target-dimension exposure
        #
        # Because energy is the deliberately corrupted
        # dimension, report it separately.
        # ----------------------------------------------------

        declared_energy_x = (
            declared_total[
                "energy"
            ]
            / reserve[
                "energy"
            ]
        )

        true_energy_x = (
            true_total[
                "energy"
            ]
            / reserve[
                "energy"
            ]
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        declared_safe = (
            vector_leq(
                declared_total,
                reserve,
            )
        )

        true_safe = (
            vector_leq(
                true_total,
                reserve,
            )
        )

        physical_violation = (
            not true_safe
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        rows.append(
            {
                "seed":
                    seed,

                "target_dimension":
                    target_dimension,

                "underestimate_fraction":
                    underestimate_fraction,

                "accepted_actions":
                    len(
                        accepted
                    ),

                # Raw energy totals
                "declared_energy":
                    declared_total[
                        "energy"
                    ],

                "true_energy":
                    true_total[
                        "energy"
                    ],

                # Energy-specific normalized values
                "declared_energy_x":
                    declared_energy_x,

                "true_energy_x":
                    true_energy_x,

                # Overall multidimensional exposure
                "declared_exposure_x":
                    declared_exposure,

                "true_exposure_x":
                    true_exposure,

                # Safety state
                "declared_within_reserve":
                    declared_safe,

                "true_within_reserve":
                    true_safe,

                "physical_violation":
                    physical_violation,
            }
        )

        # ----------------------------------------------------
        # Terminal per-seed result
        # ----------------------------------------------------

        print(
            f"seed={seed} | "
            f"accepted={len(accepted)} | "
            f"declared={declared_exposure:.3f}x | "
            f"true={true_exposure:.3f}x | "
            f"declared_energy="
            f"{declared_energy_x:.3f}x | "
            f"true_energy="
            f"{true_energy_x:.3f}x | "
            f"declared_safe="
            f"{declared_safe} | "
            f"true_safe="
            f"{true_safe} | "
            f"violation="
            f"{physical_violation}"
        )

    # ========================================================
    # Raw results
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    raw_path = (
        ROOT
        / "results"
        / "raw"
        / "rq4_bad_adapter_raw.csv"
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
    # Aggregate summary
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "target_dimension":
                    target_dimension,

                "underestimate_fraction":
                    underestimate_fraction,

                "num_seeds":
                    len(
                        df
                    ),

                # ------------------------------------------
                # Overall multidimensional exposure
                # ------------------------------------------

                "declared_exposure_mean":
                    df[
                        "declared_exposure_x"
                    ].mean(),

                "declared_exposure_sd":
                    df[
                        "declared_exposure_x"
                    ].std(),

                "true_exposure_mean":
                    df[
                        "true_exposure_x"
                    ].mean(),

                "true_exposure_sd":
                    df[
                        "true_exposure_x"
                    ].std(),

                # ------------------------------------------
                # Energy-specific exposure
                # ------------------------------------------

                "declared_energy_x_mean":
                    df[
                        "declared_energy_x"
                    ].mean(),

                "declared_energy_x_sd":
                    df[
                        "declared_energy_x"
                    ].std(),

                "true_energy_x_mean":
                    df[
                        "true_energy_x"
                    ].mean(),

                "true_energy_x_sd":
                    df[
                        "true_energy_x"
                    ].std(),

                # ------------------------------------------
                # Safety rates
                # ------------------------------------------

                "declared_safe_rate":
                    df[
                        "declared_within_reserve"
                    ].mean(),

                "true_safe_rate":
                    df[
                        "true_within_reserve"
                    ].mean(),

                "physical_violation_rate":
                    df[
                        "physical_violation"
                    ].mean(),
            }
        ]
    )

    summary_path = (
        ROOT
        / "results"
        / "tables"
        / "rq4_summary.csv"
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
    # Figure
    #
    # Show the deliberately corrupted ENERGY dimension
    # directly instead of only showing max multidimensional
    # exposure.
    # ========================================================

    figure_path = (
        ROOT
        / "results"
        / "figures"
        / "rq4_bad_adapter.png"
    )

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = list(
        range(
            len(df)
        )
    )

    width = 0.35

    plt.figure(
        figsize=(
            8,
            4.8,
        )
    )

    plt.bar(
        [
            index
            - width / 2

            for index in x
        ],

        df[
            "declared_energy_x"
        ],

        width=width,

        label=(
            "Declared energy effect"
        ),
    )

    plt.bar(
        [
            index
            + width / 2

            for index in x
        ],

        df[
            "true_energy_x"
        ],

        width=width,

        label=(
            "Synthetic ground truth"
        ),
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label=(
            "Energy reserve boundary"
        ),
    )

    plt.xticks(
        x,

        [
            str(seed)
            for seed in seeds
        ],

        rotation=15,
    )

    plt.ylabel(
        "Energy effect / energy reserve"
    )

    plt.xlabel(
        "Seed"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Terminal summary
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "RQ4 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        summary.to_string(
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
        f"[OK] figure -> "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()