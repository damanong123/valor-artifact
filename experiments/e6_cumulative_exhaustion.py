from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from workload.cumulative_exhaustion import (
    CumulativeExhaustionCase,
    generate_cumulative_exhaustion_cases,
)

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
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
# Sequential VALOR admission
# ============================================================

def evaluate_case(
    case: CumulativeExhaustionCase,
    *,
    tools,
    adapter,
    dmax,
    reserve,
) -> dict[str, object]:
    """
    Sequentially evaluate one cumulative-exhaustion trace.

    The important property is:

        each action individually fits B0

    but:

        after prior accepted actions consume reserve,
        one later action fails only because of cumulative
        depletion.
    """

    remaining = dict(
        reserve
    )

    accepted_before_reject = 0

    reject_index = None

    rejection_reason = None

    individually_admissible = True

    all_inside_envelope = True

    # --------------------------------------------------------
    # First verify every action would be individually
    # admissible against a fresh full reserve.
    # --------------------------------------------------------

    for action in (
        case.actions
    ):

        if is_normally_authorized(
            action,
            tools,
        ):

            # E6 actions should deliberately be exceptional.
            individually_admissible = False
            all_inside_envelope = False

            break

        contract = (
            adapter
            .contract(
                action
            )
        )

        effect = (
            contract
            .effect_vector()
        )

        if not (
            contract
            .hard_guard_ok
        ):

            individually_admissible = False
            all_inside_envelope = False

            break

        if not (
            contract
            .domains
            .issubset(
                dmax
            )
        ):

            individually_admissible = False
            all_inside_envelope = False

            break

        if not vector_leq(
            effect,
            reserve,
        ):

            individually_admissible = False

            break

    # --------------------------------------------------------
    # Sequential reserve consumption
    # --------------------------------------------------------

    for index, action in enumerate(
        case.actions
    ):

        contract = (
            adapter
            .contract(
                action
            )
        )

        if not (
            contract
            .hard_guard_ok
        ):

            reject_index = index

            rejection_reason = (
                "hard_guard"
            )

            break

        if not (
            contract
            .domains
            .issubset(
                dmax
            )
        ):

            reject_index = index

            rejection_reason = (
                "outside_dmax"
            )

            break

        effect = (
            contract
            .effect_vector()
        )

        if not vector_leq(
            effect,
            remaining,
        ):

            reject_index = index

            rejection_reason = (
                "cumulative_reserve_exhausted"
            )

            break

        remaining = (
            vector_subtract(
                remaining,
                effect,
            )
        )

        accepted_before_reject += 1

    # --------------------------------------------------------
    # Correct E6 result:
    #
    # - all actions individually admissible;
    # - all are inside envelope;
    # - first rejection happens exactly at analytically
    #   predicted index;
    # - rejection is only cumulative reserve exhaustion.
    # --------------------------------------------------------

    correct = (
        individually_admissible
        and all_inside_envelope
        and reject_index
        == case.expected_reject_index
        and rejection_reason
        == "cumulative_reserve_exhausted"
    )

    return {
        "case_id":
            case.case_id,

        "sequence_length":
            len(
                case.actions
            ),

        "repeat_capacity":
            case.repeat_capacity,

        "binding_dimension":
            case.binding_dimension,

        "expected_reject_index":
            case.expected_reject_index,

        "observed_reject_index":
            reject_index,

        "accepted_before_reject":
            accepted_before_reject,

        "rejection_reason":
            rejection_reason,

        "individually_admissible":
            individually_admissible,

        "all_inside_envelope":
            all_inside_envelope,

        "correct":
            correct,

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

    cumulative_cfg = (
        supplemental.get(
            "cumulative_exhaustion",
            {},
        )
    )

    cases_per_seed = int(
        cumulative_cfg.get(
            "cases_per_seed",
            300,
        )
    )

    rows = []

    # --------------------------------------------------------
    # Run seeds
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

        cases = (
            generate_cumulative_exhaustion_cases(
                seed=seed,

                tools=tools,

                adapter=adapter,

                reserve=reserve,

                num_cases=(
                    cases_per_seed
                ),
            )
        )

        correct_count = 0

        for case in cases:

            result = evaluate_case(
                case,

                tools=tools,

                adapter=adapter,

                dmax=dmax,

                reserve=reserve,
            )

            result[
                "seed"
            ] = seed

            rows.append(
                result
            )

            if result[
                "correct"
            ]:

                correct_count += 1

        print(
            f"seed={seed} | "
            f"cases={len(cases)} | "
            f"correct={correct_count}/"
            f"{len(cases)}"
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
        / "e6_cumulative_exhaustion_raw.csv"
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
    # Overall summary
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "num_seeds":
                    df[
                        "seed"
                    ].nunique(),

                "total_cases":
                    len(
                        df
                    ),

                "correct_rejection_rate":
                    df[
                        "correct"
                    ].mean(),

                "individually_admissible_rate":
                    df[
                        "individually_admissible"
                    ].mean(),

                "inside_envelope_rate":
                    df[
                        "all_inside_envelope"
                    ].mean(),

                "cumulative_rejection_rate":
                    (
                        df[
                            "rejection_reason"
                        ]
                        == (
                            "cumulative_reserve_exhausted"
                        )
                    ).mean(),

                "mean_sequence_length":
                    df[
                        "sequence_length"
                    ].mean(),

                "min_sequence_length":
                    df[
                        "sequence_length"
                    ].min(),

                "max_sequence_length":
                    df[
                        "sequence_length"
                    ].max(),

                "mean_accepted_prefix":
                    df[
                        "accepted_before_reject"
                    ].mean(),
            }
        ]
    )

    summary_path = (
        ROOT
        / "results"
        / "tables"
        / "e6_cumulative_exhaustion_summary.csv"
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
    # Per-seed summary
    # ========================================================

    seed_summary = (
        df
        .groupby(
            "seed"
        )
        .agg(
            cases=(
                "case_id",
                "count",
            ),

            correct_rate=(
                "correct",
                "mean",
            ),

            individually_admissible_rate=(
                "individually_admissible",
                "mean",
            ),

            cumulative_rejection_rate=(
                "rejection_reason",

                lambda series:
                    (
                        series
                        == (
                            "cumulative_reserve_exhausted"
                        )
                    ).mean(),
            ),

            mean_sequence_length=(
                "sequence_length",
                "mean",
            ),
        )
        .reset_index()
    )

    seed_path = (
        ROOT
        / "results"
        / "tables"
        / "e6_cumulative_exhaustion_by_seed.csv"
    )

    seed_summary.to_csv(
        seed_path,
        index=False,
    )

    # ========================================================
    # Binding-dimension summary
    # ========================================================

    dimension_summary = (
        df
        .groupby(
            "binding_dimension"
        )
        .agg(
            cases=(
                "case_id",
                "count",
            ),

            mean_sequence_length=(
                "sequence_length",
                "mean",
            ),

            correct_rate=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    dimension_path = (
        ROOT
        / "results"
        / "tables"
        / "e6_cumulative_exhaustion_by_dimension.csv"
    )

    dimension_summary.to_csv(
        dimension_path,
        index=False,
    )

    # ========================================================
    # Figure:
    # distribution of accepted prefix length
    # ========================================================

    figure_path = (
        ROOT
        / "results"
        / "figures"
        / "e6_cumulative_exhaustion.png"
    )

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            7.5,
            4.8,
        )
    )

    plt.hist(
        df[
            "accepted_before_reject"
        ],

        bins=range(
            int(
                df[
                    "accepted_before_reject"
                ].min()
            ),
            int(
                df[
                    "accepted_before_reject"
                ].max()
            )
            + 2,
        ),

        align="left",
        rwidth=0.85,
    )

    plt.xlabel(
        "Accepted exceptional actions before cumulative rejection"
    )

    plt.ylabel(
        "Number of cases"
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
        "=" * 100
    )

    print(
        "E6 CUMULATIVE EXHAUSTION SUMMARY"
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
        "By seed:"
    )

    print(
        seed_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "By binding dimension:"
    )

    print(
        dimension_summary.to_string(
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
        f"[OK] seed summary -> "
        f"{seed_path}"
    )

    print(
        f"[OK] dimension summary -> "
        f"{dimension_path}"
    )

    print(
        f"[OK] figure -> "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()