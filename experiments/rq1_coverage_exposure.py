from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt

from workload.tool_generator import (
    generate_tools,
)
from workload.trace_generator import (
    generate_feasible_traces,
)
from workload.catalog_generator import (
    select_known_profiles,
    select_unseen_traces,
)
from workload.adversary import (
    union_false_trigger_exposure,
    valor_false_trigger_exposure,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)
from valor.models import Action

from policies.baselines import (
    UnionBreakGlassPolicy,
    ValorTracePolicy,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def percentage(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def main():

    # ========================================================
    # Load configuration
    # ========================================================

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

    k_values = (
        cfg[
            "catalog"
        ]["known_profile_counts"]
    )

    attack_requests = (
        cfg[
            "false_trigger_attack"
        ]["requests_per_run"]
    )

    num_traces = (
        cfg[
            "workload"
        ]["traces_per_seed"]
    )

    min_actions = (
        cfg[
            "workload"
        ]["min_actions_per_trace"]
    )

    max_actions = (
        cfg[
            "workload"
        ]["max_actions_per_trace"]
    )

    raw_rows = []

    # ========================================================
    # Run seeds
    # ========================================================

    for seed in seeds:

        print(
            "=" * 72
        )

        print(
            f"RQ1 seed={seed}"
        )

        # ----------------------------------------------------
        # Synthetic world
        # ----------------------------------------------------

        tools = generate_tools(
            seed=seed,

            num_tools=cfg[
                "synthetic_world"
            ]["num_tools"],

            num_domains=cfg[
                "synthetic_world"
            ]["num_effect_domains"],

            num_tools_outside_dmax=cfg[
                "constitution"
            ]["num_tools_outside_dmax"],
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
        # Re-generate deterministic workload
        # ----------------------------------------------------

        traces = (
            generate_feasible_traces(
                seed=seed,
                tools=tools,
                adapter=adapter,
                reserve=reserve,
                num_traces=num_traces,
                min_actions=min_actions,
                max_actions=max_actions,
            )
        )

        # ----------------------------------------------------
        # Full VALOR candidate attack space:
        # maximum contract-bound exceptional action
        # for every allowed tool.
        # ----------------------------------------------------

        valor_attack_candidates = [
            Action(
                tool=tool.name,
                parameter=tool.exceptional_max,
            )
            for tool in tools.values()
            if tool.in_dmax
        ]

        (
            _,
            valor_exposure,
            valor_attack_accepted,
        ) = valor_false_trigger_exposure(
            candidate_actions=(
                valor_attack_candidates
            ),
            adapter=adapter,
            reserve=reserve,
            num_requests=attack_requests,
        )

        # ----------------------------------------------------
        # K sweep
        # ----------------------------------------------------

        for k in k_values:

            known = (
                select_known_profiles(
                    traces,
                    k,
                )
            )

            unseen = (
                select_unseen_traces(
                    traces,
                    k,
                )
            )

            union = (
                UnionBreakGlassPolicy(
                    tools=tools,
                    known_profiles=known,
                )
            )

            valor = (
                ValorTracePolicy(
                    tools=tools,
                    adapter=adapter,
                    dmax=dmax,
                    reserve=reserve,
                )
            )

            # ----------------------------------------------
            # Unseen coverage
            # ----------------------------------------------

            union_accepted = sum(
                union.authorize_trace(
                    trace
                )
                for trace in unseen
            )

            valor_accepted = sum(
                valor.authorize_trace(
                    trace
                )
                for trace in unseen
            )

            union_coverage = percentage(
                union_accepted,
                len(unseen),
            )

            valor_coverage = percentage(
                valor_accepted,
                len(unseen),
            )

            # ----------------------------------------------
            # Union false-trigger exposure
            # ----------------------------------------------

            union_candidates = (
                union
                .exceptional_attack_candidates()
            )

            (
                _,
                union_exposure,
                union_attack_sequence,
            ) = (
                union_false_trigger_exposure(
                    candidate_actions=(
                        union_candidates
                    ),
                    adapter=adapter,
                    reserve=reserve,
                    num_requests=(
                        attack_requests
                    ),
                )
            )

            raw_rows.append(
                {
                    "seed": seed,
                    "K": k,

                    "num_unseen":
                        len(unseen),

                    "union_accepted":
                        union_accepted,

                    "union_coverage_pct":
                        union_coverage,

                    "valor_accepted":
                        valor_accepted,

                    "valor_coverage_pct":
                        valor_coverage,

                    "union_exposure_x":
                        union_exposure,

                    "valor_exposure_x":
                        valor_exposure,

                    "union_attack_requests":
                        len(
                            union_attack_sequence
                        ),

                    "valor_attack_requests":
                        len(
                            valor_attack_accepted
                        ),

                    "union_catalog_tools":
                        len(
                            union.allowed_max
                        ),
                }
            )

            print(
                f"K={k:3d} | "
                f"union coverage="
                f"{union_coverage:6.2f}% | "
                f"VALOR="
                f"{valor_coverage:6.2f}% | "
                f"union exposure="
                f"{union_exposure:6.2f}x | "
                f"VALOR exposure="
                f"{valor_exposure:6.2f}x"
            )

    # ========================================================
    # Raw output
    # ========================================================

    raw_df = pd.DataFrame(
        raw_rows
    )

    raw_path = (
        ROOT
        / "results"
        / "raw"
        / "rq1_per_seed.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # Aggregate across seeds
    # ========================================================

    summary = (
        raw_df
        .groupby("K")
        .agg(
            union_coverage_mean=(
                "union_coverage_pct",
                "mean",
            ),

            union_coverage_sd=(
                "union_coverage_pct",
                "std",
            ),

            valor_coverage_mean=(
                "valor_coverage_pct",
                "mean",
            ),

            valor_coverage_sd=(
                "valor_coverage_pct",
                "std",
            ),

            union_exposure_mean=(
                "union_exposure_x",
                "mean",
            ),

            union_exposure_sd=(
                "union_exposure_x",
                "std",
            ),

            valor_exposure_mean=(
                "valor_exposure_x",
                "mean",
            ),

            valor_exposure_sd=(
                "valor_exposure_x",
                "std",
            ),

            union_catalog_tools_mean=(
                "union_catalog_tools",
                "mean",
            ),
        )
        .reset_index()
    )

    table_path = (
        ROOT
        / "results"
        / "tables"
        / "rq1_summary.csv"
    )

    table_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        table_path,
        index=False,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "RQ1 SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Coverage figure
    # ========================================================

    figure_dir = (
        ROOT
        / "results"
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.errorbar(
        summary["K"],
        summary[
            "union_coverage_mean"
        ],
        yerr=summary[
            "union_coverage_sd"
        ],
        marker="o",
        capsize=4,
        label="Union break-glass",
    )

    plt.errorbar(
        summary["K"],
        summary[
            "valor_coverage_mean"
        ],
        yerr=summary[
            "valor_coverage_sd"
        ],
        marker="s",
        capsize=4,
        label="VALOR",
    )

    plt.xlabel(
        "Known contingency profiles K"
    )

    plt.ylabel(
        "Unseen-trace coverage (%)"
    )

    plt.ylim(
        0,
        105,
    )

    plt.legend()

    plt.tight_layout()

    figure_path = (
        figure_dir
        / "rq1_coverage.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print()
    print(
        f"[OK] raw -> {raw_path}"
    )

    print(
        f"[OK] table -> {table_path}"
    )

    print(
        f"[OK] figure -> {figure_path}"
    )


if __name__ == "__main__":
    main()