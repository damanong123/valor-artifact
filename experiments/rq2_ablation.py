from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

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

from valor.models import (
    Action,
)

from policies.baselines import (
    UnionBreakGlassPolicy,
    CatalogReservePolicy,
    EnvelopeOnlyPolicy,
    ValorTracePolicy,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# Helpers
# ============================================================

def percentage(
    numerator: int,
    denominator: int,
) -> float:
    """
    Convert a count into percentage.
    """

    if denominator == 0:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def full_envelope_candidates(
    tools,
) -> list[Action]:
    """
    Construct the strongest single exceptional action
    available through each registered Dmax tool.

    This represents the runtime action space available to
    Envelope-only / VALOR.

    The exact action instance does not have to appear in
    any contingency catalog.
    """

    return [
        Action(
            tool=tool.name,
            parameter=tool.exceptional_max,
        )
        for tool in tools.values()
        if (
            tool.registered
            and tool.in_dmax
        )
    ]


# ============================================================
# Main experiment
# ============================================================

def main():

    # ========================================================
    # Load configuration
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

    attack_requests = (
        cfg[
            "false_trigger_attack"
        ]["requests_per_run"]
    )

    # --------------------------------------------------------
    # Paper RQ2 uses K = 50
    # --------------------------------------------------------

    K = 50

    rows = []

    # ========================================================
    # Run experiment for each seed
    # ========================================================

    for seed in seeds:

        print(
            "=" * 78
        )

        print(
            f"RQ2 seed={seed}"
        )

        print(
            "=" * 78
        )

        # ----------------------------------------------------
        # Synthetic tool universe
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
        # Deterministic feasible workload
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

        known = (
            select_known_profiles(
                traces,
                K,
            )
        )

        unseen = (
            select_unseen_traces(
                traces,
                K,
            )
        )

        # ====================================================
        # Build four policies
        # ====================================================

        union_policy = (
            UnionBreakGlassPolicy(
                tools=tools,
                known_profiles=known,
            )
        )

        catalog_reserve_policy = (
            CatalogReservePolicy(
                tools=tools,

                known_profiles=known,

                adapter=adapter,

                reserve=reserve,
            )
        )

        envelope_policy = (
            EnvelopeOnlyPolicy(
                tools=tools,

                adapter=adapter,

                dmax=dmax,
            )
        )

        valor_policy = (
            ValorTracePolicy(
                tools=tools,

                adapter=adapter,

                dmax=dmax,

                reserve=reserve,
            )
        )

        # ====================================================
        # Coverage
        # ====================================================

        union_accepted = sum(
            union_policy.authorize_trace(
                trace
            )
            for trace in unseen
        )

        catalog_reserve_accepted = sum(
            catalog_reserve_policy
            .authorize_trace(
                trace
            )
            for trace in unseen
        )

        envelope_accepted = sum(
            envelope_policy
            .authorize_trace(
                trace
            )
            for trace in unseen
        )

        valor_accepted = sum(
            valor_policy
            .authorize_trace(
                trace
            )
            for trace in unseen
        )

        union_coverage = percentage(
            union_accepted,
            len(unseen),
        )

        catalog_reserve_coverage = (
            percentage(
                catalog_reserve_accepted,
                len(unseen),
            )
        )

        envelope_coverage = percentage(
            envelope_accepted,
            len(unseen),
        )

        valor_coverage = percentage(
            valor_accepted,
            len(unseen),
        )

        # ====================================================
        # Attack candidate spaces
        # ====================================================

        # ----------------------------------------------------
        # Catalog-derived scope
        # ----------------------------------------------------

        catalog_candidates = (
            union_policy
            .exceptional_attack_candidates()
        )

        # ----------------------------------------------------
        # Runtime envelope scope
        # ----------------------------------------------------

        envelope_candidates = (
            full_envelope_candidates(
                tools
            )
        )

        # ====================================================
        # False-trigger exposure
        # ====================================================

        # ----------------------------------------------------
        # 1. Union Catalog
        #
        # Standing catalog scope
        # No cumulative reserve.
        # ----------------------------------------------------

        (
            _,
            union_exposure,
            union_sequence,
        ) = (
            union_false_trigger_exposure(
                candidate_actions=(
                    catalog_candidates
                ),

                adapter=adapter,

                reserve=reserve,

                num_requests=(
                    attack_requests
                ),
            )
        )

        # ----------------------------------------------------
        # 2. Catalog + Reserve
        #
        # Same catalog scope, but finite typed reserve.
        # ----------------------------------------------------

        (
            _,
            catalog_reserve_exposure,
            catalog_reserve_sequence,
        ) = (
            valor_false_trigger_exposure(
                candidate_actions=(
                    catalog_candidates
                ),

                adapter=adapter,

                reserve=reserve,

                num_requests=(
                    attack_requests
                ),
            )
        )

        # ----------------------------------------------------
        # 3. Envelope Only
        #
        # Runtime-instantiated envelope scope
        # No cumulative reserve.
        # ----------------------------------------------------

        (
            _,
            envelope_exposure,
            envelope_sequence,
        ) = (
            union_false_trigger_exposure(
                candidate_actions=(
                    envelope_candidates
                ),

                adapter=adapter,

                reserve=reserve,

                num_requests=(
                    attack_requests
                ),
            )
        )

        # ----------------------------------------------------
        # 4. Full VALOR
        #
        # Runtime envelope + finite reserve.
        # ----------------------------------------------------

        (
            _,
            valor_exposure,
            valor_sequence,
        ) = (
            valor_false_trigger_exposure(
                candidate_actions=(
                    envelope_candidates
                ),

                adapter=adapter,

                reserve=reserve,

                num_requests=(
                    attack_requests
                ),
            )
        )

        # ====================================================
        # Save per-seed rows
        # ====================================================

        methods = [
            (
                "Union Catalog",
                union_coverage,
                union_exposure,
                len(
                    union_sequence
                ),
            ),

            (
                "Catalog + Reserve",
                catalog_reserve_coverage,
                catalog_reserve_exposure,
                len(
                    catalog_reserve_sequence
                ),
            ),

            (
                "Envelope Only",
                envelope_coverage,
                envelope_exposure,
                len(
                    envelope_sequence
                ),
            ),

            (
                "Full VALOR",
                valor_coverage,
                valor_exposure,
                len(
                    valor_sequence
                ),
            ),
        ]

        for (
            method,
            coverage,
            exposure,
            accepted_attack_requests,
        ) in methods:

            rows.append(
                {
                    "seed":
                        seed,

                    "K":
                        K,

                    "method":
                        method,

                    "num_unseen":
                        len(unseen),

                    "coverage_pct":
                        coverage,

                    "exposure_x":
                        exposure,

                    "accepted_attack_requests":
                        accepted_attack_requests,

                    "catalog_tools":
                        len(
                            union_policy
                            .allowed_max
                        ),
                }
            )

        # ====================================================
        # Terminal output
        # ====================================================

        print(
            f"{'Method':22s} "
            f"{'Coverage':>12s} "
            f"{'Exposure':>12s} "
            f"{'Atk accepted':>14s}"
        )

        print(
            "-" * 66
        )

        for (
            method,
            coverage,
            exposure,
            accepted_attack_requests,
        ) in methods:

            print(
                f"{method:22s} "
                f"{coverage:11.2f}% "
                f"{exposure:11.2f}x "
                f"{accepted_attack_requests:14d}"
            )

        print()

    # ========================================================
    # Raw CSV
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    raw_path = (
        ROOT
        / "results"
        / "raw"
        / "rq2_per_seed.csv"
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
    # Summary
    # ========================================================

    summary = (
        raw_df
        .groupby(
            "method",
            sort=False,
        )
        .agg(
            coverage_mean=(
                "coverage_pct",
                "mean",
            ),

            coverage_sd=(
                "coverage_pct",
                "std",
            ),

            exposure_mean=(
                "exposure_x",
                "mean",
            ),

            exposure_sd=(
                "exposure_x",
                "std",
            ),

            attack_requests_mean=(
                "accepted_attack_requests",
                "mean",
            ),

            catalog_tools_mean=(
                "catalog_tools",
                "mean",
            ),
        )
        .reset_index()
    )

    # Force logical method order.
    method_order = [
        "Union Catalog",
        "Catalog + Reserve",
        "Envelope Only",
        "Full VALOR",
    ]

    summary["order"] = (
        summary["method"]
        .map(
            {
                name: index
                for index, name
                in enumerate(
                    method_order
                )
            }
        )
    )

    summary = (
        summary
        .sort_values(
            "order"
        )
        .drop(
            columns=[
                "order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    table_path = (
        ROOT
        / "results"
        / "tables"
        / "rq2_ablation_summary.csv"
    )

    table_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        table_path,
        index=False,
    )

    print(
        "=" * 78
    )

    print(
        "RQ2 ABLATION SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Figure 1:
    # Unseen coverage
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
        figsize=(8, 4.8)
    )

    plt.bar(
        summary["method"],
        summary["coverage_mean"],
        yerr=summary["coverage_sd"],
        capsize=4,
    )

    plt.ylabel(
        "Unseen-trace coverage (%)"
    )

    plt.ylim(
        0,
        105,
    )

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.tight_layout()

    coverage_figure_path = (
        figure_dir
        / "rq2_coverage_ablation.png"
    )

    plt.savefig(
        coverage_figure_path,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Figure 2:
    # Exposure
    # ========================================================

    plt.figure(
        figsize=(8, 4.8)
    )

    plt.bar(
        summary["method"],
        summary["exposure_mean"],
        yerr=summary["exposure_sd"],
        capsize=4,
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="VALOR one-session reserve",
    )

    plt.ylabel(
        "False-trigger exposure (× reserve)"
    )

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.legend()

    plt.tight_layout()

    exposure_figure_path = (
        figure_dir
        / "rq2_exposure_ablation.png"
    )

    plt.savefig(
        exposure_figure_path,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Output paths
    # ========================================================

    print()

    print(
        f"[OK] raw -> "
        f"{raw_path}"
    )

    print(
        f"[OK] table -> "
        f"{table_path}"
    )

    print(
        f"[OK] coverage figure -> "
        f"{coverage_figure_path}"
    )

    print(
        f"[OK] exposure figure -> "
        f"{exposure_figure_path}"
    )


if __name__ == "__main__":
    main()