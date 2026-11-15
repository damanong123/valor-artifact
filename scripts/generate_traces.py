from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from workload.tool_generator import (
    generate_tools,
)

from workload.trace_generator import (
    cumulative_effect,
    generate_feasible_traces,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.normal_policy import (
    is_normally_authorized,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


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

        config = yaml.safe_load(
            f
        )

    seeds = (
        config[
            "reproducibility"
        ]["seeds"]
    )

    num_tools = (
        config[
            "synthetic_world"
        ]["num_tools"]
    )

    num_domains = (
        config[
            "synthetic_world"
        ]["num_effect_domains"]
    )

    num_outside = (
        config[
            "constitution"
        ]["num_tools_outside_dmax"]
    )

    initial_reserve = (
        config[
            "constitution"
        ]["initial_reserve"]
    )

    num_traces = (
        config[
            "workload"
        ]["traces_per_seed"]
    )

    min_actions = (
        config[
            "workload"
        ]["min_actions_per_trace"]
    )

    max_actions = (
        config[
            "workload"
        ]["max_actions_per_trace"]
    )

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        ROOT
        / "results"
        / "raw"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Generate each seed
    # ========================================================

    for seed in seeds:

        print(
            "=" * 70
        )

        print(
            f"Generating traces "
            f"for seed={seed}"
        )

        # ----------------------------------------------------
        # Tool universe
        # ----------------------------------------------------

        tools = generate_tools(
            seed=seed,
            num_tools=num_tools,
            num_domains=num_domains,
            num_tools_outside_dmax=(
                num_outside
            ),
        )

        adapter = (
            SyntheticEffectAdapter(
                tools
            )
        )

        # ----------------------------------------------------
        # Trace workload
        # ----------------------------------------------------

        traces = (
            generate_feasible_traces(
                seed=seed,

                tools=tools,

                adapter=adapter,

                reserve=initial_reserve,

                num_traces=(
                    num_traces
                ),

                min_actions=(
                    min_actions
                ),

                max_actions=(
                    max_actions
                ),
            )
        )

        # ----------------------------------------------------
        # Flatten traces:
        # one CSV row per action
        # ----------------------------------------------------

        rows = []

        trace_summaries = []

        total_normal_actions = 0
        total_exceptional_actions = 0

        for trace in traces:

            trace_effect = (
                cumulative_effect(
                    trace.actions,
                    adapter,
                )
            )

            exceptional_count = 0

            for (
                action_index,
                action,
            ) in enumerate(
                trace.actions
            ):

                tool = tools[
                    action.tool
                ]

                normal_authorized = (
                    is_normally_authorized(
                        action,
                        tools,
                    )
                )

                if normal_authorized:
                    total_normal_actions += 1
                else:
                    total_exceptional_actions += 1
                    exceptional_count += 1

                contract = (
                    adapter.contract(
                        action
                    )
                )

                effect = (
                    contract
                    .effect_vector()
                )

                rows.append(
                    {
                        "seed":
                            seed,

                        "trace_id":
                            trace.trace_id,

                        "action_index":
                            action_index,

                        "tool":
                            action.tool,

                        "domain":
                            tool.domain,

                        "parameter":
                            action.parameter,

                        "normal_authorized":
                            normal_authorized,

                        "energy":
                            effect[
                                "energy"
                            ],

                        "stress":
                            effect[
                                "stress"
                            ],

                        "irreversible":
                            effect[
                                "irreversible"
                            ],

                        "external_scope":
                            effect[
                                "external_scope"
                            ],
                    }
                )

            trace_summaries.append(
                {
                    "trace_id":
                        trace.trace_id,

                    "num_actions":
                        len(
                            trace.actions
                        ),

                    "num_exceptional_actions":
                        exceptional_count,

                    "total_effect":
                        trace_effect,
                }
            )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        dataframe = (
            pd.DataFrame(
                rows
            )
        )

        csv_path = (
            output_dir
            / (
                f"traces_seed_"
                f"{seed}.csv"
            )
        )

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        # ----------------------------------------------------
        # Summary JSON
        # ----------------------------------------------------

        summary = {
            "seed":
                seed,

            "num_traces":
                len(traces),

            "num_actions":
                len(rows),

            "normal_actions":
                total_normal_actions,

            "exceptional_actions":
                total_exceptional_actions,

            "min_actions_per_trace":
                min(
                    len(trace.actions)
                    for trace in traces
                ),

            "max_actions_per_trace":
                max(
                    len(trace.actions)
                    for trace in traces
                ),

            "initial_reserve":
                initial_reserve,

            "traces":
                trace_summaries,
        }

        json_path = (
            output_dir
            / (
                f"traces_seed_"
                f"{seed}_summary.json"
            )
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Terminal summary
        # ----------------------------------------------------

        print(
            f"[OK] traces="
            f"{len(traces)}"
        )

        print(
            f"[OK] actions="
            f"{len(rows)}"
        )

        print(
            f"[OK] normal="
            f"{total_normal_actions}"
        )

        print(
            f"[OK] exceptional="
            f"{total_exceptional_actions}"
        )

        print(
            f"[OK] CSV -> "
            f"{csv_path}"
        )

        print(
            f"[OK] JSON -> "
            f"{json_path}"
        )

        print()


if __name__ == "__main__":
    main()