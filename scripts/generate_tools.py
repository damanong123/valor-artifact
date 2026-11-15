from pathlib import Path

import pandas as pd
import yaml

from workload.tool_generator import generate_tools


ROOT = Path(__file__).resolve().parents[1]


def main():

    config_path = ROOT / "config" / "experiment.yaml"

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    seeds = config["reproducibility"]["seeds"]

    num_tools = config["synthetic_world"]["num_tools"]

    num_domains = config[
        "synthetic_world"
    ]["num_effect_domains"]

    num_outside = config[
        "constitution"
    ]["num_tools_outside_dmax"]

    output_dir = ROOT / "results" / "raw"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in seeds:

        tools = generate_tools(
            seed=seed,
            num_tools=num_tools,
            num_domains=num_domains,
            num_tools_outside_dmax=num_outside,
        )

        rows = []

        for tool in tools.values():

            rows.append(
                {
                    "tool": tool.name,
                    "domain": tool.domain,

                    "normal_min":
                        tool.normal_min,

                    "normal_max":
                        tool.normal_max,

                    "exceptional_min":
                        tool.exceptional_min,

                    "exceptional_max":
                        tool.exceptional_max,

                    "energy_coeff":
                        tool.energy_coeff,

                    "stress_coeff":
                        tool.stress_coeff,

                    "irreversible_coeff":
                        tool.irreversible_coeff,

                    "external_scope_coeff":
                        tool.external_scope_coeff,

                    "registered":
                        tool.registered,

                    "in_dmax":
                        tool.in_dmax,
                }
            )

        df = pd.DataFrame(rows)

        output_path = (
            output_dir
            / f"tools_seed_{seed}.csv"
        )

        df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"[OK] seed={seed} "
            f"tools={len(df)} "
            f"outside_dmax={(~df['in_dmax']).sum()} "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()