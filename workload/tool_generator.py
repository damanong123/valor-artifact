from __future__ import annotations

from typing import Dict

import numpy as np

from valor.models import ToolSpec


def generate_tools(
    seed: int,
    num_tools: int = 30,
    num_domains: int = 8,
    num_tools_outside_dmax: int = 1,
) -> Dict[str, ToolSpec]:
    """
    Generate a deterministic synthetic tool universe.

    Paper-aligned structure:
    - 30 registered tools
    - 8 total protected effect domains
    - normal parameter range
    - larger exceptional / contract range
    - four typed-effect coefficients
    - exactly one tool outside Dmax in our current experiment

    Important design:
    The final effect domain is reserved for tools outside Dmax.
    This makes:

        D(a) subset Dmax

    an actual domain-level authorization check.
    """

    if num_tools <= 0:
        raise ValueError(
            "num_tools must be positive"
        )

    if num_domains < 2:
        raise ValueError(
            "num_domains must be at least 2"
        )

    if num_tools_outside_dmax != 1:
        raise ValueError(
            "Current experiment expects exactly "
            "one tool outside Dmax."
        )

    if num_tools_outside_dmax >= num_tools:
        raise ValueError(
            "At least one tool must remain "
            "inside Dmax."
        )

    rng = np.random.default_rng(seed)

    all_domains = [
        f"domain_{i:02d}"
        for i in range(num_domains)
    ]

    # All except the last domain belong to Dmax.
    allowed_domains = all_domains[:-1]

    # The final domain is outside Dmax.
    excluded_domain = all_domains[-1]

    excluded_index = int(
        rng.choice(num_tools)
    )

    tools: Dict[str, ToolSpec] = {}

    allowed_tool_counter = 0

    for i in range(num_tools):

        name = f"tool_{i:02d}"

        is_excluded = (
            i == excluded_index
        )

        if is_excluded:
            domain = excluded_domain
        else:
            domain = allowed_domains[
                allowed_tool_counter
                % len(allowed_domains)
            ]

            allowed_tool_counter += 1

        # ------------------------------
        # Normal operating range
        # ------------------------------

        normal_min = 0.0

        normal_max = float(
            rng.uniform(
                2.0,
                5.0,
            )
        )

        # ------------------------------
        # Exceptional contract range
        # ------------------------------

        expansion_factor = float(
            rng.uniform(
                1.5,
                2.5,
            )
        )

        exceptional_min = 0.0

        exceptional_max = (
            normal_max
            * expansion_factor
        )

        # ------------------------------
        # Effect coefficients
        # ------------------------------

        energy_coeff = float(
            rng.uniform(
                0.20,
                1.00,
            )
        )

        stress_coeff = float(
            rng.uniform(
                0.10,
                0.70,
            )
        )

        external_scope_coeff = float(
            rng.uniform(
                0.05,
                0.40,
            )
        )

        # Irreversible effects are sparse.
        if rng.random() < 0.25:
            irreversible_coeff = float(
                rng.uniform(
                    0.05,
                    0.20,
                )
            )
        else:
            irreversible_coeff = 0.0

        tool = ToolSpec(
            name=name,
            domain=domain,

            normal_min=normal_min,
            normal_max=normal_max,

            exceptional_min=(
                exceptional_min
            ),

            exceptional_max=(
                exceptional_max
            ),

            energy_coeff=(
                energy_coeff
            ),

            stress_coeff=(
                stress_coeff
            ),

            irreversible_coeff=(
                irreversible_coeff
            ),

            external_scope_coeff=(
                external_scope_coeff
            ),

            registered=True,

            in_dmax=(
                not is_excluded
            ),
        )

        tools[name] = tool

    return tools