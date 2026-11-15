import yaml
from pathlib import Path

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.kernel import (
    ValorKernel,
)

from valor.ledger import (
    InMemoryReserveLedger,
)

from valor.models import (
    Action,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def make_action(
    tool
):

    parameter = (
        tool.normal_max
        +
        (
            tool.exceptional_max
            - tool.normal_max
        )
        * 0.5
    )

    return Action(
        tool=tool.name,
        parameter=parameter,
    )


def print_vector(
    title,
    vector,
):

    print(title)

    for key, value in (
        vector.items()
    ):
        print(
            f"  {key:15s} "
            f"{value:.6f}"
        )

    print()


def main():

    with open(
        ROOT
        / "config"
        / "experiment.yaml",

        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)

    seed = (
        config[
            "reproducibility"
        ]["seeds"][0]
    )

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

    initial_reserve = (
        config[
            "constitution"
        ]["initial_reserve"]
    )

    ledger = (
        InMemoryReserveLedger(
            initial_reserve
        )
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=b"K" * 32,
        epoch=1,
    )

    allowed_tools = [
        tool
        for tool in tools.values()
        if tool.in_dmax
    ]

    # ======================================================
    # CASE 1:
    # Partial effect receipt
    # ======================================================

    print(
        "=" * 60
    )

    print(
        "CASE 1: Partial effect -> partial refund"
    )

    print(
        "=" * 60
    )

    action_1 = make_action(
        allowed_tools[0]
    )

    cap_1 = kernel.propose(
        action_1
    )

    kernel.execute(
        cap_1,
        action_1,
    )

    reserved_1 = (
        cap_1
        .reserved_effect_vector()
    )

    realized_1 = {
        dimension:
            value * 0.60

        for dimension, value
        in reserved_1.items()
    }

    print_vector(
        "Reserved b:",
        reserved_1,
    )

    print_vector(
        "Realized c:",
        realized_1,
    )

    result_1 = kernel.commit(
        cap_1,
        realized_effect=realized_1,
    )

    print_vector(
        "Refund b-c:",
        result_1[
            "refunded"
        ],
    )

    print_vector(
        "R after Commit:",
        ledger.remaining_reserve(),
    )

    print_vector(
        "C after Commit:",
        ledger.outstanding_total(),
    )

    print_vector(
        "S after Commit:",
        ledger.committed_spend(),
    )

    print(
        "Authority conservation:",
        ledger.conservation_holds(),
    )

    print()

    # ======================================================
    # CASE 2:
    # Indeterminate result
    # ======================================================

    print(
        "=" * 60
    )

    print(
        "CASE 2: Indeterminate outcome -> full charge"
    )

    print(
        "=" * 60
    )

    action_2 = make_action(
        allowed_tools[1]
    )

    cap_2 = kernel.propose(
        action_2
    )

    kernel.execute(
        cap_2,
        action_2,
    )

    reserved_2 = (
        cap_2
        .reserved_effect_vector()
    )

    print_vector(
        "Reserved b:",
        reserved_2,
    )

    result_2 = kernel.commit(
        cap_2,
        realized_effect=None,
    )

    print_vector(
        "Committed:",
        result_2[
            "committed"
        ],
    )

    print_vector(
        "Refund:",
        result_2[
            "refunded"
        ],
    )

    # ======================================================
    # Final accounting
    # ======================================================

    print(
        "=" * 60
    )

    print(
        "FINAL ACCOUNTING"
    )

    print(
        "=" * 60
    )

    print_vector(
        "R:",
        ledger.remaining_reserve(),
    )

    print_vector(
        "C:",
        ledger.outstanding_total(),
    )

    print_vector(
        "S:",
        ledger.committed_spend(),
    )

    print_vector(
        "R + C + S:",
        ledger.conservation_total(),
    )

    print_vector(
        "B0:",
        ledger.initial_reserve(),
    )

    print(
        "Authority conservation:",
        ledger.conservation_holds(),
    )


if __name__ == "__main__":
    main()