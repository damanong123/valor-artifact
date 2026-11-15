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

from valor.models import Action


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


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
        capability_ttl_seconds=5,
    )

    tool = next(
        tool
        for tool in tools.values()
        if tool.in_dmax
    )

    parameter = (
        tool.normal_max
        +
        (
            tool.exceptional_max
            - tool.normal_max
        )
        * 0.5
    )

    action = Action(
        tool=tool.name,
        parameter=parameter,
    )

    print(
        "=================================="
    )

    print(
        "VALOR Propose Demo"
    )

    print(
        "=================================="
    )

    print()

    print(
        "Action:"
    )

    print(
        f"  tool      = "
        f"{action.tool}"
    )

    print(
        f"  parameter = "
        f"{action.parameter:.4f}"
    )

    print()

    print(
        "Reserve BEFORE:"
    )

    for k, v in (
        ledger
        .remaining_reserve()
        .items()
    ):
        print(
            f"  {k:15s} "
            f"{v:.6f}"
        )

    capability = (
        kernel.propose(
            action
        )
    )

    print()

    print(
        "Capability issued:"
    )

    print(
        f"  epoch  = "
        f"{capability.epoch}"
    )

    print(
        f"  nonce  = "
        f"{capability.nonce}"
    )

    print(
        f"  tool   = "
        f"{capability.action.tool}"
    )

    print(
        f"  param  = "
        f"{capability.action.parameter:.4f}"
    )

    print()

    print(
        "Reserved effect:"
    )

    for k, v in (
        capability
        .reserved_effect_vector()
        .items()
    ):
        print(
            f"  {k:15s} "
            f"{v:.6f}"
        )

    print()

    print(
        "Reserve AFTER:"
    )

    for k, v in (
        ledger
        .remaining_reserve()
        .items()
    ):
        print(
            f"  {k:15s} "
            f"{v:.6f}"
        )

    print()

    print(
        "Outstanding C:"
    )

    print(
        ledger.outstanding_total()
    )

    print()

    print(
        "R + C + S:"
    )

    print(
        ledger.conservation_total()
    )

    print()

    print(
        "B0:"
    )

    print(
        ledger.initial_reserve()
    )

    print()

    print(
        "Authority conservation:"
    )

    print(
        ledger.conservation_holds()
    )


if __name__ == "__main__":
    main()