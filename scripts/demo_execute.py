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
    ExecutionRevalidationError,
    ReplayDetectedError,
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
    name,
    vector,
):

    print(name)

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
        capability_ttl_seconds=5,
    )

    allowed_tools = [
        tool
        for tool in tools.values()
        if tool.in_dmax
    ]

    # ======================================================
    # CASE 1
    # Successful Execute
    # ======================================================

    print(
        "=" * 60
    )

    print(
        "CASE 1: Stable state -> Execute succeeds"
    )

    print(
        "=" * 60
    )

    tool_1 = allowed_tools[0]

    action_1 = make_action(
        tool_1
    )

    cap_1 = kernel.propose(
        action_1
    )

    print(
        f"Action: "
        f"{action_1}"
    )

    print_vector(
        "Reserved:",
        cap_1.reserved_effect_vector(),
    )

    permitted = kernel.execute(
        cap_1,
        action_1,

        state={
            "effect_multiplier": 1.0
        },
    )

    print(
        f"Execute permitted: "
        f"{permitted}"
    )

    print(
        f"Nonce resolved: "
        f"{ledger.is_resolved(cap_1.nonce)}"
    )

    print(
        f"In flight: "
        f"{ledger.is_in_flight(cap_1.nonce)}"
    )

    print(
        f"Reservation still in C: "
        f"{ledger.has_reservation(cap_1.nonce)}"
    )

    print(
        f"Authority conservation: "
        f"{ledger.conservation_holds()}"
    )

    print()

    # ======================================================
    # CASE 2
    # State drift
    # ======================================================

    print(
        "=" * 60
    )

    print(
        "CASE 2: State drift -> Execute aborts"
    )

    print(
        "=" * 60
    )

    tool_2 = allowed_tools[1]

    action_2 = make_action(
        tool_2
    )

    before_propose = (
        ledger.remaining_reserve()
    )

    cap_2 = kernel.propose(
        action_2
    )

    print(
        f"Action: "
        f"{action_2}"
    )

    print_vector(
        "Reserved:",
        cap_2.reserved_effect_vector(),
    )

    try:

        kernel.execute(
            cap_2,
            action_2,

            state={
                "effect_multiplier":
                    2.0
            },
        )

    except ExecutionRevalidationError as exc:

        print(
            "Execute result: ABORT"
        )

        print(
            f"Reason: {exc}"
        )

    print()

    print(
        f"Nonce resolved: "
        f"{ledger.is_resolved(cap_2.nonce)}"
    )

    print(
        f"In flight: "
        f"{ledger.is_in_flight(cap_2.nonce)}"
    )

    print(
        f"Reservation still in C: "
        f"{ledger.has_reservation(cap_2.nonce)}"
    )

    print()

    print_vector(
        "Reserve after abort:",
        ledger.remaining_reserve(),
    )

    print(
        "Abort restored this "
        "capability's reservation:"
    )

    print(
        ledger.remaining_reserve()
        == before_propose
    )

    print()

    # ======================================================
    # CASE 3
    # Replay aborted capability
    # ======================================================

    print(
        "=" * 60
    )

    print(
        "CASE 3: Replay aborted capability"
    )

    print(
        "=" * 60
    )

    try:

        kernel.execute(
            cap_2,
            action_2,
        )

    except ReplayDetectedError as exc:

        print(
            "Replay result: REJECT"
        )

        print(
            f"Reason: {exc}"
        )

    print()

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
        "Authority conservation:"
    )

    print(
        ledger.conservation_holds()
    )


if __name__ == "__main__":
    main()