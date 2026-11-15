from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.models import Action

from valor.normal_policy import (
    is_normally_authorized,
)


def main():

    tools = generate_tools(
        seed=20260903
    )

    adapter = SyntheticEffectAdapter(
        tools
    )

    tool = tools["tool_00"]

    exceptional_parameter = (
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
        parameter=exceptional_parameter,
    )

    print(
        "================================="
    )

    print(
        "VALOR Synthetic Effect Adapter"
    )

    print(
        "================================="
    )

    print(
        f"Tool: {tool.name}"
    )

    print(
        f"Domain: {tool.domain}"
    )

    print(
        f"Normal max: "
        f"{tool.normal_max:.4f}"
    )

    print(
        f"Exceptional max: "
        f"{tool.exceptional_max:.4f}"
    )

    print(
        f"Requested parameter: "
        f"{action.parameter:.4f}"
    )

    normal = is_normally_authorized(
        action,
        tools,
    )

    print(
        f"Normally authorized: "
        f"{normal}"
    )

    contract = adapter.contract(
        action
    )

    print()
    print("Effect Contract χ(a,x)")
    print("-------------------------")

    print(
        f"D = {sorted(contract.domains)}"
    )

    print(
        "b =",
        contract.effect_vector(),
    )

    print(
        f"q = {contract.hard_guard_ok}"
    )

    print(
        f"rho = {contract.receipt_class}"
    )


if __name__ == "__main__":
    main()