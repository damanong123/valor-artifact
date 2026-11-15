from workload.negative_control import (
    generate_feasible_cases,
    generate_infeasible_cases,
)

from workload.tool_generator import (
    generate_tools,
)

from experiments.rq5_negative_control import (
    authorize_case,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)


RESERVE = {
    "energy": 40.0,
    "stress": 25.0,
    "irreversible": 3.0,
    "external_scope": 18.0,
}


def build():

    tools = generate_tools(
        seed=20260903
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

    return (
        tools,
        adapter,
        dmax,
    )


def test_feasible_negative_control_class_admitted():

    (
        tools,
        adapter,
        dmax,
    ) = build()

    cases = (
        generate_feasible_cases(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_cases=20,
        )
    )

    for case in cases:

        admitted, _ = (
            authorize_case(
                case,

                tools=tools,

                adapter=adapter,

                dmax=dmax,

                reserve=RESERVE,
            )
        )

        assert admitted


def test_outside_dmax_cases_rejected():

    (
        tools,
        adapter,
        dmax,
    ) = build()

    cases = (
        generate_infeasible_cases(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_cases=20,
        )
    )

    outside_cases = [
        case
        for case in cases
        if case.mode
        == "outside_dmax"
    ]

    assert outside_cases

    for case in outside_cases:

        admitted, reason = (
            authorize_case(
                case,

                tools=tools,

                adapter=adapter,

                dmax=dmax,

                reserve=RESERVE,
            )
        )

        assert not admitted

        assert reason == (
            "outside_dmax"
        )


def test_exceeds_initial_reserve_cases_rejected():

    (
        tools,
        adapter,
        dmax,
    ) = build()

    cases = (
        generate_infeasible_cases(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_cases=20,
        )
    )

    reserve_cases = [
        case
        for case in cases
        if case.mode
        == "exceeds_initial_reserve"
    ]

    assert reserve_cases

    for case in reserve_cases:

        admitted, reason = (
            authorize_case(
                case,

                tools=tools,

                adapter=adapter,

                dmax=dmax,

                reserve=RESERVE,
            )
        )

        assert not admitted

        assert reason == (
            "insufficient_reserve"
        )


def test_exceeds_case_is_guaranteed_above_entire_b0():

    (
        tools,
        adapter,
        _,
    ) = build()

    cases = (
        generate_infeasible_cases(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_cases=20,
        )
    )

    for case in cases:

        if case.mode != (
            "exceeds_initial_reserve"
        ):
            continue

        target_index = (
            case.target_index
        )

        action = (
            case.actions[
                target_index
            ]
        )

        state = (
            case.states[
                target_index
            ]
        )

        effect = (
            adapter
            .contract(
                action,
                state,
            )
            .effect_vector()
        )

        assert any(
            effect[
                dimension
            ]
            > RESERVE[
                dimension
            ]

            for dimension in (
                RESERVE
            )
        )