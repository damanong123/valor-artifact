from workload.attack_recovery import (
    apply_greedy_attack,
    authorize_recovery_trace,
)

from workload.tool_generator import (
    generate_tools,
)

from workload.trace_generator import (
    generate_feasible_traces,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.budget import (
    vector_leq,
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

    traces = (
        generate_feasible_traces(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_traces=100,

            min_actions=2,

            max_actions=5,
        )
    )

    return (
        tools,
        adapter,
        dmax,
        traces,
    )


def recovery_rate(
    *,
    steps: int,
):

    (
        tools,
        adapter,
        dmax,
        traces,
    ) = build()

    attack = (
        apply_greedy_attack(
            requested_steps=steps,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,
        )
    )

    admitted = 0

    for trace in traces:

        ok, _ = (
            authorize_recovery_trace(
                trace,

                tools=tools,

                adapter=adapter,

                dmax=dmax,

                remaining_reserve=(
                    attack[
                        "remaining_reserve"
                    ]
                ),
            )
        )

        admitted += int(
            ok
        )

    return (
        admitted
        / len(traces),
        attack,
    )


def test_no_attack_preserves_full_recovery():

    rate, attack = (
        recovery_rate(
            steps=0
        )
    )

    assert rate == 1.0

    assert (
        attack[
            "accepted_steps"
        ]
        == 0
    )

    assert (
        attack[
            "attack_exposure_x"
        ]
        == 0.0
    )

    assert (
        attack[
            "remaining_reserve"
        ]
        == RESERVE
    )


def test_attack_never_exceeds_b0():

    (
        tools,
        adapter,
        _,
        _,
    ) = build()

    for steps in range(
        0,
        10,
    ):

        attack = (
            apply_greedy_attack(
                requested_steps=steps,

                tools=tools,

                adapter=adapter,

                reserve=RESERVE,
            )
        )

        assert (
            attack[
                "attack_exposure_x"
            ]
            <= 1.0
            + 1e-9
        )

        assert vector_leq(
            attack[
                "attack_effect"
            ],
            RESERVE,
        )

        assert all(
            value >= -1e-9

            for value in (
                attack[
                    "remaining_reserve"
                ].values()
            )
        )


def test_remaining_reserve_is_monotone_nonincreasing():

    (
        tools,
        adapter,
        _,
        _,
    ) = build()

    previous = dict(
        RESERVE
    )

    for steps in range(
        1,
        8,
    ):

        attack = (
            apply_greedy_attack(
                requested_steps=steps,

                tools=tools,

                adapter=adapter,

                reserve=RESERVE,
            )
        )

        current = (
            attack[
                "remaining_reserve"
            ]
        )

        assert all(
            current[
                dimension
            ]
            <=
            previous[
                dimension
            ]
            + 1e-9

            for dimension in (
                RESERVE
            )
        )

        previous = current


def test_recovery_rate_is_monotone_nonincreasing():

    rates = []

    for steps in range(
        0,
        7,
    ):

        rate, _ = (
            recovery_rate(
                steps=steps
            )
        )

        rates.append(
            rate
        )

    assert all(
        rates[index + 1]
        <= rates[index]
        + 1e-9

        for index in range(
            len(rates) - 1
        )
    )