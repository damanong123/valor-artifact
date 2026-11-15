from workload.cumulative_exhaustion import (
    generate_cumulative_exhaustion_cases,
)

from workload.tool_generator import (
    generate_tools,
)

from experiments.e6_cumulative_exhaustion import (
    evaluate_case,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.budget import (
    vector_leq,
    vector_subtract,
)

from valor.normal_policy import (
    is_normally_authorized,
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

    cases = (
        generate_cumulative_exhaustion_cases(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_cases=20,
        )
    )

    return (
        tools,
        adapter,
        dmax,
        cases,
    )


def test_each_action_individually_fits_b0():

    (
        tools,
        adapter,
        _,
        cases,
    ) = build()

    for case in cases:

        for action in (
            case.actions
        ):

            assert not (
                is_normally_authorized(
                    action,
                    tools,
                )
            )

            effect = (
                adapter
                .contract(
                    action
                )
                .effect_vector()
            )

            assert vector_leq(
                effect,
                RESERVE,
            )


def test_prefix_fits_and_final_action_fails():

    (
        _,
        adapter,
        _,
        cases,
    ) = build()

    for case in cases:

        remaining = dict(
            RESERVE
        )

        # All actions before the expected rejection
        # must fit.
        for action in (
            case.actions[
                :
                case.expected_reject_index
            ]
        ):

            effect = (
                adapter
                .contract(
                    action
                )
                .effect_vector()
            )

            assert vector_leq(
                effect,
                remaining,
            )

            remaining = (
                vector_subtract(
                    remaining,
                    effect,
                )
            )

        final_action = (
            case.actions[
                case.expected_reject_index
            ]
        )

        final_effect = (
            adapter
            .contract(
                final_action
            )
            .effect_vector()
        )

        assert not vector_leq(
            final_effect,
            remaining,
        )


def test_rejection_is_only_cumulative():

    (
        tools,
        adapter,
        dmax,
        cases,
    ) = build()

    for case in cases:

        result = evaluate_case(
            case,

            tools=tools,

            adapter=adapter,

            dmax=dmax,

            reserve=RESERVE,
        )

        assert (
            result[
                "individually_admissible"
            ]
        )

        assert (
            result[
                "all_inside_envelope"
            ]
        )

        assert (
            result[
                "rejection_reason"
            ]
            == (
                "cumulative_reserve_exhausted"
            )
        )

        assert (
            result[
                "observed_reject_index"
            ]
            == (
                case
                .expected_reject_index
            )
        )

        assert result[
            "correct"
        ]


def test_same_seed_reproducible():

    tools = generate_tools(
        seed=20260903
    )

    adapter = (
        SyntheticEffectAdapter(
            tools
        )
    )

    cases_a = (
        generate_cumulative_exhaustion_cases(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_cases=20,
        )
    )

    cases_b = (
        generate_cumulative_exhaustion_cases(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_cases=20,
        )
    )

    assert (
        cases_a
        == cases_b
    )