from experiments.e7_state_drift import (
    run_trial,
)

from workload.tool_generator import (
    generate_tools,
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


def run(
    condition,
):

    (
        tools,
        adapter,
        dmax,
    ) = build()

    return run_trial(
        seed=20260903,

        trial_index=0,

        condition=condition,

        tools=tools,

        adapter=adapter,

        dmax=dmax,

        reserve=RESERVE,

        lower_multiplier=0.8,

        higher_multiplier=1.5,
    )


def test_stable_state_permitted():

    result = run(
        "stable"
    )

    assert (
        result[
            "observed_permit"
        ]
    )

    assert (
        result[
            "in_flight"
        ]
    )

    assert result[
        "correct"
    ]


def test_lower_effect_permitted():

    result = run(
        "lower_effect"
    )

    assert (
        result[
            "observed_permit"
        ]
    )

    assert (
        result[
            "execution_energy"
        ]
        <=
        result[
            "reserved_energy"
        ]
    )

    assert result[
        "correct"
    ]


def test_higher_effect_aborted_and_refunded():

    result = run(
        "higher_effect"
    )

    assert not (
        result[
            "observed_permit"
        ]
    )

    assert (
        result[
            "aborted"
        ]
    )

    assert (
        result[
            "execution_energy"
        ]
        >
        result[
            "reserved_energy"
        ]
    )

    assert (
        result[
            "refund_ok"
        ]
    )

    assert (
        result[
            "replay_rejected"
        ]
    )

    assert result[
        "correct"
    ]


def test_hard_guard_failure_aborted():

    result = run(
        "hard_guard_failure"
    )

    assert not (
        result[
            "observed_permit"
        ]
    )

    assert (
        result[
            "aborted"
        ]
    )

    assert not (
        result[
            "hard_guard_ok"
        ]
    )

    assert (
        result[
            "refund_ok"
        ]
    )

    assert (
        result[
            "replay_rejected"
        ]
    )

    assert result[
        "correct"
    ]


def test_conservation_holds_for_all_conditions():

    for condition in [
        "stable",
        "lower_effect",
        "higher_effect",
        "hard_guard_failure",
    ]:

        result = run(
            condition
        )

        assert (
            result[
                "conservation_ok"
            ]
        )