import pytest

from workload.tool_generator import (
    generate_tools,
)

from workload.trace_generator import (
    cumulative_effect,
    generate_feasible_traces,
    trace_has_exceptional_action,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.budget import (
    vector_leq,
)

from valor.normal_policy import (
    is_normally_authorized,
)


INITIAL_RESERVE = {
    "energy": 40.0,
    "stress": 25.0,
    "irreversible": 3.0,
    "external_scope": 18.0,
}


def build_workload(
    *,
    seed=20260903,
    num_traces=100,
):

    tools = generate_tools(
        seed=seed
    )

    adapter = (
        SyntheticEffectAdapter(
            tools
        )
    )

    traces = generate_feasible_traces(
        seed=seed,
        tools=tools,
        adapter=adapter,
        reserve=INITIAL_RESERVE,
        num_traces=num_traces,
        min_actions=2,
        max_actions=5,
    )

    return (
        tools,
        adapter,
        traces,
    )


def test_trace_count():

    _, _, traces = build_workload(
        num_traces=100
    )

    assert len(traces) == 100


def test_trace_length_between_2_and_5():

    _, _, traces = build_workload()

    for trace in traces:

        assert (
            2
            <= len(trace.actions)
            <= 5
        )


def test_actions_are_distinct():

    _, _, traces = build_workload()

    for trace in traces:

        assert (
            len(trace.actions)
            ==
            len(set(trace.actions))
        )


def test_tools_are_distinct_inside_trace():

    _, _, traces = build_workload()

    for trace in traces:

        tool_names = [
            action.tool
            for action in trace.actions
        ]

        assert (
            len(tool_names)
            ==
            len(set(tool_names))
        )


def test_every_trace_has_exceptional_action():

    tools, _, traces = (
        build_workload()
    )

    for trace in traces:

        assert trace_has_exceptional_action(
            trace,
            tools=tools,
        )


def test_every_trace_uses_only_dmax_tools():

    tools, _, traces = (
        build_workload()
    )

    for trace in traces:

        for action in trace.actions:

            assert tools[
                action.tool
            ].in_dmax


def test_every_action_inside_exceptional_contract():

    tools, _, traces = (
        build_workload()
    )

    for trace in traces:

        for action in trace.actions:

            tool = tools[
                action.tool
            ]

            assert (
                tool.exceptional_min
                <= action.parameter
                <= tool.exceptional_max
            )

def test_every_trace_fits_initial_reserve():

    tools, adapter, traces = (
        build_workload()
    )

    from workload.trace_generator import (
        cumulative_exceptional_effect,
    )

    for trace in traces:

        total = (
            cumulative_exceptional_effect(
                trace.actions,
                tools=tools,
                adapter=adapter,
            )
        )

        assert vector_leq(
            total,
            INITIAL_RESERVE,
        )


def test_some_actions_are_normal_and_some_exceptional():

    tools, _, traces = (
        build_workload(
            num_traces=200
        )
    )

    normal_count = 0
    exceptional_count = 0

    for trace in traces:

        for action in trace.actions:

            if is_normally_authorized(
                action,
                tools,
            ):
                normal_count += 1
            else:
                exceptional_count += 1

    assert normal_count > 0
    assert exceptional_count > 0


def test_same_seed_reproducible():

    _, _, traces_a = (
        build_workload(
            seed=20260903,
            num_traces=50,
        )
    )

    _, _, traces_b = (
        build_workload(
            seed=20260903,
            num_traces=50,
        )
    )

    assert traces_a == traces_b


def test_different_seed_changes_workload():

    _, _, traces_a = (
        build_workload(
            seed=20260903,
            num_traces=50,
        )
    )

    _, _, traces_b = (
        build_workload(
            seed=20260904,
            num_traces=50,
        )
    )

    assert traces_a != traces_b