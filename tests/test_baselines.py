from workload.tool_generator import generate_tools
from workload.trace_generator import (
    generate_feasible_traces,
)

from valor.adapter import SyntheticEffectAdapter

from policies.baselines import (
    ExactCatalogPolicy,
    UnionBreakGlassPolicy,
    ValorTracePolicy,
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

    adapter = SyntheticEffectAdapter(
        tools
    )

    traces = generate_feasible_traces(
        seed=20260903,
        tools=tools,
        adapter=adapter,
        reserve=RESERVE,
        num_traces=100,
    )

    dmax = {
        tool.domain
        for tool in tools.values()
        if tool.in_dmax
    }

    return (
        tools,
        adapter,
        traces,
        dmax,
    )


def test_exact_catalog_accepts_known_profiles():

    (
        tools,
        _,
        traces,
        _,
    ) = build()

    policy = ExactCatalogPolicy(
        tools=tools,
        known_profiles=traces[:10],
    )

    for trace in traces[:10]:

        assert policy.authorize_trace(
            trace
        )


def test_union_accepts_known_profiles():

    (
        tools,
        _,
        traces,
        _,
    ) = build()

    policy = UnionBreakGlassPolicy(
        tools=tools,
        known_profiles=traces[:10],
    )

    for trace in traces[:10]:

        assert policy.authorize_trace(
            trace
        )


def test_valor_accepts_all_generated_feasible_traces():

    (
        tools,
        adapter,
        traces,
        dmax,
    ) = build()

    policy = ValorTracePolicy(
        tools=tools,
        adapter=adapter,
        dmax=dmax,
        reserve=RESERVE,
    )

    assert all(
        policy.authorize_trace(
            trace
        )
        for trace in traces
    )


def test_union_scope_monotonic_with_k():

    (
        tools,
        _,
        traces,
        _,
    ) = build()

    policy_10 = UnionBreakGlassPolicy(
        tools=tools,
        known_profiles=traces[:10],
    )

    policy_25 = UnionBreakGlassPolicy(
        tools=tools,
        known_profiles=traces[:25],
    )

    for tool_name, limit in (
        policy_10.allowed_max.items()
    ):

        assert tool_name in (
            policy_25.allowed_max
        )

        assert (
            policy_25.allowed_max[
                tool_name
            ]
            >= limit
        )