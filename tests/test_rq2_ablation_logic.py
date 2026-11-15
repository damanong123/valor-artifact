from workload.tool_generator import (
    generate_tools,
)

from workload.trace_generator import (
    generate_feasible_traces,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from policies.baselines import (
    UnionBreakGlassPolicy,
    CatalogReservePolicy,
    EnvelopeOnlyPolicy,
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

    adapter = (
        SyntheticEffectAdapter(
            tools
        )
    )

    traces = (
        generate_feasible_traces(
            seed=20260903,

            tools=tools,

            adapter=adapter,

            reserve=RESERVE,

            num_traces=200,
        )
    )

    dmax = {
        tool.domain
        for tool in tools.values()
        if tool.in_dmax
    }

    known = traces[:50]

    unseen = traces[50:]

    return (
        tools,
        adapter,
        dmax,
        known,
        unseen,
    )


def test_full_valor_accepts_feasible_unseen_workload():

    (
        tools,
        adapter,
        dmax,
        _,
        unseen,
    ) = build()

    policy = (
        ValorTracePolicy(
            tools=tools,

            adapter=adapter,

            dmax=dmax,

            reserve=RESERVE,
        )
    )

    assert all(
        policy.authorize_trace(
            trace
        )
        for trace in unseen
    )


def test_envelope_only_accepts_feasible_unseen_workload():

    (
        tools,
        adapter,
        dmax,
        _,
        unseen,
    ) = build()

    policy = (
        EnvelopeOnlyPolicy(
            tools=tools,

            adapter=adapter,

            dmax=dmax,
        )
    )

    assert all(
        policy.authorize_trace(
            trace
        )
        for trace in unseen
    )


def test_catalog_reserve_cannot_exceed_union_coverage():

    (
        tools,
        adapter,
        _,
        known,
        unseen,
    ) = build()

    union = (
        UnionBreakGlassPolicy(
            tools=tools,
            known_profiles=known,
        )
    )

    catalog_reserve = (
        CatalogReservePolicy(
            tools=tools,

            known_profiles=known,

            adapter=adapter,

            reserve=RESERVE,
        )
    )

    union_count = sum(
        union.authorize_trace(
            trace
        )
        for trace in unseen
    )

    catalog_reserve_count = sum(
        catalog_reserve
        .authorize_trace(
            trace
        )
        for trace in unseen
    )

    assert (
        catalog_reserve_count
        <= union_count
    )