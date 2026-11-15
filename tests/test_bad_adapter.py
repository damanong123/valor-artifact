import pytest

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.bad_adapter import (
    UnderestimatingEffectAdapter,
)

from valor.models import Action


def test_bad_adapter_underestimates_energy():

    tools = generate_tools(
        seed=20260903
    )

    true_adapter = (
        SyntheticEffectAdapter(
            tools
        )
    )

    bad_adapter = (
        UnderestimatingEffectAdapter(
            tools,
            dimension="energy",
            underestimate_fraction=0.30,
        )
    )

    tool = next(
        tool
        for tool in tools.values()
        if tool.in_dmax
    )

    action = Action(
        tool=tool.name,
        parameter=tool.exceptional_max,
    )

    true_effect = (
        true_adapter
        .contract(action)
        .effect_vector()
    )

    bad_effect = (
        bad_adapter
        .contract(action)
        .effect_vector()
    )

    assert (
        bad_effect["energy"]
        ==
        pytest.approx(
            true_effect["energy"]
            * 0.70
        )
    )

    assert (
        bad_effect["stress"]
        ==
        pytest.approx(
            true_effect["stress"]
        )
    )


def test_zero_underestimate_matches_true_adapter():

    tools = generate_tools(
        seed=20260903
    )

    true_adapter = (
        SyntheticEffectAdapter(
            tools
        )
    )

    bad_adapter = (
        UnderestimatingEffectAdapter(
            tools,
            dimension="energy",
            underestimate_fraction=0.0,
        )
    )

    tool = next(
        tool
        for tool in tools.values()
        if tool.in_dmax
    )

    action = Action(
        tool=tool.name,
        parameter=tool.exceptional_max,
    )

    assert (
        bad_adapter
        .contract(action)
        .effect_vector()
        ==
        pytest.approx(
            true_adapter
            .contract(action)
            .effect_vector()
        )
    )