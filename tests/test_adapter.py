import pytest

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
    UnknownToolError,
    ParameterOutOfContractError,
    InvalidParameterError,
)

from valor.models import Action

from valor.normal_policy import (
    is_normally_authorized,
)


def make_adapter():

    tools = generate_tools(
        seed=20260903
    )

    adapter = SyntheticEffectAdapter(
        tools
    )

    return tools, adapter


def test_normal_action_contract():

    tools, adapter = make_adapter()

    tool = tools["tool_00"]

    parameter = (
        tool.normal_max * 0.5
    )

    action = Action(
        tool=tool.name,
        parameter=parameter,
    )

    contract = adapter.contract(
        action
    )

    assert contract.hard_guard_ok

    assert contract.energy >= 0
    assert contract.stress >= 0
    assert contract.irreversible >= 0
    assert contract.external_scope >= 0

    assert tool.domain in contract.domains


def test_exceptional_action_is_not_normal():

    tools, adapter = make_adapter()

    tool = tools["tool_00"]

    parameter = (
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
        parameter=parameter,
    )

    assert not is_normally_authorized(
        action,
        tools,
    )

    contract = adapter.contract(
        action
    )

    assert contract.hard_guard_ok


def test_effect_computation():

    tools, adapter = make_adapter()

    tool = tools["tool_00"]

    parameter = 1.5

    action = Action(
        tool=tool.name,
        parameter=parameter,
    )

    contract = adapter.contract(
        action
    )

    assert contract.energy == pytest.approx(
        tool.energy_coeff
        * parameter
    )

    assert contract.stress == pytest.approx(
        tool.stress_coeff
        * parameter
    )


def test_unknown_tool_rejected():

    _, adapter = make_adapter()

    action = Action(
        tool="does_not_exist",
        parameter=1.0,
    )

    with pytest.raises(
        UnknownToolError
    ):
        adapter.contract(action)


def test_parameter_outside_contract_rejected():

    tools, adapter = make_adapter()

    tool = tools["tool_00"]

    action = Action(
        tool=tool.name,
        parameter=(
            tool.exceptional_max
            + 1.0
        ),
    )

    with pytest.raises(
        ParameterOutOfContractError
    ):
        adapter.contract(action)


def test_nan_parameter_rejected():

    _, adapter = make_adapter()

    action = Action(
        tool="tool_00",
        parameter=float("nan"),
    )

    with pytest.raises(
        InvalidParameterError
    ):
        adapter.contract(action)


def test_hard_guard_failure():

    tools, adapter = make_adapter()

    tool = tools["tool_00"]

    action = Action(
        tool=tool.name,
        parameter=tool.normal_max,
    )

    contract = adapter.contract(
        action,
        state={
            "force_hard_guard_failure": True
        },
    )

    assert not contract.hard_guard_ok