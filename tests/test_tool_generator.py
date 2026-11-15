from workload.tool_generator import generate_tools


def test_tool_count():

    tools = generate_tools(
        seed=20260903
    )

    assert len(tools) == 30


def test_all_domains_present():

    tools = generate_tools(
        seed=20260903
    )

    domains = {
        tool.domain
        for tool in tools.values()
    }

    assert len(domains) == 8


def test_exceptional_range_larger():

    tools = generate_tools(
        seed=20260903
    )

    for tool in tools.values():

        assert (
            tool.exceptional_max
            >
            tool.normal_max
        )


def test_exactly_one_tool_outside_dmax():

    tools = generate_tools(
        seed=20260903
    )

    excluded = [
        tool
        for tool in tools.values()
        if not tool.in_dmax
    ]

    assert len(excluded) == 1


def test_same_seed_reproducible():

    tools_a = generate_tools(
        seed=20260903
    )

    tools_b = generate_tools(
        seed=20260903
    )

    assert tools_a == tools_b


def test_different_seed_changes_world():

    tools_a = generate_tools(
        seed=20260903
    )

    tools_b = generate_tools(
        seed=20260904
    )

    assert tools_a != tools_b

def test_excluded_domain_not_used_by_allowed_tools():

    tools = generate_tools(
        seed=20260903
    )

    excluded_tools = [
        tool
        for tool in tools.values()
        if not tool.in_dmax
    ]

    assert len(excluded_tools) == 1

    excluded_domain = (
        excluded_tools[0].domain
    )

    allowed_domains = {
        tool.domain
        for tool in tools.values()
        if tool.in_dmax
    }

    assert (
        excluded_domain
        not in allowed_domains
    )