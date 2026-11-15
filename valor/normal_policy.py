from valor.models import (
    Action,
    ToolSpec,
)


def is_normally_authorized(
    action: Action,
    tools: dict[str, ToolSpec],
) -> bool:

    if action.tool not in tools:
        return False

    tool = tools[action.tool]

    if not tool.registered:
        return False

    return (
        tool.normal_min
        <= action.parameter
        <= tool.normal_max
    )