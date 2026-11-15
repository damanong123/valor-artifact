from __future__ import annotations

from valor.models import (
    Trace,
)


def select_known_profiles(
    traces: list[Trace],
    k: int,
) -> list[Trace]:
    """
    Select the first K traces as known contingency profiles.

    This mirrors the paper-style experiment where
    catalog size K is varied:

        K = 10, 25, 50, 100
    """

    if k < 0:
        raise ValueError(
            "k cannot be negative"
        )

    if k > len(traces):
        raise ValueError(
            "k cannot exceed number of traces"
        )

    return list(
        traces[:k]
    )


def select_unseen_traces(
    traces: list[Trace],
    k: int,
) -> list[Trace]:
    """
    Traces not included among the first K known profiles.
    """

    if k < 0:
        raise ValueError(
            "k cannot be negative"
        )

    if k > len(traces):
        raise ValueError(
            "k cannot exceed number of traces"
        )

    return list(
        traces[k:]
    )