from __future__ import annotations

import json
import os
import platform
import sqlite3
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from workload.tool_generator import (
    generate_tools,
)

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.capability import (
    verify_capability,
)

from valor.kernel import (
    ValorKernel,
)

from valor.ledger import (
    InMemoryReserveLedger,
)

from valor.models import (
    Action,
)

from valor.sqlite_ledger import (
    SQLiteReserveLedger,
)

from valor.sqlite_steady_ledger import (
    SQLiteSteadyStateReserveLedger,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# Statistics
# ============================================================

def percentile(
    values: list[float],
    q: float,
) -> float:

    ordered = sorted(
        values
    )

    position = (
        q
        * (
            len(ordered)
            - 1
        )
    )

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = (
        position
        - lower
    )

    return (
        ordered[lower]
        * (1.0 - weight)
        + ordered[upper]
        * weight
    )


def summarize(
    values_us: list[float],
) -> dict[str, float]:

    return {
        "n":
            len(values_us),

        "median_us":
            statistics.median(
                values_us
            ),

        "p95_us":
            percentile(
                values_us,
                0.95,
            ),

        "p99_us":
            percentile(
                values_us,
                0.99,
            ),

        "mean_us":
            statistics.mean(
                values_us
            ),

        "min_us":
            min(
                values_us
            ),

        "max_us":
            max(
                values_us
            ),
    }


# ============================================================
# DB cleanup
# ============================================================

def remove_sqlite_files(
    path: Path,
) -> None:

    candidates = [
        path,
        Path(
            str(path)
            + "-wal"
        ),
        Path(
            str(path)
            + "-shm"
        ),
    ]

    for candidate in candidates:

        if candidate.exists():

            candidate.unlink()


# ============================================================
# World
# ============================================================

def build_world(
    seed: int,
):

    tools = generate_tools(
        seed=seed
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

    tool = next(
        tool
        for tool in tools.values()
        if tool.in_dmax
    )

    parameter = (
        tool.normal_max
        +
        (
            tool.exceptional_max
            - tool.normal_max
        )
        * 0.25
    )

    action = Action(
        tool=tool.name,
        parameter=parameter,
    )

    return (
        adapter,
        dmax,
        action,
    )


# ============================================================
# In-memory issuance
# ============================================================

def benchmark_in_memory_issuance(
    *,
    iterations: int,
    adapter,
    dmax,
    reserve,
    action,
    key,
) -> list[float]:

    # ----------------------------------------------
    # Warmup
    # ----------------------------------------------

    for _ in range(100):

        ledger = (
            InMemoryReserveLedger(
                reserve
            )
        )

        kernel = ValorKernel(
            adapter=adapter,
            ledger=ledger,
            dmax=dmax,
            capability_key=key,
        )

        kernel.propose(
            action
        )

    samples = []

    for _ in range(
        iterations
    ):

        # Fresh trial setup occurs outside timing.
        ledger = (
            InMemoryReserveLedger(
                reserve
            )
        )

        kernel = ValorKernel(
            adapter=adapter,
            ledger=ledger,
            dmax=dmax,
            capability_key=key,
        )

        start = (
            time.perf_counter_ns()
        )

        kernel.propose(
            action
        )

        end = (
            time.perf_counter_ns()
        )

        samples.append(
            (
                end - start
            )
            / 1_000.0
        )

    return samples


# ============================================================
# Capability verification
# ============================================================

def benchmark_capability_verification(
    *,
    iterations: int,
    adapter,
    dmax,
    reserve,
    action,
    key,
) -> list[float]:

    ledger = (
        InMemoryReserveLedger(
            reserve
        )
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=key,
    )

    capability = (
        kernel.propose(
            action
        )
    )

    for _ in range(1000):

        assert verify_capability(
            key=key,
            capability=capability,
        )

    samples = []

    for _ in range(
        iterations
    ):

        start = (
            time.perf_counter_ns()
        )

        ok = verify_capability(
            key=key,
            capability=capability,
        )

        end = (
            time.perf_counter_ns()
        )

        if not ok:

            raise RuntimeError(
                "Capability verification failed."
            )

        samples.append(
            (
                end - start
            )
            / 1_000.0
        )

    return samples


# ============================================================
# SQLite cold-connect durable issuance
# ============================================================

def benchmark_sqlite_cold_issuance(
    *,
    iterations: int,
    adapter,
    dmax,
    reserve,
    action,
    key,
    db_path: Path,
) -> list[float]:

    remove_sqlite_files(
        db_path
    )

    ledger = SQLiteReserveLedger(
        db_path,
        reserve,
    )

    kernel = ValorKernel(
        adapter=adapter,
        ledger=ledger,
        dmax=dmax,
        capability_key=key,
    )

    for _ in range(10):

        ledger.reset_for_benchmark()

        kernel.propose(
            action
        )

    samples = []

    for _ in range(
        iterations
    ):

        # Reset outside measured region.
        ledger.reset_for_benchmark()

        start = (
            time.perf_counter_ns()
        )

        kernel.propose(
            action
        )

        end = (
            time.perf_counter_ns()
        )

        samples.append(
            (
                end - start
            )
            / 1_000.0
        )

    return samples


# ============================================================
# SQLite STEADY-STATE durable issuance
# ============================================================

def benchmark_sqlite_steady_issuance(
    *,
    iterations: int,
    adapter,
    dmax,
    reserve,
    action,
    key,
    db_path: Path,
) -> tuple[
    list[float],
    dict[str, object],
]:

    remove_sqlite_files(
        db_path
    )

    ledger = (
        SQLiteSteadyStateReserveLedger(
            db_path,
            reserve,
        )
    )

    try:

        kernel = ValorKernel(
            adapter=adapter,
            ledger=ledger,
            dmax=dmax,
            capability_key=key,
        )

        settings = (
            ledger.sqlite_settings()
        )

        # ----------------------------------------------
        # Warmup
        # ----------------------------------------------

        for _ in range(10):

            ledger.reset_for_benchmark()

            kernel.propose(
                action
            )

        samples = []

        for _ in range(
            iterations
        ):

            # IMPORTANT:
            # benchmark reset is outside measured region.
            ledger.reset_for_benchmark()

            start = (
                time.perf_counter_ns()
            )

            kernel.propose(
                action
            )

            end = (
                time.perf_counter_ns()
            )

            samples.append(
                (
                    end - start
                )
                / 1_000.0
            )

        return (
            samples,
            settings,
        )

    finally:

        ledger.close()


# ============================================================
# Ed25519
# ============================================================

def benchmark_ed25519_verification(
    iterations: int,
) -> list[float]:

    private_key = (
        Ed25519PrivateKey
        .generate()
    )

    public_key = (
        private_key
        .public_key()
    )

    constitution = (
        b"VALOR synthetic constitution "
        b"epoch=1 "
        b"Dmax=synthetic "
        b"B0=(40,25,3,18)"
    )

    signature = (
        private_key.sign(
            constitution
        )
    )

    for _ in range(100):

        public_key.verify(
            signature,
            constitution,
        )

    samples = []

    for _ in range(
        iterations
    ):

        start = (
            time.perf_counter_ns()
        )

        public_key.verify(
            signature,
            constitution,
        )

        end = (
            time.perf_counter_ns()
        )

        samples.append(
            (
                end - start
            )
            / 1_000.0
        )

    return samples


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    with open(
        ROOT
        / "config"
        / "experiment.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(
            f
        )

    seed = (
        cfg[
            "reproducibility"
        ]["seeds"][0]
    )

    reserve = (
        cfg[
            "constitution"
        ]["initial_reserve"]
    )

    fast_iterations = (
        cfg[
            "benchmark"
        ]["in_memory_iterations"]
    )

    durable_iterations = (
        cfg[
            "benchmark"
        ]["durable_successful_issuances"]
    )

    (
        adapter,
        dmax,
        action,
    ) = build_world(
        seed
    )

    key = (
        b"K" * 32
    )

    cold_db_path = (
        ROOT
        / "data"
        / "rq3_cold.sqlite"
    )

    steady_db_path = (
        ROOT
        / "data"
        / "rq3_steady.sqlite"
    )

    # --------------------------------------------------------
    # Benchmarks
    # --------------------------------------------------------

    print(
        "=" * 78
    )

    print(
        "RQ3 — Local Enforcement Overhead"
    )

    print(
        "=" * 78
    )

    print(
        "[1/5] Capability verification..."
    )

    capability_verification = (
        benchmark_capability_verification(
            iterations=fast_iterations,
            adapter=adapter,
            dmax=dmax,
            reserve=reserve,
            action=action,
            key=key,
        )
    )

    print(
        "[2/5] In-memory issuance..."
    )

    in_memory = (
        benchmark_in_memory_issuance(
            iterations=fast_iterations,
            adapter=adapter,
            dmax=dmax,
            reserve=reserve,
            action=action,
            key=key,
        )
    )

    print(
        "[3/5] Ed25519 verification..."
    )

    ed25519 = (
        benchmark_ed25519_verification(
            fast_iterations
        )
    )

    print(
        "[4/5] SQLite steady-state issuance..."
    )

    (
        sqlite_steady,
        steady_settings,
    ) = (
        benchmark_sqlite_steady_issuance(
            iterations=durable_iterations,
            adapter=adapter,
            dmax=dmax,
            reserve=reserve,
            action=action,
            key=key,
            db_path=steady_db_path,
        )
    )

    print(
        "[5/5] SQLite cold-connect issuance..."
    )

    sqlite_cold = (
        benchmark_sqlite_cold_issuance(
            iterations=durable_iterations,
            adapter=adapter,
            dmax=dmax,
            reserve=reserve,
            action=action,
            key=key,
            db_path=cold_db_path,
        )
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    datasets = {
        "Capability verification":
            capability_verification,

        "In-memory issuance":
            in_memory,

        "Ed25519 verification":
            ed25519,

        "SQLite steady-state issuance":
            sqlite_steady,

        "SQLite cold-connect issuance":
            sqlite_cold,
    }

    # --------------------------------------------------------
    # Raw
    # --------------------------------------------------------

    raw_rows = []

    for operation, values in (
        datasets.items()
    ):

        for index, value in enumerate(
            values
        ):

            raw_rows.append(
                {
                    "operation":
                        operation,

                    "sample":
                        index,

                    "latency_us":
                        value,
                }
            )

    raw_df = pd.DataFrame(
        raw_rows
    )

    raw_path = (
        ROOT
        / "results"
        / "raw"
        / "rq3_latency_raw.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []

    for operation, values in (
        datasets.items()
    ):

        row = summarize(
            values
        )

        row[
            "operation"
        ] = operation

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )[
        [
            "operation",
            "n",
            "median_us",
            "p95_us",
            "p99_us",
            "mean_us",
            "min_us",
            "max_us",
        ]
    ]

    summary_path = (
        ROOT
        / "results"
        / "tables"
        / "rq3_latency_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    environment = {
        "platform":
            platform.platform(),

        "system":
            platform.system(),

        "release":
            platform.release(),

        "machine":
            platform.machine(),

        "processor":
            platform.processor(),

        "python":
            platform.python_version(),

        "sqlite":
            sqlite3.sqlite_version,

        "cpu_count":
            os.cpu_count(),

        "timer":
            "time.perf_counter_ns",

        "steady_state_sqlite":
            steady_settings,

        "steady_state_definition":
            (
                "SQLite connection and PRAGMA "
                "configuration are completed "
                "before the measured per-action "
                "issuance interval."
            ),

        "cold_connect_definition":
            (
                "Measured issuance uses the "
                "general SQLite ledger whose "
                "per-operation path opens and "
                "configures a connection."
            ),

        "benchmark_reset_note":
            (
                "Synthetic ledger reset occurs "
                "between independent successful "
                "trials and outside the measured "
                "issuance interval. It is not a "
                "runtime VALOR authority reset."
            ),
    }

    environment_path = (
        ROOT
        / "results"
        / "raw"
        / "rq3_environment.json"
    )

    with open(
        environment_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            environment,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    figure_path = (
        ROOT
        / "results"
        / "figures"
        / "rq3_latency.png"
    )

    labels = (
        summary_df[
            "operation"
        ]
        .tolist()
    )

    medians = (
        summary_df[
            "median_us"
        ]
        .tolist()
    )

    p95s = (
        summary_df[
            "p95_us"
        ]
        .tolist()
    )

    upper_errors = [
        max(
            0.0,
            p95 - median,
        )
        for (
            median,
            p95,
        ) in zip(
            medians,
            p95s,
        )
    ]

    plt.figure(
        figsize=(10, 5.5)
    )

    plt.bar(
        labels,
        medians,
        yerr=[
            [0.0] * len(
                labels
            ),
            upper_errors,
        ],
        capsize=4,
    )

    plt.ylabel(
        "Latency (µs)"
    )

    plt.yscale(
        "log"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------

    print()

    print(
        "=" * 78
    )

    print(
        "RQ3 SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        "SQLite steady-state settings:"
    )

    print(
        steady_settings
    )

    print()

    print(
        f"[OK] raw -> "
        f"{raw_path}"
    )

    print(
        f"[OK] summary -> "
        f"{summary_path}"
    )

    print(
        f"[OK] environment -> "
        f"{environment_path}"
    )

    print(
        f"[OK] figure -> "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()