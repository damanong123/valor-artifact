from __future__ import annotations

import sqlite3
from copy import deepcopy
from pathlib import Path
from threading import RLock

from valor.budget import (
    EFFECT_DIMENSIONS,
    validate_effect_vector,
    vector_leq,
    vector_subtract,
    zero_vector,
)

from valor.ledger import (
    DuplicateNonceError,
    UnknownReservationError,
)


class SQLiteSteadyStateReserveLedger:
    """
    Persistent-connection SQLite ledger used for the
    steady-state RQ3 issuance benchmark.

    IMPORTANT
    ---------
    This class is intentionally used for Propose/issuance
    benchmarking.

    The SQLite connection is opened and configured once:

        journal_mode = WAL
        synchronous = FULL

    Per-action timing therefore measures the steady-state
    authorization path rather than connection/PRAGMA setup.

    Runtime transaction:

        BEGIN IMMEDIATE
            read R
            check b <= R
            update R
            insert reservation C
        COMMIT
    """

    def __init__(
        self,
        path: str | Path,
        initial_reserve: dict[str, float],
    ):
        validate_effect_vector(
            initial_reserve
        )

        self.path = str(path)

        self._initial = deepcopy(
            initial_reserve
        )

        self._lock = RLock()

        Path(self.path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Connection is opened ONCE.
        # ----------------------------------------------------

        self._conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )

        self._conn.row_factory = (
            sqlite3.Row
        )

        # ----------------------------------------------------
        # Configure database ONCE.
        # These costs are not part of per-action timing.
        # ----------------------------------------------------

        self._conn.execute(
            "PRAGMA journal_mode=WAL;"
        )

        self._conn.execute(
            "PRAGMA synchronous=FULL;"
        )

        self._conn.execute(
            "PRAGMA foreign_keys=ON;"
        )

        self._conn.execute(
            "PRAGMA busy_timeout=30000;"
        )

        self._initialize_schema()

    # ========================================================
    # Schema
    # ========================================================

    def _initialize_schema(
        self,
    ) -> None:

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),

                initial_energy REAL NOT NULL,
                initial_stress REAL NOT NULL,
                initial_irreversible REAL NOT NULL,
                initial_external_scope REAL NOT NULL,

                remaining_energy REAL NOT NULL,
                remaining_stress REAL NOT NULL,
                remaining_irreversible REAL NOT NULL,
                remaining_external_scope REAL NOT NULL,

                committed_energy REAL NOT NULL,
                committed_stress REAL NOT NULL,
                committed_irreversible REAL NOT NULL,
                committed_external_scope REAL NOT NULL
            );
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reservations (
                nonce TEXT PRIMARY KEY,

                energy REAL NOT NULL,
                stress REAL NOT NULL,
                irreversible REAL NOT NULL,
                external_scope REAL NOT NULL,

                status TEXT NOT NULL
            );
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS resolved_nonces (
                nonce TEXT PRIMARY KEY,
                resolution TEXT NOT NULL
            );
            """
        )

        existing = self._conn.execute(
            """
            SELECT id
            FROM ledger_state
            WHERE id = 1;
            """
        ).fetchone()

        if existing is None:

            r = self._initial

            self._conn.execute(
                """
                INSERT INTO ledger_state (
                    id,

                    initial_energy,
                    initial_stress,
                    initial_irreversible,
                    initial_external_scope,

                    remaining_energy,
                    remaining_stress,
                    remaining_irreversible,
                    remaining_external_scope,

                    committed_energy,
                    committed_stress,
                    committed_irreversible,
                    committed_external_scope
                )
                VALUES (
                    1,

                    ?, ?, ?, ?,

                    ?, ?, ?, ?,

                    0.0, 0.0, 0.0, 0.0
                );
                """,
                (
                    r["energy"],
                    r["stress"],
                    r["irreversible"],
                    r["external_scope"],

                    r["energy"],
                    r["stress"],
                    r["irreversible"],
                    r["external_scope"],
                ),
            )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _remaining_from_row(
        row: sqlite3.Row,
    ) -> dict[str, float]:

        return {
            "energy":
                float(
                    row[
                        "remaining_energy"
                    ]
                ),

            "stress":
                float(
                    row[
                        "remaining_stress"
                    ]
                ),

            "irreversible":
                float(
                    row[
                        "remaining_irreversible"
                    ]
                ),

            "external_scope":
                float(
                    row[
                        "remaining_external_scope"
                    ]
                ),
        }

    @staticmethod
    def _committed_from_row(
        row: sqlite3.Row,
    ) -> dict[str, float]:

        return {
            "energy":
                float(
                    row[
                        "committed_energy"
                    ]
                ),

            "stress":
                float(
                    row[
                        "committed_stress"
                    ]
                ),

            "irreversible":
                float(
                    row[
                        "committed_irreversible"
                    ]
                ),

            "external_scope":
                float(
                    row[
                        "committed_external_scope"
                    ]
                ),
        }

    # ========================================================
    # Reads
    # ========================================================

    def initial_reserve(
        self,
    ) -> dict[str, float]:

        return deepcopy(
            self._initial
        )

    def remaining_reserve(
        self,
    ) -> dict[str, float]:

        with self._lock:

            row = self._conn.execute(
                """
                SELECT *
                FROM ledger_state
                WHERE id = 1;
                """
            ).fetchone()

            return (
                self._remaining_from_row(
                    row
                )
            )

    def committed_spend(
        self,
    ) -> dict[str, float]:

        with self._lock:

            row = self._conn.execute(
                """
                SELECT *
                FROM ledger_state
                WHERE id = 1;
                """
            ).fetchone()

            return (
                self._committed_from_row(
                    row
                )
            )

    def has_reservation(
        self,
        nonce: str,
    ) -> bool:

        with self._lock:

            row = self._conn.execute(
                """
                SELECT nonce
                FROM reservations
                WHERE nonce = ?;
                """,
                (nonce,),
            ).fetchone()

            return (
                row is not None
            )

    # ========================================================
    # PROPOSE — steady-state measured transaction
    # ========================================================

    def try_reserve(
        self,
        *,
        nonce: str,
        bound: dict[str, float],
    ) -> bool:

        validate_effect_vector(
            bound
        )

        with self._lock:

            try:

                self._conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                # ------------------------------------------
                # Nonce uniqueness
                # ------------------------------------------

                duplicate = (
                    self._conn.execute(
                        """
                        SELECT nonce
                        FROM reservations
                        WHERE nonce = ?

                        UNION

                        SELECT nonce
                        FROM resolved_nonces
                        WHERE nonce = ?;
                        """,
                        (
                            nonce,
                            nonce,
                        ),
                    )
                    .fetchone()
                )

                if duplicate is not None:

                    self._conn.execute(
                        "ROLLBACK;"
                    )

                    raise DuplicateNonceError(
                        f"Duplicate nonce: "
                        f"{nonce}"
                    )

                # ------------------------------------------
                # Read current reserve
                # ------------------------------------------

                row = self._conn.execute(
                    """
                    SELECT *
                    FROM ledger_state
                    WHERE id = 1;
                    """
                ).fetchone()

                remaining = (
                    self._remaining_from_row(
                        row
                    )
                )

                # ------------------------------------------
                # b <= R
                # ------------------------------------------

                if not vector_leq(
                    bound,
                    remaining,
                ):

                    self._conn.execute(
                        "ROLLBACK;"
                    )

                    return False

                new_remaining = (
                    vector_subtract(
                        remaining,
                        bound,
                    )
                )

                # ------------------------------------------
                # Debit R
                # ------------------------------------------

                self._conn.execute(
                    """
                    UPDATE ledger_state
                    SET
                        remaining_energy = ?,
                        remaining_stress = ?,
                        remaining_irreversible = ?,
                        remaining_external_scope = ?
                    WHERE id = 1;
                    """,
                    (
                        new_remaining[
                            "energy"
                        ],

                        new_remaining[
                            "stress"
                        ],

                        new_remaining[
                            "irreversible"
                        ],

                        new_remaining[
                            "external_scope"
                        ],
                    ),
                )

                # ------------------------------------------
                # Move reservation into C
                # ------------------------------------------

                self._conn.execute(
                    """
                    INSERT INTO reservations (
                        nonce,
                        energy,
                        stress,
                        irreversible,
                        external_scope,
                        status
                    )
                    VALUES (
                        ?, ?, ?, ?, ?,
                        'outstanding'
                    );
                    """,
                    (
                        nonce,

                        bound[
                            "energy"
                        ],

                        bound[
                            "stress"
                        ],

                        bound[
                            "irreversible"
                        ],

                        bound[
                            "external_scope"
                        ],
                    ),
                )

                # ------------------------------------------
                # FULL synchronous durable commit
                # ------------------------------------------

                self._conn.execute(
                    "COMMIT;"
                )

                return True

            except Exception:

                if self._conn.in_transaction:

                    self._conn.execute(
                        "ROLLBACK;"
                    )

                raise

    # ========================================================
    # Roll back if capability minting fails
    # ========================================================

    def cancel_reservation(
        self,
        nonce: str,
    ) -> None:

        with self._lock:

            try:

                self._conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                row = self._conn.execute(
                    """
                    SELECT *
                    FROM reservations
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                ).fetchone()

                if row is None:

                    raise (
                        UnknownReservationError(
                            nonce
                        )
                    )

                bound = {
                    "energy":
                        float(
                            row["energy"]
                        ),

                    "stress":
                        float(
                            row["stress"]
                        ),

                    "irreversible":
                        float(
                            row[
                                "irreversible"
                            ]
                        ),

                    "external_scope":
                        float(
                            row[
                                "external_scope"
                            ]
                        ),
                }

                state_row = (
                    self._conn.execute(
                        """
                        SELECT *
                        FROM ledger_state
                        WHERE id = 1;
                        """
                    )
                    .fetchone()
                )

                remaining = (
                    self._remaining_from_row(
                        state_row
                    )
                )

                refunded = {
                    d:
                        remaining[d]
                        + bound[d]

                    for d in (
                        EFFECT_DIMENSIONS
                    )
                }

                self._conn.execute(
                    """
                    UPDATE ledger_state
                    SET
                        remaining_energy = ?,
                        remaining_stress = ?,
                        remaining_irreversible = ?,
                        remaining_external_scope = ?
                    WHERE id = 1;
                    """,
                    (
                        refunded[
                            "energy"
                        ],

                        refunded[
                            "stress"
                        ],

                        refunded[
                            "irreversible"
                        ],

                        refunded[
                            "external_scope"
                        ],
                    ),
                )

                self._conn.execute(
                    """
                    DELETE FROM reservations
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                )

                self._conn.execute(
                    "COMMIT;"
                )

            except Exception:

                if self._conn.in_transaction:

                    self._conn.execute(
                        "ROLLBACK;"
                    )

                raise

    # ========================================================
    # Accounting
    # ========================================================

    def outstanding_total(
        self,
    ) -> dict[str, float]:

        with self._lock:

            rows = (
                self._conn.execute(
                    """
                    SELECT
                        energy,
                        stress,
                        irreversible,
                        external_scope
                    FROM reservations;
                    """
                )
                .fetchall()
            )

            total = zero_vector()

            for row in rows:

                total[
                    "energy"
                ] += float(
                    row["energy"]
                )

                total[
                    "stress"
                ] += float(
                    row["stress"]
                )

                total[
                    "irreversible"
                ] += float(
                    row["irreversible"]
                )

                total[
                    "external_scope"
                ] += float(
                    row[
                        "external_scope"
                    ]
                )

            return total

    def conservation_total(
        self,
    ) -> dict[str, float]:

        remaining = (
            self.remaining_reserve()
        )

        outstanding = (
            self.outstanding_total()
        )

        committed = (
            self.committed_spend()
        )

        return {
            d:
                remaining[d]
                + outstanding[d]
                + committed[d]

            for d in (
                EFFECT_DIMENSIONS
            )
        }

    def conservation_holds(
        self,
        epsilon: float = 1e-9,
    ) -> bool:

        total = (
            self.conservation_total()
        )

        initial = (
            self.initial_reserve()
        )

        return all(
            abs(
                total[d]
                - initial[d]
            )
            <= epsilon

            for d in (
                EFFECT_DIMENSIONS
            )
        )

    # ========================================================
    # Benchmark reset
    # ========================================================

    def reset_for_benchmark(
        self,
    ) -> None:
        """
        Reset between independent timing trials.

        IMPORTANT:
        This is NOT a VALOR runtime authority transition.

        It is called outside the measured timing interval.
        """

        with self._lock:

            try:

                self._conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                self._conn.execute(
                    """
                    DELETE FROM reservations;
                    """
                )

                self._conn.execute(
                    """
                    DELETE FROM resolved_nonces;
                    """
                )

                r = self._initial

                self._conn.execute(
                    """
                    UPDATE ledger_state
                    SET
                        remaining_energy = ?,
                        remaining_stress = ?,
                        remaining_irreversible = ?,
                        remaining_external_scope = ?,

                        committed_energy = 0.0,
                        committed_stress = 0.0,
                        committed_irreversible = 0.0,
                        committed_external_scope = 0.0

                    WHERE id = 1;
                    """,
                    (
                        r["energy"],
                        r["stress"],
                        r["irreversible"],
                        r[
                            "external_scope"
                        ],
                    ),
                )

                self._conn.execute(
                    "COMMIT;"
                )

            except Exception:

                if self._conn.in_transaction:

                    self._conn.execute(
                        "ROLLBACK;"
                    )

                raise

    # ========================================================
    # Metadata
    # ========================================================

    def sqlite_settings(
        self,
    ) -> dict[str, object]:

        with self._lock:

            journal_mode = (
                self._conn.execute(
                    "PRAGMA journal_mode;"
                )
                .fetchone()[0]
            )

            synchronous = (
                self._conn.execute(
                    "PRAGMA synchronous;"
                )
                .fetchone()[0]
            )

            return {
                "journal_mode":
                    journal_mode,

                # SQLite integer:
                # 0 OFF
                # 1 NORMAL
                # 2 FULL
                # 3 EXTRA
                "synchronous_numeric":
                    synchronous,
            }

    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ) -> None:

        with self._lock:

            self._conn.close()