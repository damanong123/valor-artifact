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
    AlreadyResolvedNonceError,
    CapabilityNotInFlightError,
    EffectReceiptExceedsReservationError,
)


class SQLiteReserveLedger:
    """
    Durable SQLite implementation of VALOR state:

        R = remaining reserve
        C = outstanding reservations
        S = committed effect
        U = resolved nonces

    SQLite configuration:

        journal_mode = WAL
        synchronous  = FULL

    Reserve-changing operations use:

        BEGIN IMMEDIATE

    so concurrent writers serialize before modifying
    the reserve row.
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

        self._initialize_database()

    # ========================================================
    # Connection
    # ========================================================

    def _connect(self) -> sqlite3.Connection:

        conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )

        conn.row_factory = (
            sqlite3.Row
        )

        conn.execute(
            "PRAGMA journal_mode=WAL;"
        )

        conn.execute(
            "PRAGMA synchronous=FULL;"
        )

        conn.execute(
            "PRAGMA foreign_keys=ON;"
        )

        conn.execute(
            "PRAGMA busy_timeout=30000;"
        )

        return conn

    # ========================================================
    # Schema
    # ========================================================

    def _initialize_database(
        self,
    ) -> None:

        Path(self.path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as conn:

            conn.execute(
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

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    nonce TEXT PRIMARY KEY,

                    energy REAL NOT NULL,
                    stress REAL NOT NULL,
                    irreversible REAL NOT NULL,
                    external_scope REAL NOT NULL,

                    status TEXT NOT NULL
                    CHECK (
                        status IN (
                            'outstanding',
                            'in_flight'
                        )
                    )
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resolved_nonces (
                    nonce TEXT PRIMARY KEY,

                    resolution TEXT NOT NULL
                    CHECK (
                        resolution IN (
                            'in_flight',
                            'aborted',
                            'committed'
                        )
                    )
                );
                """
            )

            existing = conn.execute(
                """
                SELECT id
                FROM ledger_state
                WHERE id = 1;
                """
            ).fetchone()

            if existing is None:

                r = self._initial

                conn.execute(
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
    # Vector helpers
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

    @staticmethod
    def _reservation_from_row(
        row: sqlite3.Row,
    ) -> dict[str, float]:

        return {
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
                    row["irreversible"]
                ),

            "external_scope":
                float(
                    row["external_scope"]
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

        with self._connect() as conn:

            row = conn.execute(
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

        with self._connect() as conn:

            row = conn.execute(
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

    def resolved_nonces(
        self,
    ) -> set[str]:

        with self._connect() as conn:

            rows = conn.execute(
                """
                SELECT nonce
                FROM resolved_nonces;
                """
            ).fetchall()

        return {
            str(row["nonce"])
            for row in rows
        }

    def in_flight_nonces(
        self,
    ) -> set[str]:

        with self._connect() as conn:

            rows = conn.execute(
                """
                SELECT nonce
                FROM reservations
                WHERE status = 'in_flight';
                """
            ).fetchall()

        return {
            str(row["nonce"])
            for row in rows
        }

    def is_resolved(
        self,
        nonce: str,
    ) -> bool:

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT nonce
                FROM resolved_nonces
                WHERE nonce = ?;
                """,
                (nonce,),
            ).fetchone()

        return row is not None

    def is_in_flight(
        self,
        nonce: str,
    ) -> bool:

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT nonce
                FROM reservations
                WHERE nonce = ?
                  AND status = 'in_flight';
                """,
                (nonce,),
            ).fetchone()

        return row is not None

    def has_reservation(
        self,
        nonce: str,
    ) -> bool:

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT nonce
                FROM reservations
                WHERE nonce = ?;
                """,
                (nonce,),
            ).fetchone()

        return row is not None

    def reservation_for(
        self,
        nonce: str,
    ) -> dict[str, float]:

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT *
                FROM reservations
                WHERE nonce = ?;
                """,
                (nonce,),
            ).fetchone()

        if row is None:
            raise UnknownReservationError(
                nonce
            )

        return (
            self._reservation_from_row(
                row
            )
        )

    # ========================================================
    # PROPOSE
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

            conn = self._connect()

            try:

                conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                duplicate = conn.execute(
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
                ).fetchone()

                if duplicate is not None:

                    conn.execute(
                        "ROLLBACK;"
                    )

                    raise DuplicateNonceError(
                        f"Duplicate nonce: "
                        f"{nonce}"
                    )

                row = conn.execute(
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

                if not vector_leq(
                    bound,
                    remaining,
                ):

                    conn.execute(
                        "ROLLBACK;"
                    )

                    return False

                new_remaining = (
                    vector_subtract(
                        remaining,
                        bound,
                    )
                )

                conn.execute(
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

                conn.execute(
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

                conn.execute(
                    "COMMIT;"
                )

                return True

            except Exception:

                if conn.in_transaction:
                    conn.execute(
                        "ROLLBACK;"
                    )

                raise

            finally:

                conn.close()

    # ========================================================
    # Local rollback before capability release
    # ========================================================

    def cancel_reservation(
        self,
        nonce: str,
    ) -> None:

        with self._lock:

            conn = self._connect()

            try:

                conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                resolved = conn.execute(
                    """
                    SELECT nonce
                    FROM resolved_nonces
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                ).fetchone()

                if resolved is not None:
                    raise (
                        AlreadyResolvedNonceError(
                            nonce
                        )
                    )

                row = conn.execute(
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

                bound = (
                    self._reservation_from_row(
                        row
                    )
                )

                current = (
                    self._remaining_from_row(
                        conn.execute(
                            """
                            SELECT *
                            FROM ledger_state
                            WHERE id = 1;
                            """
                        ).fetchone()
                    )
                )

                refunded = {
                    d:
                        current[d]
                        + bound[d]

                    for d in (
                        EFFECT_DIMENSIONS
                    )
                }

                conn.execute(
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
                        refunded["energy"],
                        refunded["stress"],
                        refunded[
                            "irreversible"
                        ],
                        refunded[
                            "external_scope"
                        ],
                    ),
                )

                conn.execute(
                    """
                    DELETE FROM reservations
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                )

                conn.execute(
                    "COMMIT;"
                )

            except Exception:

                if conn.in_transaction:
                    conn.execute(
                        "ROLLBACK;"
                    )

                raise

            finally:

                conn.close()

    # ========================================================
    # EXECUTE SUCCESS
    # ========================================================

    def try_resolve_for_execution(
        self,
        nonce: str,
    ) -> bool:

        with self._lock:

            conn = self._connect()

            try:

                conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                resolved = conn.execute(
                    """
                    SELECT nonce
                    FROM resolved_nonces
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                ).fetchone()

                if resolved is not None:

                    conn.execute(
                        "ROLLBACK;"
                    )

                    return False

                reservation = (
                    conn.execute(
                        """
                        SELECT nonce
                        FROM reservations
                        WHERE nonce = ?;
                        """,
                        (nonce,),
                    ).fetchone()
                )

                if reservation is None:
                    raise (
                        UnknownReservationError(
                            nonce
                        )
                    )

                conn.execute(
                    """
                    UPDATE reservations
                    SET status = 'in_flight'
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                )

                conn.execute(
                    """
                    INSERT INTO resolved_nonces (
                        nonce,
                        resolution
                    )
                    VALUES (
                        ?,
                        'in_flight'
                    );
                    """,
                    (nonce,),
                )

                conn.execute(
                    "COMMIT;"
                )

                return True

            except Exception:

                if conn.in_transaction:
                    conn.execute(
                        "ROLLBACK;"
                    )

                raise

            finally:

                conn.close()

    # ========================================================
    # EXECUTE ABORT
    # ========================================================

    def resolve_abort_and_refund(
        self,
        nonce: str,
    ) -> bool:

        with self._lock:

            conn = self._connect()

            try:

                conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                resolved = conn.execute(
                    """
                    SELECT nonce
                    FROM resolved_nonces
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                ).fetchone()

                if resolved is not None:

                    conn.execute(
                        "ROLLBACK;"
                    )

                    return False

                row = conn.execute(
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

                bound = (
                    self._reservation_from_row(
                        row
                    )
                )

                state_row = (
                    conn.execute(
                        """
                        SELECT *
                        FROM ledger_state
                        WHERE id = 1;
                        """
                    ).fetchone()
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

                conn.execute(
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
                        refunded["energy"],
                        refunded["stress"],
                        refunded[
                            "irreversible"
                        ],
                        refunded[
                            "external_scope"
                        ],
                    ),
                )

                conn.execute(
                    """
                    DELETE FROM reservations
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                )

                conn.execute(
                    """
                    INSERT INTO resolved_nonces (
                        nonce,
                        resolution
                    )
                    VALUES (
                        ?,
                        'aborted'
                    );
                    """,
                    (nonce,),
                )

                conn.execute(
                    "COMMIT;"
                )

                return True

            except Exception:

                if conn.in_transaction:
                    conn.execute(
                        "ROLLBACK;"
                    )

                raise

            finally:

                conn.close()

    # ========================================================
    # COMMIT
    # ========================================================

    def settle_commit(
        self,
        *,
        nonce: str,
        realized_effect:
            dict[str, float]
            | None,
    ) -> dict[str, dict[str, float]]:

        with self._lock:

            conn = self._connect()

            try:

                conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                resolved = conn.execute(
                    """
                    SELECT resolution
                    FROM resolved_nonces
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                ).fetchone()

                if (
                    resolved is None
                    or resolved[
                        "resolution"
                    ] != "in_flight"
                ):
                    raise (
                        CapabilityNotInFlightError(
                            "Capability is not "
                            "in-flight."
                        )
                    )

                row = conn.execute(
                    """
                    SELECT *
                    FROM reservations
                    WHERE nonce = ?
                      AND status = 'in_flight';
                    """,
                    (nonce,),
                ).fetchone()

                if row is None:
                    raise (
                        CapabilityNotInFlightError(
                            "In-flight reservation "
                            "not found."
                        )
                    )

                reserved = (
                    self._reservation_from_row(
                        row
                    )
                )

                if realized_effect is None:

                    committed = dict(
                        reserved
                    )

                else:

                    validate_effect_vector(
                        realized_effect
                    )

                    if not vector_leq(
                        realized_effect,
                        reserved,
                    ):
                        raise (
                            EffectReceiptExceedsReservationError(
                                "Realized effect "
                                "exceeds reservation."
                            )
                        )

                    committed = dict(
                        realized_effect
                    )

                refund = (
                    vector_subtract(
                        reserved,
                        committed,
                    )
                )

                state_row = (
                    conn.execute(
                        """
                        SELECT *
                        FROM ledger_state
                        WHERE id = 1;
                        """
                    ).fetchone()
                )

                remaining = (
                    self._remaining_from_row(
                        state_row
                    )
                )

                committed_total = (
                    self._committed_from_row(
                        state_row
                    )
                )

                new_remaining = {
                    d:
                        remaining[d]
                        + refund[d]

                    for d in (
                        EFFECT_DIMENSIONS
                    )
                }

                new_committed = {
                    d:
                        committed_total[d]
                        + committed[d]

                    for d in (
                        EFFECT_DIMENSIONS
                    )
                }

                conn.execute(
                    """
                    UPDATE ledger_state
                    SET
                        remaining_energy = ?,
                        remaining_stress = ?,
                        remaining_irreversible = ?,
                        remaining_external_scope = ?,

                        committed_energy = ?,
                        committed_stress = ?,
                        committed_irreversible = ?,
                        committed_external_scope = ?

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

                        new_committed[
                            "energy"
                        ],

                        new_committed[
                            "stress"
                        ],

                        new_committed[
                            "irreversible"
                        ],

                        new_committed[
                            "external_scope"
                        ],
                    ),
                )

                conn.execute(
                    """
                    DELETE FROM reservations
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                )

                conn.execute(
                    """
                    UPDATE resolved_nonces
                    SET resolution = 'committed'
                    WHERE nonce = ?;
                    """,
                    (nonce,),
                )

                conn.execute(
                    "COMMIT;"
                )

                return {
                    "reserved":
                        reserved,

                    "committed":
                        committed,

                    "refunded":
                        refund,
                }

            except Exception:

                if conn.in_transaction:
                    conn.execute(
                        "ROLLBACK;"
                    )

                raise

            finally:

                conn.close()

    # ========================================================
    # Accounting
    # ========================================================

    def outstanding_total(
        self,
    ) -> dict[str, float]:

        with self._connect() as conn:

            rows = conn.execute(
                """
                SELECT
                    energy,
                    stress,
                    irreversible,
                    external_scope
                FROM reservations;
                """
            ).fetchall()

        total = zero_vector()

        for row in rows:

            reservation = (
                self._reservation_from_row(
                    row
                )
            )

            for d in (
                EFFECT_DIMENSIONS
            ):
                total[d] += (
                    reservation[d]
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
    # Benchmark-only reset
    # ========================================================

    def reset_for_benchmark(
        self,
    ) -> None:
        """
        Reset database to a fresh epoch-equivalent state.

        This helper is ONLY for independent benchmark trials.
        It is not a VALOR runtime transition.
        """

        with self._lock:

            conn = self._connect()

            try:

                conn.execute(
                    "BEGIN IMMEDIATE;"
                )

                conn.execute(
                    "DELETE FROM reservations;"
                )

                conn.execute(
                    "DELETE FROM resolved_nonces;"
                )

                r = self._initial

                conn.execute(
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
                        r["external_scope"],
                    ),
                )

                conn.execute(
                    "COMMIT;"
                )

            except Exception:

                if conn.in_transaction:
                    conn.execute(
                        "ROLLBACK;"
                    )

                raise

            finally:

                conn.close()