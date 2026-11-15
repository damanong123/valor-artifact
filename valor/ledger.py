from __future__ import annotations

from copy import deepcopy
from threading import RLock

from valor.budget import (
    EFFECT_DIMENSIONS,
    validate_effect_vector,
    vector_leq,
    vector_subtract,
    zero_vector,
)


# ============================================================
# Ledger errors
# ============================================================

class DuplicateNonceError(RuntimeError):
    pass


class UnknownReservationError(RuntimeError):
    pass


class AlreadyResolvedNonceError(RuntimeError):
    pass


class CapabilityNotInFlightError(RuntimeError):
    pass


class EffectReceiptExceedsReservationError(RuntimeError):
    pass


# ============================================================
# In-memory VALOR ledger
# ============================================================

class InMemoryReserveLedger:
    """
    In-memory realization of VALOR state:

        (R, C, S, U)

    R:
        Remaining uncommitted exceptional reserve.

    C:
        Reserved-but-not-yet-settled effect.

    S:
        Executed and committed exceptional effect.

    U:
        Resolved nonces.

    in_flight:
        Capabilities that successfully passed Execute
        and are waiting for Commit.

    Core accounting invariant:

        R + C + S = B0
    """

    def __init__(
        self,
        initial_reserve: dict[str, float],
    ):
        validate_effect_vector(
            initial_reserve
        )

        self._initial = deepcopy(
            initial_reserve
        )

        self._remaining = deepcopy(
            initial_reserve
        )

        self._committed = (
            zero_vector()
        )

        self._outstanding: dict[
            str,
            dict[str, float],
        ] = {}

        self._resolved_nonces: set[
            str
        ] = set()

        self._in_flight: set[
            str
        ] = set()

        self._lock = RLock()

    # ========================================================
    # Read operations
    # ========================================================

    def initial_reserve(
        self,
    ) -> dict[str, float]:

        with self._lock:
            return deepcopy(
                self._initial
            )

    def remaining_reserve(
        self,
    ) -> dict[str, float]:

        with self._lock:
            return deepcopy(
                self._remaining
            )

    def committed_spend(
        self,
    ) -> dict[str, float]:

        with self._lock:
            return deepcopy(
                self._committed
            )

    def outstanding_reservations(
        self,
    ) -> dict[
        str,
        dict[str, float],
    ]:

        with self._lock:
            return deepcopy(
                self._outstanding
            )

    def resolved_nonces(
        self,
    ) -> set[str]:

        with self._lock:
            return set(
                self._resolved_nonces
            )

    def in_flight_nonces(
        self,
    ) -> set[str]:

        with self._lock:
            return set(
                self._in_flight
            )

    def is_resolved(
        self,
        nonce: str,
    ) -> bool:

        with self._lock:
            return (
                nonce
                in self._resolved_nonces
            )

    def is_in_flight(
        self,
        nonce: str,
    ) -> bool:

        with self._lock:
            return (
                nonce
                in self._in_flight
            )

    def has_reservation(
        self,
        nonce: str,
    ) -> bool:

        with self._lock:
            return (
                nonce
                in self._outstanding
            )

    def reservation_for(
        self,
        nonce: str,
    ) -> dict[str, float]:

        with self._lock:

            if nonce not in (
                self._outstanding
            ):
                raise UnknownReservationError(
                    nonce
                )

            return deepcopy(
                self._outstanding[
                    nonce
                ]
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
        """
        Atomic reservation:

            if b <= R:

                R <- R - b
                C <- C union {nonce: b}
        """

        validate_effect_vector(
            bound
        )

        with self._lock:

            if (
                nonce
                in self._outstanding
                or nonce
                in self._resolved_nonces
            ):
                raise DuplicateNonceError(
                    f"Duplicate nonce: "
                    f"{nonce}"
                )

            if not vector_leq(
                bound,
                self._remaining,
            ):
                return False

            self._remaining = (
                vector_subtract(
                    self._remaining,
                    bound,
                )
            )

            self._outstanding[
                nonce
            ] = deepcopy(
                bound
            )

            return True

    # ========================================================
    # Local rollback before capability release
    # ========================================================

    def cancel_reservation(
        self,
        nonce: str,
    ) -> None:

        with self._lock:

            if nonce in (
                self._resolved_nonces
            ):
                raise AlreadyResolvedNonceError(
                    nonce
                )

            if nonce not in (
                self._outstanding
            ):
                raise UnknownReservationError(
                    nonce
                )

            bound = (
                self._outstanding
                .pop(nonce)
            )

            for dimension in (
                EFFECT_DIMENSIONS
            ):
                self._remaining[
                    dimension
                ] += bound[
                    dimension
                ]

    # ========================================================
    # EXECUTE SUCCESS
    # ========================================================

    def try_resolve_for_execution(
        self,
        nonce: str,
    ) -> bool:
        """
        Successful execution-time revalidation.

        Transition:

            nonce -> U
            nonce -> in_flight

        Reservation remains in C until Commit.
        """

        with self._lock:

            if nonce in (
                self._resolved_nonces
            ):
                return False

            if nonce not in (
                self._outstanding
            ):
                raise UnknownReservationError(
                    nonce
                )

            self._resolved_nonces.add(
                nonce
            )

            self._in_flight.add(
                nonce
            )

            return True

    # ========================================================
    # EXECUTE ABORT
    # ========================================================

    def resolve_abort_and_refund(
        self,
        nonce: str,
    ) -> bool:
        """
        Revalidation failed before effect boundary.

        Transition:

            nonce -> U
            C <- C - b
            R <- R + b
        """

        with self._lock:

            if nonce in (
                self._resolved_nonces
            ):
                return False

            if nonce not in (
                self._outstanding
            ):
                raise UnknownReservationError(
                    nonce
                )

            bound = (
                self._outstanding
                .pop(nonce)
            )

            for dimension in (
                EFFECT_DIMENSIONS
            ):
                self._remaining[
                    dimension
                ] += bound[
                    dimension
                ]

            self._resolved_nonces.add(
                nonce
            )

            self._in_flight.discard(
                nonce
            )

            return True

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
        """
        Settle one successful execution.

        If a trusted effect receipt gives:

            c <= b

        then:

            C <- C - b
            S <- S + c
            R <- R + (b - c)

        If realized_effect is None:

            c := b

        This models an indeterminate execution outcome:
        no refund is permitted.
        """

        with self._lock:

            # Execute must already have
            # resolved this nonce.
            if nonce not in (
                self._resolved_nonces
            ):
                raise CapabilityNotInFlightError(
                    "Nonce has not passed Execute."
                )

            if nonce not in (
                self._in_flight
            ):
                raise CapabilityNotInFlightError(
                    "Capability is not in-flight."
                )

            if nonce not in (
                self._outstanding
            ):
                raise UnknownReservationError(
                    nonce
                )

            reserved = deepcopy(
                self._outstanding[
                    nonce
                ]
            )

            # ----------------------------------------------
            # Indeterminate result:
            # conservatively charge full reservation.
            # ----------------------------------------------

            if realized_effect is None:

                committed = deepcopy(
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
                            "Realized effect exceeds "
                            "reserved effect."
                        )
                    )

                committed = deepcopy(
                    realized_effect
                )

            # ----------------------------------------------
            # Refund = reserved - committed
            # ----------------------------------------------

            refund = vector_subtract(
                reserved,
                committed,
            )

            # ----------------------------------------------
            # Move committed amount into S
            # ----------------------------------------------

            for dimension in (
                EFFECT_DIMENSIONS
            ):

                self._committed[
                    dimension
                ] += committed[
                    dimension
                ]

            # ----------------------------------------------
            # Refund only proven unused capacity
            # ----------------------------------------------

            for dimension in (
                EFFECT_DIMENSIONS
            ):

                self._remaining[
                    dimension
                ] += refund[
                    dimension
                ]

            # ----------------------------------------------
            # Remove capability from C
            # ----------------------------------------------

            del self._outstanding[
                nonce
            ]

            self._in_flight.remove(
                nonce
            )

            return {
                "reserved": reserved,
                "committed": committed,
                "refunded": refund,
            }

    # ========================================================
    # Accounting
    # ========================================================

    def outstanding_total(
        self,
    ) -> dict[str, float]:

        with self._lock:

            total = zero_vector()

            for reservation in (
                self._outstanding
                .values()
            ):

                for dimension in (
                    EFFECT_DIMENSIONS
                ):

                    total[
                        dimension
                    ] += reservation[
                        dimension
                    ]

            return total

    def conservation_total(
        self,
    ) -> dict[str, float]:
        """
        Compute:

            R + C + S
        """

        with self._lock:

            outstanding = (
                self.outstanding_total()
            )

            return {
                dimension:
                    self._remaining[
                        dimension
                    ]
                    + outstanding[
                        dimension
                    ]
                    + self._committed[
                        dimension
                    ]

                for dimension in (
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
                total[
                    dimension
                ]
                - initial[
                    dimension
                ]
            )
            <= epsilon

            for dimension in (
                EFFECT_DIMENSIONS
            )
        )