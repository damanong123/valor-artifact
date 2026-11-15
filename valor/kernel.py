from __future__ import annotations

import secrets
import time
from typing import Callable

from valor.adapter import (
    AdapterError,
    SyntheticEffectAdapter,
)

from valor.budget import (
    validate_effect_vector,
    vector_leq,
)

from valor.capability import (
    mint_capability,
    verify_capability,
)

from valor.ledger import (
    InMemoryReserveLedger,
)

from valor.models import (
    Action,
    Capability,
)


# ============================================================
# PROPOSE ERRORS
# ============================================================

class ValorAuthorizationError(RuntimeError):
    """
    Base exception for exceptional authorization failures.
    """


class DomainNotAllowedError(ValorAuthorizationError):
    """
    Action touches a protected effect domain outside Dmax.
    """


class HardGuardFailedError(ValorAuthorizationError):
    """
    Hard state predicate failed.
    """


class InsufficientReserveError(ValorAuthorizationError):
    """
    Remaining exceptional reserve is insufficient.
    """


# ============================================================
# EXECUTE ERRORS
# ============================================================

class ValorExecutionError(RuntimeError):
    """
    Base exception for execution-time failures.
    """


class CapabilityVerificationError(ValorExecutionError):
    """
    Capability authentication or reservation binding failed.
    """


class ActionMismatchError(ValorExecutionError):
    """
    Capability is being used for a different exact action.
    """


class EpochMismatchError(ValorExecutionError):
    """
    Capability belongs to a different constitution epoch.
    """


class CapabilityExpiredError(ValorExecutionError):
    """
    Capability has expired.
    """


class ReplayDetectedError(ValorExecutionError):
    """
    Capability nonce has already been resolved.
    """


class ReservationNotFoundError(ValorExecutionError):
    """
    Capability does not have a matching outstanding reservation.
    """


class ExecutionRevalidationError(ValorExecutionError):
    """
    Fresh execution-time effect evaluation failed.
    """


# ============================================================
# COMMIT ERRORS
# ============================================================

class ValorCommitError(RuntimeError):
    """
    Base exception for settlement failures.
    """


class CapabilityNotInFlightCommitError(ValorCommitError):
    """
    Commit was attempted before successful Execute,
    or after the capability had already been settled.
    """


# ============================================================
# VALOR KERNEL
# ============================================================

class ValorKernel:
    """
    Minimal VALOR authorization kernel.

    Implements the paper's core lifecycle:

        Propose
            |
            v
        chi(a, xi)
            |
            v
        check D subset Dmax
        check q(a, xi)
        check b(a, xi) <= R
            |
            v
        reserve b
            |
            v
        mint exact capability
            |
            v
        Execute
            |
            v
        verify capability
        verify exact action
        verify epoch
        verify expiry
        verify nonce
            |
            v
        chi(a, xe)
            |
            v
        fresh revalidation
            |
        +---+---+
        |       |
       PASS    FAIL
        |       |
        v       v
    nonce->U   nonce->U
    in-flight  C -> R
        |
        v
      Commit
        |
        v
    C -> S
    refund b-c

    Core accounting invariant:

        R + C + S = B0
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        *,
        adapter: SyntheticEffectAdapter,
        ledger: InMemoryReserveLedger,
        dmax: set[str],
        capability_key: bytes,
        epoch: int = 1,
        capability_ttl_seconds: float = 5.0,
        nonce_factory: Callable[[], str] | None = None,
    ):
        """
        Parameters
        ----------
        adapter:
            Trusted effect adapter.

        ledger:
            Exceptional-authority reserve ledger.

        dmax:
            Human-precommitted exceptional effect-domain envelope.

        capability_key:
            Secret key used to authenticate exact capabilities.

        epoch:
            Current constitution epoch.

        capability_ttl_seconds:
            Lifetime of newly issued capabilities.

        nonce_factory:
            Optional deterministic nonce generator for tests.
        """

        if epoch < 0:
            raise ValueError(
                "epoch cannot be negative"
            )

        if capability_ttl_seconds <= 0:
            raise ValueError(
                "capability_ttl_seconds must be positive"
            )

        if not capability_key:
            raise ValueError(
                "capability_key cannot be empty"
            )

        self.adapter = adapter
        self.ledger = ledger

        self.dmax = set(
            dmax
        )

        self.capability_key = (
            capability_key
        )

        self.epoch = epoch

        self.capability_ttl_ns = int(
            capability_ttl_seconds
            * 1_000_000_000
        )

        if nonce_factory is None:
            self.nonce_factory = (
                self._default_nonce
            )
        else:
            self.nonce_factory = (
                nonce_factory
            )

    # ========================================================
    # NONCE
    # ========================================================

    @staticmethod
    def _default_nonce() -> str:
        """
        Generate a fresh operation nonce.
        """
        return secrets.token_hex(
            16
        )

    # ========================================================
    # PROPOSE
    # ========================================================

    def propose(
        self,
        action: Action,
        *,
        state: dict | None = None,
        now_ns: int | None = None,
    ) -> Capability:
        """
        Exceptional issuance transition.

        This method assumes that ordinary authorization has
        already failed and the request has entered VALOR's
        exceptional path.

        Successful transition:

            R <- R - b

            C <- C union {nonce: b}

        and only then is the exact capability released.
        """

        # ----------------------------------------------------
        # 1. Query trusted adapter at issuance-time state xi
        # ----------------------------------------------------

        contract = self.adapter.contract(
            action,
            state,
        )

        # ----------------------------------------------------
        # 2. Check exceptional effect envelope
        #
        #    D(a) subset Dmax
        # ----------------------------------------------------

        if not contract.domains.issubset(
            self.dmax
        ):
            raise DomainNotAllowedError(
                "Action touches a protected "
                "domain outside Dmax: "
                f"{sorted(contract.domains)}"
            )

        # ----------------------------------------------------
        # 3. Check hard predicate q(a, xi)
        # ----------------------------------------------------

        if not contract.hard_guard_ok:
            raise HardGuardFailedError(
                "Hard guard failed during Propose."
            )

        # ----------------------------------------------------
        # 4. Obtain conservative effect bound b(a, xi)
        # ----------------------------------------------------

        bound = (
            contract.effect_vector()
        )

        validate_effect_vector(
            bound
        )

        # ----------------------------------------------------
        # 5. Generate fresh nonce
        # ----------------------------------------------------

        nonce = (
            self.nonce_factory()
        )

        # ----------------------------------------------------
        # 6. Reserve authority BEFORE capability release
        #
        #    require:
        #
        #        b <= R
        #
        #    then:
        #
        #        R <- R - b
        #        C <- C + b
        # ----------------------------------------------------

        reservation_success = (
            self.ledger.try_reserve(
                nonce=nonce,
                bound=bound,
            )
        )

        if not reservation_success:
            raise InsufficientReserveError(
                "Insufficient exceptional reserve."
            )

        # ----------------------------------------------------
        # 7. Calculate expiry
        # ----------------------------------------------------

        if now_ns is None:
            now_ns = (
                time.monotonic_ns()
            )

        expiry_ns = (
            now_ns
            + self.capability_ttl_ns
        )

        # ----------------------------------------------------
        # 8. Mint exact one-shot capability
        #
        #    Cap(a, e, b, nonce, expiry)
        # ----------------------------------------------------

        try:

            capability = (
                mint_capability(
                    key=(
                        self.capability_key
                    ),

                    action=action,

                    epoch=self.epoch,

                    reserved=bound,

                    nonce=nonce,

                    expiry_ns=(
                        expiry_ns
                    ),
                )
            )

        except Exception:

            # Capability was never released.
            # Restore the reservation.
            self.ledger.cancel_reservation(
                nonce
            )

            raise

        return capability

    # ========================================================
    # EXECUTE
    # ========================================================

    def execute(
        self,
        capability: Capability,
        action: Action,
        *,
        state: dict | None = None,
        now_ns: int | None = None,
    ) -> bool:
        """
        Mandatory execution-time revalidation.

        Returns True only when the consequential tool may
        cross its external-effect boundary.

        Successful Execute:

            nonce -> U

            capability -> in-flight

        Reservation stays inside C until Commit.

        Failed execution-time revalidation:

            nonce -> U

            C -> R
        """

        # ----------------------------------------------------
        # 1. Verify capability authenticator
        # ----------------------------------------------------

        if not verify_capability(
            key=self.capability_key,
            capability=capability,
        ):
            raise CapabilityVerificationError(
                "Capability MAC verification failed."
            )

        # ----------------------------------------------------
        # 2. Exact action binding
        #
        # A capability for one exact tool/parameter pair
        # cannot authorize another request.
        # ----------------------------------------------------

        if capability.action != action:
            raise ActionMismatchError(
                "Capability does not match "
                "the requested exact action."
            )

        # ----------------------------------------------------
        # 3. Epoch
        # ----------------------------------------------------

        if capability.epoch != self.epoch:
            raise EpochMismatchError(
                "Capability belongs to "
                "a different epoch."
            )

        # ----------------------------------------------------
        # 4. Expiry
        # ----------------------------------------------------

        if now_ns is None:
            now_ns = (
                time.monotonic_ns()
            )

        if now_ns > capability.expiry_ns:
            raise CapabilityExpiredError(
                "Capability has expired."
            )

        nonce = (
            capability.nonce
        )

        # ----------------------------------------------------
        # 5. Replay check
        #
        # Well-formedness failures above intentionally
        # do NOT consume the nonce.
        # ----------------------------------------------------

        if self.ledger.is_resolved(
            nonce
        ):
            raise ReplayDetectedError(
                f"Nonce already resolved: "
                f"{nonce}"
            )

        # ----------------------------------------------------
        # 6. Reservation must exist
        # ----------------------------------------------------

        if not self.ledger.has_reservation(
            nonce
        ):
            raise ReservationNotFoundError(
                "Capability has no "
                "outstanding reservation."
            )

        ledger_reserved = (
            self.ledger.reservation_for(
                nonce
            )
        )

        capability_reserved = (
            capability
            .reserved_effect_vector()
        )

        # ----------------------------------------------------
        # 7. Ledger reservation must exactly match the
        #    authenticated capability reservation
        # ----------------------------------------------------

        same_reservation = (
            vector_leq(
                ledger_reserved,
                capability_reserved,
            )
            and
            vector_leq(
                capability_reserved,
                ledger_reserved,
            )
        )

        if not same_reservation:
            raise CapabilityVerificationError(
                "Capability reservation "
                "does not match ledger reservation."
            )

        # ----------------------------------------------------
        # 8. Fresh adapter query at execution-time state xe
        #
        # IMPORTANT:
        # Never reuse chi(a, xi).
        # ----------------------------------------------------

        try:

            execution_contract = (
                self.adapter.contract(
                    action,
                    state,
                )
            )

        except AdapterError as exc:

            # The adapter can no longer certify the action.
            #
            # No effect boundary was crossed.
            #
            # Therefore:
            #
            #   nonce -> U
            #   C -> R

            resolved = (
                self.ledger
                .resolve_abort_and_refund(
                    nonce
                )
            )

            if not resolved:
                raise ReplayDetectedError(
                    f"Nonce already resolved: "
                    f"{nonce}"
                )

            raise ExecutionRevalidationError(
                "Execution-time adapter "
                f"evaluation failed: {exc}"
            ) from exc

        execution_bound = (
            execution_contract
            .effect_vector()
        )

        validate_effect_vector(
            execution_bound
        )

        # ----------------------------------------------------
        # 9. Mandatory execution-time revalidation
        #
        # Require:
        #
        #   D(a) subset Dmax
        #
        #   q(a, xe) = True
        #
        #   b(a, xe) <= b_reserved
        # ----------------------------------------------------

        failure_reason: str | None = (
            None
        )

        if not (
            execution_contract
            .domains
            .issubset(self.dmax)
        ):

            failure_reason = (
                "Execution-time domain "
                "is outside Dmax."
            )

        elif not (
            execution_contract
            .hard_guard_ok
        ):

            failure_reason = (
                "Execution-time hard "
                "guard failed."
            )

        elif not vector_leq(
            execution_bound,
            capability_reserved,
        ):

            failure_reason = (
                "Execution-time effect bound "
                "exceeds the reserved effect."
            )

        # ----------------------------------------------------
        # 10. Revalidation failure
        #
        # No external effect occurred.
        #
        # Therefore:
        #
        #   nonce -> U
        #
        #   C <- C - b
        #
        #   R <- R + b
        # ----------------------------------------------------

        if failure_reason is not None:

            resolved = (
                self.ledger
                .resolve_abort_and_refund(
                    nonce
                )
            )

            if not resolved:
                raise ReplayDetectedError(
                    f"Nonce already resolved: "
                    f"{nonce}"
                )

            raise ExecutionRevalidationError(
                failure_reason
            )

        # ----------------------------------------------------
        # 11. Successful Execute
        #
        # Resolve nonce BEFORE the tool is permitted
        # to cross its effect boundary.
        #
        # Reservation remains in C, marked in-flight,
        # until Commit.
        # ----------------------------------------------------

        resolved = (
            self.ledger
            .try_resolve_for_execution(
                nonce
            )
        )

        if not resolved:
            raise ReplayDetectedError(
                f"Nonce already resolved: "
                f"{nonce}"
            )

        # Tool may now cross its effect boundary.
        return True

    # ========================================================
    # COMMIT
    # ========================================================

    def commit(
        self,
        capability: Capability,
        *,
        realized_effect:
            dict[str, float]
            | None,
    ) -> dict[str, dict[str, float]]:
        """
        Settle one successfully executed capability.

        Case 1
        ------
        A trusted effect receipt provides:

            c <= b

        Then:

            C <- C - b

            S <- S + c

            R <- R + (b - c)

        Case 2
        ------
        realized_effect is None.

        The execution outcome is indeterminate.

        Conservatively use:

            c := b

        Therefore no refund occurs.
        """

        # ----------------------------------------------------
        # 1. Authenticate capability
        # ----------------------------------------------------

        if not verify_capability(
            key=self.capability_key,
            capability=capability,
        ):
            raise CapabilityVerificationError(
                "Capability MAC verification failed."
            )

        # ----------------------------------------------------
        # 2. Epoch binding
        # ----------------------------------------------------

        if capability.epoch != self.epoch:
            raise EpochMismatchError(
                "Capability belongs to "
                "a different epoch."
            )

        nonce = (
            capability.nonce
        )

        # ----------------------------------------------------
        # 3. Capability must already have passed Execute
        # ----------------------------------------------------

        if not self.ledger.is_resolved(
            nonce
        ):
            raise (
                CapabilityNotInFlightCommitError(
                    "Capability has not "
                    "passed Execute."
                )
            )

        if not self.ledger.is_in_flight(
            nonce
        ):
            raise (
                CapabilityNotInFlightCommitError(
                    "Capability is not in-flight."
                )
            )

        # ----------------------------------------------------
        # 4. Outstanding reservation must still exist
        # ----------------------------------------------------

        if not self.ledger.has_reservation(
            nonce
        ):
            raise ReservationNotFoundError(
                "Outstanding reservation "
                "not found."
            )

        ledger_reserved = (
            self.ledger.reservation_for(
                nonce
            )
        )

        capability_reserved = (
            capability
            .reserved_effect_vector()
        )

        # ----------------------------------------------------
        # 5. Reservation binding
        # ----------------------------------------------------

        same_reservation = (
            vector_leq(
                ledger_reserved,
                capability_reserved,
            )
            and
            vector_leq(
                capability_reserved,
                ledger_reserved,
            )
        )

        if not same_reservation:
            raise CapabilityVerificationError(
                "Capability reservation "
                "does not match ledger reservation."
            )

        # ----------------------------------------------------
        # 6. Validate receipt if known
        # ----------------------------------------------------

        if realized_effect is not None:
            validate_effect_vector(
                realized_effect
            )

        # ----------------------------------------------------
        # 7. Settle:
        #
        #    C -> S
        #
        #    refund proven unused amount
        # ----------------------------------------------------

        result = (
            self.ledger
            .settle_commit(
                nonce=nonce,
                realized_effect=(
                    realized_effect
                ),
            )
        )

        return result