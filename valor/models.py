from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


# ============================================================
# Tool specification
# ============================================================

@dataclass(frozen=True)
class ToolSpec:
    """
    Definition of one registered synthetic consequential tool.

    Each tool has:
    - a protected effect domain
    - a normal operating range
    - a larger exceptional / effect-contract range
    - coefficients for four typed effects
    - registration metadata
    - membership in the constitution's Dmax envelope
    """

    name: str
    domain: str

    # Normal authorization range
    normal_min: float
    normal_max: float

    # Certified exceptional / adapter-contract range
    exceptional_min: float
    exceptional_max: float

    # Synthetic effect coefficients
    energy_coeff: float
    stress_coeff: float
    irreversible_coeff: float
    external_scope_coeff: float

    # Tool metadata
    registered: bool = True
    in_dmax: bool = True


# ============================================================
# Runtime action
# ============================================================

@dataclass(frozen=True)
class Action:
    """
    One exact runtime action proposed by the planner.

    In the current synthetic reproduction, an action is:

        a = (tool, parameter)

    Later, if needed, this can be extended to multiple parameters.
    """

    tool: str
    parameter: float


# ============================================================
# Effect contract
# ============================================================

@dataclass(frozen=True)
class EffectContract:
    """
    Synthetic representation of the paper's effect contract:

        χ(a, x) = <D(a), b(a, x), q(a, x), rho(a)>

    where:

    D(a)
        Protected effect domains touched by the action.

    b(a, x)
        Conservative typed effect bound.

    q(a, x)
        Hard state predicate.

    rho(a)
        Receipt / commitment class.
    """

    # D(a)
    domains: FrozenSet[str]

    # b(a, x)
    energy: float
    stress: float
    irreversible: float
    external_scope: float

    # q(a, x)
    hard_guard_ok: bool

    # rho(a)
    receipt_class: str = "synthetic"

    def effect_vector(self) -> dict[str, float]:
        """
        Return the typed effect vector in dictionary form.

        This representation will later be used by:
        - the reserve ledger
        - VALOR kernel admission checks
        - experiment accounting
        """
        return {
            "energy": self.energy,
            "stress": self.stress,
            "irreversible": self.irreversible,
            "external_scope": self.external_scope,
        }


# ============================================================
# Future-use reserve representation
# ============================================================

@dataclass(frozen=True)
class EffectBudget:
    """
    Typed exceptional-authority budget.

    This is included now because the next deployment step will
    implement:

        b(a, x) <= R_t

    component-wise.

    It is not yet required by the adapter itself.
    """

    energy: float
    stress: float
    irreversible: float
    external_scope: float

    def as_dict(self) -> dict[str, float]:
        return {
            "energy": self.energy,
            "stress": self.stress,
            "irreversible": self.irreversible,
            "external_scope": self.external_scope,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, float],
    ) -> "EffectBudget":
        return cls(
            energy=float(data.get("energy", 0.0)),
            stress=float(data.get("stress", 0.0)),
            irreversible=float(
                data.get("irreversible", 0.0)
            ),
            external_scope=float(
                data.get("external_scope", 0.0)
            ),
        )


# ============================================================
# Future-use capability representation
# ============================================================

@dataclass(frozen=True)
class Capability:
    """
    Exact one-shot exceptional capability.

    The paper conceptually binds a capability to:

        Cap(a, e, b, nonce, expiry)

    We define the data structure now, although the capability
    creation and verification logic will be implemented later.
    """

    action: Action

    epoch: int

    reserved_energy: float
    reserved_stress: float
    reserved_irreversible: float
    reserved_external_scope: float

    nonce: str
    expiry_ns: int

    # Authentication tag / MAC.
    # This will be populated when capability signing is added.
    mac_hex: str = ""

    def reserved_effect_vector(
        self,
    ) -> dict[str, float]:
        return {
            "energy": self.reserved_energy,
            "stress": self.reserved_stress,
            "irreversible": (
                self.reserved_irreversible
            ),
            "external_scope": (
                self.reserved_external_scope
            ),
        }
# ============================================================
# Synthetic emergency trace
# ============================================================

@dataclass(frozen=True)
class Trace:
    """
    One synthetic authorization-constrained emergency trace.

    A trace contains 2-5 exact runtime action instances.

    In the paper-aligned feasible workload:
    - at least one action is outside normal authorization;
    - all actions use registered tools inside Dmax;
    - cumulative typed effect fits B0.
    """

    trace_id: str

    actions: tuple[
        Action,
        ...
    ]