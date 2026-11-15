from __future__ import annotations

from valor.adapter import (
    SyntheticEffectAdapter,
)

from valor.models import (
    Action,
    EffectContract,
)


class UnderestimatingEffectAdapter(
    SyntheticEffectAdapter
):
    """
    Synthetic adapter that deliberately underestimates
    one selected effect dimension.

    Used ONLY for RQ4 failure-boundary evaluation.

    Example:
        underestimate_fraction = 0.30

    means:

        reported = 0.70 * ground_truth

    for the selected dimension.
    """

    def __init__(
        self,
        tools,
        *,
        dimension: str = "energy",
        underestimate_fraction: float = 0.30,
    ):
        super().__init__(
            tools
        )

        if dimension not in {
            "energy",
            "stress",
            "irreversible",
            "external_scope",
        }:
            raise ValueError(
                f"Unknown dimension: {dimension}"
            )

        if not (
            0.0
            <= underestimate_fraction
            < 1.0
        ):
            raise ValueError(
                "underestimate_fraction must "
                "be in [0, 1)"
            )

        self.dimension = dimension

        self.underestimate_fraction = (
            underestimate_fraction
        )

    def contract(
        self,
        action: Action,
        state: dict | None = None,
    ) -> EffectContract:
        """
        Return an intentionally incorrect declared contract.
        """

        true_contract = super().contract(
            action,
            state,
        )

        effects = (
            true_contract
            .effect_vector()
        )

        effects[
            self.dimension
        ] *= (
            1.0
            - self.underestimate_fraction
        )

        return EffectContract(
            domains=(
                true_contract.domains
            ),

            energy=effects[
                "energy"
            ],

            stress=effects[
                "stress"
            ],

            irreversible=effects[
                "irreversible"
            ],

            external_scope=effects[
                "external_scope"
            ],

            hard_guard_ok=(
                true_contract
                .hard_guard_ok
            ),

            receipt_class=(
                "synthetic-underestimating"
            ),
        )