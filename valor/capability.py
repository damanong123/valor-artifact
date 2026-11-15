from __future__ import annotations

import hashlib
import hmac
import json

from valor.models import (
    Action,
    Capability,
)


def _canonical_message(
    *,
    action: Action,
    epoch: int,
    reserved: dict[str, float],
    nonce: str,
    expiry_ns: int,
) -> bytes:
    """
    Build a stable representation of the
    exact authorization request.
    """

    payload = {
        "tool": action.tool,
        "parameter": action.parameter,

        "epoch": epoch,

        "reserved": {
            "energy": reserved["energy"],
            "stress": reserved["stress"],
            "irreversible":
                reserved["irreversible"],
            "external_scope":
                reserved["external_scope"],
        },

        "nonce": nonce,
        "expiry_ns": expiry_ns,
    }

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def mint_capability(
    *,
    key: bytes,
    action: Action,
    epoch: int,
    reserved: dict[str, float],
    nonce: str,
    expiry_ns: int,
) -> Capability:
    """
    Mint:

        Cap(a, e, b, nonce, expiry)
    """

    message = _canonical_message(
        action=action,
        epoch=epoch,
        reserved=reserved,
        nonce=nonce,
        expiry_ns=expiry_ns,
    )

    mac_hex = hmac.new(
        key,
        message,
        hashlib.sha256,
    ).hexdigest()

    return Capability(
        action=action,

        epoch=epoch,

        reserved_energy=(
            reserved["energy"]
        ),

        reserved_stress=(
            reserved["stress"]
        ),

        reserved_irreversible=(
            reserved["irreversible"]
        ),

        reserved_external_scope=(
            reserved["external_scope"]
        ),

        nonce=nonce,
        expiry_ns=expiry_ns,

        mac_hex=mac_hex,
    )


def verify_capability(
    *,
    key: bytes,
    capability: Capability,
) -> bool:

    reserved = (
        capability
        .reserved_effect_vector()
    )

    message = _canonical_message(
        action=capability.action,
        epoch=capability.epoch,
        reserved=reserved,
        nonce=capability.nonce,
        expiry_ns=(
            capability.expiry_ns
        ),
    )

    expected = hmac.new(
        key,
        message,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(
        expected,
        capability.mac_hex,
    )