"""External platform bridges (exports, adapters)."""

from phi.integrations.quantconnect import (
    MANIFEST_VERSION,
    write_quantconnect_bundle,
)

__all__ = [
    "MANIFEST_VERSION",
    "write_quantconnect_bundle",
]
