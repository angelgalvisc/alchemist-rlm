"""Episode-boundary isolation policies and their attestations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from alchemist_rlm.mlx_client import ServerUnavailable


@dataclass
class MLXPromptCacheIsolation:
    """Clear reusable MLX prompt state before every root episode.

    The server returns counts before and after the reset.  A successful HTTP
    response is not enough: the harness accepts the boundary only when the
    reported postcondition is empty.
    """

    base_url: str
    timeout: float = 30.0
    policy: str = "mlx_prompt_cache_reset_v1"

    @property
    def endpoint(self) -> str:
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return f"{root}/admin/prompt-cache/reset"

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.policy,
            "scope": "before_each_root_episode",
            "postcondition": {"sequences": 0, "bytes": 0},
        }

    def before_episode(self, *, run_id: str) -> dict[str, Any]:
        try:
            response = httpx.post(
                self.endpoint,
                json={"run_id": run_id, "policy": self.policy},
                timeout=self.timeout,
            )
            response.raise_for_status()
            attestation = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ServerUnavailable(
                f"episode isolation failed before {run_id}: {error}"
            ) from error
        after = attestation.get("after") or {}
        if (attestation.get("ok") is not True
                or after.get("sequences") != 0
                or after.get("bytes") != 0):
            raise ServerUnavailable(
                f"episode isolation did not establish an empty cache before "
                f"{run_id}: {attestation}"
            )
        return {
            "run_id": run_id,
            "policy": self.policy,
            "epoch": attestation.get("epoch"),
            "before": attestation.get("before"),
            "after": after,
            "verified": True,
        }
