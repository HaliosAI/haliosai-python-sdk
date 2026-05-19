"""OptimizeRecorder — HTTP client for recording optimizer runs and iterations to Halios."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


class OptimizeRecorder:
    """Records optimizer runs and iterations to the Halios backend API.

    Used by the local optimizer loop to persist state to the Halios platform
    so runs can be monitored from the UI.

    Example::

        rec = OptimizeRecorder(
            api_url="https://app.halios.ai",
            api_key="hal_...",
        )
        run_id = await rec.create_run(config=cfg, agent_id="42")
        await rec.start_run(run_id)
        await rec.record_iteration(run_id, iteration_number=0, ...)
        await rec.complete_run(run_id, final_prompt="...", accepted_iteration_id="...")
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: float = 30.0,
    ) -> None:
        self._base = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def __aenter__(self) -> "OptimizeRecorder":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    async def create_run(self, *, config: Any, agent_id: str) -> str:
        """Create an optimization run record and return the run_id."""
        from .config import OptimizeConfig

        if isinstance(config, OptimizeConfig):
            config_dict = config.to_api_config()
            run_name = config.run_name
            starting_prompt = config.starting_prompt
            dataset_id = config.dataset_id
            dataset_version = config.dataset_version
        else:
            config_dict = dict(config)
            run_name = config_dict.pop("run_name", "optimize-run")
            starting_prompt = config_dict.pop("starting_prompt", "")
            dataset_id = config_dict.pop("dataset_id", None)
            dataset_version = config_dict.pop("dataset_version", None)

        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "name": run_name,
            "starting_prompt": starting_prompt,
            "config": config_dict,
        }
        if dataset_id:
            payload["dataset_id"] = dataset_id
        if dataset_version is not None:
            payload["dataset_version"] = dataset_version
        resp = await self._client.post("/api/v1/optimization-runs", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["id"]

    async def start_run(self, run_id: str) -> None:
        """Signal the backend that the optimizer loop has started."""
        resp = await self._client.post(f"/api/v1/optimization-runs/{run_id}/start")
        resp.raise_for_status()

    async def record_iteration(
        self,
        run_id: str,
        *,
        iteration_number: int,
        verdict: str,
        prompt_before: str,
        prompt_after: str,
        scorecard_json: dict[str, Any],
        scorecard_delta_json: dict[str, Any],
        t1_gate_passed: bool,
        trace_run_tag: str | None = None,
    ) -> dict[str, Any]:
        """Record one completed iteration. Returns the created iteration object."""
        payload = {
            "iteration_number": iteration_number,
            "verdict": verdict,
            "prompt_before": prompt_before,
            "prompt_after": prompt_after,
            "scorecard_json": scorecard_json,
            "scorecard_delta_json": scorecard_delta_json,
            "t1_gate_passed": t1_gate_passed,
            "trace_run_tag": trace_run_tag,
        }
        resp = await self._client.post(
            f"/api/v1/optimization-runs/{run_id}/iterations",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def complete_run(
        self,
        run_id: str,
        *,
        final_prompt: str | None = None,
        accepted_iteration_id: str | None = None,
    ) -> None:
        """Mark the run as complete."""
        payload: dict[str, Any] = {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if final_prompt is not None:
            payload["current_prompt"] = final_prompt
        if accepted_iteration_id is not None:
            payload["accepted_iteration_id"] = accepted_iteration_id
        resp = await self._client.patch(
            f"/api/v1/optimization-runs/{run_id}",
            json=payload,
        )
        resp.raise_for_status()

    async def fail_run(self, run_id: str, *, error: str) -> None:
        """Mark the run as failed with an error message."""
        payload = {
            "status": "failed",
            "error_message": error,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        resp = await self._client.patch(
            f"/api/v1/optimization-runs/{run_id}",
            json=payload,
        )
        resp.raise_for_status()

    async def cancel_run(self, run_id: str) -> None:
        """Cancel a pending/running optimization run."""
        resp = await self._client.post(f"/api/v1/optimization-runs/{run_id}/cancel")
        resp.raise_for_status()

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch the current run state (used to poll for cancellation)."""
        resp = await self._client.get(f"/api/v1/optimization-runs/{run_id}")
        resp.raise_for_status()
        return resp.json()

    async def is_cancelled(self, run_id: str) -> bool:
        """Return True if the run has been cancelled from the UI."""
        try:
            data = await self.get_run(run_id)
            return data.get("status") == "cancelled"
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# LocalRecorder — filesystem-only recorder (no Halios account required)
# ---------------------------------------------------------------------------


class LocalRecorder:
    """Records optimizer runs to the local filesystem under ``<runs_dir>/<run_id>/``.

    Implements the same async interface as :class:`OptimizeRecorder` so it can
    be passed to :class:`~haliosai.optimize.engine.OptimizerEngine` as a
    drop-in replacement when running in local mode.

    Directory layout::

        .halios/runs/
          local-<id>/
            run.json          ← run metadata
            iterations/
              iter-0-<id>.json
              iter-1-<id>.json
              ...

    Example::

        rec = LocalRecorder()
        run_id = await rec.create_run(config=cfg, agent_id="my-agent")
        await rec.start_run(run_id)
        await rec.record_iteration(run_id, iteration_number=0, ...)
        await rec.complete_run(run_id, final_prompt="...")
    """

    def __init__(self, runs_dir: str | Path = ".halios/runs") -> None:
        self._runs_dir = Path(runs_dir)

    # ── Context-manager support (mirrors OptimizeRecorder) ────────────────────

    async def __aenter__(self) -> "LocalRecorder":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass  # nothing to close

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_path(self, run_id: str) -> Path:
        p = self._runs_dir / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _run_file(self, run_id: str) -> Path:
        return self._runs_dir / run_id / "run.json"

    def _read_run(self, run_id: str) -> dict[str, Any]:
        path = self._run_file(run_id)
        return json.loads(path.read_text())

    def _write_run(self, run_id: str, data: dict[str, Any]) -> None:
        self._run_file(run_id).write_text(json.dumps(data, indent=2))

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    async def create_run(self, *, config: Any, agent_id: str) -> str:
        """Create a local run record and return the run_id."""
        run_id = f"local-{uuid.uuid4().hex[:12]}"
        run_path = self._run_path(run_id)
        try:
            config_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)
            run_name = config_dict.get("run_name", "local-run")
        except Exception:  # noqa: BLE001
            config_dict = {}
            run_name = "local-run"

        meta: dict[str, Any] = {
            "id": run_id,
            "agent_id": agent_id,
            "name": run_name,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config_dict,
        }
        (run_path / "run.json").write_text(json.dumps(meta, indent=2))
        return run_id

    async def start_run(self, run_id: str) -> None:
        """Mark the run as running."""
        data = self._read_run(run_id)
        data["status"] = "running"
        data["started_at"] = datetime.now(timezone.utc).isoformat()
        self._write_run(run_id, data)

    async def record_iteration(
        self,
        run_id: str,
        *,
        iteration_number: int,
        verdict: str,
        prompt_before: str,
        prompt_after: str,
        scorecard_json: dict[str, Any],
        scorecard_delta_json: dict[str, Any],
        t1_gate_passed: bool,
        trace_run_tag: str | None = None,
    ) -> dict[str, Any]:
        """Record one completed iteration. Returns the created iteration object."""
        run_path = self._run_path(run_id)
        iter_dir = run_path / "iterations"
        iter_dir.mkdir(exist_ok=True)

        iter_id = f"iter-{iteration_number}-{uuid.uuid4().hex[:8]}"
        data: dict[str, Any] = {
            "id": iter_id,
            "run_id": run_id,
            "iteration_number": iteration_number,
            "verdict": verdict,
            "prompt_before": prompt_before,
            "prompt_after": prompt_after,
            "scorecard_json": scorecard_json,
            "scorecard_delta_json": scorecard_delta_json,
            "t1_gate_passed": t1_gate_passed,
            "trace_run_tag": trace_run_tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (iter_dir / f"{iter_id}.json").write_text(json.dumps(data, indent=2))
        return data

    async def complete_run(
        self,
        run_id: str,
        *,
        final_prompt: str | None = None,
        accepted_iteration_id: str | None = None,
    ) -> None:
        """Mark the run as complete."""
        data = self._read_run(run_id)
        data["status"] = "complete"
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
        if final_prompt is not None:
            data["final_prompt"] = final_prompt
        if accepted_iteration_id is not None:
            data["accepted_iteration_id"] = accepted_iteration_id
        self._write_run(run_id, data)

    async def fail_run(self, run_id: str, *, error: str) -> None:
        """Mark the run as failed with an error message."""
        data = self._read_run(run_id)
        data["status"] = "failed"
        data["error_message"] = error
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_run(run_id, data)

    async def cancel_run(self, run_id: str) -> None:
        """Cancel a pending/running run."""
        data = self._read_run(run_id)
        data["status"] = "cancelled"
        self._write_run(run_id, data)

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch the current run state."""
        return self._read_run(run_id)

    async def is_cancelled(self, run_id: str) -> bool:
        """Return True if the run has been cancelled."""
        try:
            data = await self.get_run(run_id)
            return data.get("status") == "cancelled"
        except Exception:  # noqa: BLE001
            return False
