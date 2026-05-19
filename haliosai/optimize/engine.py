"""OptimizerEngine — client-side prompt optimization loop.

Runs entirely in the CLI process.  The Halios backend API is used only for
recording run state (via OptimizeRecorder) and triggering/querying evals.

Flow
----
1. Probe ``GET /halios/config`` on the target agent → capture starting prompt.
2. Iteration 0 (baseline): run the configured scenario fixture with the current prompt.
3. For each subsequent iteration:
   a. Suggest a prompt mutation through the Halios backend LLM gateway.
   b. Apply the mutation via ``X-Halios-Prompt`` header on scenario runs.
   c. Ingest traces, run evals, build scorecard.
   d. Compare vs. baseline → ACCEPT | DISCARD | INVESTIGATE.
   e. Record iteration to backend.
4. Mark run complete / failed.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

try:
    import structlog
except ImportError:  # pragma: no cover - only used in minimal local installs
    import logging

    class _LoggerAdapter:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def info(self, event: str, **kwargs: Any) -> None:
            self._logger.info("%s %s", event, kwargs)

        def warning(self, event: str, **kwargs: Any) -> None:
            self._logger.warning("%s %s", event, kwargs)

    class _StructlogFallback:
        @staticmethod
        def get_logger() -> _LoggerAdapter:
            return _LoggerAdapter(__name__)

    structlog = _StructlogFallback()

from .config import DatasetBuildConfig, OptimizeConfig, Scenario
from .protocol import HaliosOptimizableClient
from .recorder import OptimizeRecorder
from .scorer import LocalScorer as _LocalScorer  # noqa: F401 — re-exported via __init__
from .scorecard import print_iteration_table, print_scorecard_table

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Mutation prompt template
# ---------------------------------------------------------------------------

_MUTATION_PROMPT_TEMPLATE = """\
You are an expert AI prompt engineer.

## Task
Analyze the evaluation scorecard comparison below and the current agent system \
prompt, then propose a targeted improvement.

## T1 hard-gate checks (must NOT regress):
{t1_check_names}

## Baseline scorecard:
{baseline_json}

## Candidate scorecard (current iteration):
{candidate_json}

## Failing / low-scoring areas:
{failing_areas}

## Current system prompt:
{current_prompt}

## Instructions
1. Identify what behaviors the low-scoring checks measure.
2. Proposed changes MUST preserve or improve all T1 hard-gate checks.
3. Optimize ONE area at a time — do not do unrelated rewrites.
4. Do NOT change the persona, tools, or core brand voice.
5. Return strict JSON with exactly these keys:
   - "prompt": the complete improved system prompt artifact, and only the prompt text
   - "rationale": one concise sentence explaining the change
6. Do not include markdown fences, headings, analysis notes, or rationale inside
   the "prompt" value.

Output the JSON object now:
"""

# ---------------------------------------------------------------------------
# Pure scorecard helpers (duplicated from backend so SDK has no backend dep)
# ---------------------------------------------------------------------------


def _score_value(raw: object) -> float:
    if raw in {"-inf", "-Infinity"}:
        return -math.inf
    if raw in {"inf", "+inf", "Infinity", "+Infinity"}:
        return math.inf
    if raw is None:
        return 0.0
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        if value == float("inf"):
            return "+inf"
        if value == float("-inf"):
            return "-inf"
        if math.isnan(value):
            return "nan"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _build_scorecard(
    executions: list[dict[str, Any]],
    t1_check_ids: list[str],
    t1_gate_threshold: float = 0.85,
) -> dict[str, Any]:
    if not executions:
        return {
            "overall_score": 0.0,
            "t1_passed": False,
            "check_pass_rates": {},
            "verdict": "error",
        }

    by_check: dict[str, list[dict]] = {}
    for ex in executions:
        cid = str(ex.get("check_id") or "unknown")
        by_check.setdefault(cid, []).append(ex)

    check_pass_rates: dict[str, dict] = {}
    scores: list[float] = []
    t1_passed = True

    for check_id, exs in by_check.items():
        name = exs[0].get("check_name", check_id)
        tier = exs[0].get("tier", "")
        passed_count = sum(1 for e in exs if e.get("passed"))
        total = len(exs)
        rate = passed_count / total if total else 0.0
        scored = [float(e["score"]) for e in exs if e.get("score") is not None]
        avg_score: float | None = sum(scored) / len(scored) if scored else None
        check_pass_rates[check_id] = {
            "name": name,
            "tier": tier,
            "rate": rate,
            "avg_score": avg_score,
            "total": total,
        }
        if avg_score is not None:
            scores.append(avg_score)
        if check_id in t1_check_ids and rate < t1_gate_threshold:
            t1_passed = False

    overall = sum(scores) / len(scores) if scores else 0.0
    return {
        "overall_score": _json_safe(overall),
        "t1_passed": t1_passed,
        "check_pass_rates": _json_safe(check_pass_rates),
    }


def _compare_scorecards(baseline: dict, candidate: dict) -> dict[str, Any]:
    b_score = _score_value(baseline.get("overall_score", 0))
    c_score = _score_value(candidate.get("overall_score", 0))
    delta = c_score - b_score

    check_deltas: dict[str, dict] = {}
    b_rates = baseline.get("check_pass_rates", {})
    c_rates = candidate.get("check_pass_rates", {})
    for check_id in set(list(b_rates.keys()) + list(c_rates.keys())):
        b_r = _score_value((b_rates.get(check_id) or {}).get("avg_score", 0))
        c_r = _score_value((c_rates.get(check_id) or {}).get("avg_score", 0))
        check_deltas[check_id] = {
            "name": (c_rates.get(check_id) or b_rates.get(check_id) or {}).get("name", check_id),
            "baseline": b_r,
            "candidate": c_r,
            "delta": c_r - b_r,
            "regressed": c_r < b_r - 0.01,
        }

    return {
        "baseline_score": _json_safe(b_score),
        "candidate_score": _json_safe(c_score),
        "delta": _json_safe(delta),
        "check_deltas": _json_safe(check_deltas),
    }


def _select_verdict(baseline: dict, candidate: dict, t1_check_ids: list[str]) -> str:
    if not candidate.get("t1_passed", True):
        return "discard"
    b_score = _score_value(baseline.get("overall_score", 0))
    c_score = _score_value(candidate.get("overall_score", 0))
    if c_score >= b_score + 0.01:
        return "accept"
    if c_score < b_score - 0.01:
        return "discard"
    return "investigate"


def _build_failing_areas(comparison: dict, t1_check_ids: list[str], t1_gate_threshold: float = 0.85) -> str:
    lines: list[str] = []
    for check_id, data in sorted(comparison.get("check_deltas", {}).items()):
        c_val = _score_value(data.get("candidate", 0))
        regressed = data.get("regressed", False)
        is_t1 = check_id in t1_check_ids
        if c_val < 0.8 or regressed or is_t1:
            marker = (
                " [T1 GATE — MUST FIX]" if (is_t1 and c_val < t1_gate_threshold)
                else " [T1 REGRESSED]" if (is_t1 and regressed)
                else " [T1 OK]" if is_t1
                else " [REGRESSED]" if regressed
                else " [LOW]"
            )
            lines.append(
                f"  {data.get('name', check_id)} ({check_id}): "
                f"baseline={data.get('baseline', 0):.2%}, "
                f"candidate={data.get('candidate', 0):.2%}{marker}"
            )
    return "\n".join(lines) if lines else "  (none — all checks passing)"


def _clean_prompt_artifact(value: object, fallback: str) -> str:
    """Return only the prompt artifact, never optimizer rationale prose."""
    prompt = str(value or "").strip()
    if not prompt:
        return fallback

    if prompt.startswith("```"):
        prompt = prompt.strip("`").strip()
        if prompt.lower().startswith(("text", "prompt")):
            prompt = prompt.split("\n", 1)[-1].strip()

    rationale_markers = (
        "\n**Rationale for changes:**",
        "\nRationale for changes:",
        "\n**Rationale:**",
        "\nRationale:",
        "\nThe proposed changes aim to address these by:",
    )
    for marker in rationale_markers:
        idx = prompt.find(marker)
        if idx >= 0:
            prompt = prompt[:idx].strip()

    return prompt or fallback


# ---------------------------------------------------------------------------
# Halios eval client (HTTP, not direct DB access)
# ---------------------------------------------------------------------------


class _HaliosEvalClient:
    """Lightweight async client for Halios trace ingest + eval trigger."""

    def __init__(self, api_url: str, api_key: str, agent_id: str, timeout: float = 120.0) -> None:
        self._base = api_url.rstrip("/")
        self._api_key = api_key
        self._agent_id = agent_id
        self._timeout = timeout

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        attempts = (
            {"Authorization": f"Bearer {self._api_key}"},
            {"X-API-Key": self._api_key},
        )
        last_response: httpx.Response | None = None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for headers in attempts:
                response = await client.request(
                    method,
                    f"{self._base}{path}",
                    json=json_body,
                    params=params,
                    headers=headers,
                )
                if response.status_code != 401:
                    response.raise_for_status()
                    return response
                last_response = response

        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError(f"{method} {path} failed without a response")

    async def ingest_trace(
        self,
        trace_id: str,
        spans: list[dict[str, Any]],
        run_tag: str,
        poll_interval: float = 2.0,
        max_wait: float = 120.0,
    ) -> None:
        """Ingest a trace to the Halios API and wait for it to be stored."""
        import asyncio

        payload = {
            "agent_id": self._agent_id,
            "run_tag": run_tag,
            "tags": [f"run:{run_tag}"],
            "source": "optimizer",
            "traces": [{"trace_id": trace_id, "spans": spans}],
        }
        resp = await self._request(
            "POST",
            "/api/v1/ingest/traces/bulk",
            json_body=payload,
        )
        data = resp.json()

        # If the ingest was queued (async Celery), poll until completed so the
        # trace is actually stored before trigger_eval is called.
        status_url = data.get("status_url")
        if data.get("status") != "completed" and status_url:
            deadline = datetime.now(timezone.utc).timestamp() + max_wait
            while datetime.now(timezone.utc).timestamp() < deadline:
                await asyncio.sleep(poll_interval)
                poll_resp = await self._request("GET", status_url)
                if poll_resp.status_code == 200:
                    state = poll_resp.json().get("status", "")
                    if state in ("completed", "failed"):
                        break

    async def trigger_eval(
        self,
        trace_ids: list[str],
        check_ids: list[str] | None = None,
        run_tag: str | None = None,
    ) -> dict[str, Any]:
        """Trigger an eval run; return task/run metadata."""
        payload = {
            "agent_id": self._agent_id,
            "trace_ids": trace_ids,
            "mode": "evaluator",
        }
        if check_ids:
            payload["check_ids"] = check_ids
        if run_tag:
            payload["tags"] = [run_tag]
            payload["run_name"] = run_tag.replace(":", "-")
        resp = await self._request(
            "POST",
            "/api/v1/trigger-eval",
            json_body=payload,
        )
        return resp.json()

    async def poll_eval(self, task_id: str, poll_interval: float = 3.0, max_wait: float = 300.0) -> None:
        """Poll until the eval task reaches a terminal state."""
        import asyncio

        deadline = datetime.now(timezone.utc).timestamp() + max_wait
        while datetime.now(timezone.utc).timestamp() < deadline:
            resp = await self._request(
                "GET",
                f"/api/v1/tasks/{task_id}",
            )
            if resp.status_code == 200:
                state = resp.json().get("status", "")
                # Celery states: SUCCESS/FAILURE are terminal; also accept legacy "completed"/"failed"
                if state in ("SUCCESS", "FAILURE", "completed", "failed"):
                    return
            await asyncio.sleep(poll_interval)

    async def fetch_executions(
        self,
        trace_ids: list[str],
        check_ids: list[str] | None = None,
        run_tag: str | None = None,
        max_wait: float = 60.0,
        poll_interval: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Fetch check executions for the given trace IDs."""
        deadline = datetime.now(timezone.utc).timestamp() + max_wait
        rows: list[dict[str, Any]] = []
        while datetime.now(timezone.utc).timestamp() <= deadline:
            rows = []
            cursor: str | None = None
            while True:
                params: dict[str, Any] = {
                    "limit": 100,
                    "include_progress": "false",
                    "mode": "evaluator",
                }
                if cursor:
                    params["cursor"] = cursor
                if run_tag:
                    params["tags"] = [run_tag]
                resp = await self._request(
                    "GET",
                    f"/api/v1/agents/{self._agent_id}/check-executions",
                    params=params,
                )
                data = resp.json()
                rows.extend(item for item in data.get("data", []) if isinstance(item, dict))
                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")
                if not cursor:
                    break
            if rows or not run_tag:
                break
            await asyncio.sleep(poll_interval)

        # Filter to the trace_ids and check_ids we care about
        check_id_set = set(check_ids or [])
        trace_id_set = set(trace_ids)
        return [
            {
                "check_id": r.get("check_id"),
                "check_name": r.get("check_name", ""),
                "passed": r.get("passed"),
                "score": r.get("score"),
                "tier": r.get("tier", ""),
            }
            for r in rows
            if r.get("trace_id") in trace_id_set
            and (not check_id_set or str(r.get("check_id")) in check_id_set)
        ]

    async def list_dataset_traces(self, dataset_id: str, dataset_version: int | None = None) -> list[str]:
        """Return all trace_ids stored in a Halios dataset."""
        params: dict[str, Any] = {"limit": 2000}
        if dataset_version is not None:
            params["version"] = dataset_version
        resp = await self._request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/traces",
            params=params,
        )
        items = resp.json().get("items", [])
        return [item["trace_id"] for item in items if item.get("trace_id")]

    async def create_dataset(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        provenance: str = "synthetic",
    ) -> str:
        """Create a new dataset and return its ID."""
        resp = await self._request(
            "POST",
            "/api/v1/datasets",
            json_body={
                "agent_id": agent_id,
                "name": name,
                "description": description,
                "provenance": provenance,
            },
        )
        return resp.json()["id"]

    async def add_traces_to_dataset(
        self,
        dataset_id: str,
        trace_ids: list[str],
        provenance: str = "optimizer",
        scenario_id: str | None = None,
    ) -> None:
        """Add traces to a dataset (idempotent — duplicates silently skipped)."""
        await self._request(
            "POST",
            f"/api/v1/datasets/{dataset_id}/traces",
            json_body={
                "trace_ids": trace_ids,
                "provenance": provenance,
                "scenario_id": scenario_id,
            },
        )

    async def list_scenarios(
        self,
        agent_id: str,
        *,
        scenario_ids: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch server-side scenarios for an agent."""
        resp = await self._request(
            "GET",
            "/api/v1/scenarios",
            params={"agent_id": agent_id, "limit": limit},
        )
        items = resp.json().get("items", [])
        if scenario_ids:
            allowed = set(scenario_ids)
            return [item for item in items if item.get("id") in allowed]
        return items

    async def call_llm_json(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> dict[str, Any]:
        """Call Halios backend-managed LLM routing and return parsed JSON."""
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp: httpx.Response | None = None
        for attempt in range(1, 4):
            try:
                resp = await self._request("POST", "/api/v1/llm/json", json_body=payload)
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in {429, 502, 503, 504} or attempt == 3:
                    raise
                logger.warning(
                    "llm_json_retryable_error",
                    status=status,
                    attempt=attempt,
                    model=model,
                    error=str(exc),
                )
                await asyncio.sleep(0.5 * attempt)
        if resp is None:
            raise RuntimeError("LLM JSON call failed without a response")
        data = resp.json().get("data")
        return data if isinstance(data, dict) else {}

    async def call_llm_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """Call Halios backend-managed LLM routing and return text content."""
        resp = await self._request(
            "POST",
            "/api/v1/llm/chat",
            json_body={
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        return str(resp.json().get("content") or "")


# ---------------------------------------------------------------------------
# OptimizerEngine
# ---------------------------------------------------------------------------


class OptimizerEngine:
    """Drives the full prompt optimization loop client-side.

    Instantiate, then call :meth:`run` once.  Progress is printed to the
    console via Rich (or plain text) and recorded to the Halios backend
    via :class:`OptimizeRecorder` (cloud mode) or to the local filesystem
    via :class:`~haliosai.optimize.recorder.LocalRecorder` (local mode).

    Cloud mode example (requires Halios account)::

        engine = OptimizerEngine(config=cfg, recorder=rec, verbose=True)
        final_prompt = await engine.run()

    Local mode example (OpenAI only, no Halios account)::

        from haliosai.optimize import LocalRecorder, LocalScorer

        scorer = LocalScorer(".halios/rubrics.yaml", model="gpt-4o-mini")
        async with LocalRecorder() as rec:
            engine = OptimizerEngine(
                config=cfg, recorder=rec, verbose=True, local_scorer=scorer
            )
            final_prompt = await engine.run()
    """

    def __init__(
        self,
        config: OptimizeConfig | DatasetBuildConfig,
        recorder: Any | None = None,
        verbose: bool = True,
        local_scorer: Any | None = None,
    ) -> None:
        self._cfg = config
        self._rec = recorder
        self._verbose = verbose
        self._local_scorer = local_scorer
        self._agent_client = HaliosOptimizableClient(config.target_url)
        # Conversations keyed by trace_id — populated in local mode so the
        # scorer can access the full message list without a Halios ingest.
        self._local_conversations: dict[str, list[dict[str, Any]]] = {}
        # Cloud eval client is only created when running in cloud mode.
        self._eval_client: _HaliosEvalClient | None = (
            None
            if local_scorer is not None
            else _HaliosEvalClient(
                api_url=config.halios_api_url,
                api_key=config.halios_api_key,
                agent_id=config.agent_id,
            )
        )

    async def run(self) -> str:
        """Execute the full optimization loop.

        Returns the best prompt found (the last accepted prompt, or the
        starting prompt if no iteration was accepted).
        """
        cfg = self._cfg
        if not isinstance(cfg, OptimizeConfig):
            raise RuntimeError("OptimizerEngine.run() requires an OptimizeConfig.")
        if self._rec is None:
            raise RuntimeError(
                "OptimizerEngine.run() requires a recorder. "
                "Pass an OptimizeRecorder (cloud) or a LocalRecorder (local mode)."
            )

        # Probe the target agent before creating the run so the UI records the
        # actual prompt served by the HaliosOptimizable interface, not a YAML
        # placeholder/fallback.
        current_prompt = await self._resolve_starting_prompt(cfg)

        # 1. Create & start a run record on the backend
        run_config = cfg.model_copy(update={"starting_prompt": current_prompt})
        run_id = await self._rec.create_run(config=run_config, agent_id=cfg.agent_id)
        await self._rec.start_run(run_id)

        logger.info("optimizer_started", run_id=run_id, target_url=cfg.target_url)
        self._log(f"Run started: {run_id}")

        try:
            result_prompt = await self._loop(run_id, current_prompt)
        except Exception as exc:
            await self._rec.fail_run(run_id, error=str(exc))
            raise

        return result_prompt

    async def _resolve_starting_prompt(self, cfg: OptimizeConfig) -> str:
        try:
            agent_cfg = await self._agent_client.get_config()
        except Exception as exc:
            raise RuntimeError(f"Cannot reach target agent at {cfg.target_url}: {exc}") from exc

        current_prompt: str = agent_cfg.current_prompt or cfg.starting_prompt
        if not current_prompt:
            raise RuntimeError(
                "No starting_prompt found in agent /halios/config or config YAML."
            )
        return current_prompt

    async def _loop(self, run_id: str, current_prompt: str) -> str:
        cfg = self._cfg
        if not isinstance(cfg, OptimizeConfig):
            raise RuntimeError("OptimizerEngine._loop() requires an OptimizeConfig.")

        self._log(f"Starting prompt ({len(current_prompt)} chars)")

        if not cfg.scenarios:
            raise RuntimeError(
                "Optimizer requires scenario specs. Pass --scenarios .halios/scenarios.yaml "
                "or add scenarios to optimize.yaml."
            )

        # ── Iteration 0: baseline ─────────────────────────────────────────────
        baseline_tag = f"opt:{run_id}:baseline"
        self._log(f"Scenario-fixture mode: running {len(cfg.scenarios)} scenario(s) as baseline")
        baseline_scorecard = await self._run_scenario_set(
            scenarios=cfg.scenarios,
            prompt_override=current_prompt,
            run_tag=baseline_tag,
        )
        self._log(f"Baseline score: {baseline_scorecard.get('overall_score', 0.0):.3f}")
        if self._verbose:
            print_scorecard_table(baseline_scorecard, "Baseline Scorecard")

        baseline_iteration = await self._rec.record_iteration(
            run_id,
            iteration_number=0,
            verdict="baseline",
            prompt_before=current_prompt,
            prompt_after=current_prompt,
            scorecard_json=baseline_scorecard,
            scorecard_delta_json={},
            t1_gate_passed=baseline_scorecard.get("t1_passed", True),
            trace_run_tag=baseline_tag,
        )
        last_accepted_iteration_id: str | None = None

        baseline_for_compare = baseline_scorecard
        consecutive_discards = 0
        all_iterations = [baseline_iteration]

        for iteration_number in range(1, cfg.max_iterations + 1):
            # Check for cancellation
            if await self._rec.is_cancelled(run_id):
                self._log("Run cancelled from UI. Stopping.")
                await self._rec.complete_run(
                    run_id,
                    final_prompt=current_prompt,
                    accepted_iteration_id=last_accepted_iteration_id,
                )
                break

            if consecutive_discards >= cfg.max_consecutive_discards:
                self._log(
                    f"Stopping: {consecutive_discards} consecutive discards "
                    f"(limit: {cfg.max_consecutive_discards})"
                )
                break

            self._log(f"\nIteration {iteration_number}/{cfg.max_iterations}")

            # Suggest mutation
            try:
                candidate_prompt = await self._suggest_mutation(
                    baseline_scorecard=baseline_for_compare,
                    candidate_scorecard=baseline_for_compare,
                    current_prompt=current_prompt,
                )
            except Exception as exc:
                logger.warning("mutation_failed", error=str(exc))
                candidate_prompt = current_prompt

            # Run scenarios with candidate prompt
            candidate_tag = f"opt:{run_id}:iter{iteration_number}"
            candidate_scorecard = await self._run_scenario_set(
                scenarios=cfg.scenarios,
                prompt_override=candidate_prompt,
                run_tag=candidate_tag,
            )

            comparison = _compare_scorecards(baseline_for_compare, candidate_scorecard)
            verdict = _select_verdict(baseline_for_compare, candidate_scorecard, cfg.t1_check_ids)

            self._log(
                f"  Verdict: {verdict}  "
                f"score={candidate_scorecard.get('overall_score', 0.0):.3f}  "
                f"delta={_score_value(comparison.get('delta', 0)):+.3f}"
            )
            if self._verbose:
                print_scorecard_table(candidate_scorecard, f"Iteration {iteration_number}")

            iteration_rec = await self._rec.record_iteration(
                run_id,
                iteration_number=iteration_number,
                verdict=verdict,
                prompt_before=current_prompt,
                prompt_after=candidate_prompt,
                scorecard_json=candidate_scorecard,
                scorecard_delta_json=comparison,
                t1_gate_passed=candidate_scorecard.get("t1_passed", True),
                trace_run_tag=candidate_tag,
            )
            all_iterations.append(iteration_rec)

            if verdict == "accept":
                current_prompt = candidate_prompt
                baseline_for_compare = candidate_scorecard
                last_accepted_iteration_id = iteration_rec.get("id")
                consecutive_discards = 0
            else:
                consecutive_discards += 1

        # Finalize
        await self._rec.complete_run(
            run_id,
            final_prompt=current_prompt,
            accepted_iteration_id=last_accepted_iteration_id,
        )
        self._log(f"\nOptimization complete. Run ID: {run_id}")
        if self._verbose:
            print_iteration_table([dict(it) for it in all_iterations])

        return current_prompt

    # ── Dataset helpers ───────────────────────────────────────────────────────

    async def _run_dataset_eval(
        self,
        dataset_id: str,
        _run_tag: str,
        *,
        dataset_version: int | None = None,
    ) -> dict[str, Any]:
        """Fetch traces from a dataset, trigger eval on them, and return a scorecard.

        Used by the dataset-first optimizer flow to skip baseline re-simulation.
        """
        trace_ids = await self._eval_client.list_dataset_traces(dataset_id, dataset_version)
        if not trace_ids:
            self._log(f"  Dataset {dataset_id} has no traces — returning empty scorecard")
            return _build_scorecard([], self._cfg.t1_check_ids, self._cfg.t1_gate_threshold)

        version_msg = f" at version {dataset_version}" if dataset_version is not None else ""
        self._log(f"  Found {len(trace_ids)} traces in dataset {dataset_id}{version_msg}")

        try:
            task_id = await self._eval_client.trigger_eval(
                trace_ids,
                self._cfg.check_ids or None,
                run_tag=_run_tag,
            )
            await self._eval_client.poll_eval(task_id)
        except Exception as exc:
            logger.warning("dataset_eval_trigger_failed", dataset_id=dataset_id, error=str(exc))
            return _build_scorecard([], self._cfg.t1_check_ids, self._cfg.t1_gate_threshold)

        executions = await self._eval_client.fetch_executions(
            trace_ids,
            self._cfg.check_ids or None,
            run_tag=_run_tag,
        )
        return _build_scorecard(executions, self._cfg.t1_check_ids, self._cfg.t1_gate_threshold)

    async def build_dataset(self, name: str | None = None, description: str | None = None) -> str:
        """Simulate all scenarios and store the resulting traces in a new Halios dataset.

        Supports either creating a new dataset or appending traces to an
        existing one. Scenario sources can come from the Halios API, inline
        YAML, or both.
        """
        cfg = self._cfg
        scenario_map = await self._resolve_dataset_build_scenarios()
        scenarios = list(scenario_map.values())
        if not scenarios:
            raise RuntimeError(
                "No dataset-build scenarios resolved. Add scenarios to the dataset config or create them in Halios first."
            )

        run_tag = f"dataset-build:{uuid.uuid4().hex[:8]}"

        target_dataset_id = getattr(cfg, "dataset_id", None)
        target_name = name or getattr(cfg, "dataset_name", None)
        target_description = description if description is not None else getattr(cfg, "dataset_description", "")
        provenance = getattr(cfg, "provenance", "synthetic")

        action = "Appending to dataset" if target_dataset_id else "Building dataset"
        self._log(f"{action} with {len(scenarios)} scenario(s)...")
        trace_entries: list[tuple[str, str | None]] = []

        for idx, scenario in enumerate(scenarios, start=1):
            self._log(f"  [{idx}/{len(scenarios)}] scenario: {scenario.id}")
            trace_id = await self._run_single_scenario(scenario, None, run_tag)
            if trace_id:
                trace_entries.append((trace_id, scenario.id))

        if not trace_entries:
            raise RuntimeError(
                "No traces were generated. Check that the agent is running and scenarios are valid."
            )

        self._log(f"  Ingested {len(trace_entries)} trace(s)")

        if target_dataset_id:
            dataset_id = target_dataset_id
        else:
            if not target_name:
                raise RuntimeError(
                    "Dataset name is required when creating a new dataset. Set dataset_name in the config or pass --name."
                )
            dataset_id = await self._eval_client.create_dataset(
                agent_id=cfg.agent_id,
                name=target_name,
                description=target_description or f"Synthetic dataset — {len(trace_entries)} traces, {len(scenarios)} scenario(s)",
                provenance=provenance,
            )

        for trace_id, scenario_id in trace_entries:
            await self._eval_client.add_traces_to_dataset(
                dataset_id=dataset_id,
                trace_ids=[trace_id],
                provenance=provenance,
                scenario_id=scenario_id,
            )

        self._log(f"  Dataset ready: {dataset_id}")
        return dataset_id

    async def _resolve_dataset_build_scenarios(self) -> dict[str, Scenario]:
        """Load dataset-build scenarios from Halios, inline YAML, or both."""
        cfg = self._cfg
        if not isinstance(cfg, DatasetBuildConfig):
            raise RuntimeError("Dataset building requires a DatasetBuildConfig.")

        scenarios: dict[str, Scenario] = {}

        if cfg.scenario_source in {"halios", "mixed"}:
            limit = cfg.max_scenarios or 500
            for raw in await self._eval_client.list_scenarios(
                cfg.agent_id,
                scenario_ids=cfg.scenario_ids or None,
                limit=limit,
            ):
                scenario = self._scenario_from_api(raw)
                if scenario is not None:
                    scenarios[scenario.id] = scenario

        if cfg.scenario_source in {"inline", "mixed"}:
            for scenario in cfg.scenarios:
                scenarios[scenario.id] = scenario

        if cfg.max_scenarios is not None:
            limited_ids = list(scenarios.keys())[: cfg.max_scenarios]
            scenarios = {scenario_id: scenarios[scenario_id] for scenario_id in limited_ids}

        return scenarios

    def _scenario_from_api(self, raw: dict[str, Any]) -> Scenario | None:
        """Convert a Halios scenario row into a replayable runtime scenario."""
        scenario_id = str(raw.get("id") or "")
        generation_mode = str(raw.get("generation_mode") or "simulation")
        goal = (raw.get("goal") or "").strip() or None
        persona = (raw.get("persona") or "").strip() or None
        title = (raw.get("title") or "").strip() or None
        max_turns = int(raw.get("max_turns") or 6)
        scripted_messages = raw.get("scripted_messages") or []
        if isinstance(scripted_messages, list) and scripted_messages:
            messages = [
                message
                for message in scripted_messages
                if isinstance(message, dict) and message.get("role") and message.get("content")
            ]
            if messages:
                return Scenario.model_validate(
                    {
                        "id": scenario_id,
                        "title": title,
                        "goal": goal,
                        "persona": persona,
                        "messages": messages,
                        "generation_mode": "scripted-replay" if generation_mode == "scripted-replay" else generation_mode,
                        "max_turns": max_turns,
                    }
                )

        arc_messages = raw.get("arc_messages") or []
        if isinstance(arc_messages, list) and arc_messages:
            return Scenario.model_validate(
                {
                    "id": scenario_id,
                    "title": title,
                    "goal": goal or "Follow the supplied arc messages.",
                    "persona": persona,
                    "arc_messages": [str(message) for message in arc_messages if message],
                    "generation_mode": generation_mode if generation_mode in {"simulation", "simulation-with-arc-hint"} else "simulation-with-arc-hint",
                    "max_turns": max_turns,
                }
            )

        if goal:
            return Scenario.model_validate(
                {
                    "id": scenario_id,
                    "title": title,
                    "goal": goal,
                    "persona": persona,
                    "generation_mode": generation_mode if generation_mode in {"simulation", "simulation-with-arc-hint"} else "simulation",
                    "max_turns": max_turns,
                }
            )

        logger.warning("scenario_unusable_for_dataset_build", scenario_id=scenario_id)
        return None

    # ── Scenario runner ───────────────────────────────────────────────────────

    async def _run_scenario_set(
        self,
        scenarios: list[Scenario],
        prompt_override: str | None,
        run_tag: str,
    ) -> dict[str, Any]:
        """Run all scenarios, ingest traces, trigger eval, return scorecard."""
        trace_ids: list[str] = []
        failed: list[tuple[str, str]] = []

        for scenario in scenarios:
            try:
                trace_id = await self._run_single_scenario(scenario, prompt_override, run_tag)
            except Exception as exc:
                trace_id = None
                failed.append((scenario.id, str(exc)))
                logger.warning(
                    "optimizer_scenario_failed",
                    scenario_id=scenario.id,
                    run_tag=run_tag,
                    error=str(exc),
                )
            if trace_id:
                trace_ids.append(trace_id)
            elif not any(scenario.id == item[0] for item in failed):
                failed.append((scenario.id, "no trace generated"))

        if failed:
            self._log(
                f"  Skipped {len(failed)}/{len(scenarios)} scenario(s): "
                + ", ".join(scenario_id for scenario_id, _ in failed[:5])
            )

        if not trace_ids:
            return _build_scorecard([], self._cfg.t1_check_ids, self._cfg.t1_gate_threshold)

        # -- Local scoring path: score conversations directly, skip ingest/eval --
        if self._local_scorer is not None:
            conversations = [
                self._local_conversations.pop(tid, []) for tid in trace_ids
            ]
            executions = await self._local_scorer.score_conversations(conversations)
            scorecard = _build_scorecard(
                executions, self._cfg.t1_check_ids, self._cfg.t1_gate_threshold
            )
            scorecard["trace_run_tag"] = run_tag
            return scorecard

        # Trigger eval and wait
        assert self._eval_client is not None, "eval_client must be set in cloud mode"
        try:
            eval_run = await self._eval_client.trigger_eval(
                trace_ids,
                self._cfg.check_ids or None,
                run_tag=run_tag,
            )
            task_id = str(eval_run["task_id"])
            await self._eval_client.poll_eval(task_id)
        except Exception as exc:
            logger.warning("eval_trigger_failed", error=str(exc))
            return _build_scorecard([], self._cfg.t1_check_ids, self._cfg.t1_gate_threshold)

        # Fetch executions and build scorecard
        executions = await self._eval_client.fetch_executions(
            trace_ids,
            self._cfg.check_ids or None,
            run_tag=run_tag,
        )
        scorecard = _build_scorecard(executions, self._cfg.t1_check_ids, self._cfg.t1_gate_threshold)
        if eval_run.get("run_tag"):
            scorecard["eval_run_tag"] = str(eval_run["run_tag"])
        scorecard["trace_run_tag"] = run_tag
        return scorecard

    async def _run_single_scenario(
        self,
        scenario: Scenario,
        prompt_override: str | None,
        run_tag: str,
    ) -> str | None:
        """Run one scenario, build a backend-native span, ingest, return trace_id."""
        conv_id = str(uuid.uuid4())
        request_messages: list[dict[str, str]] = []
        # Collect per-turn data: (started_at, ended_at, user_msg, turn_trace_messages)
        turns: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
        terminal_outcome = "completed"
        user_turn_count = 0

        async def _play_user_turn(user_message: dict[str, Any]) -> bool:
            nonlocal terminal_outcome, user_turn_count
            request_messages.append(user_message)
            turn_started_at = datetime.now(timezone.utc).isoformat()
            try:
                response = await self._agent_client.chat(
                    conversation_id=conv_id,
                    messages=request_messages[:],
                    prompt_override=prompt_override,
                )
                turn_ended_at = datetime.now(timezone.utc).isoformat()
                assistant_text = response.get("response", "")
                turn_trace_messages = self._coerce_trace_messages(response, assistant_text)
                request_messages.append({"role": "assistant", "content": assistant_text})
                turns.append((turn_started_at, turn_ended_at, user_message, turn_trace_messages))
                user_turn_count += 1
                return True
            except Exception as exc:
                logger.warning(
                    "scenario_chat_error",
                    scenario_id=scenario.id,
                    run_tag=run_tag,
                    error=str(exc),
                )
                terminal_outcome = "errored"
                return False

        if scenario.generation_mode == "scripted-replay":
            for msg in scenario.messages:
                user_message = {"role": msg.role, "content": msg.content}
                ok = await _play_user_turn(user_message)
                if not ok:
                    break
        else:
            while user_turn_count < scenario.max_turns:
                next_turn = await self._simulate_next_user_turn(scenario, request_messages, user_turn_count + 1)
                if next_turn is None:
                    terminal_outcome = "timed_out" if not turns else terminal_outcome
                    break
                user_message, stop_after_turn, suggested_outcome = next_turn
                if suggested_outcome and suggested_outcome != "in_progress":
                    terminal_outcome = suggested_outcome
                ok = await _play_user_turn(user_message)
                if not ok or stop_after_turn:
                    break

        try:
            await self._agent_client.delete_conversation(conv_id)
        except Exception:
            pass

        if not turns:
            return None

        # -- Local mode: store the conversation and return early (no ingest) --
        if self._local_scorer is not None:
            local_trace_id = uuid.uuid4().hex
            self._local_conversations[local_trace_id] = list(request_messages)
            return local_trace_id

        if scenario.generation_mode == "scripted-replay" and turns and len(turns) < len(scenario.messages) and terminal_outcome == "completed":
            terminal_outcome = "blocked"

        # Build one span per message event per conversation turn.
        # Each span maps to exactly one message type so checks can target them
        # individually:
        #   user message    → kind="user",  name="input"
        #   tool call       → kind="tool",  name="tool_call"
        #   tool result     → kind="tool",  name="tool_result:<fn_name>"
        #   final assistant → kind="llm",   name="output"
        # Spans within a turn are siblings parented to the user input span.
        # Turns are chained via the user input span's parent_span_id.
        trace_id = uuid.uuid4().hex
        spans: list[dict[str, Any]] = []
        prev_user_span_id: str | None = None

        for turn_idx, (ts_start, ts_end, user_message, turn_trace_messages) in enumerate(turns, start=1):
            base_attrs = {
                "gen_ai.system": "optimizer",
                "halios.run_tag": run_tag,
                "halios.scenario_id": scenario.id,
                "halios.turn": turn_idx,
                "halios.trace_kind": "synthetic_scenario",
                "halios.synthetic_terminal_outcome": terminal_outcome,
            }

            # Map tool_call_id → function_name for naming tool_result spans.
            tool_name_map: dict[str, str] = {}
            for tm in turn_trace_messages:
                for tc in tm.get("tool_calls") or []:
                    fn_name = (tc.get("function") or {}).get("name") or "tool"
                    call_id = tc.get("id") or ""
                    if call_id:
                        tool_name_map[call_id] = fn_name

            # 1. User input span — anchors the turn.
            user_span_id = uuid.uuid4().hex[:16]
            user_span: dict[str, Any] = {
                "span_id": user_span_id,
                "name": "input",
                "kind": "user",
                "status": "ok",
                "input": {"messages": [user_message]},
                "attributes": {**base_attrs},
                "started_at": ts_start,
                "ended_at": ts_start,
            }
            if prev_user_span_id:
                user_span["parent_span_id"] = prev_user_span_id
            spans.append(user_span)

            # 2. Per-message spans for the agent's response (tool calls, results, final text).
            for tm in turn_trace_messages:
                role = tm.get("role")
                span_id = uuid.uuid4().hex[:16]

                if role == "assistant" and tm.get("tool_calls"):
                    child_span: dict[str, Any] = {
                        "span_id": span_id,
                        "name": "tool_call",
                        "kind": "tool",
                        "status": "ok",
                        "input": {"messages": [tm]},
                        "attributes": {**base_attrs},
                        "started_at": ts_start,
                        "ended_at": ts_end,
                        "parent_span_id": user_span_id,
                    }
                elif role == "tool":
                    fn_name = tool_name_map.get(tm.get("tool_call_id") or "", "tool")
                    child_span = {
                        "span_id": span_id,
                        "name": f"tool_result:{fn_name}",
                        "kind": "tool",
                        "status": "ok",
                        "input": {"messages": [tm]},
                        "attributes": {**base_attrs},
                        "started_at": ts_start,
                        "ended_at": ts_end,
                        "parent_span_id": user_span_id,
                    }
                else:
                    # Final assistant text reply.
                    child_span = {
                        "span_id": span_id,
                        "name": "output",
                        "kind": "llm",
                        "status": "ok",
                        "output": {"content": tm.get("content") or ""},
                        "attributes": {
                            **base_attrs,
                            "halios.synthetic_terminal_outcome": (
                                terminal_outcome if turn_idx == len(turns) else "in_progress"
                            ),
                        },
                        "started_at": ts_start,
                        "ended_at": ts_end,
                        "parent_span_id": user_span_id,
                    }

                spans.append(child_span)

            prev_user_span_id = user_span_id

        try:
            await self._eval_client.ingest_trace(trace_id, spans, run_tag)
            return trace_id
        except Exception as exc:
            logger.warning("ingest_failed", run_tag=run_tag, error=str(exc))
            return None

    async def _simulate_next_user_turn(
        self,
        scenario: Scenario,
        transcript: list[dict[str, str]],
        turn_index: int,
    ) -> tuple[dict[str, str], bool, str] | None:
        """Generate the next user turn for simulation-based scenarios."""
        cfg = self._cfg
        if not isinstance(cfg, (DatasetBuildConfig, OptimizeConfig)):
            raise RuntimeError("Simulation mode requires an optimize or dataset-build config.")

        arc_hint = "\n".join(f"- {message}" for message in scenario.arc_messages) or "- no explicit arc hints"
        goal = scenario.goal or "Work through the scenario to completion."
        persona = scenario.persona or "You are a realistic end user."
        prompt = {
            "role": "user",
            "content": json.dumps(
                {
                    "instructions": (
                        "You are simulating the user side of a real evaluation conversation. "
                        "React to the assistant's latest reply, stay aligned to the scenario goal, "
                        "and return strict JSON with keys: user_message, should_stop, terminal_outcome. "
                        "Keep user_message under 40 words."
                    ),
                    "scenario": {
                        "id": scenario.id,
                        "title": scenario.title,
                        "goal": goal,
                        "persona": persona,
                        "generation_mode": scenario.generation_mode,
                        "arc_hints": scenario.arc_messages,
                        "max_turns": scenario.max_turns,
                    },
                    "turn_index": turn_index,
                    "conversation_so_far": transcript,
                    "required_output": {
                        "user_message": "string",
                        "should_stop": False,
                        "terminal_outcome": "completed|blocked|refused|timed_out|errored|in_progress",
                    },
                }
            ),
        }

        payload = await self._call_llm_json(
            model=cfg.simulation_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate realistic user turns for agent evaluation. "
                        "Return only a compact JSON object. No markdown, no prose.\n"
                        f"Persona: {persona}\nGoal: {goal}\nArc hints:\n{arc_hint}"
                    ),
                },
                prompt,
            ],
            temperature=cfg.simulation_temperature,
            max_tokens=700,
        )

        user_message = str(payload.get("user_message") or "").strip()
        if not user_message:
            return None
        return (
            {"role": "user", "content": user_message},
            bool(payload.get("should_stop")),
            str(payload.get("terminal_outcome") or "in_progress"),
        )

    def _coerce_trace_messages(
        self,
        response: dict[str, Any],
        assistant_text: str,
    ) -> list[dict[str, Any]]:
        raw_trace_messages = response.get("trace_messages")
        if not isinstance(raw_trace_messages, list):
            return [{"role": "assistant", "content": assistant_text}]

        trace_messages = [message for message in raw_trace_messages if isinstance(message, dict)]
        has_visible_assistant = any(
            message.get("role") == "assistant" and not message.get("tool_calls")
            for message in trace_messages
        )
        if assistant_text and not has_visible_assistant:
            trace_messages.append({"role": "assistant", "content": assistant_text})
        return trace_messages

    # ── Mutation suggester ────────────────────────────────────────────────────

    async def _suggest_mutation(
        self,
        baseline_scorecard: dict[str, Any],
        candidate_scorecard: dict[str, Any],
        current_prompt: str,
    ) -> str:
        comparison = _compare_scorecards(baseline_scorecard, candidate_scorecard)
        failing_areas = _build_failing_areas(comparison, self._cfg.t1_check_ids, self._cfg.t1_gate_threshold)
        t1_names = [
            comparison.get("check_deltas", {}).get(cid, {}).get("name", cid)
            for cid in self._cfg.t1_check_ids
        ] or ["(none configured)"]

        prompt_text = _MUTATION_PROMPT_TEMPLATE.format(
            t1_check_names="\n".join(f"  - {n}" for n in t1_names),
            baseline_json=json.dumps(_json_safe(baseline_scorecard), indent=2),
            candidate_json=json.dumps(_json_safe(candidate_scorecard), indent=2),
            failing_areas=failing_areas,
            current_prompt=current_prompt,
        )

        try:
            payload = await self._call_llm_json(
                model=self._cfg.optimizer_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.4,
                max_tokens=6000,
            )
            candidate = _clean_prompt_artifact(payload.get("prompt"), current_prompt)
            rationale = str(payload.get("rationale") or "").strip()
            if rationale:
                logger.info("mutation_rationale", rationale=rationale)
            return candidate
        except Exception as exc:
            logger.warning("mutation_json_failed_falling_back_to_text", error=str(exc))
            content = await self._call_llm_text(
                model=self._cfg.optimizer_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.4,
                max_tokens=6000,
            )
            return _clean_prompt_artifact(content, current_prompt)

    # ── LLM dispatch helpers ──────────────────────────────────────────────────
    # These helpers delegate to the cloud eval client (cloud mode) or call
    # OpenAI directly (local mode) so call sites do not need to branch.

    async def _call_llm_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        if self._eval_client is not None:
            return await self._eval_client.call_llm_json(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return await self._openai_call_json(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _call_llm_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        if self._eval_client is not None:
            return await self._eval_client.call_llm_text(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return await self._openai_call_text(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _openai_call_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for local mode. Install it with: pip install openai"
            ) from exc

        client = openai.AsyncOpenAI()
        # Use json_object response format so OpenAI always returns valid JSON.
        local_model = (
            getattr(self._cfg, "local_llm_model", None) or model
        )
        response = await client.chat.completions.create(
            model=local_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "{}").strip()
        return json.loads(content)

    async def _openai_call_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for local mode. Install it with: pip install openai"
            ) from exc

        client = openai.AsyncOpenAI()
        local_model = (
            getattr(self._cfg, "local_llm_model", None) or model
        )
        response = await client.chat.completions.create(
            model=local_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console
            Console().print(f"[dim]{msg}[/]")
        except ImportError:
            print(msg)
