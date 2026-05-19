"""DatasetBuilder — simulate scenarios and create Halios datasets."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import DatasetBuildConfig, Scenario
from .protocol import HaliosOptimizableClient

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:200]


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset_id: str
    attempted: int
    added: int
    failed: list[tuple[str, str]]


class _HaliosDatasetClient:
    """HTTP client for dataset build operations."""

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

    async def ingest_trace(self, trace_id: str, spans: list[dict[str, Any]], run_tag: str) -> None:
        payload = {
            "agent_id": self._agent_id,
            "run_tag": run_tag,
            "tags": [f"run:{run_tag}"],
            "source": "dataset_build",
            "traces": [{"trace_id": trace_id, "spans": spans}],
        }
        resp = await self._request("POST", "/api/v1/ingest/traces/bulk", json_body=payload)
        data = resp.json()
        status_url = data.get("status_url")
        if data.get("status") == "completed" or not status_url:
            return

        deadline = datetime.now(timezone.utc).timestamp() + 120.0
        while datetime.now(timezone.utc).timestamp() < deadline:
            await asyncio.sleep(2.0)
            poll_resp = await self._request("GET", status_url)
            state = poll_resp.json().get("status", "")
            if state in ("completed", "failed"):
                return

    async def create_dataset(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        provenance: str = "synthetic",
    ) -> str:
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

    async def find_dataset_by_name(self, agent_id: str, name: str) -> dict[str, Any] | None:
        resp = await self._request(
            "GET",
            "/api/v1/datasets",
            params={"agent_id": agent_id, "limit": 200, "offset": 0},
        )
        target_slug = _slugify(name)
        for item in resp.json().get("items", []):
            if item.get("slug") == target_slug or item.get("name") == name:
                return item
        return None

    async def add_traces_to_dataset(
        self,
        dataset_id: str,
        trace_ids: list[str],
        provenance: str = "synthetic",
        scenario_id: str | None = None,
    ) -> None:
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
                    "llm_json_retryable_error status=%s attempt=%s model=%s error=%s",
                    status,
                    attempt,
                    model,
                    exc,
                )
                await asyncio.sleep(0.5 * attempt)
        if resp is None:
            raise RuntimeError("LLM JSON call failed without a response")
        data = resp.json().get("data")
        return data if isinstance(data, dict) else {}


class DatasetBuilder:
    """Build a Halios dataset from scripted or simulated scenarios."""

    def __init__(self, config: DatasetBuildConfig, verbose: bool = True) -> None:
        self._cfg = config
        self._verbose = verbose
        self._server_scenario_ids: set[str] = set()
        self._agent_client = HaliosOptimizableClient(config.target_url)
        self._halios_client = _HaliosDatasetClient(
            api_url=config.halios_api_url,
            api_key=config.halios_api_key,
            agent_id=config.agent_id,
        )

    async def build(
        self,
        name: str | None = None,
        description: str | None = None,
        scenario_ids: list[str] | None = None,
    ) -> DatasetBuildResult:
        scenario_map = await self._resolve_scenarios()
        if scenario_ids:
            requested = set(scenario_ids)
            scenario_map = {
                scenario_id: scenario
                for scenario_id, scenario in scenario_map.items()
                if scenario_id in requested
            }
        scenarios = list(scenario_map.values())
        if not scenarios:
            raise RuntimeError(
                "No dataset-build scenarios resolved. Add inline scenarios, create them in Halios, "
                "or check the --scenario-id filter."
            )

        target_dataset_id = self._cfg.dataset_id
        target_name = name or self._cfg.dataset_name
        target_description = description if description is not None else self._cfg.dataset_description
        provenance = self._cfg.provenance

        if target_dataset_id:
            dataset_id = target_dataset_id
            self._log(f"Appending to dataset {dataset_id} with {len(scenarios)} scenario(s)...")
        else:
            if not target_name:
                raise RuntimeError(
                    "Dataset name is required when creating a new dataset. Set dataset_name or pass --name."
                )
            try:
                dataset_id = await self._halios_client.create_dataset(
                    agent_id=self._cfg.agent_id,
                    name=target_name,
                    description=target_description or f"Synthetic dataset - {len(scenarios)} scenario(s)",
                    provenance=provenance,
                )
                self._log(f"Created dataset {dataset_id}; building {len(scenarios)} scenario(s)...")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 409:
                    raise
                existing = await self._halios_client.find_dataset_by_name(
                    self._cfg.agent_id,
                    target_name,
                )
                if not existing or not existing.get("id"):
                    raise RuntimeError(
                        f"Dataset named '{target_name}' already exists, but its ID could not be resolved. "
                        "Set dataset_id in the config or pass --name with a unique name."
                    ) from exc
                dataset_id = str(existing["id"])
                self._log(
                    f"Dataset '{target_name}' already exists; appending to {dataset_id} "
                    f"with {len(scenarios)} scenario(s)..."
                )

        added = 0
        failed: list[tuple[str, str]] = []
        for idx, scenario in enumerate(scenarios, start=1):
            self._log(f"  [{idx}/{len(scenarios)}] scenario: {scenario.id}")
            run_tag = f"dataset-build:{dataset_id}:{idx}:{uuid.uuid4().hex[:8]}"
            try:
                trace_id = await self._run_single_scenario(scenario, run_tag)
                if not trace_id:
                    failed.append((scenario.id, "no trace generated"))
                    continue
                await self._halios_client.add_traces_to_dataset(
                    dataset_id=dataset_id,
                    trace_ids=[trace_id],
                    provenance=provenance,
                    scenario_id=scenario.id if scenario.id in self._server_scenario_ids else None,
                )
                added += 1
            except Exception as exc:
                failed.append((scenario.id, str(exc)))
                logger.warning("dataset_build_scenario_failed scenario_id=%s error=%s", scenario.id, exc)
                continue

        if added == 0:
            failure_summary = "; ".join(f"{scenario_id}: {error}" for scenario_id, error in failed[:5])
            raise RuntimeError(
                "No traces were added. "
                f"Failures: {failure_summary or 'unknown'}"
            )

        self._log(f"  Dataset ready: {dataset_id} ({added}/{len(scenarios)} scenario traces added)")
        if failed:
            self._log("  Failed scenario IDs for retry:")
            for scenario_id, error in failed:
                self._log(f"    - {scenario_id}: {error}")
        return DatasetBuildResult(
            dataset_id=dataset_id,
            attempted=len(scenarios),
            added=added,
            failed=failed,
        )

    async def _resolve_scenarios(self) -> dict[str, Scenario]:
        scenarios: dict[str, Scenario] = {}

        if self._cfg.scenario_source in {"halios", "mixed"}:
            limit = self._cfg.max_scenarios or 500
            for raw in await self._halios_client.list_scenarios(
                self._cfg.agent_id,
                scenario_ids=self._cfg.scenario_ids or None,
                limit=limit,
            ):
                scenario = self._scenario_from_api(raw)
                if scenario is not None:
                    scenarios[scenario.id] = scenario
                    self._server_scenario_ids.add(scenario.id)

        if self._cfg.scenario_source in {"inline", "mixed"}:
            for scenario in self._cfg.scenarios:
                scenarios[scenario.id] = scenario
                self._server_scenario_ids.discard(scenario.id)

        if self._cfg.max_scenarios is not None:
            limited_ids = list(scenarios.keys())[: self._cfg.max_scenarios]
            scenarios = {scenario_id: scenarios[scenario_id] for scenario_id in limited_ids}

        return scenarios

    def _scenario_from_api(self, raw: dict[str, Any]) -> Scenario | None:
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
                        "generation_mode": "scripted-replay"
                        if generation_mode == "scripted-replay"
                        else generation_mode,
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
                    "generation_mode": generation_mode
                    if generation_mode in {"simulation", "simulation-with-arc-hint"}
                    else "simulation-with-arc-hint",
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
                    "generation_mode": generation_mode
                    if generation_mode in {"simulation", "simulation-with-arc-hint"}
                    else "simulation",
                    "max_turns": max_turns,
                }
            )

        logger.warning("scenario_unusable_for_dataset_build scenario_id=%s", scenario_id)
        return None

    async def _run_single_scenario(self, scenario: Scenario, run_tag: str) -> str | None:
        conv_id = str(uuid.uuid4())
        request_messages: list[dict[str, str]] = []
        turns: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
        terminal_outcome = "completed"
        user_turn_count = 0

        async def play_user_turn(user_message: dict[str, Any]) -> bool:
            nonlocal terminal_outcome, user_turn_count
            request_messages.append(user_message)
            turn_started_at = datetime.now(timezone.utc).isoformat()
            try:
                response = await self._agent_client.chat(
                    conversation_id=conv_id,
                    messages=request_messages[:],
                    prompt_override=None,
                )
            except Exception as exc:
                logger.warning("scenario_chat_error scenario_id=%s error=%s", scenario.id, exc)
                terminal_outcome = "errored"
                return False

            turn_ended_at = datetime.now(timezone.utc).isoformat()
            assistant_text = response.get("response", "")
            turn_trace_messages = self._coerce_trace_messages(response, assistant_text)
            request_messages.append({"role": "assistant", "content": assistant_text})
            turns.append((turn_started_at, turn_ended_at, user_message, turn_trace_messages))
            user_turn_count += 1
            return True

        if scenario.generation_mode == "scripted-replay":
            for msg in scenario.messages:
                if not await play_user_turn({"role": msg.role, "content": msg.content}):
                    break
        else:
            while user_turn_count < scenario.max_turns:
                next_turn = await self._simulate_next_user_turn(
                    scenario,
                    request_messages,
                    user_turn_count + 1,
                )
                if next_turn is None:
                    terminal_outcome = "timed_out" if not turns else terminal_outcome
                    break
                user_message, stop_after_turn, suggested_outcome = next_turn
                if suggested_outcome and suggested_outcome != "in_progress":
                    terminal_outcome = suggested_outcome
                ok = await play_user_turn(user_message)
                if not ok or stop_after_turn:
                    break

        try:
            await self._agent_client.delete_conversation(conv_id)
        except Exception:
            pass

        if not turns:
            return None

        if (
            scenario.generation_mode == "scripted-replay"
            and len(turns) < len(scenario.messages)
            and terminal_outcome == "completed"
        ):
            terminal_outcome = "blocked"

        trace_id = uuid.uuid4().hex
        spans = self._build_spans(trace_id, scenario, turns, terminal_outcome, run_tag)
        try:
            await self._halios_client.ingest_trace(trace_id, spans, run_tag)
            return trace_id
        except Exception as exc:
            logger.warning("ingest_failed run_tag=%s error=%s", run_tag, exc)
            return None

    async def _simulate_next_user_turn(
        self,
        scenario: Scenario,
        transcript: list[dict[str, str]],
        turn_index: int,
    ) -> tuple[dict[str, str], bool, str] | None:
        arc_hint = "\n".join(f"- {message}" for message in scenario.arc_messages) or "- no explicit arc hints"
        goal = scenario.goal or "Work through the scenario to completion."
        persona = scenario.persona or "You are a realistic end user."
        payload = await self._halios_client.call_llm_json(
            model=self._cfg.simulation_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate realistic user turns for agent evaluation. "
                        f"Persona: {persona}\nGoal: {goal}\nArc hints:\n{arc_hint}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instructions": (
                                "You are simulating the user side of a real evaluation conversation. "
                                "React to the assistant's latest reply, stay aligned to the scenario goal, "
                                "and return strict compact JSON with keys: user_message, should_stop, "
                                "terminal_outcome. Keep user_message under 40 words."
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
                },
            ],
            temperature=self._cfg.simulation_temperature,
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

    def _build_spans(
        self,
        trace_id: str,
        scenario: Scenario,
        turns: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]],
        terminal_outcome: str,
        run_tag: str,
    ) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        prev_user_span_id: str | None = None

        for turn_idx, (ts_start, ts_end, user_message, turn_trace_messages) in enumerate(turns, start=1):
            base_attrs = {
                "gen_ai.system": "dataset_builder",
                "halios.run_tag": run_tag,
                "halios.scenario_id": scenario.id,
                "halios.turn": turn_idx,
                "halios.trace_kind": "synthetic_scenario",
                "halios.synthetic_terminal_outcome": terminal_outcome,
            }
            tool_name_map: dict[str, str] = {}
            for trace_message in turn_trace_messages:
                for tool_call in trace_message.get("tool_calls") or []:
                    fn_name = (tool_call.get("function") or {}).get("name") or "tool"
                    call_id = tool_call.get("id") or ""
                    if call_id:
                        tool_name_map[call_id] = fn_name

            user_span_id = uuid.uuid4().hex[:16]
            user_span: dict[str, Any] = {
                "span_id": user_span_id,
                "name": "input",
                "kind": "user",
                "status": "ok",
                "input": {"messages": [user_message]},
                "attributes": {**base_attrs, "halios.trace_id": trace_id},
                "started_at": ts_start,
                "ended_at": ts_start,
            }
            if prev_user_span_id:
                user_span["parent_span_id"] = prev_user_span_id
            spans.append(user_span)

            for trace_message in turn_trace_messages:
                role = trace_message.get("role")
                span_id = uuid.uuid4().hex[:16]
                if role == "assistant" and trace_message.get("tool_calls"):
                    child_span = {
                        "span_id": span_id,
                        "name": "tool_call",
                        "kind": "tool",
                        "status": "ok",
                        "input": {"messages": [trace_message]},
                        "attributes": base_attrs,
                        "started_at": ts_start,
                        "ended_at": ts_end,
                        "parent_span_id": user_span_id,
                    }
                elif role == "tool":
                    fn_name = tool_name_map.get(trace_message.get("tool_call_id") or "", "tool")
                    child_span = {
                        "span_id": span_id,
                        "name": f"tool_result:{fn_name}",
                        "kind": "tool",
                        "status": "ok",
                        "input": {"messages": [trace_message]},
                        "attributes": base_attrs,
                        "started_at": ts_start,
                        "ended_at": ts_end,
                        "parent_span_id": user_span_id,
                    }
                else:
                    child_span = {
                        "span_id": span_id,
                        "name": "output",
                        "kind": "llm",
                        "status": "ok",
                        "output": {"content": trace_message.get("content") or ""},
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

        return spans

    def _coerce_trace_messages(self, response: dict[str, Any], assistant_text: str) -> list[dict[str, Any]]:
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

    def _log(self, msg: str) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console

            Console().print(f"[dim]{msg}[/]")
        except ImportError:
            print(msg)
