"""Scenario generation client helpers for dataset-first evaluation flows."""

from __future__ import annotations

import re
from typing import Any

import httpx

from .config import ScenarioHintsConfig


def normalize_target_base_url(target_url: str) -> str:
    cleaned = target_url.strip()
    if not cleaned:
        return cleaned
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = f"http://{cleaned}"
    cleaned = cleaned.rstrip("/")
    for suffix in ("/halios/config", "/halios/chat"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.rstrip("/")


async def generate_scenarios(cfg: ScenarioHintsConfig) -> list[dict[str, Any]]:
    """Ask the Halios backend to generate scenarios using server-side LLM routing."""
    async with httpx.AsyncClient(
        base_url=cfg.halios_api_url.rstrip("/"),
        headers={"Authorization": f"Bearer {cfg.halios_api_key}"},
        timeout=90.0,
    ) as client:
        response = await client.post(
            "/api/v1/scenarios/generate",
            json={
                "agent_id": cfg.agent_id,
                "target_url": normalize_target_base_url(cfg.target_url),
                "scenario_count": cfg.scenario_count,
                "hints": cfg.hints,
                "generation_mode": cfg.generation_mode,
                "max_turns": cfg.max_turns,
            },
        )
        response.raise_for_status()
        payload = response.json()
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list):
        raise RuntimeError("Halios did not return a scenarios list")
    return [item for item in scenarios if isinstance(item, dict)]


def build_scenario_set_payload(
    cfg: ScenarioHintsConfig,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": 1,
        "generation_mode": cfg.generation_mode,
        "max_turns": cfg.max_turns,
        "scenarios": [
            {
                "id": item["id"],
                "title": item.get("title"),
                "goal": item["goal"],
                "persona": item.get("persona"),
                "generation_mode": item.get("generation_mode") or cfg.generation_mode,
                "arc_messages": item.get("arc_messages") or [],
                "max_turns": item.get("max_turns") or cfg.max_turns,
            }
            for item in scenarios
        ],
    }


# Backwards-compatible name. The payload is now intentionally a pure scenario
# fixture rather than a dataset-build config with credentials and IDs.
build_dataset_config_payload = build_scenario_set_payload
