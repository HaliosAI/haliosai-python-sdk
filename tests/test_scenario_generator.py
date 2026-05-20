from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from haliosai.optimize.config import ScenarioHintsConfig
from haliosai.optimize.scenario_generator import (
    build_scenario_set_payload,
    generate_scenarios,
    normalize_target_base_url,
)


def test_normalizes_target_url():
    assert normalize_target_base_url("localhost:8001/halios/config") == "http://localhost:8001"
    assert normalize_target_base_url("https://agent.test/halios/chat") == "https://agent.test"


@pytest.mark.asyncio
@respx.mock
async def test_generate_scenarios_delegates_to_halios_backend():
    route = respx.post("http://localhost:8000/api/v1/scenarios/generate").mock(
        return_value=Response(
            200,
            json={
                "scenarios": [
                    {
                        "id": "booking_success",
                        "title": "Booking success",
                        "goal": "Book a trip with all required information.",
                        "persona": "Friendly traveler",
                        "generation_mode": "simulation-with-arc-hint",
                        "arc_messages": ["Ask to book the trip.", "Provide dates when asked."],
                        "max_turns": 6,
                    }
                ]
            },
        )
    )
    cfg = ScenarioHintsConfig(
        target_url="localhost:8001/halios/config",
        halios_api_url="http://localhost:8000",
        halios_api_key="hlk_test",
        agent_id="agent-123",
        scenario_count=3,
        hints=["cover booking happy paths"],
    )

    scenarios = await generate_scenarios(cfg)

    assert scenarios[0]["id"] == "booking_success"
    assert route.called
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer hlk_test"
    posted = json.loads(request.content)
    assert posted["target_url"] == "http://localhost:8001"
    assert posted["hints"] == ["cover booking happy paths"]


@pytest.mark.asyncio
@respx.mock
async def test_generate_scenarios_posts_hints_and_normalized_target_url():
    route = respx.post("http://localhost:8000/api/v1/scenarios/generate").mock(
        return_value=Response(200, json={"scenarios": [{"id": "s1", "goal": "Complete task"}]})
    )
    cfg = ScenarioHintsConfig(
        target_url="localhost:8001/halios/config",
        halios_api_url="http://localhost:8000",
        halios_api_key="hlk_test",
        agent_id="agent-123",
        hints=["exercise missing info"],
    )

    await generate_scenarios(cfg)

    posted = json.loads(route.calls.last.request.content)
    assert posted["target_url"] == "http://localhost:8001"
    assert posted["hints"] == ["exercise missing info"]


def test_build_scenario_set_payload_writes_only_fixture_scenarios():
    cfg = ScenarioHintsConfig(
        target_url="localhost:8001/halios/config",
        halios_api_url="http://localhost:8000",
        halios_api_key="hlk_test",
        agent_id="agent-123",
        dataset_name="generated",
    )
    scenarios = [
        {
            "id": "booking_success",
            "title": "Booking success",
            "goal": "Book a trip with all required information.",
            "persona": "Friendly traveler",
            "arc_messages": ["Ask to book the trip.", "Provide dates when asked."],
        }
    ]

    payload = build_scenario_set_payload(cfg, scenarios)

    assert "target_url" not in payload
    assert "halios_api_key" not in payload
    assert "agent_id" not in payload
    assert "dataset_id" not in payload
    assert payload["version"] == 1
    assert payload["scenarios"][0]["goal"] == "Book a trip with all required information."
