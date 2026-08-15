from __future__ import annotations

import pytest

from haliosai.client import HaliosClient

REPORT = {
    "run_id": "run-1",
    "run_tag": "run:sdk-eval",
    "status": "completed",
    "attempted_trial_count": 2,
    "completed_trial_count": 2,
    "evaluated_trial_count": 2,
    "telemetry_incomplete_count": 0,
    "check_execution_error_count": 0,
    "pass_at_k": 1.0,
    "threshold": 0.9,
    "protected_failure": False,
    "gate_passed": True,
    "trials": [],
}


@pytest.mark.asyncio
async def test_evaluate_existing_traces_uses_immutable_run_api(client: HaliosClient) -> None:
    client._transport.responses["POST /api/v1/runs/evaluations"] = {"run_id": "run-1"}
    client._transport.responses["GET /api/v1/runs/evaluations/run-1"] = REPORT

    report = await client.evaluate_traces(["1" * 32, "2" * 32], fail_below=0.9)

    payload = client._transport.calls[0]["json"]
    assert payload["trace_ids"] == ["1" * 32, "2" * 32]
    assert payload["gate"] == {"fail_below": 0.9}
    assert report.gate_passed is True
