"""Tests for local mode — no Halios backend required.

Covers:
- LocalRecorder: filesystem run/iteration persistence
- LocalScorer: LLM-as-judge via openai (mocked)
- OptimizeConfig: mode="local" creation and validation
- OptimizerEngine: one iteration with local_scorer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haliosai.optimize.config import OptimizeConfig, Scenario, ScenarioMessage
from haliosai.optimize.recorder import LocalRecorder
from haliosai.optimize.scorer import LocalScorer


# ---------------------------------------------------------------------------
# LocalRecorder tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_recorder_create_run_returns_id(tmp_path: Path):
    recorder = LocalRecorder(runs_dir=tmp_path / "runs")
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
        run_name="my-run",
    )
    run_id = await recorder.create_run(config=cfg, agent_id="agent-1")

    assert run_id.startswith("local-")
    assert (tmp_path / "runs" / run_id / "run.json").exists()


@pytest.mark.asyncio
async def test_local_recorder_start_run_sets_status(tmp_path: Path):
    recorder = LocalRecorder(runs_dir=tmp_path / "runs")
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
    )
    run_id = await recorder.create_run(config=cfg, agent_id="agent-1")
    await recorder.start_run(run_id)

    run_data = json.loads((tmp_path / "runs" / run_id / "run.json").read_text())
    assert run_data["status"] == "running"
    assert "started_at" in run_data


@pytest.mark.asyncio
async def test_local_recorder_record_iteration_creates_file(tmp_path: Path):
    recorder = LocalRecorder(runs_dir=tmp_path / "runs")
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
    )
    run_id = await recorder.create_run(config=cfg, agent_id="agent-1")
    await recorder.start_run(run_id)

    result = await recorder.record_iteration(
        run_id,
        iteration_number=0,
        verdict="accepted",
        prompt_before="old prompt",
        prompt_after="new prompt",
        scorecard_json={"t1_pass_rate": 1.0},
        scorecard_delta_json={"delta": 0.1},
        t1_gate_passed=True,
        trace_run_tag="run-tag-1",
    )

    iter_dir = tmp_path / "runs" / run_id / "iterations"
    assert iter_dir.is_dir()
    iter_files = list(iter_dir.glob("iter-0-*.json"))
    assert len(iter_files) == 1
    assert result["verdict"] == "accepted"
    assert result["prompt_after"] == "new prompt"


@pytest.mark.asyncio
async def test_local_recorder_complete_run(tmp_path: Path):
    recorder = LocalRecorder(runs_dir=tmp_path / "runs")
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
    )
    run_id = await recorder.create_run(config=cfg, agent_id="agent-1")
    await recorder.start_run(run_id)
    await recorder.complete_run(
        run_id,
        final_prompt="final",
        accepted_iteration_id=None,
    )

    run_data = json.loads((tmp_path / "runs" / run_id / "run.json").read_text())
    assert run_data["status"] == "complete"
    assert run_data["final_prompt"] == "final"


@pytest.mark.asyncio
async def test_local_recorder_fail_run(tmp_path: Path):
    recorder = LocalRecorder(runs_dir=tmp_path / "runs")
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
    )
    run_id = await recorder.create_run(config=cfg, agent_id="agent-1")
    await recorder.start_run(run_id)
    await recorder.fail_run(run_id, error="something went wrong")

    run_data = json.loads((tmp_path / "runs" / run_id / "run.json").read_text())
    assert run_data["status"] == "failed"
    assert "something went wrong" in run_data.get("error_message", "")


@pytest.mark.asyncio
async def test_local_recorder_context_manager(tmp_path: Path):
    """LocalRecorder should work as an async context manager."""
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
    )
    async with LocalRecorder(runs_dir=tmp_path / "runs") as recorder:
        run_id = await recorder.create_run(config=cfg, agent_id="agent-1")
    # After context exit the run file should still be there
    assert (tmp_path / "runs" / run_id / "run.json").exists()


# ---------------------------------------------------------------------------
# LocalScorer tests
# ---------------------------------------------------------------------------


def _make_rubrics_yaml(tmp_path: Path) -> Path:
    rubrics_file = tmp_path / "rubrics.yaml"
    rubrics_file.write_text(
        """
rubrics:
  - id: task_completion
    name: Task Completion
    tier: T1
    weight: 2.0
    prompt: >
      Did the assistant fully complete the user's request?
      Return 1.0 for yes, 0.0 for no.
  - id: tone_quality
    name: Tone & Clarity
    tier: T2
    weight: 1.0
    prompt: >
      Was the response clear and helpful? Score between 0.0 and 1.0.
"""
    )
    return rubrics_file


@pytest.mark.asyncio
async def test_local_scorer_returns_empty_for_missing_rubrics_file(tmp_path: Path):
    scorer = LocalScorer(rubrics_path=tmp_path / "nonexistent.yaml")
    conversations = [[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]]
    result = await scorer.score_conversations(conversations)
    assert result == []


@pytest.mark.asyncio
async def test_local_scorer_returns_empty_for_empty_conversations(tmp_path: Path):
    rubrics_file = _make_rubrics_yaml(tmp_path)
    scorer = LocalScorer(rubrics_path=rubrics_file)

    # Mock openai
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=None)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        result = await scorer.score_conversations([])

    assert result == []


@pytest.mark.asyncio
async def test_local_scorer_scores_conversations(tmp_path: Path):
    rubrics_file = _make_rubrics_yaml(tmp_path)
    scorer = LocalScorer(rubrics_path=rubrics_file, model="gpt-4o-mini", pass_threshold=0.7)

    conversations = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    ]

    # Build a mock openai response for each rubric call
    def make_mock_response(score: float, reason: str) -> MagicMock:
        msg = MagicMock()
        msg.content = json.dumps({"score": score, "reason": reason})
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            make_mock_response(1.0, "Fully correct"),
            make_mock_response(0.9, "Very clear"),
        ]
    )

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        executions = await scorer.score_conversations(conversations)

    # Two rubrics × one conversation = 2 execution dicts
    assert len(executions) == 2
    ids = {e["check_id"] for e in executions}
    assert ids == {"task_completion", "tone_quality"}

    for ex in executions:
        assert "passed" in ex
        assert "score" in ex
        assert "tier" in ex


@pytest.mark.asyncio
async def test_local_scorer_marks_failed_when_below_threshold(tmp_path: Path):
    rubrics_file = _make_rubrics_yaml(tmp_path)
    scorer = LocalScorer(rubrics_path=rubrics_file, pass_threshold=0.8)

    conversations = [
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    ]

    def make_mock_response(score: float) -> MagicMock:
        msg = MagicMock()
        msg.content = json.dumps({"score": score, "reason": "test"})
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            make_mock_response(0.5),  # below 0.8 → failed
            make_mock_response(0.9),  # above 0.8 → passed
        ]
    )

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        executions = await scorer.score_conversations(conversations)

    scores = {e["check_id"]: e["passed"] for e in executions}
    assert scores["task_completion"] is False
    assert scores["tone_quality"] is True


# ---------------------------------------------------------------------------
# OptimizeConfig local mode tests
# ---------------------------------------------------------------------------


def test_optimize_config_local_mode_no_api_key_required():
    """Local mode should work without an API key."""
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
    )
    assert cfg.mode == "local"
    assert cfg.halios_api_key == ""


def test_optimize_config_local_mode_from_yaml(tmp_path: Path):
    config_path = tmp_path / "optimize.yaml"
    config_path.write_text(
        """
target_url: "http://localhost:8080"
agent_id: "agent-1"
mode: local
local_llm_model: "gpt-4o-mini"
run_name: "local-test-run"
"""
    )
    cfg = OptimizeConfig.from_yaml(config_path)
    assert cfg.mode == "local"
    assert cfg.local_llm_model == "gpt-4o-mini"
    assert cfg.run_name == "local-test-run"


def test_optimize_config_cloud_mode_default():
    """Cloud mode is the default."""
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="hal_test",
    )
    assert cfg.mode == "cloud"


def test_optimize_config_to_api_config_includes_mode():
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="hal_test",
        mode="local",
    )
    api_cfg = cfg.to_api_config()
    assert api_cfg["mode"] == "local"


def test_optimize_config_local_mode_with_inline_scenario():
    """Local mode supports inline scenarios."""
    cfg = OptimizeConfig(
        target_url="http://localhost:8080",
        agent_id="agent-1",
        halios_api_key="",
        mode="local",
        scenarios=[
            Scenario(
                id="s1",
                messages=[
                    ScenarioMessage(role="user", content="Hello"),
                    ScenarioMessage(role="assistant", content="Hi there"),
                ],
            )
        ],
    )
    assert len(cfg.scenarios) == 1
    assert cfg.scenarios[0].id == "s1"


# ---------------------------------------------------------------------------
# LocalScorer rubric loading from "checks:" key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_scorer_accepts_checks_key(tmp_path: Path):
    """LocalScorer should also accept a `checks:` YAML key (alias for rubrics)."""
    rubrics_file = tmp_path / "checks.yaml"
    rubrics_file.write_text(
        """
checks:
  - id: accuracy
    name: Accuracy
    tier: T1
    weight: 1.0
    prompt: "Is the answer accurate? Score 0.0-1.0."
"""
    )
    scorer = LocalScorer(rubrics_path=rubrics_file)

    def make_mock_response() -> MagicMock:
        msg = MagicMock()
        msg.content = json.dumps({"score": 0.95, "reason": "accurate"})
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_response())

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        executions = await scorer.score_conversations(
            [[{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]]
        )

    assert len(executions) == 1
    assert executions[0]["check_id"] == "accuracy"
    assert executions[0]["passed"] is True


# ---------------------------------------------------------------------------
# LocalScorer handles malformed LLM JSON gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_scorer_handles_malformed_json_response(tmp_path: Path):
    rubrics_file = _make_rubrics_yaml(tmp_path)
    scorer = LocalScorer(rubrics_path=rubrics_file)

    def make_bad_response() -> MagicMock:
        msg = MagicMock()
        msg.content = "not json at all"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[make_bad_response(), make_bad_response()]
    )

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        executions = await scorer.score_conversations(
            [[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]]
        )

    # Should still return execution rows — score defaults to 0.0
    assert len(executions) == 2
    for ex in executions:
        assert ex["score"] == 0.0
        assert ex["passed"] is False
