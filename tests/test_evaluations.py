"""Tests for evaluations module."""

from __future__ import annotations

import pytest

from haliosai.evaluations import EvalRun, _list_check_executions, _trigger_eval
from haliosai.types import CheckExecutionListResult, CheckExecutionResponse


class TestTriggerEval:
    @pytest.mark.asyncio
    async def test_trigger_eval_returns_eval_run(self, mock_transport):
        run = await _trigger_eval(
            mock_transport,
            agent_id="test-agent",
            tags=["nightly"],
        )
        assert isinstance(run, EvalRun)
        assert run.task_id == "task-abc"
        assert run.run_tag == "run:test-abc"
        assert run.status == "completed"
        assert run.trace_count == 5
        assert run.check_count == 10

    @pytest.mark.asyncio
    async def test_trigger_eval_with_trace_ids(self, mock_transport):
        await _trigger_eval(
            mock_transport,
            agent_id="test-agent",
            trace_ids=["t1", "t2"],
            check_ids=["c1"],
            tags=["ci"],
            mode="evaluator",
        )
        call = mock_transport._calls[-1]
        assert call["json"]["trace_ids"] == ["t1", "t2"]
        assert call["json"]["check_ids"] == ["c1"]
        assert call["json"]["tags"] == ["ci"]

    @pytest.mark.asyncio
    async def test_trigger_eval_with_dataset(self, mock_transport):
        await _trigger_eval(
            mock_transport,
            agent_id="test-agent",
            dataset_id="dataset-1",
            dataset_version=3,
            run_name="nightly dataset",
            run_comment="CI",
        )
        call = mock_transport._calls[-1]
        assert call["json"]["dataset_id"] == "dataset-1"
        assert call["json"]["dataset_version"] == 3
        assert call["json"]["run_name"] == "nightly dataset"
        assert call["json"]["run_comment"] == "CI"


class TestListCheckExecutions:
    @pytest.mark.asyncio
    async def test_list_returns_paginated(self, mock_transport):
        page = await _list_check_executions(
            mock_transport,
            agent_id="test-agent",
        )
        assert isinstance(page, CheckExecutionListResult)
        assert len(page.data) == 1
        assert isinstance(page.data[0], CheckExecutionResponse)
        assert page.data[0].check_name == "toxicity"
        assert page.data[0].task_name == "Safety"
        assert page.data[0].task_slug == "safety"
        assert page.has_more is False

    @pytest.mark.asyncio
    async def test_list_with_filters(self, mock_transport):
        await _list_check_executions(
            mock_transport,
            agent_id="test-agent",
            tags=["nightly"],
            triggered=True,
            mode="evaluator",
            limit=10,
            include_progress=True,
        )
        call = mock_transport._calls[-1]
        assert call["params"]["tags"] == ["nightly"]
        assert call["params"]["triggered"] == "true"
        assert call["params"]["mode"] == "evaluator"
        assert call["params"]["limit"] == 10
        assert call["params"]["include_progress"] == "true"
        # Check endpoint uses agent-scoped URL
        assert "/agents/test-agent/check-executions" in call["path"]


class TestEvalRun:
    @pytest.mark.asyncio
    async def test_succeeded_property(self, mock_transport):
        run = EvalRun(
            transport=mock_transport,
            agent_id="test-agent",
            task_id="task-1",
            status="completed",
            trace_count=1,
            check_count=1,
        )
        assert run.succeeded is True
        assert run.is_done is True

    @pytest.mark.asyncio
    async def test_wait_already_done(self, mock_transport):
        run = EvalRun(
            transport=mock_transport,
            agent_id="test-agent",
            task_id="task-1",
            status="completed",
            trace_count=1,
            check_count=1,
        )
        result = await run.wait(timeout=1)
        assert result is run

    @pytest.mark.asyncio
    async def test_results_streaming(self, mock_transport):
        run = EvalRun(
            transport=mock_transport,
            agent_id="test-agent",
            task_id="task-1",
            status="completed",
            trace_count=1,
            check_count=1,
            tags=["nightly"],
        )
        # results() is now a generator
        results = []
        async for result in run.results():
            results.append(result)
        assert len(results) >= 1
        # Check that progress is attached
        assert hasattr(results[0], "_progress")

    @pytest.mark.asyncio
    async def test_results_streaming_reads_all_pages_when_progress_exceeds_check_count(
        self,
        mock_transport,
    ):
        def paged_response(method, path, json_body, params):
            cursor = (params or {}).get("cursor")
            if cursor is None:
                return 200, {
                    "data": [
                        {
                            "id": "3",
                            "trace_id": "trace-003",
                            "span_id": "span-003",
                            "check_name": "check",
                            "mode": "evaluator",
                            "tags": ["run:test"],
                            "passed": True,
                        }
                    ],
                    "next_cursor": "3",
                    "has_more": True,
                    "progress": {"total": 3, "completed": 3, "pending": 0},
                }
            return 200, {
                "data": [
                    {
                        "id": "2",
                        "trace_id": "trace-002",
                        "span_id": "span-002",
                        "check_name": "check",
                        "mode": "evaluator",
                        "tags": ["run:test"],
                        "passed": False,
                    },
                    {
                        "id": "1",
                        "trace_id": "trace-001",
                        "span_id": "span-001",
                        "check_name": "check",
                        "mode": "evaluator",
                        "tags": ["run:test"],
                        "passed": True,
                    },
                ],
                "next_cursor": None,
                "has_more": False,
                "progress": {"total": 3, "completed": 3, "pending": 0},
            }

        mock_transport.set_response("GET /api/v1/agents/", paged_response)
        run = EvalRun(
            transport=mock_transport,
            agent_id="test-agent",
            task_id="task-1",
            status="completed",
            trace_count=1,
            check_count=1,
            tags=["run:test"],
        )

        results = []
        async for result in run.results(timeout=1):
            results.append(result)

        assert [result.id for result in results] == ["3", "2", "1"]
        assert mock_transport._calls[-1]["params"]["cursor"] == "3"

    @pytest.mark.asyncio
    async def test_results_streaming_waits_for_materialized_results(self, mock_transport):
        calls = {"count": 0}

        def delayed_response(method, path, json_body, params):
            calls["count"] += 1
            if calls["count"] == 1:
                return 200, {
                    "data": [],
                    "next_cursor": None,
                    "has_more": False,
                    "progress": {"total": 0, "completed": 0, "pending": 0},
                }
            return 200, {
                "data": [
                    {
                        "id": "1",
                        "trace_id": "trace-001",
                        "span_id": "span-001",
                        "check_name": "check",
                        "mode": "evaluator",
                        "tags": ["run:test"],
                        "passed": True,
                    }
                ],
                "next_cursor": None,
                "has_more": False,
                "progress": {"total": 1, "completed": 1, "pending": 0},
            }

        mock_transport.set_response("GET /api/v1/agents/", delayed_response)
        run = EvalRun(
            transport=mock_transport,
            agent_id="test-agent",
            task_id="task-1",
            status="completed",
            trace_count=1,
            check_count=1,
            tags=["run:test"],
        )

        results = []
        async for result in run.results(timeout=1, poll_interval=0.01):
            results.append(result)

        assert [result.id for result in results] == ["1"]
        assert calls["count"] == 2

    def test_repr(self, mock_transport):
        run = EvalRun(
            transport=mock_transport,
            agent_id="a",
            task_id="t",
            status="pending",
            trace_count=5,
            check_count=10,
        )
        assert "task_id='t'" in repr(run)
        assert "traces=5" in repr(run)
