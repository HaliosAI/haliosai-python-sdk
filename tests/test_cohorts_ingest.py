"""Tests for cohort utilities and bulk ingest helper classes."""

from __future__ import annotations

import pytest

from haliosai.cohorts import CohortValidator
from haliosai.exceptions import CohortTagValidationError
from haliosai.ingest import BulkEvalPusher, BulkIngester
from haliosai.types import CohortDefinition


def _strict_cohort() -> CohortDefinition:
    return CohortDefinition.model_validate(
        {
            "id": "c1",
            "name": "Prod Auto",
            "slug": "prod-auto",
            "type": "system",
            "match": {
                "include_tags_any": ["prod", "auto"],
                "include_tags_all": [],
                "exclude_tags_any": ["test"],
                "agent_ids": [],
                "mode_scope": "all",
            },
            "strict_match": True,
            "auto_stamp": True,
            "tag_allowlist": ["prod", "auto", "spark"],
            "require_tags_all": ["prod"],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )


class TestCohortValidator:
    def test_validate_or_raise_strict_mode(self):
        validator = CohortValidator(_strict_cohort())
        with pytest.raises(CohortTagValidationError) as exc:
            validator.validate_or_raise(["PROD", "spak"])

        reasons = exc.value.reasons
        assert any("did you mean 'spark'" in reason for reason in reasons)

    def test_validate_or_raise_passes_for_allowed_tags(self):
        validator = CohortValidator(_strict_cohort())
        validator.validate_or_raise(["prod", "auto"])


class TestBulkHelpers:
    @pytest.mark.asyncio
    async def test_bulk_ingester_flushes_and_waits(self, mock_transport):
        mock_transport.set_response(
            "POST /api/v1/ingest/traces/bulk",
            {
                "task_id": "task-traces-1",
                "accepted_count": 2,
                "duplicate_count": 0,
                "rejected_count": 0,
                "rejected_reasons": [],
                "run_tag": "run:ci-1",
                "status_url": "/api/v1/tasks/task-traces-1",
            },
            status=202,
        )
        mock_transport.set_response(
            "GET /api/v1/ingest/tasks/task-traces-1",
            {
                "id": "task-traces-1",
                "task_id": "task-traces-1",
                "task_type": "trace_ingest",
                "status": "completed",
                "accepted_count": 2,
                "duplicate_count": 0,
                "rejected_count": 0,
                "errors": [],
            },
        )

        async with BulkIngester(
            mock_transport,
            agent_id="agent-1",
            run_tag="run:ci-1",
            tags=["prod", "auto"],
            batch_size=100,
        ) as ingester:
            ingester.submit_trace(messages=[{"role": "user", "content": "hello"}], external_trace_id="e1")
            ingester.submit_trace(messages=[{"role": "user", "content": "world"}], external_trace_id="e2")

        result = await ingester.wait_for_results(timeout=5)
        assert result.accepted_count == 2
        assert result.rejected_count == 0
        assert result.task_ids == ["task-traces-1"]

    @pytest.mark.asyncio
    async def test_bulk_eval_pusher_batches_and_completes(self, mock_transport):
        mock_transport.set_response(
            "POST /api/v1/ingest/check-executions/bulk",
            {
                "task_id": "task-evals-1",
                "accepted_count": 2,
                "unknown_trace_count": 0,
                "validation_errors": [],
                "status_url": "/api/v1/tasks/task-evals-1",
            },
            status=202,
        )
        mock_transport.set_response(
            "GET /api/v1/ingest/tasks/task-evals-1",
            {
                "id": "task-evals-1",
                "task_id": "task-evals-1",
                "task_type": "check_execution_ingest",
                "status": "completed",
                "accepted_count": 2,
                "duplicate_count": 0,
                "rejected_count": 0,
                "errors": [],
            },
        )

        pusher = BulkEvalPusher(
            mock_transport,
            agent_id="agent-1",
            run_tag="run:spark-1",
            tags=["spark"],
            source="spark",
        )

        task_id = await pusher.push(
            [
                {"trace_id": "t1", "check_slug": "response-quality", "triggered": False, "score": 0.9},
                {"trace_id": "t2", "check_slug": "safety", "triggered": False, "score": 0.95},
            ],
            batch_size=500,
        )
        assert task_id == "task-evals-1"

        result = await pusher.wait_for_completion(timeout=5)
        assert result.accepted_count == 2
        assert result.rejected_count == 0
