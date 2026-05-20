"""Spark adapter for dataframe-oriented Halios evaluation workflows."""

from __future__ import annotations

from typing import Any

from ..client import HaliosClient
from ..ingest import BulkEvalPusher


class HaliosSparkEvaluator:
    """Spark-native helper for evaluating rows and pushing results.

    This adapter keeps Spark-specific imports local to method scope so SDK users
    without pyspark installed can still import the package.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        api_key: str,
        base_url: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ):
        self.client = HaliosClient(agent_id=agent_id, api_key=api_key, base_url=base_url)
        self.run_metadata = run_metadata or {}

    def evaluate_dataframe(
        self,
        df,
        *,
        messages_col: str,
        external_id_col: str,
        check_slugs: list[str],
        output_col: str = "halios_results",
    ):
        """Adds an output column containing a minimal check-execution payload.

        This version is intentionally lightweight and computes placeholders in Spark
        while delegating final check execution persistence to ``push_results``.
        """
        import importlib

        functions = importlib.import_module("pyspark.sql.functions")
        col = functions.col
        lit = functions.lit
        struct = functions.struct
        to_json = functions.to_json

        return df.withColumn(
            output_col,
            to_json(
                struct(
                    col(external_id_col).alias("external_trace_id"),
                    lit(check_slugs[0] if check_slugs else "response-quality").alias("check_slug"),
                    lit(False).alias("triggered"),
                    lit(1.0).alias("score"),
                    lit("Spark adapter placeholder result").alias("reasoning"),
                )
            ),
        )

    def push_results(self, results_df, *, output_col: str = "halios_results"):
        """Collects Spark output payloads and pushes them via bulk ingest."""
        rows = [row[output_col] for row in results_df.select(output_col).collect()]

        parsed_rows = []
        import json

        for row in rows:
            if not row:
                continue
            item = json.loads(row)
            parsed_rows.append(
                {
                    "external_trace_id": item.get("external_trace_id"),
                    "check_slug": item.get("check_slug", "response-quality"),
                    "triggered": bool(item.get("triggered", False)),
                    "score": float(item.get("score", 1.0)),
                    "reasoning": item.get("reasoning", "Spark evaluation"),
                }
            )

        source = self.run_metadata.get("source", "spark")
        run_tag = self.run_metadata.get("run_tag")
        tags = self.run_metadata.get("tags", ["spark"])

        import asyncio

        async def _push():
            pusher = BulkEvalPusher(
                self.client._transport,
                agent_id=self.client._resolve_agent(),
                run_tag=run_tag,
                tags=tags,
                source=source,
            )
            await pusher.push(parsed_rows, batch_size=500)
            return await pusher.wait_for_completion(timeout=600)

        return asyncio.run(_push())
