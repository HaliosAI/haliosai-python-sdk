"""Start and wait for an immutable evaluation run over existing traces."""

import asyncio
import os

import haliosai


async def release_gate(trace_ids: list[str]) -> None:
    async with haliosai.Client(agent_id=os.getenv("HALIOS_AGENT_ID", "support-agent")) as client:
        run = await client.evaluate_traces(trace_ids, run_name="release", fail_below=0.95)
        report = await client.wait_for_evaluation_run(run.run_id)
        if not report.gate_passed:
            raise RuntimeError(f"Halios gate failed with pass@k={report.pass_at_k:.3f}")


if __name__ == "__main__":
    trace_id = os.getenv("HALIOS_SAMPLE_TRACE_ID", "0123456789abcdef0123456789abcdef")
    if os.getenv("HALIOS_API_KEY"):
        asyncio.run(release_gate([trace_id]))
    else:
        print("Set HALIOS_API_KEY to execute evaluation against Halios cloud.")
