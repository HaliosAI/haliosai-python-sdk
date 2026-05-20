"""OpenAI Agents framework integration for HaliosAI V2.

Provides :class:`HaliosInputGuardrail` and :class:`HaliosOutputGuardrail`
that plug directly into the OpenAI Agents ``Agent`` definition.

Usage::

    from haliosai.openai_agents import HaliosInputGuardrail, HaliosOutputGuardrail
    from agents import Agent

    agent = Agent(
        name="my_agent",
        instructions="You are a helpful assistant",
        input_guardrails=[HaliosInputGuardrail(agent_id="your-agent")],
        output_guardrails=[HaliosOutputGuardrail(agent_id="your-agent")],
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any, Union

logger = logging.getLogger("haliosai")

try:
    from agents.guardrail import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
    from agents.run_context import RunContextWrapper

    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False

    # Minimal stubs so the module can be imported without the agents package
    class GuardrailFunctionOutput:  # type: ignore[no-redef]
        def __init__(self, output_info: Any, tripwire_triggered: bool):
            self.output_info = output_info
            self.tripwire_triggered = tripwire_triggered

    class InputGuardrail:  # type: ignore[no-redef]
        def __init__(self, guardrail_function, name=None):
            self.guardrail_function = guardrail_function
            self.name = name

    class OutputGuardrail:  # type: ignore[no-redef]
        def __init__(self, guardrail_function, name=None):
            self.guardrail_function = guardrail_function
            self.name = name

    class RunContextWrapper:  # type: ignore[no-redef]
        pass


def _require_agents() -> None:
    if not _AGENTS_AVAILABLE:
        raise ImportError(
            "The OpenAI Agents framework is required for this integration. "
            "Install it with: pip install 'haliosai[agents]'"
        )


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------


class HaliosInputGuardrail(InputGuardrail):  # type: ignore[misc]
    """Evaluate user input through HaliosAI guardrails before the agent runs.

    If a guardrail is triggered the tripwire fires and the Agents framework
    will halt execution.

    Args:
        agent_id: HaliosAI agent slug or UUID.
        api_key: API key (falls back to ``HALIOS_API_KEY``).
        base_url: API base URL (falls back to ``HALIOS_BASE_URL``).
        name: Display name for tracing (auto-generated if omitted).
        tags: Tags to attach to every guardrail evaluation.
        fail_open: If ``True`` (default), allow the request when guardrail
            evaluation itself errors. Set to ``False`` for strict mode.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        fail_open: bool = True,
    ):
        _require_agents()

        self.agent_id = agent_id
        self._api_key = api_key
        self._base_url = base_url
        self._display_name = name or f"halios_input_{agent_id}"
        self._tags = tags
        self._fail_open = fail_open

        super().__init__(
            guardrail_function=self._evaluate_input,
            name=self._display_name,
        )

    def _make_client(self):
        """Lazy-import and construct a HaliosClient."""
        from .client import HaliosClient

        return HaliosClient(
            agent_id=self.agent_id,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    async def _evaluate_input(
        self,
        context: Any,
        agent: Any,
        input_data: Any,
    ) -> GuardrailFunctionOutput:
        try:
            input_text = _extract_input_text(input_data)
            messages = [{"role": "user", "content": input_text}]

            client = self._make_client()
            try:
                result = await client.evaluate(messages=messages, tags=self._tags)
            finally:
                await client.close()

            triggered = result.triggered and result.action == "block"

            if triggered:
                logger.warning("Input blocked by guardrail for agent %s", self.agent_id)

            return GuardrailFunctionOutput(
                output_info={
                    "guardrail_type": "halios_input",
                    "agent_id": self.agent_id,
                    "triggered": triggered,
                    "violations": [v.model_dump() for v in result.violations] if triggered else [],
                },
                tripwire_triggered=triggered,
            )

        except Exception as exc:
            logger.error("Input guardrail error for agent %s: %s", self.agent_id, exc)
            if not self._fail_open:
                return GuardrailFunctionOutput(
                    output_info={"error": str(exc), "triggered": True},
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(
                output_info={"error": str(exc), "triggered": False},
                tripwire_triggered=False,
            )


# ---------------------------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------------------------


class HaliosOutputGuardrail(OutputGuardrail):  # type: ignore[misc]
    """Evaluate agent output through HaliosAI guardrails after generation.

    If a guardrail is triggered the tripwire fires and the Agents framework
    will raise an error.

    Args:
        agent_id: HaliosAI agent slug or UUID.
        api_key: API key (falls back to ``HALIOS_API_KEY``).
        base_url: API base URL (falls back to ``HALIOS_BASE_URL``).
        name: Display name for tracing (auto-generated if omitted).
        tags: Tags to attach to every guardrail evaluation.
        fail_open: If ``True`` (default), allow the response when guardrail
            evaluation itself errors.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        fail_open: bool = True,
    ):
        _require_agents()

        self.agent_id = agent_id
        self._api_key = api_key
        self._base_url = base_url
        self._display_name = name or f"halios_output_{agent_id}"
        self._tags = tags
        self._fail_open = fail_open

        super().__init__(
            guardrail_function=self._evaluate_output,
            name=self._display_name,
        )

    def _make_client(self):
        from .client import HaliosClient

        return HaliosClient(
            agent_id=self.agent_id,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    async def _evaluate_output(
        self,
        context: Any,
        agent: Any,
        agent_output: Any,
    ) -> GuardrailFunctionOutput:
        try:
            output_text = _extract_output_text(agent_output)
            messages = [
                {"role": "user", "content": "(previous conversation)"},
                {"role": "assistant", "content": output_text},
            ]

            client = self._make_client()
            try:
                result = await client.evaluate(messages=messages, tags=self._tags)
            finally:
                await client.close()

            triggered = result.triggered and result.action == "block"

            if triggered:
                logger.warning("Output blocked by guardrail for agent %s", self.agent_id)

            return GuardrailFunctionOutput(
                output_info={
                    "guardrail_type": "halios_output",
                    "agent_id": self.agent_id,
                    "triggered": triggered,
                    "violations": [v.model_dump() for v in result.violations] if triggered else [],
                },
                tripwire_triggered=triggered,
            )

        except Exception as exc:
            logger.error("Output guardrail error for agent %s: %s", self.agent_id, exc)
            if not self._fail_open:
                return GuardrailFunctionOutput(
                    output_info={"error": str(exc), "triggered": True},
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(
                output_info={"error": str(exc), "triggered": False},
                tripwire_triggered=False,
            )


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_input_text(input_data: Any) -> str:
    """Convert various input formats to a plain text string."""
    if isinstance(input_data, str):
        return input_data

    if isinstance(input_data, list):
        parts: list[str] = []
        for item in input_data:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "content") and isinstance(item.content, str):
                parts.append(item.content)
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
            elif isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(input_data)


def _extract_output_text(output: Any) -> str:
    """Convert various output formats to a plain text string."""
    if isinstance(output, str):
        return output
    if hasattr(output, "content"):
        return str(output.content)
    if hasattr(output, "text"):
        return str(output.text)
    if isinstance(output, dict):
        return str(output.get("content", output))
    return str(output)
