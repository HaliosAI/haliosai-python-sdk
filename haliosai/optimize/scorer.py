"""LocalScorer — LLM-as-judge scorer that runs entirely locally via OpenAI.

Reads rubric definitions from a YAML file and scores optimizer conversations
without any Halios backend connection.  Returns execution dicts in the same
shape expected by :func:`~haliosai.optimize.engine._build_scorecard`.

Rubrics YAML format::

    rubrics:
      - id: task_completion
        name: Task Completion
        tier: T1
        weight: 2.0
        prompt: >
          Did the assistant fully complete what the user asked?
          Return 1.0 for full completion, 0.0 for complete failure.

      - id: tone_quality
        name: Tone & Clarity
        tier: T2
        weight: 1.0
        prompt: >
          Was the assistant response clear, professional, and appropriately
          helpful?  Score between 0.0 and 1.0.

Each rubric entry is converted into a check execution row with keys:
  check_id, check_name, passed, score, tier

Example::

    scorer = LocalScorer(
        rubrics_path=".halios/rubrics.yaml",
        model="gpt-4o-mini",
        pass_threshold=0.7,
    )
    executions = await scorer.score_conversations(conversations)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalScorer:
    """Scores optimizer conversations locally using OpenAI as an LLM judge.

    Parameters
    ----------
    rubrics_path:
        Path to the YAML file containing rubric definitions.
        Defaults to ``.halios/rubrics.yaml``.
    model:
        OpenAI model to use for scoring (e.g. ``"gpt-4o-mini"``).
    pass_threshold:
        Minimum score (0–1) for a rubric to be considered passing.
        Defaults to ``0.7``.
    """

    def __init__(
        self,
        rubrics_path: str | Path = ".halios/rubrics.yaml",
        model: str = "gpt-4o-mini",
        pass_threshold: float = 0.7,
    ) -> None:
        self._rubrics_path = Path(rubrics_path)
        self._model = model
        self._pass_threshold = pass_threshold
        self._rubrics: list[dict[str, Any]] | None = None

    # ── Rubric loading ────────────────────────────────────────────────────────

    def _load_rubrics(self) -> list[dict[str, Any]]:
        """Load and cache rubrics from the YAML file.

        Falls back to an empty list when the file does not exist or PyYAML is
        not installed so the rest of the optimizer loop can still run (it will
        produce a zero-score scorecard which immediately triggers the discard
        path — a safe default).
        """
        if self._rubrics is not None:
            return self._rubrics

        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            self._rubrics = []
            return self._rubrics

        try:
            with open(self._rubrics_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            self._rubrics = []
            return self._rubrics

        # Accept both ``rubrics:`` and ``checks:`` as the top-level key so
        # users can reuse an existing checks file with minimal edits.
        rubrics = data.get("rubrics") or data.get("checks") or []
        self._rubrics = [r for r in rubrics if isinstance(r, dict)]
        return self._rubrics

    # ── Public API ────────────────────────────────────────────────────────────

    async def score_conversations(
        self,
        conversations: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Score a list of conversations against every rubric.

        Each element of *conversations* is a list of message dicts
        (``{"role": "user"|"assistant", "content": "..."}``).

        Returns a flat list of execution dicts that can be fed directly into
        :func:`~haliosai.optimize.engine._build_scorecard`.
        """
        rubrics = self._load_rubrics()
        if not rubrics:
            return []

        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for local mode. "
                "Install it with: pip install openai"
            ) from exc

        client = openai.AsyncOpenAI()
        executions: list[dict[str, Any]] = []

        for conv in conversations:
            if not conv:
                continue
            conv_text = self._format_conversation(conv)
            for rubric in rubrics:
                execution = await self._score_one(client, conv_text, rubric)
                executions.append(execution)

        return executions

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _format_conversation(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "unknown").upper()
            content = str(msg.get("content") or "")
            # Truncate very long messages to avoid blowing the context window.
            if len(content) > 2000:
                content = content[:2000] + "…"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _score_one(
        self,
        client: Any,
        conv_text: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        rubric_id = str(rubric.get("id") or "check")
        rubric_name = str(rubric.get("name") or rubric_id)
        rubric_prompt = str(
            rubric.get("prompt")
            or "Evaluate the quality of the assistant's response. Score between 0.0 and 1.0."
        )
        tier = str(rubric.get("tier") or "T2")

        system_content = (
            f"{rubric_prompt}\n\n"
            "Return ONLY a JSON object with exactly two keys:\n"
            "  score  — a float between 0.0 and 1.0\n"
            "  reason — one short sentence explaining the score\n"
            "No markdown, no code fences, no extra keys."
        )

        score = 0.0
        for attempt in range(1, 4):
            try:
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {
                            "role": "user",
                            "content": f"Conversation to evaluate:\n\n{conv_text}",
                        },
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                content = (response.choices[0].message.content or "{}").strip()
                parsed = json.loads(content)
                score = float(parsed.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                break
            except (json.JSONDecodeError, ValueError):
                score = 0.0
                break
            except Exception:  # noqa: BLE001 — retry on transient API errors
                if attempt == 3:
                    score = 0.0

        return {
            "check_id": rubric_id,
            "check_name": rubric_name,
            "passed": score >= self._pass_threshold,
            "score": score,
            "tier": tier,
        }
