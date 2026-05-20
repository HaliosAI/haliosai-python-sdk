"""Cohort APIs and client-side validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable

from ._transport import HaliosTransport
from .exceptions import CohortTagValidationError
from .types import CohortDefinition, CohortValidateResult


@dataclass
class CohortValidator:
    """Validator built from a specific cohort definition."""

    cohort: CohortDefinition

    def suggest_correction(self, misspelled_tag: str) -> str | None:
        allowlist = [t for t in (self.cohort.tag_allowlist or []) if isinstance(t, str)]
        if not allowlist:
            return None
        matches = get_close_matches(misspelled_tag, allowlist, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def validate_or_raise(self, tags: list[str]) -> None:
        normalized = [str(t).strip().lower() for t in tags if str(t).strip()]

        reasons: list[str] = []
        required = [str(t).strip().lower() for t in (self.cohort.require_tags_all or []) if str(t).strip()]
        if required:
            missing = [tag for tag in required if tag not in normalized]
            if missing:
                reasons.append(f"Missing required tags: {', '.join(missing)}")

        allowlist = [str(t).strip().lower() for t in (self.cohort.tag_allowlist or []) if str(t).strip()]
        if allowlist:
            disallowed = [tag for tag in normalized if tag not in allowlist]
            for bad in disallowed:
                suggestion = self.suggest_correction(bad)
                if suggestion:
                    reasons.append(f"Tag '{bad}' is not allowed (did you mean '{suggestion}'?)")
                else:
                    reasons.append(f"Tag '{bad}' is not allowed")

        if self.cohort.strict_match and reasons:
            raise CohortTagValidationError("Cohort tag validation failed", reasons=reasons)


class CohortCollection(list[CohortDefinition]):
    """Convenient collection wrapper for cohort lookups."""

    def get_by_slug(self, slug: str) -> CohortDefinition:
        for cohort in self:
            if cohort.slug == slug:
                return cohort
        raise KeyError(f"Cohort '{slug}' not found")


async def _get_cohorts(
    transport: HaliosTransport,
    *,
    agent_id: str | None = None,
) -> CohortCollection:
    params = {"agent_id": agent_id} if agent_id else None
    resp = await transport.request("GET", f"{transport.api_prefix}/cohorts", params=params)
    rows = resp.json().get("cohorts", [])
    return CohortCollection([CohortDefinition.model_validate(item) for item in rows])


async def _validate_cohort_tags(
    transport: HaliosTransport,
    *,
    tags: Iterable[str],
    cohort_slug: str | None = None,
    mode: str | None = None,
    agent_id: str | None = None,
) -> CohortValidateResult:
    payload = {
        "tags": [str(t) for t in tags],
        "cohort_slug": cohort_slug,
        "mode": mode,
        "agent_id": agent_id,
    }
    resp = await transport.request("POST", f"{transport.api_prefix}/cohorts/validate", json=payload)
    return CohortValidateResult.model_validate(resp.json())
