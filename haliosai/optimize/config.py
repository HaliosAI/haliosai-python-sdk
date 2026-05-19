"""YAML-loadable configuration models for optimization and dataset generation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _load_local_env(path: str | Path) -> None:
    """Load .env files for CLI configs when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    config_path = Path(path)
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / ".halios" / ".env",
        config_path.parent / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            return os.environ.get(name, match.group(0))

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _load_yaml(path: str | Path) -> dict[str, Any]:
    _load_local_env(path)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load Halios YAML configs. "
            "Install it with: pip install pyyaml"
        ) from exc

    with open(path) as fh:
        return _expand_env_vars(yaml.safe_load(fh) or {})


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _is_missing_or_unresolved(value: str | None) -> bool:
    return not value or bool(_ENV_PATTERN.search(value))


def _normalize_api_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    if cleaned and not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = f"http://{cleaned}"
    return cleaned


class ScenarioMessage(BaseModel):
    role: str
    content: str


class Scenario(BaseModel):
    id: str
    messages: list[ScenarioMessage] = Field(default_factory=list)
    generation_mode: Literal["scripted-replay", "simulation", "simulation-with-arc-hint"] = "scripted-replay"
    title: str | None = None
    goal: str | None = None
    persona: str | None = None
    arc_messages: list[str] = Field(default_factory=list)
    max_turns: int = Field(6, ge=1, le=20)

    @model_validator(mode="after")
    def _validate_shape(self) -> "Scenario":
        if self.generation_mode == "scripted-replay":
            if not self.messages:
                raise ValueError("scripted-replay scenarios require messages")
            return self

        if not self.goal and not self.arc_messages and not self.messages:
            raise ValueError(
                "simulation scenarios require at least one of: goal, arc_messages, or seed messages"
            )
        return self


class OptimizeConfig(BaseModel):
    """Configuration for a scenario-fixture optimization run."""

    target_url: str = Field(
        default_factory=lambda: os.getenv("HALIOS_TARGET_URL", ""),
        description="Base URL of the HaliosOptimizable agent",
    )
    halios_api_url: str = Field(
        default_factory=lambda: _env_first("HALIOS_BASE_URL", "HALIOS_API_URL", default="https://app.halios.ai"),
        description="Halios API base URL",
    )
    halios_api_key: str = Field(
        default_factory=lambda: os.getenv("HALIOS_API_KEY", ""),
        description="Halios API key (hal_...)",
    )
    agent_id: str = Field(
        default_factory=lambda: os.getenv("HALIOS_AGENT_ID", ""),
        description="Halios agent ID",
    )

    run_name: str = Field("optimize-run", description="Human-readable run name")
    starting_prompt: str = Field("", description="Initial prompt (fetched from agent if empty)")

    check_ids: list[str] = Field(default_factory=list, description="Check UUIDs to run each iteration")
    t1_check_ids: list[str] = Field(default_factory=list, description="Hard-gate check IDs")
    t1_gate_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Minimum T1 pass-rate")

    max_iterations: int = Field(5, ge=1, le=50)
    max_consecutive_discards: int = Field(3, ge=1, le=20)
    optimizer_model: str = Field("fast", description="Halios LLM route or LiteLLM model for prompt mutations")
    simulation_model: str = Field(
        "gemini/gemini-2.5-flash",
        description="Model name routed by Halios backend for simulated-user turns",
    )
    simulation_temperature: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for simulated-user turns",
    )

    dataset_id: str | None = Field(None, description="Optional baseline/reference dataset ID")
    dataset_version: int | None = Field(None, ge=1, description="Optional dataset snapshot version")

    scenarios: list[Scenario] = Field(default_factory=list, description="Inline scenario fixture specs")

    mode: Literal["cloud", "local"] = Field(
        "cloud",
        description="Execution mode: 'cloud' uses the Halios backend; 'local' uses OpenAI directly (no Halios account needed)",
    )
    local_llm_model: str = Field(
        "gpt-4o-mini",
        description="OpenAI model used for mutation and simulation in local mode",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OptimizeConfig":
        return cls(**_load_yaml(path))

    @model_validator(mode="after")
    def _normalize_legacy_optimizer_model(self) -> "OptimizeConfig":
        # Early starter configs used bare OpenAI model names. In the current
        # gateway model, "fast" means "use the org-configured Halios route".
        if self.optimizer_model in {"gpt-4o-mini", "gpt-4o"}:
            self.optimizer_model = "fast"
        return self

    def to_api_config(self) -> dict[str, Any]:
        return {
            "target_url": self.target_url,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "scenario_count": len(self.scenarios),
            "check_ids": self.check_ids,
            "t1_check_ids": self.t1_check_ids,
            "max_iterations": self.max_iterations,
            "max_consecutive_discards": self.max_consecutive_discards,
            "optimizer_model": self.optimizer_model,
            "simulation_model": self.simulation_model,
            "simulation_temperature": self.simulation_temperature,
            "mode": self.mode,
        }


class DatasetBuildConfig(BaseModel):
    """Configuration for bulk dataset generation / augmentation."""

    target_url: str = Field(..., description="Base URL of the HaliosOptimizable agent")
    halios_api_url: str = Field(
        default_factory=lambda: _env_first("HALIOS_BASE_URL", "HALIOS_API_URL", default="https://app.halios.ai"),
        description="Halios API base URL",
    )
    halios_api_key: str = Field(..., description="Halios API key (hal_...)")
    agent_id: str = Field(..., description="Halios agent ID")

    dataset_name: str | None = Field(None, description="Name for a new dataset")
    dataset_description: str = Field("", description="Description for a new dataset")
    dataset_id: str | None = Field(None, description="Existing dataset ID to append to")
    dataset_version: int | None = Field(None, ge=1, description="Optional version snapshot for source reads")
    provenance: str = Field("synthetic", description="Trace provenance tag for dataset membership")
    simulation_model: str = Field("gemini/gemini-2.5-flash", description="Model name routed by Halios backend for simulated-user turns")
    simulation_temperature: float = Field(0.4, ge=0.0, le=1.0, description="Sampling temperature for simulated-user turns")

    scenario_source: Literal["halios", "inline", "mixed"] = Field(
        "halios",
        description="Where dataset-build scenarios come from",
    )
    scenario_ids: list[str] = Field(default_factory=list, description="Optional scenario IDs to fetch from Halios")
    max_scenarios: int | None = Field(None, ge=1, description="Optional cap on number of scenarios to execute")
    scenarios: list[Scenario] = Field(default_factory=list, description="Optional inline scenario overrides")

    @model_validator(mode="after")
    def _validate_source(self) -> "DatasetBuildConfig":
        if self.scenario_source == "inline" and not self.scenarios:
            raise ValueError("scenario_source=inline requires at least one inline scenario")
        if self.scenario_source == "mixed" and not (self.scenarios or self.scenario_ids):
            raise ValueError("scenario_source=mixed requires scenario_ids and/or inline scenarios")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetBuildConfig":
        return cls(**_load_yaml(path))


class ScenarioSetConfig(BaseModel):
    """Reusable scenario fixture file.

    This file intentionally contains no Halios credentials, agent URL, agent ID,
    or dataset ID. Those belong to command config, environment, or CLI flags.
    """

    version: int = 1
    generation_mode: Literal["simulation", "simulation-with-arc-hint"] = "simulation-with-arc-hint"
    max_turns: int = Field(6, ge=1, le=20)
    scenarios: list[Scenario] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults_to_scenarios(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        generation_mode = data.get("generation_mode")
        max_turns = data.get("max_turns")
        scenarios = data.get("scenarios")
        if isinstance(scenarios, list):
            normalized = []
            for item in scenarios:
                if isinstance(item, dict):
                    item = dict(item)
                    if generation_mode and "generation_mode" not in item:
                        item["generation_mode"] = generation_mode
                    if max_turns and "max_turns" not in item:
                        item["max_turns"] = max_turns
                normalized.append(item)
            data = {**data, "scenarios": normalized}
        return data

    @model_validator(mode="after")
    def _require_scenarios(self) -> "ScenarioSetConfig":
        if not self.scenarios:
            raise ValueError("scenario fixture must contain at least one scenario")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioSetConfig":
        data = _load_yaml(path)
        raw_scenarios = data.get("scenarios")
        if raw_scenarios is None and isinstance(data, list):
            data = {"scenarios": data}
        return cls(**data)


class ScenarioHintsConfig(BaseModel):
    """User hints for backend-managed scenario generation."""

    target_url: str = Field(
        default_factory=lambda: os.getenv("HALIOS_TARGET_URL", ""),
        description="Base URL of the HaliosOptimizable agent",
    )
    halios_api_url: str = Field(
        default_factory=lambda: _env_first("HALIOS_BASE_URL", "HALIOS_API_URL", default="https://app.halios.ai"),
        description="Halios API base URL",
    )
    halios_api_key: str = Field(
        default_factory=lambda: os.getenv("HALIOS_API_KEY", ""),
        description="Halios API key (hal_...)",
    )
    agent_id: str = Field(
        default_factory=lambda: os.getenv("HALIOS_AGENT_ID", ""),
        description="Halios agent ID",
    )

    dataset_name: str = Field(
        "Generated scenario dataset",
        description="Dataset name for output config",
    )
    dataset_description: str = Field(
        "Synthetic dataset generated from scenario coverage goals",
        description="Dataset description for output config",
    )

    scenario_count: int = Field(12, ge=1, le=100)
    generation_mode: Literal["simulation", "simulation-with-arc-hint"] = "simulation-with-arc-hint"
    max_turns: int = Field(6, ge=1, le=20)
    hints: list[str] = Field(
        default_factory=list,
        description="Natural-language guidance, coverage goals, or examples.",
    )

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "ScenarioHintsConfig":
        if _is_missing_or_unresolved(self.halios_api_url):
            self.halios_api_url = _env_first(
                "HALIOS_BASE_URL",
                "HALIOS_API_URL",
                default="https://app.halios.ai",
            )
        self.halios_api_url = _normalize_api_base_url(self.halios_api_url)

        if _is_missing_or_unresolved(self.target_url):
            raise ValueError("target_url is required. Set it in YAML or HALIOS_TARGET_URL.")
        if _is_missing_or_unresolved(self.agent_id):
            raise ValueError("agent_id is required. Set it in YAML or HALIOS_AGENT_ID.")
        if _is_missing_or_unresolved(self.halios_api_key):
            raise ValueError("halios_api_key is required. Set it in YAML or HALIOS_API_KEY.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioHintsConfig":
        return cls(**_load_yaml(path))


# Backwards-compatible import name for older callers.
ScenarioGenerationConfig = ScenarioHintsConfig
