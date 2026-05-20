from __future__ import annotations

import textwrap

from haliosai.optimize.config import (
    DatasetBuildConfig,
    OptimizeConfig,
    ScenarioHintsConfig,
    ScenarioSetConfig,
)


def test_optimize_config_allows_scenario_fixture_without_dataset_id(tmp_path):
    config_path = tmp_path / "optimize.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            target_url: "http://localhost:8080"
            halios_api_url: "http://localhost:8000"
            halios_api_key: "hlk_test"
            agent_id: "agent-123"
            check_ids: ["check-1"]
            """
        )
    )

    cfg = OptimizeConfig.from_yaml(config_path)

    assert cfg.dataset_id is None
    assert cfg.check_ids == ["check-1"]


def test_optimize_config_parses_dataset_snapshot(tmp_path):
    config_path = tmp_path / "optimize.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            target_url: "http://localhost:8080"
            halios_api_url: "http://localhost:8000"
            halios_api_key: "hlk_test"
            agent_id: "agent-123"
            dataset_id: "dataset-abc"
            dataset_version: 3
            check_ids: ["check-1"]
            t1_check_ids: ["check-1"]
            """
        )
    )

    cfg = OptimizeConfig.from_yaml(config_path)

    assert cfg.dataset_id == "dataset-abc"
    assert cfg.dataset_version == 3
    assert cfg.to_api_config()["dataset_version"] == 3
    assert cfg.to_api_config()["simulation_model"] == "gemini/gemini-2.5-flash"


def test_optimize_config_maps_legacy_openai_starter_model_to_halios_route(tmp_path):
    config_path = tmp_path / "optimize.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            target_url: "http://localhost:8080"
            halios_api_url: "http://localhost:8000"
            halios_api_key: "hlk_test"
            agent_id: "agent-123"
            optimizer_model: "gpt-4o-mini"
            scenarios:
              - id: "hello"
                messages:
                  - role: "user"
                    content: "Hello"
            """
        )
    )

    cfg = OptimizeConfig.from_yaml(config_path)

    assert cfg.optimizer_model == "fast"


def test_dataset_build_config_supports_inline_scenarios(tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            target_url: "http://localhost:8080"
            halios_api_url: "http://localhost:8000"
            halios_api_key: "hlk_test"
            agent_id: "agent-123"
            dataset_name: "test-dataset"
            scenario_source: "inline"
            scenarios:
              - id: "hello"
                messages:
                  - role: "user"
                    content: "Hello"
            """
        )
    )

    cfg = DatasetBuildConfig.from_yaml(config_path)

    assert cfg.dataset_name == "test-dataset"
    assert cfg.scenario_source == "inline"
    assert cfg.scenarios[0].id == "hello"


def test_scenario_set_config_contains_only_scenarios(tmp_path):
    config_path = tmp_path / "scenarios.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            version: 1
            generation_mode: "simulation-with-arc-hint"
            max_turns: 5
            scenarios:
              - id: "find_item"
                title: "Find item"
                goal: "Find a chair"
                persona: "Careful shopper"
            """
        )
    )

    cfg = ScenarioSetConfig.from_yaml(config_path)

    assert cfg.version == 1
    assert cfg.scenarios[0].id == "find_item"
    assert cfg.scenarios[0].max_turns == 5


def test_scenario_hints_config_parses_hints(tmp_path):
    config_path = tmp_path / "scenario-hints.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            target_url: "http://localhost:8080"
            halios_api_url: "http://localhost:8000"
            halios_api_key: "hlk_test"
            agent_id: "agent-123"
            dataset_name: "generated"
            scenario_count: 6
            hints:
              - "booking"
              - "refund policy"
            """
        )
    )

    cfg = ScenarioHintsConfig.from_yaml(config_path)

    assert cfg.dataset_name == "generated"
    assert cfg.scenario_count == 6
    assert cfg.hints == ["booking", "refund policy"]


def test_scenario_hints_config_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("HALIOS_TARGET_URL", "http://localhost:8100")
    monkeypatch.setenv("HALIOS_API_URL", "http://localhost:8000")
    monkeypatch.setenv("HALIOS_API_KEY", "hlk_env")
    monkeypatch.setenv("HALIOS_AGENT_ID", "agent-env")

    config_path = tmp_path / "scenario-hints.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            target_url: "${HALIOS_TARGET_URL}"
            halios_api_url: "$HALIOS_API_URL"
            halios_api_key: "$HALIOS_API_KEY"
            agent_id: "${HALIOS_AGENT_ID}"
            scenario_count: 2
            """
        )
    )

    cfg = ScenarioHintsConfig.from_yaml(config_path)

    assert cfg.target_url == "http://localhost:8100"
    assert cfg.halios_api_url == "http://localhost:8000"
    assert cfg.halios_api_key == "hlk_env"
    assert cfg.agent_id == "agent-env"


def test_scenario_hints_config_uses_base_url_alias_for_unresolved_api_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HALIOS_BASE_URL", "localhost:8000")
    monkeypatch.setenv("HALIOS_TARGET_URL", "http://localhost:8100")
    monkeypatch.setenv("HALIOS_API_KEY", "hlk_env")
    monkeypatch.setenv("HALIOS_AGENT_ID", "agent-env")

    config_path = tmp_path / "scenario-hints.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            target_url: "${HALIOS_TARGET_URL}"
            halios_api_url: "${HALIOS_API_URL}"
            halios_api_key: "$HALIOS_API_KEY"
            agent_id: "${HALIOS_AGENT_ID}"
            """
        )
    )

    cfg = ScenarioHintsConfig.from_yaml(config_path)

    assert cfg.halios_api_url == "http://localhost:8000"
