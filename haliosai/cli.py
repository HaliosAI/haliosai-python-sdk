"""Halios CLI.

Entry point: ``halios`` (configured in pyproject.toml).

Subcommands
-----------
halios optimize run    — Run an optimization loop from a YAML config file
halios optimize status — Show the status of a run
halios optimize cancel — Cancel a running run
halios optimize list   — List runs for an agent
halios optimize init   — Generate a starter optimizer YAML config
halios dataset init    — Generate a starter dataset-build YAML config
halios dataset build   — Simulate scenarios and create/augment a dataset
halios evaluate dataset — Trigger evaluator checks for a dataset snapshot
halios scenario init   — Generate a starter scenario-hints YAML config
halios scenario generate — Generate a scenarios.yaml dataset-build config
"""

from __future__ import annotations

import asyncio
import os

import typer

# Top-level app
app = typer.Typer(
    name="halios",
    help="Halios AI CLI — guardrails, evaluations, and prompt optimization.",
    no_args_is_help=True,
    add_completion=False,
)

# ── `optimize` sub-group ──────────────────────────────────────────────────────
optimize_app = typer.Typer(
    name="optimize",
    help="Prompt optimization commands.",
    no_args_is_help=True,
)
app.add_typer(optimize_app, name="optimize")

# ── `dataset` sub-group ───────────────────────────────────────────────────────
dataset_app = typer.Typer(
    name="dataset",
    help="Dataset management commands.",
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")

# ── `evaluate` sub-group ──────────────────────────────────────────────────────
evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluation trigger commands.",
    no_args_is_help=True,
)
app.add_typer(evaluate_app, name="evaluate")

# ── `scenario` sub-group ─────────────────────────────────────────────────────
scenario_app = typer.Typer(
    name="scenario",
    help="Scenario generation commands.",
    no_args_is_help=True,
)
app.add_typer(scenario_app, name="scenario")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_rich() -> None:
    try:
        import rich  # noqa: F401
    except ImportError:
        typer.echo(
            "Rich is required for CLI output. Install it with:\n"
            "  pip install haliosai[cli]",
            err=True,
        )
        raise typer.Exit(1)


def _env_or(env_var: str, override: str | None) -> str | None:
    return override or os.environ.get(env_var)


def _require_env(env_var: str, override: str | None, flag_name: str) -> str:
    value = _env_or(env_var, override)
    if not value:
        typer.echo(
            f"Missing {flag_name}. Set ${env_var} or pass --{flag_name.replace('_', '-')}.",
            err=True,
        )
        raise typer.Exit(1)
    return value


def _resolve_halios_base_url(override: str | None = None) -> str | None:
    value = override or os.environ.get("HALIOS_BASE_URL") or os.environ.get("HALIOS_API_URL")
    return _normalize_cli_url(value) if value else None


def _require_halios_base_url(override: str | None = None) -> str:
    value = _resolve_halios_base_url(override)
    if not value:
        typer.echo(
            "Missing base URL. Set $HALIOS_BASE_URL or pass --api-url. "
            "$HALIOS_API_URL is still accepted as a legacy alias.",
            err=True,
        )
        raise typer.Exit(1)
    return value


def _resolve_halios_credentials(
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    env_file: str | None = None,
) -> tuple[str, str]:
    _load_cli_env(env_file)
    return (
        _require_halios_base_url(api_url),
        _require_env("HALIOS_API_KEY", api_key, "api_key"),
    )


def _normalize_run_id(value: str | None) -> str:
    run_id = (value or "").strip()
    if run_id.startswith("run_id="):
        run_id = run_id.split("=", 1)[1].strip()
    if not run_id:
        typer.echo("Missing run ID. Pass it as the RUN_ID argument.", err=True)
        raise typer.Exit(1)
    return run_id


def _load_cli_env(env_file: str | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    from pathlib import Path

    for candidate in (Path.cwd() / ".env", Path.cwd() / ".halios" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
    if env_file:
        load_dotenv(env_file, override=True)


def _normalize_cli_url(url: str) -> str:
    if "://" in url:
        return url.rstrip("/")
    return f"http://{url.rstrip('/')}"


# ---------------------------------------------------------------------------
# `halios optimize run`
# ---------------------------------------------------------------------------

@optimize_app.command("run")
def run_optimize(
    config_path: str = typer.Option(".halios/optimize.yaml", "--config", "-c", help="Path to optimize.yaml"),
    scenarios_path: str | None = typer.Option(
        None,
        "--scenarios",
        "-s",
        help="Path to scenarios.yaml fixture. Overrides inline scenarios in optimize.yaml.",
    ),
    agent_id: str | None = typer.Option(None, "--agent-id", help="Halios agent ID override."),
    target_url: str | None = typer.Option(None, "--target-url", help="Optimizable agent URL override."),
    dataset_id: str | None = typer.Option(None, "--dataset-id", help="Optional baseline/reference dataset ID override."),
    dataset_version: int | None = typer.Option(None, "--dataset-version", help="Optional dataset snapshot version override."),
    max_iterations: int | None = typer.Option(None, "--max-iterations", help="Maximum optimizer iterations override."),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL override."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key override."),
    env_file: str | None = typer.Option(None, "--env-file", help="Extra env file to load after default .env files."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print scorecard tables."),
    local: bool = typer.Option(
        False,
        "--local/--cloud",
        help=(
            "Run in local mode: score conversations with an LLM judge directly "
            "via OpenAI (no Halios account needed). "
            "Reads rubrics from .halios/rubrics.yaml. "
            "Results are saved to .halios/runs/."
        ),
    ),
) -> None:
    """Run a prompt optimization loop from a YAML config file."""
    _require_rich()
    _load_cli_env(env_file)

    from rich.console import Console
    console = Console()

    from haliosai.optimize import (
        OptimizeConfig,
        OptimizeRecorder,
        OptimizerEngine,
        ScenarioSetConfig,
    )

    console.print(f"[bold]Loading config:[/] {config_path}")
    try:
        cfg = OptimizeConfig.from_yaml(config_path)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/] {config_path}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/] {exc}")
        raise typer.Exit(1)

    update: dict[str, object] = {}
    resolved_agent = agent_id or os.environ.get("HALIOS_AGENT_ID")
    resolved_target = target_url or os.environ.get("HALIOS_TARGET_URL")
    resolved_api_url = _resolve_halios_base_url(api_url)
    resolved_api_key = api_key or os.environ.get("HALIOS_API_KEY")
    resolved_dataset = dataset_id or os.environ.get("HALIOS_DATASET_ID")

    if resolved_agent:
        update["agent_id"] = resolved_agent
    if resolved_target:
        update["target_url"] = _normalize_cli_url(resolved_target)
    if resolved_api_url:
        update["halios_api_url"] = _normalize_cli_url(resolved_api_url)
    if resolved_api_key:
        update["halios_api_key"] = resolved_api_key
    if resolved_dataset:
        update["dataset_id"] = resolved_dataset
    if dataset_version is not None:
        update["dataset_version"] = dataset_version
    if max_iterations is not None:
        update["max_iterations"] = max_iterations
    if scenarios_path:
        try:
            scenario_set = ScenarioSetConfig.from_yaml(scenarios_path)
        except FileNotFoundError:
            console.print(f"[red]Scenarios file not found:[/] {scenarios_path}")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]Failed to load scenarios:[/] {exc}")
            raise typer.Exit(1)
        update["scenarios"] = scenario_set.scenarios

    if local:
        # In local mode, override the config mode field so downstream code knows.
        update["mode"] = "local"

    if update:
        cfg = cfg.model_copy(update=update)

    missing = []
    if not cfg.agent_id or "${" in cfg.agent_id or cfg.agent_id.startswith("$"):
        missing.append("agent_id (--agent-id or HALIOS_AGENT_ID)")
    if not cfg.target_url or "${" in cfg.target_url or cfg.target_url.startswith("$"):
        missing.append("target_url (--target-url or HALIOS_TARGET_URL)")

    # Cloud credentials are only required in cloud mode.
    if not local:
        if not cfg.halios_api_key or "${" in cfg.halios_api_key or cfg.halios_api_key.startswith("$"):
            missing.append("halios_api_key (--api-key or HALIOS_API_KEY)")
        if not cfg.halios_api_url or "${" in cfg.halios_api_url or cfg.halios_api_url.startswith("$"):
            missing.append("halios_api_url (--api-url or HALIOS_BASE_URL)")

    if missing:
        console.print("[red]Missing optimizer execution settings:[/]")
        for item in missing:
            console.print(f"  - {item}")
        raise typer.Exit(1)

    if not cfg.scenarios:
        console.print(
            "[red]No scenario fixture configured.[/] Pass --scenarios .halios/scenarios.yaml "
            "or add scenarios to optimize.yaml."
        )
        raise typer.Exit(1)

    async def _run() -> str:
        if local:
            from haliosai.optimize import LocalRecorder, LocalScorer

            rubrics_path = ".halios/rubrics.yaml"
            scorer = LocalScorer(
                rubrics_path=rubrics_path,
                model=getattr(cfg, "local_llm_model", "gpt-4o-mini"),
            )
            async with LocalRecorder() as recorder:
                engine = OptimizerEngine(
                    config=cfg,
                    recorder=recorder,
                    verbose=verbose,
                    local_scorer=scorer,
                )
                return await engine.run()
        else:
            async with OptimizeRecorder(
                api_url=cfg.halios_api_url,
                api_key=cfg.halios_api_key,
            ) as recorder:
                engine = OptimizerEngine(config=cfg, recorder=recorder, verbose=verbose)
                return await engine.run()

    try:
        final_prompt = asyncio.run(_run())
    except Exception as exc:
        console.print(f"[red]Optimization failed:[/] {exc}")
        raise typer.Exit(1)

    console.print("\n[bold green]Optimization complete.[/]")
    console.print(f"Final prompt ({len(final_prompt)} chars) ready.")


# ---------------------------------------------------------------------------
# `halios optimize status`
# ---------------------------------------------------------------------------

@optimize_app.command("status")
def status_run(
    run_id: str | None = typer.Argument(None, help="Optimization run ID (UUID)."),
    run_id_opt: str | None = typer.Option(
        None,
        "--run-id",
        help="Legacy alias for RUN_ID.",
        hidden=True,
    ),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key."),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Extra env file to load after default .env files.",
    ),
) -> None:
    """Show the status of an optimization run."""
    _require_rich()
    from rich import print as rprint
    from rich.table import Table

    from haliosai.optimize import OptimizeRecorder

    resolved_run_id = _normalize_run_id(run_id_opt or run_id)
    resolved_url, resolved_key = _resolve_halios_credentials(
        api_url=api_url,
        api_key=api_key,
        env_file=env_file,
    )

    async def _get() -> dict:
        async with OptimizeRecorder(api_url=resolved_url, api_key=resolved_key) as rec:
            return await rec.get_run(resolved_run_id)

    try:
        data = asyncio.run(_get())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    t = Table(title=f"Run {resolved_run_id}")
    t.add_column("Field")
    t.add_column("Value")
    for field in ("id", "name", "status", "agent_id", "created_at", "started_at", "completed_at"):
        if data.get(field) is not None:
            t.add_row(field, str(data[field]))

    iterations = data.get("iterations") or []
    t.add_row("iterations", str(len(iterations)))
    rprint(t)


# ---------------------------------------------------------------------------
# `halios optimize cancel`
# ---------------------------------------------------------------------------

@optimize_app.command("cancel")
def cancel_run(
    run_id: str | None = typer.Argument(None, help="Optimization run ID (UUID)."),
    run_id_opt: str | None = typer.Option(
        None,
        "--run-id",
        help="Legacy alias for RUN_ID.",
        hidden=True,
    ),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key."),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Extra env file to load after default .env files.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Cancel a running optimization run."""
    resolved_run_id = _normalize_run_id(run_id_opt or run_id)
    resolved_url, resolved_key = _resolve_halios_credentials(
        api_url=api_url,
        api_key=api_key,
        env_file=env_file,
    )

    if not yes:
        confirmed = typer.confirm(f"Cancel run {resolved_run_id}?")
        if not confirmed:
            raise typer.Abort()

    from haliosai.optimize import OptimizeRecorder

    async def _cancel() -> None:
        async with OptimizeRecorder(api_url=resolved_url, api_key=resolved_key) as rec:
            await rec.cancel_run(resolved_run_id)

    try:
        asyncio.run(_cancel())
        typer.echo(f"Run {resolved_run_id} cancelled.")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# `halios optimize list`
# ---------------------------------------------------------------------------

@optimize_app.command("list")
def list_runs(
    agent_id: str | None = typer.Option(None, "--agent-id", help="Filter by agent ID."),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key."),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Extra env file to load after default .env files.",
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum number of runs to display."),
) -> None:
    """List recent optimization runs."""
    _require_rich()
    import httpx
    from rich import print as rprint
    from rich.table import Table

    resolved_url, resolved_key = _resolve_halios_credentials(
        api_url=api_url,
        api_key=api_key,
        env_file=env_file,
    )

    params: dict[str, object] = {"limit": limit}
    if agent_id:
        params["agent_id"] = agent_id

    async def _list() -> list[dict]:
        async with httpx.AsyncClient(
            base_url=resolved_url,
            headers={"Authorization": f"Bearer {resolved_key}"},
            timeout=30.0,
        ) as client:
            resp = await client.get("/api/v1/optimization-runs", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("items", data.get("data", []))

    try:
        rows = asyncio.run(_list())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if not rows:
        typer.echo("No runs found.")
        return

    t = Table(title="Optimization Runs")
    t.add_column("ID", no_wrap=True)
    t.add_column("Name")
    t.add_column("Status")
    t.add_column("Agent")
    t.add_column("Created")
    t.add_column("Iters", justify="right")

    for r in rows:
        t.add_row(
            str(r.get("id", ""))[:8] + "…",
            str(r.get("name", "")),
            str(r.get("status", "")),
            str(r.get("agent_id", "")),
            str(r.get("created_at", ""))[:19],
            str(len(r.get("iterations") or [])),
        )
    rprint(t)


# ---------------------------------------------------------------------------
# `halios optimize init`
# ---------------------------------------------------------------------------

_OPTIMIZE_STARTER_YAML = """\
# Halios Prompt Optimizer — scenario-fixture configuration
# Run: halios optimize run --config {output} --scenarios .halios/scenarios.yaml

target_url: "${{HALIOS_TARGET_URL}}"     # URL of your agent (must mount mount_halios())
halios_api_url: "{api_url}"              # Halios backend
halios_api_key: "{api_key}"              # Your Halios API key; can also come from env
agent_id: "{agent_id}"                  # Your Halios agent ID; can also come from env

run_name: "My first optimizer run"
starting_prompt: |
  You are a helpful assistant.

# Optional reference dataset for UI/lineage. Optimizer iterations use scenarios.
# dataset_id: "${{HALIOS_DATASET_ID}}"
# dataset_version: 1

check_ids: []            # UUIDs of Halios checks to run each iteration
t1_check_ids: []         # Subset of check_ids that are hard gates

max_iterations: 5
max_consecutive_discards: 2
optimizer_model: "fast"                 # Halios backend route, using org LLM config
simulation_model: "gemini/gemini-2.5-flash"
simulation_temperature: 0.4
"""

_DATASET_STARTER_YAML = """\
# Halios Dataset Build — configuration
# Run: halios dataset build --config {output}

target_url: "http://localhost:8001"       # URL of your agent (must mount mount_halios())
halios_api_url: "{api_url}"              # Halios backend
halios_api_key: "{api_key}"              # Your Halios API key
agent_id: "{agent_id}"                  # Your Halios agent ID

dataset_name: "My first dataset"        # Used when creating a new dataset
dataset_description: "Synthetic baseline dataset"
# dataset_id: "<existing-dataset-id>"   # Uncomment to append to an existing dataset instead

provenance: "synthetic"
scenario_source: "halios"               # halios | inline | mixed
# scenario_ids: []                      # Optional subset of server-side scenario IDs
# max_scenarios: 20                     # Optional cap for bulk generation

scenarios:
  - id: "hello"
    messages:
      - role: user
        content: "Hello, how are you?"
"""

_SCENARIO_STARTER_YAML = """\
# Halios Scenario Hints — input to backend-managed scenario generation
# Run: halios scenario generate --hints {output} --output .halios/scenarios.yaml
# Then: halios optimize run --config .halios/optimize.yaml --scenarios .halios/scenarios.yaml
#
# CLI loads .env and .halios/.env automatically. Use --env-file for overrides.
# Credentials and IDs can live in env files instead of this file:
#   HALIOS_BASE_URL, HALIOS_API_KEY, HALIOS_AGENT_ID, HALIOS_TARGET_URL
#   HALIOS_API_URL is accepted as a legacy alias.

scenario_count: 12
generation_mode: "simulation-with-arc-hint"
max_turns: 6

hints:
  - "Cover happy paths, missing-information cases, and recovery from user confusion."
  - "Include policy or capability boundary cases if they are relevant to this agent."
"""


@optimize_app.command("init")
def init_config(
    agent_id: str | None = typer.Option(None, "--agent-id", help="Halios agent ID."),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key."),
    output: str = typer.Option(".halios/optimize.yaml", "--output", "-o", help="Output path."),
) -> None:
    """Generate a starter optimize.yaml config file."""
    import pathlib

    resolved_url = _resolve_halios_base_url(api_url) or "https://app.halios.ai"
    resolved_key = _env_or("HALIOS_API_KEY", api_key) or "<your-api-key>"
    resolved_agent = agent_id or "<your-agent-id>"

    content = _OPTIMIZE_STARTER_YAML.format(
        output=output,
        api_url=resolved_url,
        api_key=resolved_key,
        agent_id=resolved_agent,
    )

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        overwrite = typer.confirm(f"{output} already exists — overwrite?")
        if not overwrite:
            raise typer.Abort()

    out_path.write_text(content)
    typer.echo(f"Created {output}")


@dataset_app.command("init")
def init_dataset_config(
    agent_id: str | None = typer.Option(None, "--agent-id", help="Halios agent ID."),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key."),
    output: str = typer.Option(".halios/dataset.yaml", "--output", "-o", help="Output path."),
) -> None:
    """Generate a starter dataset.yaml config file."""
    import pathlib

    resolved_url = _resolve_halios_base_url(api_url) or "https://app.halios.ai"
    resolved_key = _env_or("HALIOS_API_KEY", api_key) or "<your-api-key>"
    resolved_agent = agent_id or "<your-agent-id>"

    content = _DATASET_STARTER_YAML.format(
        output=output,
        api_url=resolved_url,
        api_key=resolved_key,
        agent_id=resolved_agent,
    )

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        overwrite = typer.confirm(f"{output} already exists — overwrite?")
        if not overwrite:
            raise typer.Abort()

    out_path.write_text(content)
    typer.echo(f"Created {output}")


# ---------------------------------------------------------------------------
# `halios scenario init`
# ---------------------------------------------------------------------------

@scenario_app.command("init")
def init_scenario_config(
    agent_id: str | None = typer.Option(None, "--agent-id", help="Halios agent ID."),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key."),
    output: str = typer.Option(".halios/scenario-hints.yaml", "--output", "-o", help="Output path."),
) -> None:
    """Generate a starter scenario-hints.yaml config file."""
    import pathlib

    resolved_url = _resolve_halios_base_url(api_url) or "https://app.halios.ai"
    resolved_key = _env_or("HALIOS_API_KEY", api_key) or "<your-api-key>"
    resolved_agent = agent_id or "<your-agent-id>"

    content = _SCENARIO_STARTER_YAML.format(
        output=output,
        api_url=resolved_url,
        api_key=resolved_key,
        agent_id=resolved_agent,
    )

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        overwrite = typer.confirm(f"{output} already exists — overwrite?")
        if not overwrite:
            raise typer.Abort()

    out_path.write_text(content)
    typer.echo(f"Created {output}")


# ---------------------------------------------------------------------------
@scenario_app.command("generate")
def generate_scenarios_command(
    hints: str = typer.Option(
        ".halios/scenario-hints.yaml",
        "--hints",
        "--config",
        "-c",
        help="Path to scenario-hints.yaml input file.",
    ),
    output: str = typer.Option(
        ".halios/scenarios.yaml",
        "--output",
        "-o",
        help="Generated scenarios.yaml fixture to write.",
    ),
    env_file: str | None = typer.Option(None, "--env-file", help="Extra env file to load after default .env files."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite output without prompting."),
) -> None:
    """Generate scenarios.yaml using Halios-managed LLM scenario generation."""
    _require_rich()
    import pathlib

    import yaml
    from rich.console import Console
    from rich.table import Table

    if env_file:
        try:
            from dotenv import load_dotenv
        except ImportError:
            typer.echo("python-dotenv is required for --env-file. Install haliosai[cli].", err=True)
            raise typer.Exit(1)
        load_dotenv(env_file, override=True)

    from haliosai.optimize.config import ScenarioHintsConfig
    from haliosai.optimize.scenario_generator import build_scenario_set_payload, generate_scenarios

    console = Console()

    try:
        cfg = ScenarioHintsConfig.from_yaml(hints)
    except FileNotFoundError:
        typer.echo(f"Hints file not found: {hints}", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"Failed to load hints: {exc}", err=True)
        raise typer.Exit(1)

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not yes:
        overwrite = typer.confirm(f"{output} already exists — overwrite?")
        if not overwrite:
            raise typer.Abort()

    try:
        scenarios = asyncio.run(generate_scenarios(cfg))
        payload = build_scenario_set_payload(cfg, scenarios)
    except Exception as exc:
        typer.echo(f"Scenario generation failed: {exc}", err=True)
        raise typer.Exit(1)

    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))

    table = Table(title="Generated scenarios")
    table.add_column("ID")
    table.add_column("Risk")
    table.add_column("Title")
    for item in scenarios[:12]:
        table.add_row(
            str(item.get("id") or "")[:28],
            str(item.get("risk_label") or ""),
            str(item.get("title") or "")[:60],
        )
    console.print(table)
    if len(scenarios) > 12:
        console.print(f"...and {len(scenarios) - 12} more")
    console.print(f"\n[bold green]Wrote scenario fixture:[/] {output}")
    console.print("This file contains only scenario specs; credentials, agent URL, and dataset IDs stay in env/config.")
    console.print("Next:")
    console.print(f"  halios optimize run --config .halios/optimize.yaml --scenarios {output}")


# ---------------------------------------------------------------------------
# `halios dataset build`
# ---------------------------------------------------------------------------

@dataset_app.command("build")
def build_dataset(
    config: str = typer.Option(
        ".halios/dataset.yaml",
        "--config",
        "-c",
        help="Path to dataset.yaml config file.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Optional name override when creating a new dataset.",
    ),
    description: str | None = typer.Option(
        None,
        "--description",
        "-d",
        help="Optional description override.",
    ),
    scenario_ids: list[str] | None = typer.Option(
        None,
        "--scenario-id",
        help="Only run the named scenario ID. Repeat to retry multiple failed scenarios.",
    ),
) -> None:
    """Simulate scenarios and create or augment a Halios dataset.

    Reads dataset-build settings from a dedicated config file. Scenarios can
    come from the Halios scenario bank, inline YAML, or both.

    Example::

        halios dataset build --config .halios/dataset.yaml --name "tau2-baseline-v1"
        # → Dataset created: <id>
        # → Add to optimize.yaml:
        #     dataset_id: <id>
    """
    _require_rich()

    from haliosai.optimize.config import DatasetBuildConfig
    from haliosai.optimize.dataset_builder import DatasetBuilder

    try:
        cfg = DatasetBuildConfig.from_yaml(config)
    except FileNotFoundError:
        typer.echo(f"Config file not found: {config}", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"Failed to load config: {exc}", err=True)
        raise typer.Exit(1)

    builder = DatasetBuilder(config=cfg, verbose=True)

    async def _run():
        return await builder.build(name=name, description=description, scenario_ids=scenario_ids)

    try:
        result = asyncio.run(_run())
    except Exception as exc:
        typer.echo(f"\nDataset build failed: {exc}", err=True)
        raise typer.Exit(1)

    if cfg.dataset_id:
        typer.echo(f"\nDataset updated: {result.dataset_id}")
    else:
        typer.echo(f"\nDataset created: {result.dataset_id}")
        typer.echo("Add to optimize.yaml:")
        typer.echo(f"  dataset_id: {result.dataset_id}")
    typer.echo(f"Scenario traces added: {result.added}/{result.attempted}")
    if result.failed:
        typer.echo("Failed scenarios:")
        for scenario_id, error in result.failed:
            typer.echo(f"  - {scenario_id}: {error}")
        typer.echo("Retry only failures with repeated --scenario-id flags.")


# ---------------------------------------------------------------------------
# `halios evaluate dataset`
# ---------------------------------------------------------------------------

@evaluate_app.command("dataset")
def evaluate_dataset(
    dataset_id: str | None = typer.Option(
        None,
        "--dataset-id",
        "-d",
        help="Dataset ID to evaluate. Falls back to HALIOS_DATASET_ID.",
    ),
    agent_id: str | None = typer.Option(None, "--agent-id", help="Halios agent ID or slug."),
    dataset_version: int | None = typer.Option(None, "--dataset-version", help="Dataset snapshot version."),
    check_ids: list[str] | None = typer.Option(
        None,
        "--check-id",
        help="Specific check UUID to run. Repeat for multiple checks.",
    ),
    tags: list[str] | None = typer.Option(
        None,
        "--tag",
        help="Additional tag for resulting check executions. Repeat for multiple tags.",
    ),
    run_name: str | None = typer.Option(None, "--run-name", help="Human-readable evaluation run name."),
    run_comment: str | None = typer.Option(None, "--comment", help="Optional run note."),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for completion and print a result summary."),
    timeout: float = typer.Option(300.0, "--timeout", help="Seconds to wait when --wait is enabled."),
    fail_on_fail: bool = typer.Option(
        False,
        "--fail-on-failed-checks",
        "--fail-on-fail",
        help="Exit non-zero if any check execution fails.",
    ),
    min_pass_rate: float | None = typer.Option(
        None,
        "--min-pass-rate",
        help="Minimum pass rate percentage required for exit code 0.",
    ),
    min_average_score: float | None = typer.Option(
        None,
        "--min-average-score",
        help="Minimum average score required for exit code 0.",
    ),
    max_examples: int = typer.Option(
        10,
        "--max-examples",
        help="Maximum failed or inconclusive result examples to print.",
    ),
    api_url: str | None = typer.Option(None, "--api-url", help="Halios API URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="Halios API key."),
    env_file: str | None = typer.Option(None, "--env-file", help="Extra env file to load after default .env files."),
) -> None:
    """Trigger evaluator checks for all traces in a dataset snapshot."""
    _require_rich()
    _load_cli_env(env_file)

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    from haliosai.client import HaliosClient

    console = Console()
    resolved_agent = _require_env("HALIOS_AGENT_ID", agent_id, "agent_id")
    resolved_dataset = _require_env("HALIOS_DATASET_ID", dataset_id, "dataset_id")
    resolved_key = _require_env("HALIOS_API_KEY", api_key, "api_key")
    resolved_url = _normalize_cli_url(
        _resolve_halios_base_url(api_url) or "https://app.halios.ai"
    )

    async def _run() -> tuple[object, list[object]]:
        async with HaliosClient(
            agent_id=resolved_agent,
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=max(timeout, 30.0),
        ) as client:
            run = await client.trigger_dataset_eval(
                dataset_id=resolved_dataset,
                dataset_version=dataset_version,
                check_ids=check_ids or None,
                tags=tags or None,
                run_name=run_name,
                run_comment=run_comment,
            )
            if not wait:
                return run, []
            rows = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("Waiting for evaluator task", total=None)
                await run.wait(timeout=timeout)
                progress.update(task_id, description="Collecting eval results")
                async for item in run.results(timeout=timeout):
                    rows.append(item)
                    item_progress = getattr(item, "_progress", None)
                    progress.update(
                        task_id,
                        total=(
                            item_progress.total
                            if item_progress and getattr(item_progress, "total", 0) > 0
                            else None
                        ),
                        completed=len(rows),
                    )
                progress.update(task_id, description="Eval results collected", completed=len(rows))
            return run, rows

    try:
        run, rows = asyncio.run(_run())
    except Exception as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(1)

    run_tag = getattr(run, "run_tag", "")
    console.print(f"[bold green]Evaluation triggered[/] task={getattr(run, 'task_id', '')}")
    if run_tag:
        console.print(f"Run tag: {run_tag}")
    console.print(
        f"Dataset: {resolved_dataset}"
        + (f" @ v{dataset_version}" if dataset_version is not None else "")
        + f" · traces={getattr(run, 'trace_count', 0)} checks={getattr(run, 'check_count', 0)}"
    )

    if not wait:
        console.print("Use the run tag in the UI or API to inspect results.")
        return

    passed = sum(1 for item in rows if getattr(item, "passed", None) is True)
    failed = sum(1 for item in rows if getattr(item, "passed", None) is False)
    errors = sum(1 for item in rows if getattr(item, "error", None))
    total = len(rows)
    inconclusive = total - passed - failed
    determinate = passed + failed
    pass_rate = (passed / determinate * 100.0) if determinate else None
    scores = [float(score) for item in rows if (score := getattr(item, "score", None)) is not None]
    average_score = sum(scores) / len(scores) if scores else None

    summary = Table(title="Evaluation Run Summary")
    summary.add_column("Results", justify="right")
    summary.add_column("Passed", justify="right")
    summary.add_column("Failed", justify="right")
    summary.add_column("Inconclusive", justify="right")
    summary.add_column("Pass Rate", justify="right")
    summary.add_column("Avg Score", justify="right")
    summary.add_column("Errors", justify="right")
    summary.add_row(
        str(total),
        str(passed),
        str(failed),
        str(inconclusive),
        "n/a" if pass_rate is None else f"{pass_rate:.1f}%",
        "n/a" if average_score is None else f"{average_score:.3f}",
        str(errors),
    )
    console.print(summary)

    task_stats: dict[str, dict[str, object]] = {}
    for item in rows:
        task_key = (
            getattr(item, "task_slug", None)
            or getattr(item, "task_name", None)
            or getattr(item, "check_name", None)
            or getattr(item, "check_id", None)
            or "unknown"
        )
        task = task_stats.setdefault(
            str(task_key),
            {
                "name": getattr(item, "task_name", None)
                or getattr(item, "check_name", None)
                or str(task_key),
                "results": 0,
                "passed": 0,
                "failed": 0,
                "scores": [],
                "traces": set(),
            },
        )
        task["results"] = int(task["results"]) + 1
        if getattr(item, "passed", None) is True:
            task["passed"] = int(task["passed"]) + 1
        elif getattr(item, "passed", None) is False:
            task["failed"] = int(task["failed"]) + 1
        score = getattr(item, "score", None)
        if score is not None:
            task["scores"].append(float(score))  # type: ignore[union-attr]
        trace_id = getattr(item, "trace_id", None)
        if trace_id:
            task["traces"].add(str(trace_id))  # type: ignore[union-attr]

    if task_stats:
        task_table = Table(title="Task Scores")
        task_table.add_column("Task")
        task_table.add_column("Traces", justify="right")
        task_table.add_column("Results", justify="right")
        task_table.add_column("Pass Rate", justify="right")
        task_table.add_column("Avg Score", justify="right")
        for task in sorted(task_stats.values(), key=lambda item: str(item["name"]).lower()):
            task_passed = int(task["passed"])
            task_failed = int(task["failed"])
            task_determinate = task_passed + task_failed
            task_pass_rate = (
                f"{(task_passed / task_determinate * 100.0):.1f}%"
                if task_determinate
                else "n/a"
            )
            task_scores = task["scores"]
            task_avg = (
                f"{(sum(task_scores) / len(task_scores)):.3f}"  # type: ignore[arg-type]
                if task_scores
                else "n/a"
            )
            task_table.add_row(
                str(task["name"])[:48],
                str(len(task["traces"])),  # type: ignore[arg-type]
                str(task["results"]),
                task_pass_rate,
                task_avg,
            )
        console.print(task_table)

    problem_rows = [
        item
        for item in rows
        if getattr(item, "passed", None) is not True or getattr(item, "error", None)
    ]
    if problem_rows and max_examples > 0:
        details = Table(title="Failed / Inconclusive Examples")
        details.add_column("Trace")
        details.add_column("Check")
        details.add_column("Passed")
        details.add_column("Score", justify="right")
        details.add_column("Reason")
        for item in problem_rows[:max_examples]:
            score = getattr(item, "score", None)
            reason = getattr(item, "error", None) or getattr(item, "reasoning", None) or ""
            details.add_row(
                str(getattr(item, "trace_id", ""))[:16],
                str(getattr(item, "check_name", None) or getattr(item, "check_id", ""))[:40],
                str(getattr(item, "passed", None)),
                "" if score is None else f"{float(score):.3f}",
                str(reason)[:60],
            )
        console.print(details)
        if len(problem_rows) > max_examples:
            console.print(f"...and {len(problem_rows) - max_examples} more failed/inconclusive results")

    threshold_failed = False
    if min_pass_rate is not None and (pass_rate is None or pass_rate < min_pass_rate):
        threshold_failed = True
        console.print(
            f"[red]Pass rate threshold failed:[/] "
            f"{'n/a' if pass_rate is None else f'{pass_rate:.1f}%'} < {min_pass_rate:.1f}%"
        )
    if min_average_score is not None and (
        average_score is None or average_score < min_average_score
    ):
        threshold_failed = True
        console.print(
            f"[red]Average score threshold failed:[/] "
            f"{'n/a' if average_score is None else f'{average_score:.3f}'} < {min_average_score:.3f}"
        )

    if (fail_on_fail and (failed > 0 or errors > 0)) or threshold_failed:
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
