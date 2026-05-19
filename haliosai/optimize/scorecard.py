"""Scorecard and iteration display helpers for the Halios optimizer CLI."""

from __future__ import annotations

from typing import Any


def _try_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def print_scorecard_table(scorecard: dict[str, Any], title: str = "Scorecard") -> None:
    """Pretty-print a scorecard dict using Rich (or plain text fallback)."""
    if _try_rich():
        _rich_scorecard(scorecard, title)
    else:
        _plain_scorecard(scorecard, title)


def print_iteration_table(iterations: list[dict[str, Any]]) -> None:
    """Pretty-print a list of iteration summary dicts."""
    if _try_rich():
        _rich_iterations(iterations)
    else:
        _plain_iterations(iterations)


# ---------------------------------------------------------------------------
# Rich implementations
# ---------------------------------------------------------------------------

def _rich_scorecard(scorecard: dict[str, Any], title: str) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Tier", justify="center")

    check_rates = scorecard.get("check_pass_rates", {})
    for _check_id, data in sorted(check_rates.items(), key=lambda x: x[1].get("name", "")):
        rate = data.get("rate", 0.0)
        avg_score = data.get("avg_score")
        tier = str(data.get("tier", ""))
        rate_str = f"{rate:.1%}"
        score_str = f"{avg_score:.3f}" if avg_score is not None else "n/a"
        style = "red" if rate < 0.85 else "green"
        table.add_row(
            data.get("name", _check_id),
            f"[{style}]{rate_str}[/]",
            f"[{style}]{score_str}[/]",
            tier,
        )

    overall = scorecard.get("overall_score", 0.0)
    t1 = scorecard.get("t1_passed", True)
    console.print(table)
    console.print(
        f"  Overall: [bold]{overall:.3f}[/]  |  "
        f"T1 Gate: {'[green]PASS[/]' if t1 else '[red]FAIL[/]'}"
    )


def _rich_iterations(iterations: list[dict[str, Any]]) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    table = Table(title="Optimization Iterations", box=box.ROUNDED)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Verdict")
    table.add_column("Score", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("T1 Gate", justify="center")
    table.add_column("Run Tag", style="dim")

    verdict_colors = {
        "baseline": "blue",
        "accept": "green",
        "investigate": "yellow",
        "discard": "red",
        "error": "red bold",
    }

    for it in iterations:
        verdict = it.get("verdict", "")
        color = verdict_colors.get(verdict, "white")
        scorecard = it.get("scorecard_json", {})
        delta_sc = it.get("scorecard_delta_json", {})
        score = scorecard.get("overall_score", 0.0)
        delta = delta_sc.get("delta", "")
        t1 = it.get("t1_gate_passed", True)
        delta_str = f"{delta:+.3f}" if isinstance(delta, (int, float)) else str(delta)
        table.add_row(
            str(it.get("iteration_number", "")),
            f"[{color}]{verdict}[/]",
            f"{score:.3f}",
            delta_str,
            "[green]PASS[/]" if t1 else "[red]FAIL[/]",
            it.get("trace_run_tag", ""),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Plain-text fallbacks
# ---------------------------------------------------------------------------

def _plain_scorecard(scorecard: dict[str, Any], title: str) -> None:
    print(f"\n--- {title} ---")
    check_rates = scorecard.get("check_pass_rates", {})
    for _cid, data in sorted(check_rates.items(), key=lambda x: x[1].get("name", "")):
        rate = data.get("rate", 0.0)
        avg_score = data.get("avg_score")
        score_str = f"{avg_score:.3f}" if avg_score is not None else "n/a"
        print(f"  {data.get('name', _cid):<50} rate={rate:.1%}  score={score_str}")
    overall = scorecard.get("overall_score", 0.0)
    t1 = scorecard.get("t1_passed", True)
    print(f"  Overall: {overall:.3f}  T1 Gate: {'PASS' if t1 else 'FAIL'}")


def _plain_iterations(iterations: list[dict[str, Any]]) -> None:
    print("\n--- Iterations ---")
    for it in iterations:
        sc = it.get("scorecard_json", {})
        delta_sc = it.get("scorecard_delta_json", {})
        print(
            f"  #{it.get('iteration_number', '')}  {it.get('verdict', ''):<12}"
            f"  score={sc.get('overall_score', 0.0):.3f}"
            f"  delta={delta_sc.get('delta', 0):+.3f}"
            f"  t1={'PASS' if it.get('t1_gate_passed', True) else 'FAIL'}"
        )
