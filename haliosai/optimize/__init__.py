"""HaliosAI Optimize — SDK subpackage for prompt optimization.

Provides:
  - HaliosOptimizableConfig / HaliosOptimizableClient: protocol types
  - mount_halios(): FastAPI route mounter for optimizable agents
  - OptimizeConfig: YAML-loadable configuration for optimization runs
  - OptimizeRecorder: HTTP client for recording runs to Halios backend
  - LocalRecorder: filesystem recorder for local/offline mode
  - LocalScorer: LLM-as-judge scorer for local/offline mode
  - OptimizerEngine: client-side prompt optimization loop
  - print_scorecard / print_iteration_table: Rich console helpers
"""

from .config import OptimizeConfig, ScenarioSetConfig
from .engine import OptimizerEngine
from .protocol import HaliosOptimizableClient, HaliosOptimizableConfig
from .recorder import LocalRecorder, OptimizeRecorder
from .scorecard import print_iteration_table, print_scorecard_table
from .scorer import LocalScorer
from .server import mount_halios

__all__ = [
    "HaliosOptimizableConfig",
    "HaliosOptimizableClient",
    "OptimizeConfig",
    "ScenarioSetConfig",
    "OptimizeRecorder",
    "LocalRecorder",
    "LocalScorer",
    "OptimizerEngine",
    "mount_halios",
    "print_scorecard_table",
    "print_iteration_table",
]
