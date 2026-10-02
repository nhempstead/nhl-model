"""Shared paths, configuration loading, and research-contract constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUCCESS_HIERARCHY = (
    "engineering_validity",
    "data_integrity",
    "forecasting_quality",
    "uncertainty_assessment",
    "paper_trading_evidence",
)

PRIMARY_MARKET = "moneyline_full_game"
PRIMARY_HORIZON_MINUTES = 60
SIGNIFICANCE_LEVEL = 0.05
POWER_TARGETS = (0.80, 0.90)
DEFAULT_SEED = 20261002


def repo_root() -> Path:
    env = os.environ.get("NHL_RESEARCH_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "nhl_research").exists():
            return parent
    return Path.cwd()


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (repo_root() / "configs" / "default.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} is not a mapping")
    return data


@dataclass(frozen=True)
class Paths:
    root: Path
    warehouse: Path
    reports: Path
    registry: Path
    raw: Path
    processed: Path

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "Paths":
        root = repo_root()
        cfg = config or load_config()
        paths = cfg.get("paths", {})
        obj = cls(
            root=root,
            warehouse=root / paths.get("warehouse", "data/warehouse"),
            reports=root / paths.get("reports", "reports"),
            registry=root / paths.get("registry", "models/registry"),
            raw=root / paths.get("raw", "data/raw"),
            processed=root / paths.get("processed", "data/processed"),
        )
        obj.warehouse.mkdir(parents=True, exist_ok=True)
        obj.reports.mkdir(parents=True, exist_ok=True)
        (obj.registry / "versions").mkdir(parents=True, exist_ok=True)
        return obj
