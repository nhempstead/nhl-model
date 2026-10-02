"""Model registry, champion/challenger comparison, and immutability helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nhl_research.config import Paths

UTC = timezone.utc


@dataclass
class ModelVersion:
    model_id: str
    family: str
    created_at: str
    code_version: str
    data_snapshot: str
    feature_names: list[str]
    seed: int
    notes: str
    metrics: dict[str, Any]
    origin: str


class ModelRegistry:
    def __init__(self, paths: Paths):
        self.root = paths.registry
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)
        self.champion_path = self.root / "champion.json"
        self.history_path = self.root / "history.jsonl"

    def record(self, version: ModelVersion) -> Path:
        path = self.root / "versions" / f"{version.model_id}.json"
        path.write_text(json.dumps(asdict(version), indent=2), encoding="utf-8")
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(version)) + "\n")
        return path

    def champion(self) -> dict[str, Any] | None:
        if not self.champion_path.exists():
            return None
        return json.loads(self.champion_path.read_text(encoding="utf-8"))

    def set_champion(self, version: ModelVersion) -> None:
        self.champion_path.write_text(json.dumps(asdict(version), indent=2), encoding="utf-8")

    def rollback(self, model_id: str) -> dict[str, Any]:
        path = self.root / "versions" / f"{model_id}.json"
        if not path.exists():
            raise FileNotFoundError(model_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.champion_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "rollback", "model_id": model_id, "at": datetime.now(tz=UTC).isoformat()}) + "\n")
        return payload
