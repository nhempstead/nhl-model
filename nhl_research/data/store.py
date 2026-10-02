"""Parquet warehouse with manifests, origin labels, and idempotent writes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nhl_research.config import Paths
from nhl_research.entities import DatasetOrigin
from nhl_research.timeutil import to_iso

UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(tz=UTC)


class Warehouse:
    def __init__(self, paths: Paths):
        self.paths = paths
        self.root = paths.warehouse
        self.root.mkdir(parents=True, exist_ok=True)

    def table_path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def manifest_path(self, name: str) -> Path:
        return self.root / f"{name}.manifest.json"

    def write_table(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        origin: DatasetOrigin,
        source: str,
        note: str,
        primary_key: list[str],
    ) -> Path:
        if "dataset_origin" not in frame.columns:
            frame = frame.copy()
            frame["dataset_origin"] = origin.value
        path = self.table_path(name)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, frame], ignore_index=True)
            combined = combined.drop_duplicates(subset=primary_key, keep="first")
        else:
            combined = frame.drop_duplicates(subset=primary_key, keep="first")
        combined.to_parquet(path, index=False)
        manifest = {
            "table": name,
            "path": str(path),
            "origin": origin.value,
            "source": source,
            "note": note,
            "primary_key": primary_key,
            "rows": int(len(combined)),
            "written_at_utc": to_iso(_now()),
            "columns": list(combined.columns),
            "content_sha256": _hash_file(path),
        }
        self.manifest_path(name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def read_table(self, name: str) -> pd.DataFrame:
        path = self.table_path(name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def replace_table(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        origin: DatasetOrigin,
        source: str,
        note: str,
        primary_key: list[str],
    ) -> Path:
        path = self.table_path(name)
        if path.exists():
            path.unlink()
        if self.manifest_path(name).exists():
            self.manifest_path(name).unlink()
        return self.write_table(
            name, frame, origin=origin, source=source, note=note, primary_key=primary_key
        )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
