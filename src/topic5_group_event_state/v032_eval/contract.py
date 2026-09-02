"""Frozen configuration, paths and atomic IO for the v0.3.2 evaluation package."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "topic5_group_event_state_v032_eval.json"
CONFIG_FORMAT = "group_event_state_v0_3_2_eval_config"


def load_eval_config(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen evaluation configuration and check its format tag."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    payload = json.loads(config_path.read_text())
    if payload.get("format") != CONFIG_FORMAT:
        raise ValueError(f"{config_path}: not a {CONFIG_FORMAT} file")
    payload["_config_path"] = str(config_path)
    payload["_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return payload


@dataclass(frozen=True)
class EvalPaths:
    data_root: Path
    measurement: Path
    evaluation: Path
    shared: Path
    results: Path

    @classmethod
    def from_config(cls, config: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> "EvalPaths":
        root = Path(config["data_root"])
        return cls(
            data_root=root,
            measurement=root / "measurement",
            evaluation=root / "evaluation",
            shared=root / "shared",
            results=Path(repo_root) / "results" / "group_event_state" / "v0_3_2",
        )

    def ensure(self) -> None:
        for path in (self.measurement, self.evaluation, self.shared, self.results):
            path.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a temporary file + rename so readers never see a torn file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """``np.savez`` appends ``.npz`` to bare paths; hand it an open handle instead."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **{k: np.asarray(v) for k, v in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def source_commit(repo: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance is never faked
        return f"unavailable:{type(exc).__name__}"


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
