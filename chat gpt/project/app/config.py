"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _as_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _project_path(name: str, default: Path) -> Path:
    """Resolve relative environment paths from the project root."""

    path = Path(os.getenv(name, str(default))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings kept in one place for fast competition tuning."""

    project_root: Path
    documents_dir: Path
    index_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    relevance_threshold: float
    debug: bool
    thailmm_api_key: str
    thailmm_base_url: str
    thailmm_model: str
    thailmm_timeout_seconds: float


def get_settings() -> Settings:
    """Build settings from the current environment."""

    documents_dir = _project_path("DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents")
    index_dir = _project_path("INDEX_DIR", PROJECT_ROOT / "data" / "index")

    return Settings(
        project_root=PROJECT_ROOT,
        documents_dir=documents_dir,
        index_dir=index_dir,
        chunk_size=_as_int("CHUNK_SIZE", 1000),
        chunk_overlap=_as_int("CHUNK_OVERLAP", 150),
        top_k=_as_int("TOP_K", 5),
        relevance_threshold=_as_float("RELEVANCE_THRESHOLD", 0.35),
        debug=_as_bool("DEBUG", False),
        thailmm_api_key=os.getenv("THAILLM_API_KEY", ""),
        thailmm_base_url=os.getenv("THAILLM_BASE_URL", ""),
        thailmm_model=os.getenv("THAILLM_MODEL", ""),
        thailmm_timeout_seconds=_as_float("THAILLM_TIMEOUT_SECONDS", 60.0),
    )
