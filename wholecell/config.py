"""User-level settings for the wholecell package.

Settings are stored in ~/.wholecell/settings.json and are loaded once per
process.  The file is created automatically on the first save.
"""

from __future__ import annotations

import json
import logging
import tempfile
import os
from pathlib import Path

log = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".wholecell" / "settings.json"

DEFAULT_SETTINGS: dict = {
    "default_data_directory": None,  # None → OS default in file dialogs
    "lowpass_hz": 2000.0,
    "dvdt_threshold_mv_per_ms": 5.0,
    "peak_window_ms": 20.0,
}

_cache: dict | None = None


def load_settings() -> dict:
    """Return current settings merged with defaults.

    Missing keys fall back to DEFAULT_SETTINGS.  A missing or malformed
    settings file is treated as empty (defaults used, warning logged).
    """
    global _cache
    if _cache is not None:
        return _cache

    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open(encoding="utf-8") as fh:
                on_disk = json.load(fh)
            settings.update(on_disk)
        except Exception as exc:
            log.warning("Could not read %s: %s — using defaults", SETTINGS_PATH, exc)

    _cache = settings
    return _cache


def save_settings(settings: dict) -> None:
    """Write *settings* to disk, creating the directory if needed.

    Uses an atomic rename so readers never see a partial file.
    """
    global _cache
    _cache = dict(DEFAULT_SETTINGS)
    _cache.update(settings)

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=SETTINGS_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, SETTINGS_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_default_data_dir() -> Path | None:
    """Return the configured default data directory, or None if unset."""
    val = load_settings().get("default_data_directory")
    if val:
        p = Path(val)
        return p if p.is_dir() else None
    return None


def update_setting(key: str, value) -> None:
    """Update a single key and persist immediately."""
    settings = dict(load_settings())
    settings[key] = value
    save_settings(settings)
