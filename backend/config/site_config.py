"""Per-site configuration load/save."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_VALID_SITE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")

SITES_DIR = Path(__file__).parent / "sites"


@dataclass
class SiteConfig:
    """Configuration for a single junction site."""

    site_id: str
    roi_polygon: list[tuple[float, float]] = field(default_factory=list)
    entry_exit_lines: dict = field(default_factory=dict)

    def has_roi(self) -> bool:
        return len(self.roi_polygon) >= 3


def _validate_site_id(site_id: str) -> None:
    """Validate site_id to prevent path traversal."""
    if not _VALID_SITE_ID.match(site_id):
        raise ValueError(
            f"Invalid site_id: {site_id!r}. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )


def load_site_config(site_id: str, sites_dir: Path = SITES_DIR) -> SiteConfig:
    """Load site config from JSON file."""
    _validate_site_id(site_id)
    path = sites_dir / f"{site_id}.json"
    with open(path) as f:
        data = json.load(f)
    return SiteConfig(
        site_id=data["site_id"],
        roi_polygon=[tuple(p) for p in data.get("roi_polygon", [])],
        entry_exit_lines=data.get("entry_exit_lines", {}),
    )


def save_site_config(config: SiteConfig, sites_dir: Path = SITES_DIR) -> Path:
    """Save site config to JSON file. Returns the path written."""
    _validate_site_id(config.site_id)
    sites_dir.mkdir(parents=True, exist_ok=True)
    path = sites_dir / f"{config.site_id}.json"
    data = {
        "site_id": config.site_id,
        "roi_polygon": [list(p) for p in config.roi_polygon],
        "entry_exit_lines": config.entry_exit_lines,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
