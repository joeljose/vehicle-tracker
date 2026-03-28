"""Per-site configuration load/save."""

import json
from dataclasses import dataclass, field
from pathlib import Path

SITES_DIR = Path(__file__).parent / "sites"


@dataclass
class SiteConfig:
    """Configuration for a single junction site."""

    site_id: str
    roi_polygon: list[tuple[float, float]] = field(default_factory=list)
    entry_exit_lines: dict = field(default_factory=dict)

    def has_roi(self) -> bool:
        return len(self.roi_polygon) >= 3


def load_site_config(site_id: str, sites_dir: Path = SITES_DIR) -> SiteConfig:
    """Load site config from JSON file."""
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
