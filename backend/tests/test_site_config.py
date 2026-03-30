"""Unit tests for site config load/save (pure Python, no GPU needed)."""

import json

from backend.config.site_config import SiteConfig, load_site_config, save_site_config


def test_save_and_load(tmp_path):
    """Round-trip: save then load preserves data."""
    config = SiteConfig(
        site_id="test_junction",
        roi_polygon=[(100, 200), (500, 200), (500, 800), (100, 800)],
        entry_exit_lines={
            "north": {"label": "North", "start": [200, 210], "end": [400, 210]},
        },
    )

    path = save_site_config(config, sites_dir=tmp_path)
    assert path.exists()

    loaded = load_site_config("test_junction", sites_dir=tmp_path)
    assert loaded.site_id == "test_junction"
    assert loaded.roi_polygon == [(100, 200), (500, 200), (500, 800), (100, 800)]
    assert loaded.entry_exit_lines["north"]["label"] == "North"


def test_has_roi():
    config = SiteConfig(site_id="test", roi_polygon=[(0, 0), (1, 0), (1, 1)])
    assert config.has_roi() is True

    config_no_roi = SiteConfig(site_id="test", roi_polygon=[(0, 0)])
    assert config_no_roi.has_roi() is False


def test_load_741_73():
    """The shipped 741_73.json config loads correctly."""
    config = load_site_config("741_73")
    assert config.site_id == "741_73"
    assert config.has_roi()
    assert len(config.roi_polygon) >= 3
    assert "north" in config.entry_exit_lines


def test_save_creates_directory(tmp_path):
    """Save creates the sites directory if it doesn't exist."""
    nested = tmp_path / "deep" / "nested"
    config = SiteConfig(site_id="x", roi_polygon=[(0, 0), (1, 0), (1, 1)])
    path = save_site_config(config, sites_dir=nested)
    assert path.exists()


def test_json_format(tmp_path):
    """Saved JSON is human-readable (indented)."""
    config = SiteConfig(site_id="fmt_test", roi_polygon=[(0, 0), (1, 1), (2, 2)])
    path = save_site_config(config, sites_dir=tmp_path)
    content = path.read_text()
    assert "\n" in content  # indented, not single-line
    data = json.loads(content)
    assert data["site_id"] == "fmt_test"
