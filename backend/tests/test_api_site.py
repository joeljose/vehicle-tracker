"""Tests for site config REST endpoints."""

import json

import pytest


@pytest.fixture
def sites_dir(tmp_path, monkeypatch):
    """Redirect site config storage to a temp directory."""
    import backend.config.site_config as sc
    monkeypatch.setattr(sc, "SITES_DIR", tmp_path)
    return tmp_path


VALID_CONFIG = {
    "site_id": "test_junction",
    "roi_polygon": [[100, 200], [300, 400], [500, 600], [700, 800]],
    "entry_exit_lines": {
        "north": {"label": "741-North", "start": [100, 50], "end": [200, 50]},
    },
}


class TestSaveConfig:
    def test_save_valid(self, client, sites_dir):
        resp = client.post("/site/config", json=VALID_CONFIG)
        assert resp.status_code == 200
        assert resp.json() == {"status": "saved"}
        # Verify file written
        path = sites_dir / "test_junction.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["site_id"] == "test_junction"
        assert len(data["roi_polygon"]) == 4

    def test_save_missing_site_id(self, client, sites_dir):
        resp = client.post("/site/config", json={"roi_polygon": [[0, 0], [1, 1], [2, 2]]})
        assert resp.status_code == 422

    def test_save_too_few_roi_vertices(self, client, sites_dir):
        resp = client.post("/site/config", json={
            "site_id": "bad",
            "roi_polygon": [[0, 0], [1, 1]],
        })
        assert resp.status_code == 422


class TestLoadConfig:
    def test_load_saved(self, client, sites_dir):
        client.post("/site/config", json=VALID_CONFIG)
        resp = client.get("/site/config?site_id=test_junction")
        assert resp.status_code == 200
        data = resp.json()
        assert data["site_id"] == "test_junction"
        assert len(data["roi_polygon"]) == 4
        assert "north" in data["entry_exit_lines"]

    def test_load_not_found(self, client, sites_dir):
        resp = client.get("/site/config?site_id=nonexistent")
        assert resp.status_code == 404


class TestListSites:
    def test_list_empty(self, client, sites_dir):
        resp = client.get("/site/configs")
        assert resp.status_code == 200
        assert resp.json() == {"sites": []}

    def test_list_after_save(self, client, sites_dir):
        client.post("/site/config", json=VALID_CONFIG)
        resp = client.get("/site/configs")
        assert "test_junction" in resp.json()["sites"]

    def test_list_multiple(self, client, sites_dir):
        client.post("/site/config", json=VALID_CONFIG)
        config2 = {**VALID_CONFIG, "site_id": "another_site"}
        client.post("/site/config", json=config2)
        sites = client.get("/site/configs").json()["sites"]
        assert len(sites) == 2
        assert "another_site" in sites
        assert "test_junction" in sites
