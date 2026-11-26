"""Declared response models keep every field the web client reads."""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    wiki = tmp_path / "wiki"
    battery = tmp_path / "battery"
    (wiki / "papers").mkdir(parents=True)
    (wiki / "graph").mkdir()
    (wiki / "figures" / "example").mkdir(parents=True)
    (battery / "电芯").mkdir(parents=True)
    (wiki / "papers" / "example.md").write_text(
        "---\nnode_id: paper:example\nmethod: [\"Transformer\"]\n---\n# 时序预测范式\n\n跨电芯泛化证据。",
        encoding="utf-8",
    )
    (wiki / "graph" / "edges.jsonl").write_text(
        json.dumps({"from": "paper:example", "to": "paper:example", "type": "cites"}), encoding="utf-8"
    )
    (wiki / "figures" / "example" / "fig1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (wiki / "figures" / "example" / "fig1.md").write_text(
        "# Figure 1\n\nSource: paper:example\nPDF page: 3\n\n容量预测结果。", encoding="utf-8"
    )
    (battery / "电芯" / "Tongji.md").write_text("# Tongji 数据集\n\n寿命标签。", encoding="utf-8")
    monkeypatch.setenv("BATTERYSCIENTIST_PROJECT_ROOTS", str(projects))
    monkeypatch.setenv("BATTERYSCIENTIST_WEB_DATA", str(tmp_path / "web"))
    monkeypatch.setenv("BATTERYSCIENTIST_WIKI_ROOT", str(wiki))
    monkeypatch.setenv("BATTERYSCIENTIST_BATTERY_ROOT", str(battery))
    monkeypatch.delenv("BATTERYSCIENTIST_PASSWORD_HASH", raising=False)
    from batteryscientist.api import app as api

    monkeypatch.setattr(api, "scan_terminal_projects", lambda: frozenset())
    monkeypatch.setattr(api, "_terminal_cache", None)
    with TestClient(api.app, base_url="http://127.0.0.1") as test_client:
        yield test_client, projects


def make_project(client, projects):
    response = client.post("/api/projects", json={"name": "响应模型"})
    assert response.status_code == 201, response.text
    project = response.json()
    root = projects / project["path"].rsplit("/", 1)[-1]
    (root / "literature").mkdir()
    (root / "literature" / "RESEARCH_REPORT.md").write_text("# 报告\n\n内容。", encoding="utf-8")
    (root / "figures").mkdir()
    (root / "figures" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return project["id"]


def test_project_file_endpoints_keep_their_fields(client):
    test_client, projects = client
    project_id = make_project(test_client, projects)

    files = test_client.get(f"/api/projects/{project_id}/files").json()
    assert files["truncated"] is False
    entry = next(item for item in files["files"] if item["path"] == "literature/RESEARCH_REPORT.md")
    assert set(entry) == {"path", "size", "modified"}

    text = test_client.get(
        f"/api/projects/{project_id}/file", params={"path": "literature/RESEARCH_REPORT.md"}
    ).json()
    assert set(text) == {"kind", "path", "content", "sha256", "editable"}
    assert text["kind"] == "text" and text["editable"] is True

    binary = test_client.get(f"/api/projects/{project_id}/file", params={"path": "figures/a.png"}).json()
    assert binary == {"kind": "binary", "path": "figures/a.png", "size": 8}

    saved = test_client.put(
        f"/api/projects/{project_id}/file",
        params={"path": "literature/RESEARCH_REPORT.md"},
        json={"content": "# 报告\n\n已修改。", "sha256": text["sha256"]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["content"] == "# 报告\n\n已修改。"

    assert test_client.get(f"/api/projects/{project_id}/log").json() == {"content": ""}
    activity = test_client.get(f"/api/projects/{project_id}/activity").json()
    assert activity["has_more"] is False
    event = activity["events"][-1]
    assert set(event) == {"id", "project_id", "run_id", "kind", "data", "created_at"}
    assert event["kind"] == "file_saved" and event["data"]["path"] == "literature/RESEARCH_REPORT.md"


def test_library_and_auth_endpoints_keep_their_fields(client):
    test_client, _ = client
    assert test_client.get("/api/auth/session").json() == {"enabled": False, "authenticated": True}

    index = test_client.get("/api/libraries/battery").json()
    assert index["available"] is True and index["total"] == 1
    assert set(index["items"][0]) == {"path", "title", "category", "excerpt", "modified"}
    assert index["categories"] == [{"id": "电芯", "count": 1}]

    document = test_client.get("/api/libraries/battery/document", params={"path": "电芯/Tongji.md"}).json()
    assert document["title"] == "Tongji 数据集" and "寿命标签" in document["content"]

    graph = test_client.get("/api/libraries/paradigms/graph").json()
    node = graph["nodes"][0]
    assert node["id"] == "paper:example" and node["tags"] == ["Transformer"]
    assert graph["edges"] == [{"source": "paper:example", "target": "paper:example", "type": "cites", "evidence": ""}]

    figures = test_client.get("/api/libraries/paradigms/figures").json()
    figure = figures["items"][0]
    assert figure["paper_title"] == "时序预测范式" and figure["page"] == 3
    assert figure["paper_path"] == "papers/example.md"
