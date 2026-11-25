"""A leftover proposal-review draft is reported and can be archived to unblock approval."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from batteryscientist.proposal_review import (
    CANVAS,
    RECORD,
    archive_proposal_review,
    pending_proposal_review,
    require_resolved_proposal_review,
)

PROPOSAL_TEXT = "# 最终研究方案\n\n## 模型输入\n\n电压曲线。\n"


def write_draft(root, *, changes=("调整输入",), feedback=""):
    proposal = root / "refine-logs" / "FINAL_PROPOSAL.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(PROPOSAL_TEXT, encoding="utf-8")
    draft = {
        "version": 1,
        "sourceHash": hashlib.sha256(proposal.read_bytes()).hexdigest(),
        "nodes": [],
        "edges": [],
        "changes": list(changes),
        "review": {"sections": [], "feedback": feedback},
    }
    (root / CANVAS).write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")


def test_untouched_or_missing_draft_does_not_block(tmp_path):
    assert pending_proposal_review(tmp_path) is None
    write_draft(tmp_path, changes=())
    assert pending_proposal_review(tmp_path) is None
    require_resolved_proposal_review(tmp_path)


def test_edited_draft_is_reported_and_blocks(tmp_path):
    write_draft(tmp_path, changes=("调整输入", "补充约束"), feedback="请说明标签来源")
    assert pending_proposal_review(tmp_path) == {
        "changes": 2,
        "editedSections": 0,
        "hasFeedback": True,
        "hasRevision": False,
        "unreadable": False,
    }
    with pytest.raises(ValueError):
        require_resolved_proposal_review(tmp_path)


def test_unreadable_draft_blocks_and_is_reported(tmp_path):
    (tmp_path / "refine-logs").mkdir()
    (tmp_path / CANVAS).write_text("{not json", encoding="utf-8")
    assert pending_proposal_review(tmp_path)["unreadable"] is True


def test_archive_moves_draft_and_record_aside(tmp_path):
    write_draft(tmp_path)
    (tmp_path / RECORD).write_text(json.dumps({"id": "r1"}), encoding="utf-8")
    archived = archive_proposal_review(tmp_path)
    assert len(archived) == 2 and all(path.startswith("refine-logs/archive/idea-review-") for path in archived)
    assert all((tmp_path / path).is_file() for path in archived)
    assert not (tmp_path / CANVAS).exists() and not (tmp_path / RECORD).exists()
    assert pending_proposal_review(tmp_path) is None
    require_resolved_proposal_review(tmp_path)


def test_overview_reports_draft_and_archive_endpoint_unblocks(tmp_path, monkeypatch):
    monkeypatch.setenv("BATTERYSCIENTIST_PROJECT_ROOTS", str(tmp_path / "projects"))
    monkeypatch.setenv("BATTERYSCIENTIST_WEB_DATA", str(tmp_path / "web"))
    monkeypatch.delenv("BATTERYSCIENTIST_PASSWORD_HASH", raising=False)
    from batteryscientist.api import app as api

    with TestClient(api.app, base_url="http://127.0.0.1") as client:
        project = client.post("/api/projects", json={"name": "草稿"}).json()
        root = tmp_path / "projects" / project["path"].rsplit("/", 1)[-1]
        write_draft(root)
        overview = client.get(f"/api/projects/{project['id']}/proposal-overview").json()
        assert overview["pendingReview"]["changes"] == 1

        result = client.post(
            f"/api/projects/{project['id']}/idea-canvas/archive", headers={"Origin": "http://127.0.0.1"}
        )
        assert result.status_code == 200, result.text
        assert result.json()["archived"][0].endswith("IDEA_CANVAS.json")
        overview = client.get(f"/api/projects/{project['id']}/proposal-overview").json()
        assert overview["pendingReview"] is None
        kinds = [e["kind"] for e in client.get(f"/api/projects/{project['id']}/activity").json()["events"]]
        assert "proposal_review_archived" in kinds
