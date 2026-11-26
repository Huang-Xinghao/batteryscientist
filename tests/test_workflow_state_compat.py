"""Legacy project goals survive loading and expert-review continuation."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from batteryscientist.agent.workflow import WorkflowState
from batteryscientist.agent.workflow_cli import (
    approve_checkpoint,
    checkpoint_snapshot,
    run_coordinated_workflow,
)


def legacy_review(project):
    report = project / "refine-logs" / "FINAL_PROPOSAL.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Research proposal\n\nExisting evidence and method.\n")
    state = {
        "run_id": "legacy-review",
        "task_prompt": "Original battery research goal",
        "mode": "expert",
        "current_node": "refine",
        "status": "awaiting_user",
        "artifacts": {"refine": [{"path": "refine-logs/FINAL_PROPOSAL.md"}]},
        "history": [{"node": "ideas", "metadata": {"final": "Existing research"}}],
        "expert_checkpoint": {
            "node": "refine",
            "next_node": "experiment_plan",
            "status": "pending",
            "report": "refine-logs/FINAL_PROPOSAL.md",
            "files": ["refine-logs/FINAL_PROPOSAL.md"],
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "questions": [],
        },
    }
    (project / "workflow-state.json").write_text(json.dumps(state))
    return state


def test_legacy_goal_roundtrip_preserves_review_and_does_not_mutate_input(tmp_path):
    original = legacy_review(tmp_path)
    before = copy.deepcopy(original)
    state = WorkflowState.from_dict(original)
    assert state.project_goal == original["task_prompt"]
    assert state.status == "awaiting_user"
    assert state.expert_checkpoint == original["expert_checkpoint"]
    assert state.history == original["history"]
    assert state.artifacts == original["artifacts"]
    assert original == before
    assert "task_prompt" not in state.to_dict()
    assert WorkflowState.from_dict(state.to_dict()) == state


@pytest.mark.parametrize("goal,expected", [("Current goal", "Current goal"), ("", "Legacy goal")])
def test_current_nonempty_goal_takes_precedence(goal, expected):
    state = WorkflowState.from_dict({"run_id": "r1", "project_goal": goal, "task_prompt": "Legacy goal"})
    assert state.project_goal == expected


def test_unknown_fields_still_fail_explicitly():
    with pytest.raises(TypeError, match="unexpected_field"):
        WorkflowState.from_dict({"run_id": "r1", "unexpected_field": True})


def test_retired_stage_options_are_preserved_as_history_without_replaying(tmp_path):
    original = legacy_review(tmp_path)
    legacy = {"node_instructions": "Previously requested figure changes",
              "experiment_expansion_requested": True, "experiment_expansion_count": 2}
    original.update(legacy)
    before = copy.deepcopy(original)
    state = WorkflowState.from_dict(original)
    assert state.history[:-1] == original["history"]
    assert state.history[-1] == {"event": "legacy_state_metadata", "node": "refine", "metadata": legacy}
    assert state.status == "awaiting_user"
    assert state.current_node == "refine"
    assert state.expert_checkpoint == original["expert_checkpoint"]
    assert original == before
    assert WorkflowState.from_dict(state.to_dict()) == state


def test_revision_continuation_loads_legacy_goal_without_approving(tmp_path, monkeypatch):
    original = legacy_review(tmp_path)
    from batteryscientist.agent import coordinator

    def make_agent(project, **options):
        assert project == tmp_path
        assert options["project_goal"] == original["task_prompt"]
        assert options["wait_for_user"] is True

        def revise(prompt):
            assert "Clarify the input window" in prompt
            return SimpleNamespace(status="awaiting_user", error=None)

        return SimpleNamespace(go=revise)

    monkeypatch.setattr(coordinator, "CoordinatorAgent", make_agent)
    state = run_coordinated_workflow(tmp_path, revise="Clarify the input window", wait_for_user=True)
    assert state.project_goal == original["task_prompt"]
    assert state.status == "awaiting_user"
    assert state.expert_checkpoint["status"] == "pending"
    assert json.loads((tmp_path / "workflow-state.json").read_text()) == original


def mixed_legacy_artifacts(project):
    original = legacy_review(project)
    original.update(status="failed", current_node="experiment_plan", failed_node="experiment_plan")
    original["artifacts"].update({
        "literature": "# Existing literature report\n\nEvidence, not a filename.",
        "data_search": "data-search/DATA_REPORT.md",
        "ideas": "# Existing ideas\n\nResearch summary.",
    })
    original["node_results"] = {"ideas": {"canonical_path": "idea-stage/CUSTOM_IDEA.md"}}
    for relative in ("literature/RESEARCH_REPORT.md", "data-search/DATA_REPORT.md", "idea-stage/CUSTOM_IDEA.md"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Existing research\n")
    (project / "workflow-state.json").write_text(json.dumps(original))
    return original


def test_mixed_legacy_artifacts_keep_text_and_normalize_file_references(tmp_path):
    original = mixed_legacy_artifacts(tmp_path)
    before = copy.deepcopy(original)
    state = WorkflowState.from_dict(original)
    assert state.artifacts["literature"] == [{"path": "literature/RESEARCH_REPORT.md"}]
    assert state.artifacts["data_search"] == [{"path": "data-search/DATA_REPORT.md"}]
    assert state.artifacts["ideas"] == [{"path": "idea-stage/CUSTOM_IDEA.md"}]
    assert state.artifacts["refine"] == original["artifacts"]["refine"]
    assert state.node_results["literature"]["legacy_artifact"] == original["artifacts"]["literature"]
    assert original == before
    assert WorkflowState.from_dict(state.to_dict()) == state


@pytest.mark.parametrize("node,raw,metadata,expected", [
    ("paper_write", "paper/main.tex", {}, "paper/main.tex"),
    ("refine", "Proposal written; see the report for details.", {}, "refine-logs/FINAL_PROPOSAL.md"),
    ("novelty", '{"verdict": "needs evidence"}', {}, "idea-stage/NOVELTY.json"),
    ("ideas", "# Summary", {"files": ["notes/custom.md"]}, "notes/custom.md"),
])
def test_legacy_paths_prose_and_recorded_deliveries(node, raw, metadata, expected):
    state = WorkflowState.from_dict({"run_id": "legacy", "artifacts": {node: raw},
                                     "node_results": {node: metadata}})
    assert state.artifacts[node] == [{"path": expected}]
    assert state.node_results[node]["legacy_artifact"] == raw


@pytest.mark.parametrize("path", ["/outside/report.md", "../report.md", "C:\\outside\\report.md", "https://example.com/report.md"])
def test_migration_does_not_accept_paths_outside_project(path):
    with pytest.raises(ValueError, match="项目之外"):
        WorkflowState.from_dict({"run_id": "legacy", "artifacts": {"ideas": "# Summary"},
                                 "node_results": {"ideas": {"canonical_path": path}}})


@pytest.mark.parametrize("full_stage", [True, False])
def test_legacy_artifacts_reach_real_experiment_planning_entry(tmp_path, monkeypatch, full_stage):
    from batteryscientist.agent import workflow_cli

    original = mixed_legacy_artifacts(tmp_path)
    report = tmp_path / "refine-logs" / "FINAL_PROPOSAL.md"
    before = report.read_bytes()
    plan = tmp_path / "refine-logs" / "EXPERIMENT_PLAN.md"

    def plan_reply(prompt, **kwargs):
        if full_stage:
            for relative in ("literature/RESEARCH_REPORT.md", "data-search/DATA_REPORT.md", "idea-stage/CUSTOM_IDEA.md"):
                assert relative in prompt
        plan.write_text("# Test plan\n\nNo training was executed.\n")
        return SimpleNamespace(files=["refine-logs/EXPERIMENT_PLAN.md"], iterations=1,
                               final="Plan drafted", delivery_status="completed", remaining="")

    agent = SimpleNamespace(project=tmp_path, last_state=None, go=Mock(side_effect=plan_reply))
    monkeypatch.setattr(workflow_cli, "PhaseAgent", lambda *args, **kwargs: agent)
    result = workflow_cli.execute_stage(tmp_path, "experiment_plan", "Only write the experiment plan",
                                       role="experiment", project_goal=original["task_prompt"],
                                       full_stage=full_stage)
    assert result.status == ("awaiting_user" if full_stage else "completed")
    assert agent.go.call_count == 1
    assert report.read_bytes() == before
    assert not (tmp_path / "experiment-runs").exists()
    if full_stage:
        saved = json.loads((tmp_path / "workflow-state.json").read_text())
        assert saved["current_node"] == "experiment_plan"
        assert saved["expert_checkpoint"]["status"] == "pending"
        assert saved["artifacts"]["literature"] == [{"path": "literature/RESEARCH_REPORT.md"}]


def test_approval_loads_legacy_state_and_keeps_stale_review_guard(tmp_path):
    original = legacy_review(tmp_path)
    expected = checkpoint_snapshot(tmp_path)
    report = tmp_path / "refine-logs" / "FINAL_PROPOSAL.md"
    report.write_text(report.read_text() + "\nChanged before approval.\n")
    with pytest.raises(ValueError, match="检查点已变化"):
        approve_checkpoint(tmp_path, expected, stop_after="refine")
    assert json.loads((tmp_path / "workflow-state.json").read_text()) == original
    state, stop_after = approve_checkpoint(tmp_path, checkpoint_snapshot(tmp_path), stop_after="refine")
    assert stop_after == "refine"
    assert state.status == "completed"
    assert state.project_goal == original["task_prompt"]
    persisted = json.loads((tmp_path / "workflow-state.json").read_text())
    assert "task_prompt" not in persisted
    assert persisted["project_goal"] == original["task_prompt"]
    assert persisted["history"][:-1] == original["history"]


def test_api_worker_review_reaches_main_with_legacy_project(tmp_path, monkeypatch):
    from batteryscientist.agent import coordinator
    from batteryscientist.api.app import checkpoint_token
    from batteryscientist.api import store, worker

    original = legacy_review(tmp_path)
    monkeypatch.setenv("BATTERYSCIENTIST_WEB_DATA", str(tmp_path / "web"))
    store.initialize()
    project = {"id": "legacy", "path": str(tmp_path)}
    options = {"action": "review", "answer": "Clarify the input window using the archived draft",
               "checkpoint_token": checkpoint_token(project, original), "max_tool_rounds": 3}
    with store.connection() as db:
        db.execute("INSERT INTO projects(id,name,path,created_at) VALUES(?,?,?,?)",
                   (project["id"], "Legacy review", project["path"], store.now()))
        db.execute("INSERT INTO runs(id,project_id,status,options,created_at) VALUES(?,?,?,?,?)",
                   ("review-run", project["id"], "running", json.dumps(options), store.now()))
    folder = store.data_root() / "runs" / "review-run"
    folder.mkdir(parents=True)
    called = []

    def make_agent(project, **kwargs):
        assert kwargs["project_goal"] == original["task_prompt"]
        assert kwargs["wait_for_user"] is True

        def go(prompt):
            assert options["answer"] in prompt
            called.append(prompt)
            return SimpleNamespace(status="awaiting_user", error=None)

        return SimpleNamespace(go=go)

    monkeypatch.setattr(coordinator, "CoordinatorAgent", make_agent)
    worker._execute("review-run")
    assert len(called) == 1
    assert json.loads((folder / "result.json").read_text()) == {"status": "awaiting_user", "error": None}
    assert json.loads((tmp_path / "workflow-state.json").read_text()) == original
