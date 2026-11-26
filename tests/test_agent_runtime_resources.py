"""Deployment library configuration is honored by the actual agent tools."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from batteryscientist.agent.coordinator import CoordinatorAgent
from batteryscientist.agent.loop import PhaseAgent
from batteryscientist.tools.data_access import resolve_data_read


def agents(project):
    return (CoordinatorAgent(project, model_client=SimpleNamespace()),
            PhaseAgent(project, phase="experiment_plan", model_client=SimpleNamespace()))


def test_configured_library_is_shared_by_api_and_real_agent_tools(tmp_path, monkeypatch):
    library = tmp_path / "shared-library"
    document = library / "电芯" / "数据集" / "Tongji_README.md"
    document.parent.mkdir(parents=True)
    document.write_text("# Tongji\n\nDataset description.\n")
    monkeypatch.setenv("BATTERYSCIENTIST_BATTERY_ROOT", str(library))
    from batteryscientist.api.library import library_root

    assert library_root("battery") == library
    project = tmp_path / "project"
    project.mkdir()
    for agent in agents(project):
        assert agent.knowledge_roots == (library,)
        spec = agent._by_name["battery_resources"]
        result = asyncio.run(agent._invoke(spec, {"action": "search", "query": "Tongji", "top_k": 6}))
        assert not result.is_error
        value = json.loads(result.output)
        assert value["count"] == 1
        assert value["knowledge_root"] == str(library)


def test_explicit_library_and_disabled_library_keep_their_meaning(tmp_path, monkeypatch):
    monkeypatch.setenv("BATTERYSCIENTIST_BATTERY_ROOT", str(tmp_path / "shared"))
    explicit = tmp_path / "specific"
    agent = PhaseAgent(tmp_path, phase="experiment_plan", model_client=SimpleNamespace(), knowledge_root=explicit)
    assert agent.knowledge_roots == (explicit,)
    disabled = PhaseAgent(tmp_path, phase="experiment_plan", model_client=SimpleNamespace(), knowledge_root=None)
    assert disabled.knowledge_roots == ()


def test_absent_optional_project_config_returns_effective_defaults_without_writing(tmp_path):
    agent = CoordinatorAgent(tmp_path, model_client=SimpleNamespace())
    spec = agent._by_name["read_project_file"]
    result = asyncio.run(agent._invoke(spec, {"path": "project-config.json", "limit": 12000}))
    value = json.loads(result.output)
    assert not result.is_error
    assert value["exists"] is False
    assert value["source"] == "defaults"
    assert value["compute"]["device"] == "auto"
    assert not (tmp_path / "project-config.json").exists()
    assert not agent.turn.read_ledger


def test_existing_config_still_reads_original_content_and_blocks_escape(tmp_path):
    path = tmp_path / "project-config.json"
    path.write_text('{"compute":{"device":"cpu"}}')
    agent = CoordinatorAgent(tmp_path, model_client=SimpleNamespace())
    value = json.loads(agent._read(None, path.name).output)
    assert value["content"] == path.read_text()
    assert value["sha256"]
    with pytest.raises(ValueError):
        agent._read(None, "../project-config.json")


def test_data_context_does_not_present_unmounted_server_paths_as_available(tmp_path, monkeypatch):
    library = tmp_path / "library"
    library.mkdir()
    missing = tmp_path / "unmounted-data" / "Tongji"
    (library / "data_sources.json").write_text(json.dumps({"sources": [
        {"resource_id": "Tongji", "kind": "dataset", "path": str(missing)}
    ]}))
    monkeypatch.setenv("BATTERYSCIENTIST_BATTERY_ROOT", str(library))
    project = tmp_path / "project"
    project.mkdir()
    main, phase = agents(project)
    context = json.loads(main._context(None).output)
    assert context["local_data_sources"][0]["exists"] is False
    prompt = phase._system_prompt()
    assert str(missing) in prompt and '"exists": false' in prompt
    with pytest.raises(FileNotFoundError, match="未挂载"):
        resolve_data_read(project, (library,), str(missing))
    with pytest.raises(ValueError):
        resolve_data_read(project, (library,), str(tmp_path / "unregistered"))
