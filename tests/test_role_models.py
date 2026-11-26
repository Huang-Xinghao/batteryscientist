"""Per-role models survive dispatch and phase changes without changing injected clients."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from batteryscientist.agent.coordinator import CoordinatorAgent
from batteryscientist.agent.loop import PhaseAgent
from batteryscientist.integrations.model import role_model, role_connection


@pytest.fixture(autouse=True)
def model_environment(monkeypatch):
    for role in ('main', 'idea_generator', 'experiment', 'paper_writer', 'wiki_agent'):
        for setting in ('MODEL', 'API_KEY', 'BASE_URL'):
            monkeypatch.setenv(f'BATTERYSCIENTIST_{role.upper()}_{setting}', '')
    monkeypatch.setenv('BATTERYSCIENTIST_MODEL', 'shared-model')
    monkeypatch.setenv('BATTERYSCIENTIST_API_KEY', 'shared-test-key')
    monkeypatch.setenv('BATTERYSCIENTIST_BASE_URL', 'https://shared.example/v1')


def test_role_override_then_run_model_then_shared_default(monkeypatch):
    assert role_model('experiment') == 'shared-model'
    assert role_model('experiment', 'run-model') == 'run-model'
    monkeypatch.setenv('BATTERYSCIENTIST_EXPERIMENT_MODEL', ' coding-model ')
    assert role_model('experiment', 'run-model') == 'coding-model'
    assert role_model('paper_writer', 'run-model') == 'run-model'
    monkeypatch.setenv('BATTERYSCIENTIST_EXPERIMENT_MODEL', ' ')
    assert role_model('experiment', 'run-model') == 'run-model'
    monkeypatch.delenv('BATTERYSCIENTIST_MODEL')
    assert role_model('experiment') == 'gpt-4o'


def test_phase_changes_select_matching_role_model(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTERYSCIENTIST_IDEA_GENERATOR_MODEL', 'research-model')
    monkeypatch.setenv('BATTERYSCIENTIST_EXPERIMENT_MODEL', 'coding-model')
    monkeypatch.setenv('BATTERYSCIENTIST_PAPER_WRITER_MODEL', 'writing-model')
    monkeypatch.setenv('BATTERYSCIENTIST_WIKI_AGENT_MODEL', 'wiki-model')
    with patch('batteryscientist.agent.loop.ModelClient', side_effect=lambda model, **kw: SimpleNamespace(model=model)):
        agent = PhaseAgent(tmp_path, model='run-model', phase='literature')
    assert agent.model.model == 'research-model'
    for phase, expected in [('experiment_bridge', 'coding-model'), ('paper_write', 'writing-model'),
                            ('wiki', 'wiki-model'), ('ideas', 'research-model')]:
        agent.configure(phase=phase)
        assert agent.model.model == expected


def test_injected_model_client_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTERYSCIENTIST_EXPERIMENT_MODEL', 'coding-model')
    client = SimpleNamespace(model='injected')
    agent = PhaseAgent(tmp_path, model_client=client, phase='experiment_bridge')
    agent.configure(phase='paper_write')
    assert agent.model is client
    assert client.model == 'injected'


def test_role_credentials_are_isolated_across_phase_changes(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTERYSCIENTIST_PAPER_WRITER_MODEL', 'writer-model')
    monkeypatch.setenv('BATTERYSCIENTIST_PAPER_WRITER_API_KEY', 'writer-test-key')
    monkeypatch.setenv('BATTERYSCIENTIST_PAPER_WRITER_BASE_URL', 'https://writer.example/v1')
    assert role_connection('experiment')['api_key'] == 'shared-test-key'
    assert role_connection('paper_writer')['api_key'] == 'writer-test-key'
    with patch('batteryscientist.agent.loop.ModelClient',
               side_effect=lambda model, **kw: SimpleNamespace(model=model, connection=kw)) as factory:
        agent = PhaseAgent(tmp_path, phase='experiment_bridge')
        assert agent.model.connection['api_key'] == 'shared-test-key'
        agent.configure(phase='paper_write')
        assert agent.model.model == 'writer-model'
        assert agent.model.connection['api_key'] == 'writer-test-key'
        assert agent.model.connection['base_url'] == 'https://writer.example/v1'
        agent.configure(phase='paper_plan')
        assert factory.call_count == 2
        agent.configure(phase='experiment_bridge')
        assert agent.model.connection['api_key'] == 'shared-test-key'
        assert agent.model.connection['base_url'] == 'https://shared.example/v1'


@pytest.mark.parametrize('node,role,expected', [
    ('ideas', 'idea_generator', 'research-model'),
    ('experiment_bridge', 'experiment', 'coding-model'),
    ('paper_write', 'paper_writer', 'writing-model'),
])
def test_dispatch_records_selected_worker_model(tmp_path, monkeypatch, node, role, expected):
    monkeypatch.setenv(f'BATTERYSCIENTIST_{role.upper()}_MODEL', expected)
    main = CoordinatorAgent(tmp_path, model='run-model', model_client=SimpleNamespace(), project_goal='Research')
    outcome = SimpleNamespace(status='completed', error=None, task_id='task', final='Done', files=[])
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=outcome) as execute:
        main._delegate(None, dict(node=node, role=role, instruction='Complete scoped task', full_stage=False))
    assert execute.call_args.kwargs['model'] == expected


def test_wiki_dispatch_uses_wiki_model(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTERYSCIENTIST_WIKI_AGENT_MODEL', 'wiki-model')
    (tmp_path / 'paper.pdf').write_bytes(b'paper')
    main = CoordinatorAgent(tmp_path, model='run-model', model_client=SimpleNamespace())
    outcome = SimpleNamespace(status='completed', error=None, wiki_result={'status': 'completed'})
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=outcome) as execute:
        result = main._curate_wiki(None, paper_path='paper.pdf')
    assert json.loads(result.output)['status'] == 'completed'
    assert execute.call_args.kwargs['model'] == 'wiki-model'


def test_main_override_does_not_replace_worker_default(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTERYSCIENTIST_MAIN_MODEL', 'main-model')
    main = CoordinatorAgent(tmp_path, model='run-model')
    with patch('batteryscientist.agent.coordinator.ModelClient',
               return_value=SimpleNamespace(model='main-model')) as client, \
         patch.object(main, 'run', return_value=SimpleNamespace(final='Answer')):
        assert main.go('Explain the project').final == 'Answer'
    assert client.call_args.kwargs['model'] == 'main-model'
    assert main.worker_model == 'run-model'


def test_discussion_window_uses_role_override(tmp_path, monkeypatch):
    from batteryscientist.agent.runtime.host import chat

    monkeypatch.setenv('BATTERYSCIENTIST_PAPER_WRITER_MODEL', 'writing-model')
    root = tmp_path / '.batteryscientist/herdr'
    root.mkdir(parents=True)
    client = SimpleNamespace(model='writing-model', complete=lambda *a: SimpleNamespace(content='Reply'))
    with patch('batteryscientist.integrations.model.ModelClient', return_value=client) as factory, \
         patch('batteryscientist.agent.runtime.herdr.ensure_no_pending'), \
         patch('batteryscientist.agent.runtime.sessions.RoleSession.compact',
               side_effect=lambda data, history, model, budget: history):
        assert chat(tmp_path, 'paper_writer', 'Discuss the paper') == 'Reply'
    assert factory.call_args.kwargs['model'] == 'writing-model'
