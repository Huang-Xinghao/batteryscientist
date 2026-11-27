"""User-request updates and completion follow current-turn execution receipts."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from batteryscientist.agent.coordinator import CoordinatorAgent
from batteryscientist.agent.loop import ArtifactValidationError
from batteryscientist.integrations.model import ModelResponse, ToolCall


def make_agent(tmp_path, **kwargs):
    agent = CoordinatorAgent(tmp_path, model_client=SimpleNamespace(), **kwargs)
    agent.connect_mcp = False
    return agent


def test_missing_stage_tools_rejected_before_dispatch_or_budget_charge(tmp_path):
    agent = make_agent(tmp_path)
    with patch('batteryscientist.agent.coordinator.execute_stage') as execute:
        result = agent._delegate(None, {
            'node': 'paper_plan', 'role': 'experiment', 'full_stage': False,
            'instruction': 'Run the existing verification script',
            'required_tools': ['bash'],
        })
    assert result.is_error
    assert 'bash' in result.output
    assert agent.turn.assignments == 0
    execute.assert_not_called()


def test_scoped_verification_with_capable_phase_returns_to_main(tmp_path):
    agent = make_agent(tmp_path)
    worker = SimpleNamespace(status='completed', error=None, task_id='verify-1',
                             final='Hashes verified', files=['verification.json'])
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=worker) as execute:
        result = agent._delegate(None, {
            'node': 'experiment_bridge', 'role': 'experiment', 'full_stage': False,
            'instruction': 'Only run the existing verification script; no training',
            'required_tools': ['bash'],
        })
    assert not result.is_error
    assert execute.call_args.kwargs['full_stage'] is False
    assert json.loads(result.output)['artifacts'] == ['verification.json']


def test_update_existing_goal_preserves_workflow_and_session(tmp_path):
    path = tmp_path / 'workflow-state.json'
    path.write_text(json.dumps({'project_goal': 'Old', 'status': 'completed',
                                'artifacts': {'refine': []}}))
    agent = make_agent(tmp_path)
    data = {}
    agent._active = SimpleNamespace(data=data, session=SimpleNamespace(save=lambda d: None))
    agent._set_research_goal(None, 'Multiple datasets')
    assert agent.project_goal == data['project_goal'] == 'Multiple datasets'
    assert json.loads(path.read_text()) == {'project_goal': 'Multiple datasets',
                                           'status': 'completed', 'artifacts': {'refine': []}}


def test_only_updating_proposal_cannot_complete_experiments(tmp_path):
    agent = make_agent(tmp_path)
    agent._set_request_scope(None, 'execution', ['data_search', 'experiment_bridge'])
    agent.turn.edited_paths.add('refine-logs/FINAL_PROPOSAL.md')
    agent.turn.receipts['data_search'] = {'status': 'completed'}
    with pytest.raises(ArtifactValidationError, match='experiment_bridge'):
        agent._complete_research()
    agent.turn.receipts['experiment_bridge'] = {'status': 'completed'}
    agent._complete_research()


def test_exhausted_budget_reports_partial_without_user_approval(tmp_path):
    agent = make_agent(tmp_path, max_assignments=1)
    agent._set_request_scope(None, 'execution', ['data_search', 'experiment_bridge'])
    agent.turn.assignments = 1
    agent._complete_research()
    assert agent.turn.status == 'partial'
    assert agent.turn.stop_reason == 'Assignment budget exhausted'
    assert agent._dispatch_budget() == {'limit': 1, 'used': 1, 'remaining': 0}


def test_scope_does_not_turn_execution_into_discussion(tmp_path):
    agent = make_agent(tmp_path)
    agent._set_request_scope(None, 'execution', ['experiment_bridge'])
    with pytest.raises(ValueError):
        agent._set_request_scope(None, 'discussion')


def test_discussion_and_document_outcomes(tmp_path):
    agent = make_agent(tmp_path)
    agent._set_request_scope(None, 'discussion')
    agent._complete_research()
    agent = make_agent(tmp_path)
    agent._set_request_scope(None, 'document')
    with pytest.raises(ArtifactValidationError):
        agent._complete_research()
    agent.turn.edited_paths.add('plan.md')
    agent._complete_research()


@pytest.mark.parametrize('query', ['你是谁？', '现在进度如何？'])
def test_plain_reply_finishes_without_scope_registration(tmp_path, query):
    events = []
    agent = make_agent(tmp_path, on_event=events.append)
    agent._request_model = AsyncMock(return_value=ModelResponse(content='我是 BatteryScientist。'))
    result = agent.go(query)
    assert result.outcome == 'answered'
    assert agent._request_model.await_count == 1
    assert not any(event['kind'] == 'validation_failed' for event in events)
    assert not (tmp_path / 'workflow-state.json').exists()


def test_read_and_edit_automatically_record_document_outcome(tmp_path):
    (tmp_path / 'plan.md').write_text('# Old plan')
    agent = make_agent(tmp_path)
    agent._request_model = AsyncMock(side_effect=[
        ModelResponse(tool_calls=(ToolCall('read', 'read_project_file', {'path': 'plan.md'}),)),
        ModelResponse(tool_calls=(ToolCall('edit', 'edit_project_markdown', {
            'node': 'paper_plan', 'path': 'plan.md',
            'old_text': '# Old plan', 'new_text': '# New plan'}),)),
        ModelResponse(content='已更新。'),
    ])
    result = agent.go('修改计划标题')
    assert result.outcome == 'document_updated'
    assert (tmp_path / 'plan.md').read_text() == '# New plan'


def test_read_only_query_needs_no_scope(tmp_path):
    (tmp_path / 'plan.md').write_text('# Research plan')
    agent = make_agent(tmp_path)
    agent._request_model = AsyncMock(side_effect=[
        ModelResponse(tool_calls=(ToolCall('read', 'read_project_file', {'path': 'plan.md'}),)),
        ModelResponse(content='这是当前研究计划。'),
    ])
    assert agent.go('看看计划标题是什么').outcome == 'answered'


def test_dispatch_without_scope_cannot_be_downgraded_to_discussion(tmp_path):
    agent = make_agent(tmp_path)
    worker = SimpleNamespace(status='completed', error=None, task_id='task-1',
                             final='Done', files=['report.md'])
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=worker):
        agent._delegate(None, {'node': 'experiment_bridge', 'role': 'experiment',
                               'instruction': 'Verify results', 'full_stage': False})
    assert agent.turn.request_kind == 'execution'
    assert agent.turn.required_nodes == ['experiment_bridge']
    with pytest.raises(ValueError):
        agent._set_request_scope(None, 'discussion')
    agent._complete_research()


@pytest.mark.parametrize('receipt_status', ['completed', 'skipped'])
def test_wiki_dispatch_tracks_success_and_unchanged_sources(tmp_path, receipt_status):
    (tmp_path / 'paper').mkdir()
    (tmp_path / 'paper/main.pdf').write_bytes(b'%PDF-1.4')
    agent = make_agent(tmp_path)
    agent.turn.archive_due = True
    worker = SimpleNamespace(status='completed', error=None,
                             wiki_result={'status': receipt_status})
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=worker):
        result = agent._curate_wiki(None, project_evidence=True)
    assert not result.is_error
    assert agent.turn.request_kind == 'execution'
    assert agent.turn.required_nodes == ['wiki']
    assert not agent.turn.archive_due
    agent._complete_research()


def test_premature_summary_resumes_dispatch_and_saves_receipts(tmp_path):
    agent = make_agent(tmp_path, max_assignments=3)
    responses = [
        ModelResponse(tool_calls=(ToolCall('scope', 'set_request_scope', {
            'kind': 'execution', 'nodes': ['experiment_bridge']}),)),
        ModelResponse(content='派单额度已用完。'),
        ModelResponse(tool_calls=(ToolCall('run', 'delegate_stage', {'assignment': {
            'node': 'experiment_bridge', 'role': 'experiment',
            'instruction': 'Add datasets and execute experiments', 'full_stage': False}}),)),
        ModelResponse(content='新增实验完成。'),
    ]
    agent._request_model = AsyncMock(side_effect=responses)
    worker = SimpleNamespace(status='completed', error=None, task_id='worker-1',
                             final='Experiments completed', files=['report.md'])
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=worker):
        result = agent.go('补充不同材料的数据集并运行实验')
    assert result.status == 'completed'
    assert result.outcome == 'execution_completed'
    assert agent.turn.assignments == 1
    task = json.loads((tmp_path / '.batteryscientist/role-sessions/main.json').read_text())['tasks'][-1]
    assert task['receipts']['experiment_bridge']['task_id'] == 'worker-1'
    assert task['dispatch_budget']['remaining'] == 2


def test_current_budget_refreshed_before_each_model_response(tmp_path):
    agent = make_agent(tmp_path, max_assignments=3)
    state = SimpleNamespace(messages=[{'role': 'system', 'content': 'Main prompt'}])
    async def check(state, schemas, context):
        current = json.loads(state.messages[0]['content'].split('\n本轮执行状态：')[1])
        assert current['dispatch_budget']['used'] == agent.turn.assignments
    with patch('batteryscientist.agent.loop.AgentLoop._request_model', side_effect=check):
        import asyncio
        asyncio.run(agent._request_model(state, [], None))
        agent.turn.assignments = 2
        asyncio.run(agent._request_model(state, [], None))
    assert state.messages[0]['content'].count('本轮执行状态：') == 1
