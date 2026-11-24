"""Real queue execution and delivery outcomes across full/scoped assignments."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from batteryscientist.agent.coordinator import CoordinatorTurn
from batteryscientist.agent.loop import PhaseAgent
from batteryscientist.agent.workflow import WorkflowRunner
from batteryscientist.agent.workflow_cli import execute_stage
from batteryscientist.integrations.model import ModelResponse, ToolCall
from batteryscientist.tools import ToolContext
from batteryscientist.tools.experiments.run import run_experiments


def make_agent(project, *, full_stage=True, status='partial'):
    agent = PhaseAgent(project, model_client=SimpleNamespace(), phase='experiment_bridge',
                       full_stage=full_stage)
    agent.connect_mcp = False
    responses = [ModelResponse(tool_calls=(ToolCall('delivery', 'submit_delivery', {
        'files': ['result.md'], 'status': status,
        'remaining': 'Missing external dataset' if status == 'partial' else '',
    }),)), ModelResponse(content='Result available; external data is missing.' if status == 'partial' else 'Done.')]
    agent._request_model = AsyncMock(side_effect=responses)
    return agent


@pytest.mark.parametrize('full_stage', [True, False])
def test_experiment_tool_available_without_workflow_injection(tmp_path, full_stage):
    agent = make_agent(tmp_path, full_stage=full_stage)
    assert 'run_experiments' in {spec.name for spec in agent.tools}
    agent.configure(phase='experiment_plan')
    assert 'run_experiments' not in {spec.name for spec in agent.tools}


def test_real_queue_reports_failure_resumes_and_reuses_outputs(tmp_path):
    (tmp_path/'experiments').mkdir()
    (tmp_path/'refine-logs').mkdir()
    (tmp_path/'refine-logs/EXPERIMENT_PLAN.md').write_text('## B1: queue integration test\n')
    script = tmp_path/'job.py'
    script.write_text("from pathlib import Path\nassert Path('ready').exists(), 'input not ready'\nPath('results').mkdir(exist_ok=True)\nPath('results/output.txt').write_text('ok')\n")
    job = {'id': 'B1_job', 'block_id': 'B1', 'argv': ['python', 'job.py'],
           'resources': {'device': 'cpu'}, 'depends_on': [], 'timeout_seconds': 10,
           'expected_outputs': [{'path': 'results/output.txt', 'nonempty': True}]}
    (tmp_path/'experiments/manifest.json').write_text(json.dumps({
        'backend': 'local', 'required_inputs': ['job.py'], 'jobs': [job]}))
    progress = []
    ctx = ToolContext(tmp_path, on_progress=lambda kind, details: progress.append((kind, details)))
    failed = run_experiments(ctx)
    assert failed.is_error
    assert failed.structured_content['job_id'] == 'B1_job'
    assert 'input not ready' in (tmp_path/failed.structured_content['log']).read_text()
    assert (tmp_path/failed.structured_content['receipt']).is_file()
    (tmp_path/'ready').touch()
    completed = run_experiments(ctx)
    assert not completed.is_error
    assert completed.structured_content['completed_jobs'] == 1
    assert progress
    count = len(list((tmp_path/'experiment-runs').glob('*/status.json')))
    assert not run_experiments(ctx).is_error
    assert len(list((tmp_path/'experiment-runs').glob('*/status.json'))) == count == 2
    assert not list(tmp_path.glob('*delivery*'))


@pytest.mark.parametrize('mode', ['auto', 'expert'])
def test_partial_full_stage_keeps_artifacts_without_success_or_approval(tmp_path, mode):
    (tmp_path/'result.md').write_text('Current available results')
    agent = make_agent(tmp_path)
    runner = WorkflowRunner(agent)
    state = runner.start_from_project('experiment_bridge', 'Run experiments')
    state.mode = mode
    result = runner.execute_node('experiment_bridge', 'Run experiments', state=state)
    assert result.status == 'partial'
    assert result.error == 'Missing external dataset'
    assert result.transitions == 0
    assert result.expert_checkpoint is None
    assert result.artifacts['experiment_bridge'][0]['path'] == 'result.md'
    assert result.node_results['experiment_bridge']['remaining'] == result.error
    session = json.loads((tmp_path/'.batteryscientist/role-sessions/experiment.json').read_text())
    assert session['tasks'][-1]['status'] == 'partial'
    assert session['tasks'][-1]['remaining'] == result.error
    turn = CoordinatorTurn()
    turn.accept_result(result.status, result.error)
    assert turn.stop_reason is None  # Main can dispatch the repair without a user approval.


def test_partial_scoped_assignment_reaches_existing_task_result(tmp_path):
    (tmp_path/'result.md').write_text('Current available results')
    agent = make_agent(tmp_path, full_stage=False)
    with patch('batteryscientist.agent.workflow_cli.PhaseAgent', return_value=agent):
        result = execute_stage(tmp_path, 'experiment_bridge', 'Finish pending work',
                               role='experiment', project_goal='Research', full_stage=False)
    assert result.status == 'partial'
    assert result.error == 'Missing external dataset'
    assert result.files == ['result.md']
    assert not (tmp_path/'workflow-state.json').exists()


def test_completed_delivery_preserves_normal_stage_outcome(tmp_path):
    (tmp_path/'result.md').write_text('Completed report')
    agent = make_agent(tmp_path, status='completed')
    runner = WorkflowRunner(agent)
    state = runner.execute_node('literature', 'Research')
    assert state.status == 'completed'
    assert state.transitions == 1
    assert state.error is None


def test_partial_delivery_still_requires_declared_files_to_exist(tmp_path):
    agent = make_agent(tmp_path)
    from batteryscientist.agent.loop import AgentState
    agent.last_state = AgentState([], 'experiment_bridge')
    with pytest.raises(ValueError, match='不存在'):
        agent._submit_delivery(None, ['missing.md'], status='partial', remaining='Missing data')
    with pytest.raises(ValueError, match='简述'):
        agent._submit_delivery(None, [], status='partial')


@pytest.mark.parametrize('full_stage', [True, False])
def test_transport_preserves_partial_result(tmp_path, full_stage):
    from batteryscientist.agent.runtime.execution import ExecutionTask, _result
    task = ExecutionTask(id='a'*32, project=str(tmp_path), node='experiment_bridge',
                         role='experiment', instruction='Complete experiments',
                         project_goal='Research', full_stage=full_stage)
    outcome = {'status': 'partial', 'error': 'Missing external dataset',
               'final': 'Available results saved', 'artifacts': [{'path': 'result.md'}]}
    result = _result(task, outcome)
    assert result.status == 'partial'
    assert result.error == 'Missing external dataset'
    files = ([ref['path'] for ref in result.artifacts['experiment_bridge']]
             if full_stage else result.files)
    assert files == ['result.md']


def test_resubmitting_completed_delivery_clears_previous_gap(tmp_path):
    from batteryscientist.agent.loop import AgentState
    agent = make_agent(tmp_path)
    agent.last_state = AgentState([], 'experiment_bridge')
    agent._submit_delivery(None, [], status='partial', remaining='Missing data')
    agent._submit_delivery(None, [])
    assert agent.last_state.delivery_status == 'completed'
    assert agent.last_state.remaining == ''


def test_completed_expert_stage_still_requires_existing_review(tmp_path):
    (tmp_path/'result.md').write_text('Completed report')
    agent = make_agent(tmp_path, status='completed')
    runner = WorkflowRunner(agent)
    state = runner.start_from_project('literature', 'Research')
    state.mode = 'expert'
    state = runner.execute_node('literature', 'Research', state=state)
    assert state.status == 'awaiting_user'
    assert state.expert_checkpoint['report'] == 'result.md'
