"""Review rounds yield to main before any revision work starts."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from batteryscientist.agent.coordinator import CoordinatorAgent
from batteryscientist.agent.workflow import WorkflowConfig, WorkflowRunner, WorkflowState
from batteryscientist.agent.workflow_cli import execute_stage


@pytest.mark.parametrize('rounds', [1, 2])
def test_reviews_return_to_main_and_resume_after_other_stages(tmp_path, rounds):
    paper = tmp_path / 'paper'
    paper.mkdir()
    (paper / 'main.tex').write_text('Initial manuscript')
    (paper / 'main.pdf').write_bytes(b'Original PDF')
    reviewed = []
    compiled = []

    def review(_runner, request):
        reviewed.append((paper / 'main.pdf').read_bytes())
        (tmp_path / request.output).write_text(f'Review {len(reviewed)}: revise text and plots')
        return SimpleNamespace(status='completed')

    def compile_pdf(_runner):
        compiled.append((paper / 'main.tex').read_text())
        (paper / 'main.pdf').write_bytes(compiled[-1].encode())
        return paper / 'main.pdf'

    agent = SimpleNamespace(project=tmp_path, last_state=None)
    with (patch('batteryscientist.agent.workflow_cli.PhaseAgent', return_value=agent),
          patch('batteryscientist.integrations.review.paper_review_inputs', return_value=['paper/main.tex']),
          patch('batteryscientist.integrations.review.ReviewRunner.run', review),
          patch.object(WorkflowRunner, '_compile_pdf', compile_pdf),
          patch.object(WorkflowRunner, '_go', side_effect=AssertionError('Revision bypassed main'))):
        def run():
            return execute_stage(tmp_path, 'improvement', 'Continue paper improvement',
                                 role='paper_writer', project_goal='Research', mode='auto',
                                 config=WorkflowConfig(improvement_rounds=rounds))

        first = run()
        assert first.status == 'partial'
        assert first.improvement_checkpoint == {'round': 1, 'stage': 'revision'}
        assert first.node_results['improvement']['review_handoff']['content'].startswith('Review 1:')
        assert reviewed == [b'Original PDF']
        assert compiled == []

        # Main may run another full stage while collecting revisions. Its status
        # and current node must not discard the saved improvement round.
        saved = json.loads((tmp_path / 'workflow-state.json').read_text())
        saved.update(status='completed', current_node='paper_write')
        (tmp_path / 'workflow-state.json').write_text(json.dumps(saved))
        (paper / 'main.tex').write_text('Round one revised by roles')
        second = run()
        if rounds == 1:
            assert second.status == 'completed'
            assert second.improvement_checkpoint == {'round': 1, 'stage': 'completed'}
            assert len(reviewed) == 1
            assert compiled == ['Round one revised by roles']
            assert (paper / 'main_round1.pdf').read_bytes() == b'Round one revised by roles'
            assert not (paper / 'review_round2.md').exists()
            assert not (paper / 'main_round2.pdf').exists()
            return
        assert second.status == 'partial'
        assert second.improvement_checkpoint == {'round': 2, 'stage': 'revision'}
        assert second.node_results['improvement']['review_handoff']['round'] == 2
        assert reviewed == [b'Original PDF', b'Round one revised by roles']
        assert (paper / 'main_round1.pdf').read_bytes() == reviewed[1]

        (paper / 'main.tex').write_text('Round two revised by roles')
        final = run()
        assert final.status == 'completed'
        assert final.improvement_checkpoint == {'round': 2, 'stage': 'completed'}
        assert len(reviewed) == 2
        assert (paper / 'main_round2.pdf').read_bytes() == b'Round two revised by roles'
        assert (paper / 'main_round0_original.pdf').read_bytes() == b'Original PDF'
        log = (paper / 'PAPER_IMPROVEMENT_LOG.md').read_text()
        assert log.count('## 第 1 轮原始审稿') == log.count('## 第 2 轮原始审稿') == 1


def test_explicit_improvement_allows_main_to_delegate_scoped_revisions(tmp_path):
    (tmp_path / 'workflow-state.json').write_text(json.dumps({
        'improvement_checkpoint': {'round': 1, 'stage': 'revision'},
    }))
    main = CoordinatorAgent(tmp_path, model_client=SimpleNamespace(), stage_task={
        'node': 'improvement', 'role': 'paper_writer', 'full_stage': True,
        'instruction': 'Improve the compiled paper in two rounds',
    })
    worker = SimpleNamespace(status='completed', error=None, task_id='revision',
                             final='Plots revised', files=['figure.pdf'])
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=worker):
        result = main._delegate(None, {
            'node': 'paper_figure', 'role': 'paper_writer', 'full_stage': False,
            'instruction': 'Resolve round one plot readability comments',
        })
    assert not result.is_error
    assert main.turn.stop_reason is None
    assert main.turn.receipts['paper_figure']['status'] == 'completed'


def test_explicit_improvement_keeps_full_stage_scope_outside_revision(tmp_path):
    main = CoordinatorAgent(tmp_path, model_client=SimpleNamespace(), stage_task={
        'node': 'improvement', 'role': 'paper_writer', 'full_stage': True,
        'instruction': 'Improve the paper',
    })
    with pytest.raises(ValueError, match='explicit assignment'):
        main._delegate(None, {
            'node': 'experiment_bridge', 'role': 'experiment', 'full_stage': True,
            'instruction': 'Rerun the entire experiment stage',
        })


def test_main_receives_full_review_and_keeps_improvement_pending(tmp_path):
    main = CoordinatorAgent(tmp_path, model_client=SimpleNamespace())
    main.project_goal = 'Research'
    handoff = {'round': 1, 'report': 'paper/review_round1.md',
               'content': 'MAJOR: revise plot labels', 'next_action': 'Delegate plot revision'}
    state = WorkflowState('review-task', status='partial', error='Revisions pending',
                          node_results={'improvement': {'review_handoff': handoff, 'final': 'Review ready'}})
    state.artifacts['improvement'] = [{'path': handoff['report']}]
    with patch('batteryscientist.agent.coordinator.execute_stage', return_value=state):
        result = main._delegate(None, {
            'node': 'improvement', 'role': 'paper_writer', 'full_stage': True,
            'instruction': 'Review the paper',
        })
    assert json.loads(result.output)['review_handoff'] == handoff
    assert main.turn.receipts['improvement']['status'] == 'partial'
    assert main.turn.stop_reason is None
    assert not main.turn.archive_due
