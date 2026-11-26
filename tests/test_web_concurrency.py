"""Cross-project admission with real transactions and simulated workers."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from batteryscientist.api import store

api = importlib.import_module('batteryscientist.api.app')


class WebConcurrency(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.projects = {}
        for i in range(8):
            path = self.root / f'project-{i}'
            path.mkdir()
            self.projects[str(i)] = {'id': str(i), 'path': str(path), 'name': str(i)}
        self.env = patch.dict(os.environ, {'BATTERYSCIENTIST_WEB_DATA': str(self.root / 'web')})
        self.env.start()
        self.addCleanup(self.env.stop)
        store.initialize()
        with store.connection() as db:
            db.executemany('INSERT INTO projects(id,name,path,created_at) VALUES(?,?,?,?)',
                           [(key, key, project['path'], store.now()) for key, project in self.projects.items()])
        for target, kwargs in [
            ('project_record', {'side_effect': lambda key: self.projects[key]}),
            ('reconcile', {}),
            ('terminal_projects', {'return_value': set()}),
        ]:
            mock = patch.object(api, target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)
        self.worker = patch.object(api.subprocess, 'Popen', side_effect=lambda *a, **k:
                                   SimpleNamespace(pid=os.getpid(), wait=lambda: 0))
        self.spawn = self.worker.start()
        self.addCleanup(self.worker.stop)

    def start(self, project):
        try:
            return api.create_run(project, api.RunCreate(action='coordinate', prompt='concurrency test'))['status']
        except HTTPException as exc:
            return exc.status_code

    def test_eight_different_projects_can_start_concurrently(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(self.start, self.projects))
        self.assertEqual(result, ['queued'] * 8)
        self.assertEqual(self.spawn.call_count, 8)

    def test_same_project_admits_only_one_concurrent_request(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(self.start, ['0'] * 8))
        self.assertEqual(result.count('queued'), 1)
        self.assertEqual(result.count(409), 7)
        self.assertEqual(self.spawn.call_count, 1)

    def test_terminal_only_blocks_its_own_project(self):
        with patch.object(api, 'terminal_projects', return_value={self.projects['0']['path']}):
            self.assertEqual(self.start('1'), 'queued')
            self.assertEqual(self.start('0'), 409)
        self.assertEqual(self.spawn.call_count, 1)

    def test_live_question_displays_wait_without_releasing_active_run(self):
        from batteryscientist.agent.runtime.questions import UserQuestions

        self.assertEqual(self.start('0'), 'queued')
        project = self.projects['0']
        questions = UserQuestions(project['path'])
        question = questions.create('Which input?', role='experiment', task_id='task-1',
                                    phase='experiment_bridge')
        view = api.detail(project, terminal_paths=set())
        self.assertEqual(view['status'], 'awaiting_answer')
        self.assertEqual(view['progress']['status'], 'awaiting_answer')
        self.assertEqual(view['runs'][0]['status'], 'queued')
        self.assertEqual(self.start('0'), 409)
        questions.answer(question['id'], 'Use current input')
        self.assertEqual(api.detail(project, terminal_paths=set())['status'], 'queued')

    def test_stopping_run_keeps_stopping_display_with_pending_question(self):
        from batteryscientist.agent.runtime.questions import UserQuestions

        self.assertEqual(self.start('0'), 'queued')
        project = self.projects['0']
        UserQuestions(project['path']).create('Which input?', role='experiment',
                                              task_id='task-1', phase='experiment_bridge')
        run = store.rows('SELECT id FROM runs WHERE project_id=?', ('0',))[0]
        store.run_update(run['id'], status='stopping')
        self.assertEqual(api.detail(project, terminal_paths=set())['status'], 'stopping')


if __name__ == '__main__':
    unittest.main()
