"""Budgeted queues admit independent jobs on different GPUs."""
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import json

from batteryscientist.project_config import validate_execution_config
from batteryscientist.tools.experiments import queue
from batteryscientist.tools.experiments.executor import ExperimentExecutor, AwaitingResults


class ParallelBudget(unittest.TestCase):
    def schedule(self, total):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            barrier, lock = Barrier(2), Lock()
            state = {'spent': 0, 'active': 0, 'peak': 0, 'reserved': 0, 'peak_reserved': 0}
            allocations = []
            def execute(job, *, resources, cancel_event, budget_allocation):
                with lock:
                    ordinal = len(allocations)
                    allocations.append((resources['gpu_index'], budget_allocation[1]))
                    state['active'] += 1
                    state['peak'] = max(state['peak'], state['active'])
                    state['reserved'] += budget_allocation[1]
                    state['peak_reserved'] = max(state['peak_reserved'], state['reserved'] + state['spent'])
                if ordinal < 2:
                    barrier.wait(timeout=5)
                with lock:
                    state['active'] -= 1
                    state['reserved'] -= budget_allocation[1]
                    state['spent'] += 1
                return {'status': 'completed', 'job': job, 'elapsed_seconds': 1}
            executor = SimpleNamespace(project=root, seconds_used=lambda **kw: state['spent'],
                resolve_resources=lambda job: {'allocated_device': 'cuda'}, execute=execute)
            for target, value in {
                'read_plan_blocks': [], 'recover_receipts': [],
                'load_compute_config': {'device': 'auto', 'gpu_index': None, 'total_experiment_seconds': total},
                'load_execution_config': validate_execution_config({'gpu_ids': [0, 1], 'poll_seconds': 0.01}),
                'discover_gpus': {i: {'token': f'test-gpu-{i}', 'memory_used_mib': 0} for i in range(2)},
            }.items():
                stack.enter_context(patch.object(queue, target, return_value=value))
            stack.enter_context(patch.object(queue.Path, 'home', return_value=root))
            jobs = [{'id': str(i), 'depends_on': [], 'timeout_seconds': 10,
                     'resources': {'device': 'cuda'}} for i in range(3)]
            results = queue._schedule(executor, None, {'jobs': jobs}, None, track=False)
            self.assertEqual(len(results), 3)
            self.assertEqual(state['peak'], 2)
            self.assertEqual({g for g, _ in allocations[:2]}, {0, 1})
            self.assertLessEqual(state['peak_reserved'], total)
            self.assertFalse(json.loads((root / 'experiment-queue-state.json').read_text())['budget_serialized'])
            return allocations

    def test_parallel_with_total_budget(self):
        self.schedule(30)

    def test_small_budget_caps_second_job_and_recycles_unused_time(self):
        allocations = self.schedule(15)
        self.assertEqual(sorted(limit for _, limit in allocations[:2]), [5, 10])
        self.assertGreater(allocations[2][1], 0)

    def test_accounting_skips_active_receipts_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, receipt in [('active', {'status': 'running'}),
                                  ('done', {'status': 'completed', 'elapsed_seconds': 3})]:
                folder = root / 'experiment-runs' / name
                folder.mkdir(parents=True)
                (folder / 'status.json').write_text(json.dumps(receipt))
            executor = ExperimentExecutor(root)
            with self.assertRaises(AwaitingResults):
                executor.seconds_used()
            self.assertEqual(executor.seconds_used(ignore_running=True), 3)


if __name__ == '__main__':
    unittest.main()
