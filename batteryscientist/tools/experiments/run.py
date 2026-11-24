"""The same experiment queue tool for full stages and scoped follow-up work."""
from collections import Counter
import json

from ..core import ToolContext, ToolResult, tool
from .executor import ExperimentExecutor, AwaitingResults, ExperimentExecutionError
from .queue import run_queue


@tool()
def run_experiments(ctx: ToolContext) -> ToolResult:
    """Execute experiments/manifest.json through the resumable project queue.

    Waits until the queue finishes; model HTTP timeouts do not limit job duration.
    Reuses completed jobs with matching definitions and valid outputs. Uses the
    existing device, dependency, timeout and cumulative-budget settings. Returns
    a short summary and failure log paths; repair and call again to resume.
    """
    executor = ExperimentExecutor(ctx.workspace, knowledge_root=next(iter(ctx.knowledge_roots), None))
    value = {'queue': 'experiment-queue-state.json', 'tracker': 'refine-logs/EXPERIMENT_TRACKER.md'}
    try:
        manifest = executor.load_manifest()
        executor.validate_inputs(manifest)

        def progress(node, state, details):
            if ctx.on_progress:
                ctx.on_progress('node_progress', {'node': node, **details})

        attempts = run_queue(executor, None, manifest, on_progress=progress)
        value.update(status='completed', completed_jobs=len(attempts), total_jobs=len(manifest['jobs']))
    except (AwaitingResults, ValueError, RuntimeError, OSError) as exc:
        value.update(status='awaiting_results' if isinstance(exc, AwaitingResults) else 'failed',
                     error=str(exc), next_step='Read the indicated log, repair the cause and call run_experiments again.')
        if isinstance(exc, ExperimentExecutionError) and exc.receipt:
            receipt = exc.receipt
            value.update(job_id=receipt['job']['id'], log=receipt.get('log'))
            if receipt.get('log'):
                value['receipt'] = executor.output_path(receipt['log']).with_name('status.json').relative_to(ctx.workspace.resolve()).as_posix()
    queue = ctx.workspace / value['queue']
    if queue.is_file():
        snapshot = json.loads(queue.read_text())
        value['job_counts'] = dict(Counter(job['status'] for job in snapshot.get('jobs', [])))
    return ToolResult(json.dumps(value, ensure_ascii=False),
                      is_error=value['status'] != 'completed', structured_content=value)
