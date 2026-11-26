"""Read legacy artifact payloads through the current file-reference contract."""

from pathlib import PurePosixPath, PureWindowsPath


# 仅供旧状态迁移：这些路径来自旧 WorkflowRunner 的交付约定。
# 新任务仍以实际登记的 files 为准，不强制使用固定文件名。
LEGACY_REPORTS = {
    'literature': 'literature/RESEARCH_REPORT.md',
    'data_search': 'data-search/DATA_REPORT.md',
    'ideas': 'idea-stage/IDEA_REPORT.md',
    'novelty': 'idea-stage/NOVELTY.json',
    'refine': 'refine-logs/FINAL_PROPOSAL.md',
    'experiment_plan': 'refine-logs/EXPERIMENT_PLAN.md',
    'experiment_bridge': 'refine-logs/EXPERIMENT_RESULTS.md',
    'paper_plan': 'PAPER_PLAN.md',
    'paper_figure': 'figures/latex_includes.tex',
    'paper_write': 'paper/main.tex',
    'paper_compile': 'paper/main.pdf',
    'improvement': 'paper/main.pdf',
}


def _reference(path):
    if not isinstance(path, str) or not path or any(c in path for c in ('\n', '\r', '\x00')):
        raise ValueError('工作流产物引用必须是项目内的文件路径')
    normalized = path.replace('\\', '/')
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or PureWindowsPath(path).drive or '..' in relative.parts or '://' in path:
        raise ValueError('工作流产物引用不能位于项目之外：' + path)
    return {'path': normalized}


def artifact_references(state):
    """Normalize old text/path payloads without treating report prose as a path.

    References identify files to read; they do not assert file existence, content
    validity, or approval. Existing file/approval checks remain authoritative.
    """
    result = {}
    for node, refs in state.get('artifacts', {}).items():
        if isinstance(refs, str):
            metadata = state.get('node_results', {}).get(node, {})
            checkpoint = state.get('expert_checkpoint') or {}
            paths = metadata.get('files') or []
            if not paths and metadata.get('canonical_path'):
                paths = [metadata['canonical_path']]
            if not paths and checkpoint.get('node') == node:
                paths = checkpoint.get('files') or ([checkpoint['report']] if checkpoint.get('report') else [])
            # 旧 artifacts 既可能是正文/总结，也可能是单个相对路径。
            if not paths and refs and len(refs) < 1024 and not any(c.isspace() for c in refs) and PurePosixPath(refs).suffix:
                paths = [refs]
            if not paths and node in LEGACY_REPORTS:
                paths = [LEGACY_REPORTS[node]]
            if not isinstance(paths, list) or not paths:
                raise ValueError('旧阶段产物缺少文件引用，请登记交付文件：' + node)
            result[node] = [_reference(path) for path in dict.fromkeys(paths)]
        elif isinstance(refs, list):
            normalized = []
            for ref in refs:
                if not isinstance(ref, dict) or 'path' not in ref:
                    raise ValueError('工作流产物必须是包含 path 的记录：' + node)
                normalized.append({**ref, **_reference(ref['path'])})
            result[node] = normalized
        else:
            raise ValueError('无法识别工作流产物格式：' + node)
    return result
