"""Persistent main agent: serial, bounded assignments through existing validators."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from batteryscientist.tools import BUILTIN_TOOLS, ToolContext, ToolResult, ToolSpec
from batteryscientist.project_config import load_compute_config, load_execution_config, resolve_battery_root
from batteryscientist.tools.data_access import registered_data_sources
from batteryscientist.project_config import AUTO_QUESTION_REQUIREMENT, interaction_instructions
from .loop import AgentLoop, ArtifactValidationError
from .context import CONTEXT_MAX_TOKENS
from batteryscientist.integrations.model import ModelClient, load_env_file, role_model, role_connection
from batteryscientist.agent.runtime.sessions import RoleSession, now
from .stages import BY_NODE, WORKFLOW_NODE_NAMES, ROLE_DESCRIPTIONS
from .state_compat import artifact_references
from batteryscientist.agent.runtime.execution import dispatch_stage as execute_stage
from batteryscientist.agent.runtime.execution import StageAssignment
from batteryscientist.storage import write_json as _write_json
from batteryscientist.storage import read_json_or_default as read_json


def project_context(project, query=''):
    root = Path(project['path'])
    state = read_json(root / 'workflow-state.json', {})
    files = []
    # Bounded inventory; no paper archives, datasets, checkpoints or credentials.
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in {
            'audit', 'backup', 'backups', '__pycache__', 'node_modules', 'experiment-runs', 'results'})
        for name in sorted(names):
            p = Path(base) / name
            relative = p.relative_to(root).as_posix()
            if (not p.is_symlink() and not name.startswith('.') and
                    p.suffix.lower() in {'.md', '.tex', '.bib', '.py', '.png', '.pdf', '.svg', '.jpg'} and
                    (not query or query.lower() in relative.lower())):
                files.append(relative)
                if len(files) >= 400:
                    break
        if len(files) >= 400:
            break
    return {'name': project['name'], 'status': project.get('status', state.get('status', 'idle')),
            'current_node': state.get('current_node'), 'failed_node': state.get('failed_node'),
            'checkpoint': (state.get('expert_checkpoint') or {}).get('node'),
            'research_goal': state.get('project_goal', '')[:6000],
            'progress': read_json(root / 'workflow-progress.json', {}).get('last_event'),
            'files': files, 'truncated': len(files) >= 400}


SYSTEM = """## 1. 主控 Agent 的定义
- 你是研究项目的主控 Agent（main）。{main}
- 维护项目全局视角，向用户解释进展和研究取舍，协调执行角色完成研究。
- 简单问答、概念解释、进度查询和讨论无需调用 set_request_scope，直接回答即可；文档修改由编辑工具自动记录。
- 执行研究任务前用 set_request_scope(kind="execution", nodes=[...]) 记录需要完成的阶段，例如补数据并跑实验包含 data_search 和 experiment_bridge；已登记的执行范围不能降级为问答来提前结束。

## 2. 执行 Agent 的定义
- 执行角色各自保留独立会话，按派单加载阶段 Skill。主控确定目标和交付范围，执行角色确定具体实现，不模糊各个执行agent的每个阶段都的边界（例如让literature去做datasearch相关报告）。
- idea_generator：{idea_generator}
  - literature / research-lit：梳理已有研究、技术路线和研究缺口，从宽泛和交叉的视角整理研究问题，不对数据进行深入分析。
  - ideas / idea-creator：生成新的研究想法，不考虑预算和资源限制。
  - refine / research-refine：明确问题、方法机制、输入输出和贡献，不负责实验设计、论文定位、训练设计。
- experiment：{experiment}
  - experiment_plan / experiment-plan：围绕研究问题设计有故事性的对照、消融和诊断等实验，不负责实验具体执行。
  - experiment_bridge / experiment-bridge：基于项目模板实现方法，完成代码审阅、运行验证、实验执行和结果分析，交付实验报告。
  - run-experiment：experiment_bridge 内部的执行子 Skill，负责运行检查、队列、诊断和续跑。
- paper_writer：{paper_writer}
  - paper_plan / paper-plan：依据结果组织论文论点、章节、图表和引用。
  - paper_figure / paper-figure：生成图表、图注和 LaTeX 引用，检查视觉质量。
  - paper_write / paper-write：撰写正文、整合图表和引用，交付 LaTeX 源码。
  - paper_compile：程序编译节点；编译问题交由执行角色修复。
  - improvement / auto-paper-improvement-loop：默认一轮独立审稿，返回 review_handoff，由主控按意见派发相关角色修改；收集修改结果后继续 improvement 编译并交付。
- wiki_agent：{wiki_agent} 通过 curate_wiki 派单，按 wiki-curation Skill 整理来源、修订和发布 Wiki。

## 3. 研究目标与任务范围
- 用户提出研究任务后，结合对话和澄清答案调用 set_research_goal 记录总目标，再开始派单。
- 按用户意图组织研究：方法设计围绕具体场景、瓶颈和机制展开；评测、复现或应用任务围绕用户指定的问题展开。
- 文献与数据核对后，由 ideas 收敛研究方向，再细化方法和实验。评价方案时关注已有方法的不足、核心机制、与最近方法的区别及可验证的对照；依据不足时定向完善，如实保留负面或不确定结果。
- 用户扩展研究范围时更新总目标，当前要求写入派单。按本次请求完成完整研究、单阶段、局部修改或查询；历史记录用于理解背景。
- 默认阶段顺序：{workflow_order}。

## 4. 派单与持续推进
- 研究执行和代码、图表等产物修改使用 delegate_stage。完整执行一个阶段设 full_stage=true；局部修改、补充、检查和定向任务设 full_stage=false。研究完成后的修改默认按局部任务处理，除非用户要求完整重做阶段或继续完整研究。
- 局部任务完成后按本次要求交付；历史阶段的 partial 和 Skill 中完整阶段的其他要求，不自动成为本次待办。只有实际影响本次结果有效性的缺项才需处理，不为关闭旧回执重跑已有效完成的工作。
- 派单需要特定工具时在 assignment.required_tools 列出工具名（如 bash、run_experiments）；程序会按 node 对应阶段检查能力，role 不改变阶段工具权限。能力不匹配时调整到具备能力的阶段并保持局部任务范围，不反复向用户索取执行授权。
- 项目总目标由程序传递；assignment.instruction 写明本次目标、用户要求、必要输入和预期交付，保留用户对效果的原始描述。具体实验设计、实现、工具选择和视觉设计由执行角色依据 Skill 与指南确定，主控补充跨阶段约束和已确认的研究决定。
- 交付后读取关键产物，判断是否回答了本阶段的问题，再推进下一阶段。阴性或不确定结果也是有效结论；仅在缺项影响结论解释时安排必要补充。
- 研究收尾与 Wiki 整理遵循后附 research-memory Skill。
- 不派单关于预算、验收、回执等，由执行角色内部处理。

## 5. 产物检查与修改
- Markdown 修改先读取，再用 edit_project_markdown 精确编辑。修改 FINAL_PROPOSAL 时结合已有 IDEA_CANVAS.json 和 IDEA_REVIEW.json。
- 局部实现修改通过 delegate_stage(full_stage=false) 交给对应角色，说明修改范围和验证要求；完整重跑待审阶段使用 revise_checkpoint。
- 图表修改先确认 LaTeX 引用及对应的生成脚本或原图。
- 工具失败或结果不明时，查看回执、文件和错误原因，修复后恢复执行，复用已完成且有效的工作。

## 6. 用户审阅与协作
- 按当前协作模式推进；需要用户提供信息或作出取舍时使用 request_user_decision，答案写入相关文档并传给执行角色。
- 专家模式下，refine 完成后解释方案，用 review_checkpoint 整理摘要和关于模型输入、架构、输出的必要问题；收到用户明确批准后通过 approve_checkpoint 推进。修改意见交回当前检查点修订。
- 自动模式下，由主控评估方案并继续下一阶段，任务内的常规选择自行决定。
- awaiting_user 时处理材料和用户反馈，收到答案后恢复派单；awaiting_results、failed 或 interrupted 时，说明原因并安排恢复。

## 7. 角色记录与研究资源
- role_cli 查询 list、status、history，当前项目已绑定。例如：history --role paper_writer --channel discussion。按 next_offset/next_char_offset 续读，status 可查看 archive 路径。
- 用户提到角色窗口讨论时，读取对应 discussion，将确认的结论用于文档和派单。
- battery_resources 的 search 检索资源，read 读取 resource_id，inspect 查看真实数据结构。

## 8. 面向用户的表达
- 耗时操作前和获得关键结果后，简要说明目标、发现和下一步。
- 围绕可核查的进展、证据和研究取舍交流。
- 说明完成内容、证据位置、尚缺内容和下一步，区分实际结果与后续计划。

## 9. 代码设计与实验执行
- 代码实现遵循资源库 `代码和绘图指南/代码设计`，由 experiment_bridge 根据研究方案和实验计划完成。
- 基于项目模板扩展模型、数据处理和任务逻辑，复用训练、验证、checkpoint 和结果导出流程；具体模块设计由执行角色确定。
- 实验参数与计算资源按指南和当前计划配置，通过 run_experiments 组织训练、评估和后处理，结合实际结果推进研究。
""".format(workflow_order=" → ".join(WORKFLOW_NODE_NAMES), **ROLE_DESCRIPTIONS)


@dataclass
class CoordinatorTurn:
    """Mutable state of one user request; project workflow state stays on disk."""
    status: str = 'completed'
    error: str | None = None
    stop_reason: str | None = None
    assignments: int = 0
    archive_due: bool = False
    approval_snapshot: dict | None = None
    checkpoint_revision: str | None = None
    approve_revision: bool = False
    edited_paths: set[str] = field(default_factory=set)
    read_ledger: dict = field(default_factory=dict)
    invalid_batch: bool = False
    request_kind: str | None = None
    required_nodes: list[str] = field(default_factory=list)
    receipts: dict[str, dict] = field(default_factory=dict)

    def accept_result(self, status, error=None):
        self.status, self.error = status, error
        # Definite outcomes may be inspected and retried as fresh assignments.
        self.stop_reason = (None if status in {'completed', 'partial', 'failed', 'interrupted', 'awaiting_results'}
                            else f'Worker returned {status}: {error or ""}')

    def pause(self, reason, *, status='awaiting_user'):
        self.status, self.stop_reason = status, reason

    def fail(self, error):
        self.status, self.error = 'failed', str(error)


class CoordinatorAgent(AgentLoop):
    auto_continue_iterations = True

    def __init__(self, project, *, model=None, model_client=None, max_iterations=None,
                 max_assignments=None, worker_max_iterations=300, timeout=600,
                 mode=None, on_event=None, context_max_tokens=CONTEXT_MAX_TOKENS,
                 execution_guard=None, stage_task=None,
                 workflow_options=None, checkpoint_revision=None, stop_after=None, first_node=None, approve_revision=False, project_goal=None,
                 wait_for_user=False):
        self.workflow_options = workflow_options or {}
        self.turn = CoordinatorTurn(checkpoint_revision=checkpoint_revision,
                                    approve_revision=approve_revision)
        self.stop_after = stop_after
        self.first_node = first_node
        self.project_goal = project_goal
        self.wait_for_user = wait_for_user
        self.project = Path(project).resolve()
        from batteryscientist.project_config import resolve_mode
        mode = resolve_mode(self.project, mode)
        repo = Path(__file__).resolve().parents[2]
        load_env_file(repo / ".env")
        load_env_file(repo / ".env.example")
        self.worker_model = model
        self.worker_max_iterations = worker_max_iterations
        self.timeout, self.mode = timeout, mode
        self.max_assignments = max_assignments if max_assignments is not None else (
            len(WORKFLOW_NODE_NAMES) + 8 if mode == 'auto' else 6)
        if max_iterations is None:
            max_iterations = 4 * len(WORKFLOW_NODE_NAMES) + 8 if mode == 'auto' else 24
        if mode not in {"auto", "expert"} or self.max_assignments < 1 or context_max_tokens < 1000:
            raise ValueError("Invalid coordinator mode or budget")
        self.context_max_tokens = context_max_tokens
        self.on_event = on_event
        self.execution_guard = execution_guard
        self.stage_task = StageAssignment.model_validate(stage_task) if stage_task else None
        self._active = None
        self.research_memory_skill = (repo / "batteryscientist/skills/research-memory/SKILL.md").read_text(encoding="utf-8")
        super().__init__(model_client, workspace=self.project, tools=self._registered_tools(),
                         max_iterations=max_iterations)
        self.knowledge_roots = (resolve_battery_root(self.project),)

    def _registered_tools(self):
        """Select exposed capabilities; AgentLoop derives parameter schemas from handlers."""
        return tuple(spec for spec in BUILTIN_TOOLS if spec.name in {
            'battery_resources', 'find', 'extract_writing_card',
            'search_writing_cards', 'read_writing_card',
        }) + (
            ToolSpec("role_cli", "只读查询角色：仅支持list/status/history。参数command是命令文本，无需--project。执行任务使用delegate_stage，Wiki整理使用curate_wiki。", self._role_cli, True),
            ToolSpec("get_project_context", "读取最新项目状态、产物目录和阶段任务。", self._context, True),
            ToolSpec("inspect_compute", "只读检查当前执行环境的CPU、CUDA/MPS可用性和项目计算限制。", self._inspect_compute, True),
            ToolSpec("set_research_goal", "依据用户明确的研究需求记录项目总目标goal；可整合此前的澄清答案。仅设置目标，不派单、不增加用户确认。", self._set_research_goal),
            ToolSpec("set_request_scope", "登记研究执行范围：kind=execution，nodes列出本次需要完成的阶段（Wiki用wiki）。普通问答和只读查询无需调用；文档编辑自动记录。兼容discussion/document类型，其nodes为空。", self._set_request_scope),
            ToolSpec("read_project_file", "读取项目中的文本产物。offset和limit均以字符计数，不是行数；默认读取12000字符，单次最多20000字符。通常只需提供真实path；需要续读时使用返回的next_offset。普通read工具才按行读取。", self._read, True),
            ToolSpec("edit_project_markdown", "主控直接落实用户要求：先读文件，再以path、old_text、new_text精确替换一处Markdown。node为所属工作流节点。修改不会批准或派单。", self._edit_markdown),
            ToolSpec("review_checkpoint", "主控理解待审产物后更新面向用户的summary、questions及可选decision_options；沿用当前检查点，不批准、不派单。", self._review_checkpoint),
            ToolSpec("approve_checkpoint", "仅落实用户对当前待审版本的明确批准，沿用工作流校验；无明确批准或版本变化时拒绝。", self._approve_checkpoint),
            ToolSpec("curate_wiki", "派单给正式wiki_agent角色，阅读、生成、独立审核、修订并发布共享Wiki。单篇PDF用project_evidence=false；研究收尾用true。无需先设置研究目标。", self._curate_wiki),
            ToolSpec("delegate_stage", "统一串行派单：assignment包含node、role、instruction、full_stage。完整研究阶段设full_stage=true，普通修改或检查设false（默认），按本次指令工作，不套整阶段验收。", self._delegate),
            ToolSpec("revise_checkpoint", "复杂研究修订才发回原角色；普通Markdown修改用edit_project_markdown。生成新版后等待审批，node必须匹配待审节点。", self._revise_checkpoint),
            ToolSpec("request_user_decision", "提问用户；context_paths 填写本问题依赖的项目内文件相对路径，文件变化会使回答失效。自动模式仅限已查证且无法自行解决的必要输入阻塞，必须填写blocking_reason。", self._ask),
        )

    def _tool_schemas(self):
        schemas = super()._tool_schemas()
        for schema in schemas:
            name = schema["function"]["name"]
            if name == "delegate_stage":
                key, model = "assignment", StageAssignment
                schema["function"]["parameters"] = {
                    "type": "object", "properties": {key: model.model_json_schema()},
                    "required": [key], "additionalProperties": False}
            elif name == 'review_checkpoint':
                schema['function']['parameters']['properties']['decision_options'] = {
                    'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}}}
            elif name == 'read_project_file':
                properties = schema['function']['parameters']['properties']
                properties['offset']['description'] = '跳过的字符数，从0开始；续读请使用返回的next_offset，不是行号。'
                properties['limit']['description'] = '读取的字符数，不是行数；默认12000，最多20000，通常省略即可。'
        return schemas

    def _context(self, ctx) -> ToolResult:
        value = project_context({"path": str(self.project), "name": self.project.name})
        value["phase_roles"] = {node: stage.role for node, stage in BY_NODE.items()}
        from batteryscientist.tools.profiles import PHASE_TOOL_NAMES
        value['phase_tools'] = {node: sorted(PHASE_TOOL_NAMES[stage.phase])
                                for node, stage in BY_NODE.items()}
        value["decision"] = self._load_decision()
        from batteryscientist.agent.runtime.questions import UserQuestions
        value['pending_questions'] = UserQuestions(self.project).pending()
        value["research_goal"] = self.project_goal or value["research_goal"]
        value['mode'] = self.mode
        value['dispatch_budget'] = self._dispatch_budget()
        value['battery_resource_roots'] = [str(path) for path in self.knowledge_roots]
        value['local_data_sources'] = registered_data_sources(self.knowledge_roots)
        value['data_access_note'] = ('只读取已登记且exists=true的数据源；列表为空表示没有已登记的本地原始数据。'
                                     '历史报告中的服务器路径不代表当前容器已挂载，不要反复猜测路径。')
        value['compute_policy'] = load_compute_config(self.project)
        value['execution_policy'] = load_execution_config(self.project)
        memory = read_json(self.project / '.batteryscientist/research-memory.json', {})
        value['research_memory'] = {key: memory.get(key) for key in
                                    ('status', 'wiki_path', 'idea_path', 'experiment_paths', 'error', 'updated_at')}
        return ToolResult(json.dumps(value, ensure_ascii=False))

    def _inspect_compute(self, ctx) -> ToolResult:
        from batteryscientist.tools.experiment_resources import inspect_compute
        return inspect_compute(ToolContext(self.project))

    def _set_research_goal(self, ctx, goal: str) -> ToolResult:
        goal = goal.strip()
        if not goal:
            raise ValueError('研究目标不能为空')
        from batteryscientist.agent.runtime.execution import project_write_lock
        with project_write_lock(self.project):
            path = self.project / 'workflow-state.json'
            if path.exists():
                saved = read_json(path, {})
                saved['project_goal'] = goal
                _write_json(path, saved)
        session, data = self._active.session, self._active.data
        self.project_goal = data['project_goal'] = goal
        session.save(data)
        return ToolResult(json.dumps({'project_goal': goal}, ensure_ascii=False))

    def _set_request_scope(self, ctx, kind: str, nodes: list[str] | None = None) -> ToolResult:
        nodes = list(dict.fromkeys(nodes or []))
        if kind not in {'discussion', 'document', 'execution'}:
            raise ValueError('kind应为discussion、document或execution')
        if any(node not in {*WORKFLOW_NODE_NAMES, 'wiki'} for node in nodes):
            raise ValueError('nodes应使用工作流阶段名称或wiki')
        if (kind == 'execution') != bool(nodes):
            raise ValueError('execution需要列出执行阶段；discussion和document的nodes为空')
        if self.turn.request_kind not in {None, 'discussion'} and (
                kind != self.turn.request_kind or nodes != self.turn.required_nodes):
            raise ValueError('本次请求范围已记录，按当前范围继续完成')
        self.turn.request_kind, self.turn.required_nodes = kind, nodes
        return ToolResult(json.dumps({'kind': kind, 'nodes': nodes}, ensure_ascii=False))

    def _dispatch_budget(self):
        return {'limit': self.max_assignments, 'used': self.turn.assignments,
                'remaining': max(0, self.max_assignments - self.turn.assignments)}

    async def _request_model(self, state, schemas, working_context):
        marker = '\n本轮执行状态：'
        state.messages[0]['content'] = state.messages[0]['content'].split(marker)[0] + marker + json.dumps({
            'dispatch_budget': self._dispatch_budget(), 'request_kind': self.turn.request_kind,
            'required_nodes': self.turn.required_nodes, 'receipts': self.turn.receipts,
            'stop_reason': self.turn.stop_reason,
        }, ensure_ascii=False)
        return await super()._request_model(state, schemas, working_context)


    def _curate_wiki(self, ctx, paper_path: str = "paper/main.pdf", project_evidence: bool = False,
                     source_paths: list[str] | None = None, instruction: str = "",
                     force: bool = False) -> ToolResult:
        if self.stage_task or self.stop_after not in (None, 'improvement'):
            raise ValueError('A scoped stage/revision request does not include Wiki publication')
        self._execution_gate(count_assignment=not project_evidence)
        self._record_execution_scope('wiki')
        ToolContext(self.project).resolve_read(paper_path)
        task_instruction = instruction.strip() or (
            '将最终论文与相关项目证据整理入共享Wiki，保留重要负结果与适用边界。' if project_evidence else
            '完整阅读指定论文，整理Paper、Idea和重要Experiment，经独立审核后发布到共享Wiki。')
        self._progress('archive_start', role='wiki_agent', paper_path=paper_path)
        if project_evidence:
            self.turn.archive_due = False
        try:
            state = execute_stage(
                self.project, 'wiki', task_instruction, kind='wiki', role='wiki_agent',
                project_goal=self.project_goal or '', paper_path=paper_path,
                project_evidence=project_evidence, source_paths=source_paths, force=force,
                model=role_model('wiki_agent', self.worker_model), timeout=self.timeout, max_iterations=self.worker_max_iterations,
                interaction_mode=self.mode, on_event=self._worker_event)
            result = state.wiki_result or {'status': state.status, 'error': state.error}
            if not project_evidence:
                self.turn.accept_result(state.status, state.error)
            # An unchanged, already curated source is a successful stage too.
            self.turn.receipts['wiki'] = {**result, 'status': state.status}
            result['dispatch_budget'] = self._dispatch_budget()
            self._progress('archive_end', **result)
            return ToolResult(json.dumps(result, ensure_ascii=False), state.status != 'completed')
        except Exception as exc:
            result = {'status': 'failed', 'role': 'wiki_agent', 'error': str(exc),
                      'message': 'Research artifacts are preserved. Retry curate_wiki separately.'}
            if not project_evidence:
                self.turn.fail(exc)
                self.turn.stop_reason = str(exc)
            self._progress('archive_end', **result)
            return ToolResult(json.dumps(result, ensure_ascii=False), True)

    def _complete_research(self):
        if not self.turn.request_kind:
            self.turn.request_kind = 'document' if self.turn.edited_paths else 'discussion'
        if self.turn.status == 'completed' and not self.turn.stop_reason:
            missing = [node for node in self.turn.required_nodes
                       if self.turn.receipts.get(node, {}).get('status') != 'completed']
            if missing:
                if self.turn.assignments >= self.max_assignments:
                    self.turn.pause('Assignment budget exhausted', status='partial')
                else:
                    raise ArtifactValidationError('本次执行尚未完成，请继续派单：' + '、'.join(missing)
                                                  + '。当前派单余额：' + str(self._dispatch_budget()['remaining']))
            if self.turn.request_kind == 'document' and not self.turn.edited_paths:
                raise ArtifactValidationError('本次文档修改尚未落实，请更新对应文件。')
        if self.turn.archive_due:
            raise ArtifactValidationError('完整研究已完成，请调用 curate_wiki(project_evidence=true) 沉淀成果后交付总结。')

    def _role_cli(self, ctx, command: str) -> ToolResult:
        from .role_cli import run
        value = run(command, project=self.project, read_only=True)
        return value if isinstance(value, ToolResult) else ToolResult(json.dumps(value, ensure_ascii=False))

    def _read(self, ctx, path: str, offset: int = 0, limit: int = 12000) -> ToolResult:
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("offset must be a nonnegative integer; limit must be a positive integer")
        limit = min(limit, 20000)
        relative = Path(path)
        target = (self.project / relative).resolve()
        if (relative.is_absolute() or not target.is_relative_to(self.project)
                or any(part.startswith(".") or part in {"audit", "backup", "backups"} for part in relative.parts)
                or any(part.startswith(".") for part in target.relative_to(self.project).parts)
                or target.suffix.lower() not in {".md", ".txt", ".json", ".csv", ".tex", ".bib", ".py"}):
            raise ValueError("Only bounded project text artifacts may be read")
        if relative.as_posix() == 'project-config.json' and not target.exists():
            # 配置文件可省略；返回有效默认值，不伪造文件或写入读取凭证。
            return ToolResult(json.dumps({
                'path': path, 'exists': False, 'source': 'defaults',
                'compute': load_compute_config(self.project),
                'execution': load_execution_config(self.project),
            }, ensure_ascii=False))
        if not target.is_file():
            raise FileNotFoundError('项目文件不存在：' + path)
        if target.stat().st_size > 512000:
            raise ValueError('项目文件过大，请使用专用工具读取：' + path)
        content = target.read_bytes()
        text = content.decode('utf-8')
        digest = hashlib.sha256(content).hexdigest()
        key = relative.as_posix()
        ledger = self.turn.read_ledger.get(key)
        end = min(len(text), offset + limit)
        if ledger and ledger["sha256"] == digest:
            covered = any(start <= offset and finish >= end for start, finish in ledger["ranges"])
            if covered:
                return ToolResult(json.dumps({"path": path, "cached": True,
                                              "sha256": digest,
                                              "message": "This unchanged range was already read in this coordinator invocation; use the prior tool result instead.",
                                              "next_offset": None if end >= len(text) else end},
                                             ensure_ascii=False))
        if not ledger or ledger["sha256"] != digest:
            ledger = {"sha256": digest, "ranges": []}
            self.turn.read_ledger[key] = ledger
        ledger["ranges"].append((offset, end))
        return ToolResult(json.dumps({"path": path, "content": text[offset:offset + limit],
                                     "sha256": digest,
                                     "next_offset": offset + limit if len(text) > offset + limit else None},
                                    ensure_ascii=False))

    def _require_current_read(self, path):
        target = (self.project / path).resolve()
        ledger = self.turn.read_ledger.get(path)
        if (not target.is_relative_to(self.project) or not target.is_file()
                or not ledger or ledger['sha256'] != hashlib.sha256(target.read_bytes()).hexdigest()):
            raise ValueError(f'先读取当前文件再操作：{path}')
        return target

    def _require_answered_questions(self):
        from batteryscientist.agent.runtime.questions import UserQuestions
        if UserQuestions(self.project).pending():
            self.turn.status = 'awaiting_user'
            raise ValueError('请先在网页问题卡片或用 /answer 回答待处理问题；查看进度不会解除等待')

    @contextmanager
    def _project_update(self, *, require_answers=False):
        """Serialize main-owned writes and exclude unfinished role execution."""
        from batteryscientist.agent.runtime.execution import project_write_lock
        from batteryscientist.agent.runtime.herdr import ensure_no_pending
        with project_write_lock(self.project):
            ensure_no_pending(self.project)
            if require_answers:
                self._require_answered_questions()
            yield

    def _edit_markdown(self, ctx, node: str, path: str, old_text: str, new_text: str) -> ToolResult:
        from .workflow_cli import refresh_revision_checkpoint
        from batteryscientist.proposal_review import CANVAS, RECORD, PROPOSAL, write_group
        relative = Path(path)
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or any(part.startswith('.') or part in {'audit', 'backup', 'backups'} for part in relative.parts)
                or relative.parts[0] in {'results', 'experiment-runs', 'papers'}
                or relative.name.startswith(('workflow-', 'experiment-queue', 'project-config'))):
            raise ValueError('不允许修改此路径：' + path)
        saved = read_json(self.project / 'workflow-state.json', {})
        owners = [n for n, refs in artifact_references(saved).items()
                  if any(ref['path'] == path for ref in refs)]
        if owners and node not in owners:
            raise ValueError('node必须是此产物所属的阶段')
        if self.stop_after and WORKFLOW_NODE_NAMES.index(node) > WORKFLOW_NODE_NAMES.index(self.stop_after):
            raise ValueError('修改超出本次用户授权的最后阶段')
        if Path(path).suffix.lower() != '.md':
            raise ValueError('主控直接编辑仅支持Markdown；代码或实验修改请委派角色')
        if self.stage_task and self.stage_task.node != node:
            raise ValueError('修改超出本次用户的精确任务范围')
        with self._project_update(require_answers=True):
            target = self._require_current_read(path)
            before = target.read_bytes().decode('utf-8')
            if not old_text or before.count(old_text) != 1:
                raise ValueError('old_text必须准确匹配当前文件中的唯一一处内容')
            after = before.replace(old_text, new_text, 1)
            if not after.strip() or len(after.encode('utf-8')) > 512000:
                raise ValueError('Markdown不能为空或超过512KB')
            if after == before:
                return ToolResult(json.dumps({'path': path, 'changed': False}))
            changes = {path: after.encode('utf-8')}
            if path == PROPOSAL:
                # The MD becomes the current source. Preserve old canvas edits for
                # traceability, rather than leaving a stale candidate able to replace it.
                for name in (CANVAS, RECORD):
                    draft = self.project / name
                    if draft.exists():
                        self._require_current_read(name)
                        changes[f'refine-logs/review-history/main-{uuid4().hex}-{draft.name}'] = draft.read_bytes()
            write_group(self.project, changes)
            if path == PROPOSAL:
                for name in (CANVAS, RECORD):
                    (self.project / name).unlink(missing_ok=True)
            self.turn.edited_paths.add(path)
            if self.turn.request_kind in {None, 'discussion'}:
                self.turn.request_kind = 'document'
            pending = refresh_revision_checkpoint(self.project, node, self.project_goal, files=[path])
            self.turn.approval_snapshot = None  # An answer approving the old text never approves this edit.
            if pending:
                self.turn.pause('Markdown已修订，等待用户审阅当前版本。')
            result = {'path': path, 'node': node, 'changed': True, 'status': self.turn.status,
                      'sha256': hashlib.sha256(after.encode('utf-8')).hexdigest(),
                      'message': '主控已直接更新MD；修改意见未被当作批准。'}
            self._progress('artifact_edited', **result)
            return ToolResult(json.dumps(result, ensure_ascii=False))

    def _review_checkpoint(self, ctx, summary: str, questions: list[str] | None = None,
                           decision_options: list[list[str]] | None = None) -> ToolResult:
        questions = questions or []
        if (not summary.strip() or any(not q.strip() for q in questions)
                or decision_options is not None and len(decision_options) != len(questions)):
            raise ValueError('请提供简短摘要及与实际取舍相关的问题，选项应逐题对应')
        with self._project_update():
            state = read_json(self.project / 'workflow-state.json', {})
            checkpoint = state.get('expert_checkpoint') or {}
            if state.get('status') != 'awaiting_user' or checkpoint.get('status') != 'pending':
                raise ValueError('没有待审检查点')
            checkpoint.update(summary=summary, questions=questions)
            checkpoint.pop('decision_options', None)
            if decision_options is not None:
                checkpoint['decision_options'] = decision_options
            _write_json(self.project / 'workflow-state.json', state)
            self.turn.status = 'awaiting_user'
            self._progress('coordinator_decision', checkpoint=checkpoint)
            return ToolResult(json.dumps(checkpoint, ensure_ascii=False))

    def _approve_checkpoint(self, ctx) -> ToolResult:
        from .workflow_cli import approve_checkpoint
        if self.turn.approval_snapshot is None:
            raise ValueError('需要用户明确批准当前版本；修改、讨论或“继续”不等于批准')
        self._require_answered_questions()
        state, boundary = approve_checkpoint(self.project, self.turn.approval_snapshot)
        self.turn.approval_snapshot = None
        self.turn.accept_result('completed')
        self.stop_after = self.stop_after or boundary
        if (state.status == "completed" and state.current_node == "improvement"
                and not self.stage_task):
            self.turn.archive_due = True
        self._progress('checkpoint_approved', node=state.expert_checkpoint['node'])
        return ToolResult(json.dumps({'approved': True, 'next_node': state.current_node,
                                     'status': state.status, 'stop_after': self.stop_after}, ensure_ascii=False))

    def _execution_gate(self, *, checkpoint_node=None, revision=False, count_assignment=True):
        state = read_json(self.project / "workflow-state.json", {})
        if self.turn.stop_reason and not (revision and state.get('status') == 'awaiting_user'
                                         and self.turn.status == 'awaiting_user'):
            raise ValueError(f"Dispatch stopped: {self.turn.stop_reason}")
        self._require_answered_questions()
        revising_upstream = (revision and checkpoint_node in WORKFLOW_NODE_NAMES
                             and state.get('current_node') in WORKFLOW_NODE_NAMES
                             and WORKFLOW_NODE_NAMES.index(checkpoint_node) <= WORKFLOW_NODE_NAMES.index(state['current_node']))
        if state.get("status") == "awaiting_user" and not (
                self.turn.checkpoint_revision and checkpoint_node == state.get("current_node") or revising_upstream):
            self.turn.status = "awaiting_user"
            raise ValueError("User checkpoint must be approved through the existing review endpoint")
        if not count_assignment:
            return
        if self.turn.assignments >= self.max_assignments:
            self.turn.pause("Assignment budget exhausted", status='partial')
            raise ValueError(self.turn.stop_reason)
        self.turn.assignments += 1

    def _record_execution_scope(self, node):
        # Actual dispatch cannot be reported as discussion, even when the model
        # omitted scope registration. Keep every declared stage for completion checks.
        self.turn.request_kind = 'execution'
        if node not in self.turn.required_nodes:
            self.turn.required_nodes.append(node)

    def _delegate(self, ctx, assignment: dict) -> ToolResult:
        task = StageAssignment.model_validate(assignment)
        from batteryscientist.tools.profiles import PHASE_TOOL_NAMES
        available = PHASE_TOOL_NAMES[BY_NODE[task.node].phase]
        missing = sorted(set(task.required_tools) - available)
        if missing:
            return ToolResult(
                f"未启动任务：阶段 {task.node} 缺少工具 {', '.join(missing)}。"
                "工具权限由阶段决定，切换角色或用户再次授权不会增加工具。"
                "请改派到具备所需工具的阶段，局部任务保持 full_stage=false 和原任务范围。", True)
        improvement_revision = (self.stage_task and self.stage_task.node == 'improvement'
                                and not task.full_stage
                                and read_json(self.project / 'workflow-state.json', {})
                                .get('improvement_checkpoint', {}).get('stage') == 'revision')
        if self.stage_task and not improvement_revision and (
                task.node != self.stage_task.node or task.role != self.stage_task.role
                or task.full_stage != self.stage_task.full_stage):
            raise ValueError("Execution must match the user's explicit assignment")
        if self.stage_task and task.instruction != self.stage_task.instruction:
            task = task.model_copy(update={"instruction": (
                "原始授权任务（必须遵守）：\n" + self.stage_task.instruction
                + "\n\n主控补充的阶段执行要求（不得改变原始授权范围）：\n" + task.instruction
            )})
        if self.first_node and self.turn.assignments == 0 and task.node != self.first_node:
            raise ValueError("First assignment must match the requested starting node")
        if self.stop_after and WORKFLOW_NODE_NAMES.index(task.node) > WORKFLOW_NODE_NAMES.index(self.stop_after):
            raise ValueError("Assignment exceeds the user's stop boundary")
        if task.full_stage and not self.project_goal:
            return ToolResult('研究目标尚未确定；根据用户的研究需求及澄清答案调用set_research_goal后再派单。', True)
        self._execution_gate(checkpoint_node=task.node, revision=not task.full_stage or bool(self.turn.checkpoint_revision))
        self._record_execution_scope(task.node)
        if task.full_stage and self.execution_guard is not None:
            self.execution_guard(task.node)
        self._progress("assignment_start", assignment=task.model_dump())
        state = execute_stage(self.project, task.node, task.instruction,
                             revise=self.turn.checkpoint_revision, project_goal=self.project_goal or "",
                             full_stage=task.full_stage,
                             role=task.role, model=role_model(task.role, self.worker_model),
                             mode=None if self.turn.checkpoint_revision else self.mode,
                             timeout=self.timeout, max_iterations=self.worker_max_iterations,
                             on_event=self._worker_event, workflow_stop_after=self.stop_after,
                             approve_revision=self.turn.approve_revision,
                             **self.workflow_options)
        self.turn.checkpoint_revision = None
        checkpoint = getattr(state, "expert_checkpoint", None)
        result = {"status": state.status, "error": state.error,
                  "task_id": state.run_id if task.full_stage else state.task_id,
                  "final": (state.node_results.get(task.node, {}).get("final", "")
                            if task.full_stage else state.final),
                  "artifacts": ([ref["path"] for ref in state.artifacts.get(task.node, [])]
                                if task.full_stage else state.files),
                  "checkpoint": {key: checkpoint[key] for key in
                                 ("node", "status", "summary", "questions", "decision_options")
                                 if key in checkpoint} if checkpoint else None}
        if task.full_stage:
            handoff = state.node_results.get(task.node, {}).get('review_handoff')
            if handoff:
                result['review_handoff'] = handoff
        self.turn.accept_result(state.status, state.error)
        self.turn.receipts[task.node] = dict(result)
        result['dispatch_budget'] = self._dispatch_budget()
        if (task.full_stage and task.node == 'improvement' and state.status == 'completed' and
                not self.stage_task and self.stop_after in (None, 'improvement')):
            self.turn.archive_due = True
            result['research_memory'] = 'Call curate_wiki(project_evidence=true) before summarizing.'
        explicit_task_done = (self.stage_task and task.node == self.stage_task.node
                              and task.full_stage == self.stage_task.full_stage)
        if state.status == 'completed' and (explicit_task_done or self.stop_after == task.node and not self.turn.archive_due):
            self.turn.stop_reason = "The explicitly assigned stage has ended; summarize and stop."
        self._progress("assignment_end", node=task.node, role=task.role, **result)
        return ToolResult(json.dumps(result, ensure_ascii=False), state.status in {"failed", "interrupted"})

    def _revise_checkpoint(self, ctx, node: str, instruction: str) -> ToolResult:
        state = read_json(self.project / 'workflow-state.json', {})
        checkpoint = state.get('expert_checkpoint') or {}
        if (state.get('status') != 'awaiting_user' or checkpoint.get('status') != 'pending'
                or checkpoint.get('node') != node or state.get('current_node') != node):
            raise ValueError('修订必须对应当前待审批节点')
        if self.stage_task:
            raise ValueError('本次已有精确任务，不可改为整阶段修订')
        self.turn.checkpoint_revision = instruction
        # The existing revision path preserves checkpoint guards and regenerates
        # the role's review summary; never autoapprove this newly revised version.
        self.turn.approve_revision = False
        from .workflow_cli import checkpoint_author
        result = self._delegate(ctx, {'node': node, 'role': checkpoint_author(self.project, node),
                                      'instruction': instruction, 'full_stage': True})
        self.turn.stop_reason = '新版已交付，请用户审阅并批准后再继续。'
        return result

    def _ask(self, ctx, question: str, options: list[str] | None = None,
             blocking_reason: str = '', context_paths: list[str] | None = None) -> ToolResult:
        if self.mode == 'auto' and not blocking_reason.strip():
            return ToolResult(AUTO_QUESTION_REQUIREMENT, True)
        from batteryscientist.agent.runtime.questions import UserQuestions
        from .workflow_cli import checkpoint_snapshot, is_approval
        snapshot = checkpoint_snapshot(self.project)
        store = UserQuestions(self.project)
        value = store.create(question, role='main', task_id=self._active.record['id'],
                             phase='coordinator', options=options, live=self.wait_for_user,
                             blocking_reason=blocking_reason.strip(), context_paths=context_paths)
        if self.wait_for_user:
            session, data, task = self._active.session, self._active.data, self._active.record
            task.update(status='awaiting_user', question_id=value['id'])
            session.save(data)
            answer = store.wait(value, self._progress)
            self.turn.approval_snapshot = snapshot if is_approval(answer) else None
            task.update(status='running')
            session.save(data)
            return ToolResult(json.dumps({'question_id': value['id'], 'answer': answer}, ensure_ascii=False))
        self.turn.pause(question)
        self._progress("user_question", question=value)
        return ToolResult("Question recorded. Stop dispatching and present this question to the user.")

    def _load_decision(self):
        from batteryscientist.agent.runtime.questions import UserQuestions
        return UserQuestions(self.project).latest(role='main')

    async def _invoke(self, spec, arguments):
        # Stage runtimes own asyncio.run(); keep their loops out of this event loop.
        critical = self._critical_operation(spec.name)
        if critical and self.turn.invalid_batch:
            self.turn.stop_reason = "Multiple critical operations in one response"
            self.turn.fail(self.turn.stop_reason)
            raise ValueError("Submit at most one critical operation per model response")
        try:
            ctx = ToolContext(self.project, self.skill_dirs, self.knowledge_roots)
            result = await asyncio.to_thread(spec.handler, ctx, **arguments)
            result.metadata['dispatch_budget'] = self._dispatch_budget()
            try:
                output = json.loads(result.output)
            except (ValueError, TypeError):
                output = None
            if isinstance(output, dict):
                output['dispatch_budget'] = self._dispatch_budget()
                result.output = json.dumps(output, ensure_ascii=False)
            return result
        except asyncio.CancelledError:
            # asyncio's first SIGINT cancels the coroutine. Release any question
            # wait in the executor before asyncio.run joins its worker threads.
            import os
            from batteryscientist.agent.runtime.questions import UserQuestions
            UserQuestions(self.project).interrupt_owned(os.getpid())
            raise
        except Exception as exc:
            if critical and spec.name != 'edit_project_markdown':
                if self.turn.status == "completed":
                    self.turn.fail(exc)
            raise

    def _worker_event(self, record):
        if self.on_event:
            self.on_event(record)

    @staticmethod
    def _critical_operation(name):
        return name in {'delegate_stage', 'revise_checkpoint', 'request_user_decision',
                        'curate_wiki', 'edit_project_markdown', 'review_checkpoint', 'approve_checkpoint'}

    def _prepare_tool_batch(self, calls):
        self.turn.invalid_batch = sum(self._critical_operation(call.name) for call in calls) > 1

    def _progress(self, kind, **data):
        if self.on_event:
            identifiers = {}
            if self._active:
                identifiers = {"session_id": self._active.data["session_id"],
                               "task_id": self._active.record["id"]}
            self.on_event({"kind": kind, "time": now(), "role": "main", **identifiers, **data})

    def _save_conversation(self, state):
        if not self._active:
            return
        session, data, task = self._active.session, self._active.data, self._active.record
        session.save_conversation(data, task, state)

    def _working_context_compacted(self, phase, iteration):
        # A summary may replace the earlier read result. Allow evidence to be
        # fetched again rather than directing the model to invisible history.
        self.turn.read_ledger = {}
        super()._working_context_compacted(phase, iteration)

    def dispatch(self, task: StageAssignment) -> ToolResult:
        """Execute an explicit CLI assignment without another main-model turn."""
        with RoleSession(self.project, 'main').locked():
            saved = read_json(self.project / 'workflow-state.json', {})
            main = RoleSession(self.project, 'main').load()
            self.project_goal = saved.get('project_goal') or main.get('project_goal') or task.instruction
            return self._delegate(None, task)

    def go(self, prompt, *, archive_completed=False):
        if not prompt.strip():
            raise ValueError("Main assignment must not be empty")
        if self.model is None:
            self.model = ModelClient(model=role_model('main', self.worker_model),
                                     **role_connection('main'), timeout=self.timeout)
        self.project.mkdir(parents=True, exist_ok=True)
        saved = read_json(self.project / "workflow-state.json", {})
        from .workflow_cli import checkpoint_snapshot, is_approval
        self.turn.approval_snapshot = checkpoint_snapshot(self.project) if is_approval(prompt) else None
        self.project_goal = saved.get("project_goal") or self.project_goal
        session = RoleSession(self.project, "main")
        def prepare_session(data, task):
            nonlocal prompt
            self.project_goal = self.project_goal or data.get('project_goal') or ''
            data['project_goal'] = self.project_goal
            if prompt.startswith('/answer '):
                from batteryscientist.agent.runtime.questions import answer_command
                answered = answer_command(self.project, prompt)
                self.turn.approval_snapshot = checkpoint_snapshot(self.project) if is_approval(answered['answer']) else None
                prompt = f"用户已正式回答问题 [{answered['id']}]：{answered['answer']}"
            task['instruction'] = prompt

        task = {"id": uuid4().hex, "instruction": prompt}
        self.last_state = None
        try:
            with session.task(task, model=self.model, max_tokens=self.context_max_tokens,
                              get_state=lambda: self.last_state, before_start=prepare_session,
                              on_compacting=lambda: self._context_compaction_event(True),
                              on_compacted=lambda: self._context_compaction_event(False)) as active:
                self._active = active
                data, history = active.data, active.history
                recovered_results = active.recovered_results
                self.turn = CoordinatorTurn(
                    status='awaiting_user' if saved.get('status') == 'awaiting_user' else 'completed',
                    archive_due=archive_completed, approval_snapshot=self.turn.approval_snapshot,
                    checkpoint_revision=self.turn.checkpoint_revision,
                    approve_revision=self.turn.approve_revision)
                if self.stage_task:
                    self.turn.request_kind = 'execution'
                    self.turn.required_nodes = [self.stage_task.node]
                decision = self._load_decision()
                if decision.get('status') == 'awaiting_user':
                    self.turn.status = 'awaiting_user'
                context = json.dumps({
                    "project_path": str(self.project), "research_goal": self.project_goal,
                    "mode": self.mode, "workflow_state_path": "workflow-state.json",
                    "project_config_path": "project-config.json",
                    "project_config_exists": (self.project / 'project-config.json').is_file(),
                    "local_data_sources": registered_data_sources(self.knowledge_roots),
                    "data_access_note": "仅exists=true的数据源已在本环境可用；空列表表示未登记本地原始数据，历史服务器路径不代表已挂载。",
                }, ensure_ascii=False)
                context += '\n派单次数按本次请求重新计数，以本轮执行状态和工具回执中的dispatch_budget为准。'
                if recovered_results:
                    context += "\n上次中断有待恢复的工具结果，请结合当前状态、文件和任务回执复用已有工作并继续。"
                if self.first_node:
                    context += f"\n本次从 {self.first_node} 阶段开始，逐节点分配。"
                if self.stop_after:
                    context += f"\n本次任务到 {self.stop_after} 阶段结束，完成后总结。"
                if self.stage_task:
                    context += "\n本次指定任务（按给定角色、阶段和范围执行，可补充实现细节，完成后总结）：\n" + json.dumps(
                        {"delegate_stage": self.stage_task.model_dump()}, ensure_ascii=False)
                result = self.run(prompt, system_prompt=SYSTEM + "\n" + self.research_memory_skill + "\n当前项目：\n" + context
                                  + interaction_instructions(self.mode),
                                  phase="coordinator", history=history, completion_check=self._complete_research)
                if (self.stage_task and not self.turn.edited_paths) and self.turn.assignments == 0:
                    raise RuntimeError("Main agent did not execute the explicit assignment")
                if not result.final.strip():
                    raise RuntimeError("Main agent returned an empty summary")
                outcome = ('execution_completed' if self.turn.request_kind == 'execution' else
                           'document_updated' if self.turn.request_kind == 'document' else 'answered')
                if self.turn.status != 'completed':
                    outcome = self.turn.status
                task.update(status=self.turn.status, final=result.final, error=self.turn.error,
                            outcome=outcome, request_kind=self.turn.request_kind,
                            required_nodes=self.turn.required_nodes, receipts=self.turn.receipts,
                            dispatch_budget=self._dispatch_budget())
                return SimpleNamespace(status=self.turn.status, error=self.turn.error,
                                       final=result.final, outcome=outcome, task_id=task["id"], session_id=data["session_id"])
        finally:
            self._active = None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the persistent research coordinator")
    parser.add_argument("prompt")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=["auto", "expert"], default="auto")
    parser.add_argument("--max-assignments", type=int, default=None)
    parser.add_argument("--max-tool-rounds", type=int, default=300)
    args = parser.parse_args()

    def guard(node):
        if node == "experiment_bridge":
            from batteryscientist.tools.experiments.executor import check_runtime
            check_runtime()

    result = CoordinatorAgent(args.project, model=args.model, mode=args.mode,
                              max_assignments=args.max_assignments,
                              worker_max_iterations=args.max_tool_rounds,
                              execution_guard=guard).go(args.prompt)
    print(result.final)
    if result.status in {"failed", "interrupted"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
