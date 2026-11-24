"""Public response contracts, independent from the evolving workflow internals."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from batteryscientist.proposal_review import CanvasDraft


class NodeView(BaseModel):
    id: str
    label: str


class HealthView(BaseModel):
    status: str
    runtime: str
    project_roots: list[str]
    nodes: list[NodeView]
    concurrency: int


class CheckpointView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    node: str
    summary: str = ""
    questions: list[str] = Field(default_factory=list)
    report: str
    files: list[str] = Field(default_factory=list)
    decision_options: list[list[str]] | None = None
    report_sha256: str | None = None


class HistoryView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    node: str | None = None
    timestamp: str | None = None


class WorkflowView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str | None = None
    current_node: str | None = None
    project_goal: str = ""
    mode: str | None = None
    history: list[HistoryView] = Field(default_factory=list)
    expert_checkpoint: CheckpointView | None = None
    error: str | None = None
    paper_status: str | None = None


class RunView(BaseModel):
    id: str
    status: str
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    options: dict[str, Any]


class QuestionView(BaseModel):
    id: str
    task_id: str
    role: str
    phase: str
    question: str
    options: list[str] = Field(default_factory=list)
    status: str
    live: bool
    created_at: str


class QuestionAnswer(BaseModel):
    task_id: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=20000)


class AgentView(BaseModel):
    role: str
    status: str = 'idle'
    phase: str | None = None
    task_id: str | None = None
    instruction: str = ''
    updated_at: str | None = None
    error: str | None = None
    is_active: bool = False


class ProjectView(BaseModel):
    id: str
    name: str
    path: str
    created_at: str
    status: str
    state: WorkflowView
    progress: dict[str, Any]
    runs: list[RunView]
    checkpoint_token: str
    pending_questions: list[QuestionView] = Field(default_factory=list)
    agents: list[AgentView] = Field(default_factory=list)


# --- Responses of the remaining JSON endpoints --------------------------------
# Field sets mirror exactly what the handlers return, so declaring them changes
# no payload; the OpenAPI schema (and the generated web types) now covers them.


class RunAccepted(BaseModel):
    id: str
    status: str


class MessageBusy(BaseModel):
    """A message sent while a run is active is not executed."""

    intent: Literal["status"]
    reply: str
    accepted: bool
    run_id: str
    status: str
    progress: dict[str, Any]
    pending_questions: list[QuestionView] = Field(default_factory=list)


class AnswerAccepted(BaseModel):
    status: str
    question_id: str
    task_id: str


class StopAccepted(BaseModel):
    status: str


class ProjectFileView(BaseModel):
    path: str
    size: int
    modified: float


class FileListView(BaseModel):
    files: list[ProjectFileView]
    truncated: bool


class FileContentView(BaseModel):
    kind: Literal["text", "binary"]
    path: str
    size: int | None = None
    content: str | None = None
    sha256: str | None = None
    editable: bool | None = None


class LogView(BaseModel):
    content: str


class ActivityEventView(BaseModel):
    id: int
    project_id: str
    run_id: str | None = None
    kind: str
    data: dict[str, Any]
    created_at: str


class ActivityPageView(BaseModel):
    events: list[ActivityEventView]
    has_more: bool


class SessionView(BaseModel):
    enabled: bool
    authenticated: bool


class AuthResult(BaseModel):
    authenticated: bool


class LibraryEntryView(BaseModel):
    path: str
    title: str
    category: str
    excerpt: str
    modified: float


class LibraryCategoryView(BaseModel):
    id: str
    count: int


class LibraryIndexView(BaseModel):
    available: bool
    items: list[LibraryEntryView]
    categories: list[LibraryCategoryView]
    total: int
    matched: int
    truncated: bool


class LibraryDocumentView(LibraryEntryView):
    content: str


class GraphNodeView(LibraryEntryView):
    id: str
    tags: list[str] = Field(default_factory=list)


class GraphEdgeView(BaseModel):
    source: str
    target: str
    type: str
    evidence: str


class ParadigmGraphView(BaseModel):
    available: bool
    nodes: list[GraphNodeView]
    edges: list[GraphEdgeView]
    truncated: bool


class LibraryFigureView(BaseModel):
    id: str
    image_path: str
    paper_id: str
    paper_title: str
    paper_path: str | None = None
    label: str
    page: int | None = None
    caption: str


class LibraryFiguresView(BaseModel):
    available: bool
    items: list[LibraryFigureView]
    total: int
    truncated: bool


class ProposalOverviewView(BaseModel):
    sha256: str
    diagram: str
    input: str
    architecture: str
    output: str


class ReviewImpactView(BaseModel):
    kind: Literal["direct", "related"]
    title: str
    detail: str
    sectionIds: list[str] = Field(default_factory=list)


class ReviewQuestionView(BaseModel):
    id: str
    question: str
    sectionIds: list[str] = Field(default_factory=list)


class RevisionDiffView(BaseModel):
    id: str
    title: str
    beforeSummary: str
    afterSummary: str
    detailChanged: bool


class RevisionView(BaseModel):
    id: str | None = None
    summary: str | None = None
    impacts: list[ReviewImpactView] | None = None
    questions: list[ReviewQuestionView] | None = None
    diff: list[RevisionDiffView] | None = None
    createdAt: str | None = None
    approvedAt: str | None = None
    formalHash: str | None = None
    frozenPath: str | None = None
    synced: bool
    approved: bool


class RevisionProgressView(BaseModel):
    stage: str | None = None
    receivedChars: int = 0
    updatedAt: str


class CanvasOperationView(BaseModel):
    id: str
    status: str
    error: str | None = None
    createdAt: str
    finishedAt: str | None = None
    action: str
    progress: RevisionProgressView | None = None


class CanvasView(BaseModel):
    operation: CanvasOperationView | None = None
    sha256: str
    sourceHash: str
    draft: CanvasDraft | None = None
    revision: RevisionView | None = None
    canApprove: bool
    canApply: bool = False
    context: Literal["checkpoint", "auto", "historical", "unavailable"]
