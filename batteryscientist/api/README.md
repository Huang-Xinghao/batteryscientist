# Web API

FastAPI 服务层，供 `batteryscientist-web` 前端使用。启动：`./run-web-api`，默认监听 `127.0.0.1:8000`（`BATTERYSCIENTIST_API_PORT`）。

| 文件 | 内容 |
| --- | --- |
| `app.py` | 项目、运行、文件、执行记录、SSE 事件流 |
| `schemas.py` | 所有 JSON 接口的响应模型 |
| `auth.py` | 单用户登录与会话 |
| `library.py` | 电池范式库、电池资源库只读浏览 |
| `idea_canvas.py` | 方案画布草稿与方案总览 |
| `laboratory.py` | 物理实验室设备只读接入 |
| `store.py` / `worker.py` | 管理数据库、独立运行进程 |

## 响应模型

每个返回 JSON 的接口都在装饰器上声明 `response_model`，模型集中在 `schemas.py`。只有文件下载、资源图片和 SSE 事件流不返回 JSON，不声明模型。

- 模型字段必须与处理函数实际返回的字段一致：FastAPI 会按模型过滤输出，漏写的字段会被静默丢掉。新增或修改返回字段时同步修改模型，并在 `tests/test_api_response_models.py` 中覆盖前端依赖的字段。
- 处理函数总会返回的字段写成必填（不给默认值），生成的前端类型才不会把它标成可选。
- 前端通过 `npm run api:generate` 从 `app.openapi()` 生成类型，因此改动模型后需要在前端仓库重新生成并提交 `src/api/schema.d.ts`。

## 轮询相关的开销

- 判断项目是否由终端命令行运行，需要遍历本机进程（`terminal_projects()`）。状态接口会被前端频繁读取，所以扫描结果缓存 5 秒（`TERMINAL_SCAN_TTL`）；启动运行前的冲突检查始终重新扫描（`fresh=True`）。
- `/api/projects/{id}/events` 建立连接后立即发送一行注释，让代理马上转发响应头、浏览器马上确认连接；之后每约 15 秒发送一次心跳。前端据此在连接正常时停止定时轮询。

## 方案审阅草稿

旧版方案审阅页会在项目里留下 `refine-logs/IDEA_CANVAS.json`（草稿）和 `IDEA_REVIEW.json`（修订记录）。草稿里有未处理的修改时，`require_resolved_proposal_review()` 会拒绝在研究精炼检查点直接批准方案。

- `GET /api/projects/{id}/proposal-overview` 的 `pendingReview` 字段说明是否有这样的草稿（修改条数、改过的章节、是否有修改意见或修订记录、是否无法读取），前端据此提示用户；判断逻辑与拦截共用 `pending_proposal_review()`。
- `POST /api/projects/{id}/idea-canvas/archive` 把草稿和修订记录移到 `refine-logs/archive/idea-review-<时间>/`，不删除；运行期间不可调用，成功后记录 `proposal_review_archived` 事件。
