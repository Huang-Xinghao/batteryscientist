# Wiki 代码入口

模型调用的工具名称不变，内部按职责组织：

| 修改内容 | 文件 |
| --- | --- |
| 搜索、读取工具的参数和返回结果 | `search.py` |
| Markdown 解析、关键词排名和证据摘录 | `retrieval.py` |
| 语义搜索、向量缓存和沿关系图推荐 | `hybrid.py` |
| 入库任务、原文准备、进度恢复和工具入口 | `ingest.py` |
| 原文证据校验、独立模型审核 | `review.py` |
| 发布草稿校验、配图、页面发布和旧产物归档 | `publish.py` |
| 关系类型、论文关系和参考文献匹配 | `relations.py` |
| 目录、元数据路径、页面读写和索引 | `storage.py` |
| 模型提示词 | `prompts.py` |
| 按需提取、检索和读取论文写作卡片 | `writing.py` |
| 知识库维护命令 | `cli.py` |

查询流程：`search.py` 调用 `retrieval.py` 或 `hybrid.py`，再读取选中条目。

入库流程：`ingest.py` 准备原文 → Agent 编写草稿 → `review.py` 审核 →
`publish.py` 发布。页面文件由 `storage.py` 管理，引用和关系由
`relations.py` 管理。

通用调用方从 `batteryscientist.tools.wiki` 导入 `search_wiki`、`read_wiki`
及知识库路径入口；Wiki 角色从 `batteryscientist.tools.wiki.ingest` 导入
`WikiIngestion`。

维护命令：`python -m batteryscientist.tools.wiki.cli --help`。

写法分支独立于知识入库：`extract_writing_card(paper_id, focus, user_goal)` 复用已入库原文，
用 wiki_agent 的模型配置生成卡片，校验来源 ID 与英文摘录后保存到
`writing/<collection>/<paper_slug>/<focus>-<goal_hash>.json` 及同名中文 Markdown。
同一来源和目标复用缓存；源文件变动后重新生成。`search_writing_cards` 支持词法检索，
`read_writing_card` 返回完整分析、依据与过期状态。主控可直接提取，写作阶段仅检索读取。
没有原文缓存时先整理入库，不根据 Wiki 摘要冒充原论文拆解。程序检查不替代语义审核。

命令行示例（沿用仓库 `.env` 中的模型配置）：

```bash
python -m batteryscientist.tools.wiki.cli writing_card paper:example \
  --focus introduction --user-goal '学习如何从场景约束引出方法'
```

`read_wiki` 会附带该论文已有写作卡片的路径；卡片不纳入科学证据的 Wiki 排名。
