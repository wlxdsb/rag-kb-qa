# 本地知识库问答助手（Local RAG QA）

把一批本地文档（PDF / Markdown / TXT）建成知识库，用自然语言提问，得到**基于文档、带出处**的回答。

面向智算/运维场景：可导入运维手册、故障 SOP 等垂直领域资料，作为现场问答助手。

## 架构与流程

```
                 ┌──────────── 离线：建库 (ingest.py) ────────────┐
 data/ 文档  →  加载  →  文本切分(chunk)  →  BGE 向量化  →  Chroma 向量库
                 └────────────────────────────────────────────────┘

                 ┌──────────── 在线：问答 (qa.py / app.py) ───────┐
 用户提问  →  BGE 向量化  →  相似度检索 top-k  →  拼 Prompt  →  DeepSeek 生成
                                                                    ↓
                                                        带出处的答案
                 └────────────────────────────────────────────────┘
```

## 技术选型（及原因）

| 模块 | 选型 | 原因 |
| --- | --- | --- |
| 编排框架 | LangChain | 生态成熟，RAG 组件齐全 |
| 向量库 | Chroma | 本地持久化、零部署，适合单机知识库 |
| Embedding | BAAI/bge-small-zh-v1.5（fastembed） | 中文语义强；ONNX 运行，免装 torch |
| 大模型 | DeepSeek（deepseek-chat） | 中文好、成本低 |
| 界面 | Streamlit | 快速做出可演示网页 |

## 目录结构

```
rag-kb-qa/
├─ config.py          参数与模型工厂（切分大小、top-k、embedding、LLM）
├─ ingest.py          建库：加载 → 切分 → 向量化 → 存 Chroma
├─ qa.py              问答：检索 → 拼 Prompt → 生成
├─ app.py             Streamlit 网页界面
├─ eval/run_eval.py   检索命中率评测
├─ eval/questions.jsonl  评测集
├─ data/              知识库文档（含 1 份示例）
├─ requirements.txt
└─ .env.example       环境变量模板
```

## 快速开始

```bash
# 0) 用 Python 3.13 建虚拟环境（不要用 3.14）
py -3.13 -m venv .venv
.venv\Scripts\activate

# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置 API Key：复制 .env.example 为 .env，填入 DeepSeek Key
copy .env.example .env

# 3) 建库（首次会下载 BGE 模型，约 90MB）
python ingest.py

# 4a) 命令行问答
python qa.py

# 4b) 或网页问答
streamlit run app.py

# 5) 检索评测
python eval/run_eval.py
```

## 关键概念（面试要能讲）

- **切分（chunking）**：把长文档切成 ~500 字的小块。块太大 → 检索噪声多；太小 → 语义被切断。用重叠（overlap）缓解。
- **Embedding**：把文本映射成向量，语义相近的文本向量距离更近。
- **向量检索**：把问题也变成向量，在库中找最相似的 top-k 个块。
- **RAG 的意义**：让大模型基于**外部私有知识**回答，缓解幻觉、无需重新训练模型。
- **幻觉护栏**：Prompt 里强制"只能依据资料回答 + 不会就说不会 + 标注出处"。

## 评测

`eval/run_eval.py` 统计 top-k 检索是否命中预期关键词，输出**检索命中率**。

实测结果（示例知识库，5 题）：**检索命中率 5/5 = 100%**。

```
✅ 命中 | GPU 显存不足时应该怎么处理？
✅ 命中 | 磁盘写满、提示 No space left on device，怎么排查？
✅ 命中 | 推理服务时延升高、TTFT 劣化，如何优化？
✅ 命中 | 向量检索召回不准，可以从哪些方面优化？
✅ 命中 | 日志文件怎么做轮转和清理？
检索命中率：5/5 = 100%
```

## 下一步可扩展（面试加分项）

- 加入 **rerank**（重排序）提升召回质量；
- 把向量库换成 **Milvus**，支持更大规模；
- 用 **LangGraph** 把问答升级为可多步推理的 Agent；
- 接入 **MCP**，让助手能调用外部工具（如查监控、执行脚本）；
- 增加"答案正确率"的模型/人工评测。
