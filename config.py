"""集中配置：读取 .env，构造大模型（DeepSeek）与 embedding（BGE）。

这些参数都是"面试可讲"的：为什么设 500？为什么要重叠？为什么用 BGE？
"""

import os

from dotenv import load_dotenv

load_dotenv()  # 从同目录的 .env 读取 DEEPSEEK_API_KEY

# 国内直连 HuggingFace 常超时，默认走镜像下载模型（可用环境变量覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------- 路径 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")        # 你的知识库文档放这里
DB_DIR = os.path.join(ROOT, "chroma_db")     # 向量库持久化目录

# ---------- 检索参数 ----------
# chunk_size：每个文本块约 500 字。太大 → 检索噪声多；太小 → 上下文被切断。
CHUNK_SIZE = 500
# chunk_overlap：相邻块重叠 80 字，避免句子被切断导致语义丢失。
CHUNK_OVERLAP = 80
# top_k：每次检索返回最相似的 4 个块喂给大模型。
TOP_K = 4

# ---------- 模型 ----------
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文语义向量模型（本地、免费）
LLM_MODEL = "deepseek-chat"             # DeepSeek 对话模型


def get_embeddings():
    """返回一个 LangChain 兼容的 BGE embedding 对象。

    直接用 fastembed（底层 ONNX），不依赖 torch，装得快、跑得轻。
    """
    from fastembed import TextEmbedding
    from langchain_core.embeddings import Embeddings

    class FastEmbedBGE(Embeddings):
        def __init__(self, model_name: str):
            self._model = TextEmbedding(model_name=model_name)

        def embed_documents(self, texts):
            # 文档侧：批量向量化
            return [e.tolist() for e in self._model.embed(texts)]

        def embed_query(self, text):
            # 查询侧：BGE 会自动加检索指令前缀，效果更好
            return [e.tolist() for e in self._model.query_embed(text)][0]

    return FastEmbedBGE(EMBED_MODEL)


def get_llm():
    """构造 DeepSeek 聊天模型；缺少 Key 时给出清晰报错。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key.startswith("sk-在此"):
        raise RuntimeError(
            "未找到有效的 DEEPSEEK_API_KEY。请把 .env.example 复制为 .env 并填入你的 Key。"
        )
    from langchain_deepseek import ChatDeepSeek

    # temperature=0.2：问答场景要收敛、少发挥，降低胡编的概率
    return ChatDeepSeek(model=LLM_MODEL, api_key=api_key, temperature=0.2)
