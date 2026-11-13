"""检索增强问答：相似度检索 top-k → 拼 Prompt → DeepSeek 生成带出处的答案。

运行：python qa.py   （命令行问答，先确保已跑过 ingest.py）
"""

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from config import DB_DIR, TOP_K, get_embeddings, get_llm

# Prompt 是 RAG 的"灵魂"：约束模型只用资料回答 + 必须给出处 + 不会就说不会。
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的知识库问答助手。只能依据【参考资料】回答问题，"
            "不得编造资料中没有的内容。如果资料不足以回答，就直接说"
            "「根据已有资料无法回答」。回答末尾用「出处：」列出你引用的文件名。",
        ),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ]
)


def get_store():
    """打开已持久化的向量库。"""
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
        collection_name="kb_docs",
    )


def format_context(docs):
    """把检索到的块拼成带编号和来源的上下文文本。"""
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "未知")
        blocks.append(f"[{i}] 来源：{src}\n{d.page_content}")
    return "\n\n".join(blocks)


def retrieve(store, question, k=TOP_K):
    """向量相似度检索，返回最相关的 k 个块。"""
    return store.similarity_search(question, k=k)


def answer(question, store=None, k=TOP_K):
    """完整问答：检索 → 拼上下文 → 调大模型 → 返回(答案, 命中的块)。"""
    store = store or get_store()
    docs = retrieve(store, question, k)
    context = format_context(docs)
    chain = PROMPT | get_llm()
    result = chain.invoke({"context": context, "question": question})
    return result.content, docs


if __name__ == "__main__":
    question = input("请输入问题：")
    ans, docs = answer(question)
    print("\n================ 回答 ================\n")
    print(ans)
    print("\n============ 命中的原始片段 ============")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.metadata.get('source')}")
