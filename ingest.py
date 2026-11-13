"""文档摄取：加载 data/ 下的文档 → 切分 → 向量化 → 存入 Chroma。

运行：python ingest.py
每次改完 data/ 里的文档都要重新跑一次。
"""

import glob
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DB_DIR, get_embeddings

# 中文友好的切分分隔符：优先按段落，再按句号/问号等，最后才按空格
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def load_documents():
    """读取 data/ 下所有 .pdf / .md / .txt 文件。"""
    docs = []
    pattern = os.path.join(DATA_DIR, "**", "*")
    for path in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        elif ext in (".md", ".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def main():
    print(f"[1/3] 从 {DATA_DIR} 读取文档 ...")
    documents = load_documents()
    if not documents:
        raise SystemExit("data/ 下没有可用的 .pdf/.md/.txt 文档，请先放几份进来。")
    print(f"      加载到 {len(documents)} 个文档片段。")

    print("[2/3] 切分文本 ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"      切分为 {len(chunks)} 个文本块。")

    print("[3/3] 向量化并写入向量库（首次会下载 BGE 模型，约 90MB）...")
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
        collection_name="kb_docs",
    )
    print(f"      完成！向量库已保存到：{DB_DIR}")


if __name__ == "__main__":
    main()
