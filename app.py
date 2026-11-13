"""Streamlit 网页界面：输入问题 → 显示答案与出处。

运行：streamlit run app.py
"""

import streamlit as st

from qa import answer, get_store

st.set_page_config(page_title="本地知识库问答", page_icon="📚")

st.title("📚 本地知识库问答（RAG）")
st.caption("LangChain · Chroma · BGE · DeepSeek")


@st.cache_resource
def load_store():
    # 只加载一次向量库与模型，避免每次交互都重新初始化
    return get_store()


store = load_store()

question = st.text_input("输入你的问题：", placeholder="例如：GPU 显存不足怎么处理？")

if question:
    with st.spinner("检索并生成中 ..."):
        ans, docs = answer(question, store)
    st.markdown("### 回答")
    st.write(ans)
    with st.expander(f"查看命中的 {len(docs)} 个原始片段（出处）"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**[{i}] {d.metadata.get('source')}**")
            st.write(d.page_content)
