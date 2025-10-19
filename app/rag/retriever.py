# app/rag/retriever.py
import os
from dotenv import load_dotenv, find_dotenv

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import DashScopeEmbeddings
from app.config import SETTINGS

def build_retriever(persist_dir: str = "./.chroma_mof", top_k: int = 5):
    # 统一加载 .env，无论从哪里启动
    load_dotenv(find_dotenv(usecwd=True), override=True)

    # 取 key（优先 SETTINGS，兜底环境变量），并 strip
    key = (getattr(SETTINGS, "dashscope_api_key", "") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("❌ 未读取到 DASHSCOPE_API_KEY，请在 .env 配置或 export 环境变量。")

    persist_abs = os.path.abspath(persist_dir)
    print(f"[Retriever] persist_dir={persist_abs}  top_k={top_k}  key_len={len(key)}")

    embed = DashScopeEmbeddings(
        model=getattr(SETTINGS, "embedding_model", "text-embedding-v1"),
        dashscope_api_key=key,
    )
    db = Chroma(persist_directory=persist_abs, embedding_function=embed)
    return db.as_retriever(search_kwargs={"k": top_k})
