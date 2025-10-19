# app/embeddings/dashscope_intl.py
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings.base import Embeddings

load_dotenv()  # 读取 .env

class DashScopeIntlEmbeddings(Embeddings):
    """DashScope 国际站 Embedding（OpenAI 兼容接口）"""
    def __init__(self, model: str = "text-embedding-v2", api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = (api_key or os.getenv("DASHSCOPE_API_KEY") or "").strip()
        self.base_url = base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI 兼容接口支持批量
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding
