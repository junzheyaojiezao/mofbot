# app/config.py
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

# 统一加载 .env（无论从哪里运行，都能找到）
load_dotenv(find_dotenv(usecwd=True), override=True)

class Settings(BaseModel):
    dashscope_api_key: str = Field(default_factory=lambda: (os.getenv("DASHSCOPE_API_KEY") or "").strip())
    embedding_model: str = "text-embedding-v1"
    top_k: int = 5

SETTINGS = Settings()

# 兼容 dashscope 官方 SDK（有的库会从这里取）
try:
    import dashscope
    if SETTINGS.dashscope_api_key:
        dashscope.api_key = SETTINGS.dashscope_api_key
except Exception:
    pass

