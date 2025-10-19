# app/graph.py
from __future__ import annotations

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# 兼容新老 LangChain 的 ChatPromptTemplate
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover
    from langchain.prompts import ChatPromptTemplate  # type: ignore

# Qwen（阿里百炼 OpenAI 兼容）客户端
from langchain_openai import ChatOpenAI

# 本项目内的相对导入
from .config import SETTINGS
from .tools.mof_tools import maybe_tool_call
from .memory.memory import Memory
from .rag.retriever import build_retriever

# ========= 环境变量加载（稳健） =========
def _load_env():
    """尽量稳地加载 .env（在 CWD 或项目根），不抛异常。"""
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = os.getenv("ENV_FILE") or find_dotenv(usecwd=True)
        if not env_path:
            # 兜底：尝试当前目录和上级目录
            for p in [".env", os.path.join(os.path.dirname(__file__), "..", ".env")]:
                p = os.path.abspath(p)
                if os.path.exists(p):
                    env_path = p
                    break
        if env_path:
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()  # 最后兜底
    except Exception:
        pass


# =========================
# State
# =========================
class GraphState(BaseModel):
    question: str
    docs: List[Dict[str, Any]] = Field(default_factory=list)       # [{"text":..., "source":...}, ...]
    tool_result: Dict[str, Any] = Field(default_factory=dict)
    answer: str = ""
    # sources: 原始去重后的路径列表（用于打印）
    sources: List[str] = Field(default_factory=list)
    # id_map: 路径 -> 编号（L1/L2/...）
    id_map: Dict[str, str] = Field(default_factory=dict)
    # source_map_str: 编号清单字符串（用于提示词展示）
    source_map_str: str = ""


# =========================
# Nodes
# =========================
def parse_query(state: GraphState) -> GraphState:
    # 如后续要做意图识别/重写查询，可在此扩展
    return state


def retrieve_docs(state: GraphState, retriever) -> GraphState:
    try:
        # 兼容老版本（<0.2）与新版本（>=0.2）
        if hasattr(retriever, "get_relevant_documents"):
            hits = retriever.get_relevant_documents(state.question)  # old API
        else:
            hits = retriever.invoke(state.question)  # new API (Runnable)
        print(f"[Retrieve] q='{state.question}'  hits={len(hits)}")
        for i, h in enumerate(hits[:3], 1):
            print(f"  [{i}] src={h.metadata.get('source')}")
    except Exception as e:
        print("[Retrieve][ERROR]", repr(e))
        hits = []

    docs = [{"text": h.page_content, "source": h.metadata.get("source", "local")} for h in hits]
    state.docs = docs

    # 去重并编号
    uniq_paths: List[str] = []
    for d in docs:
        p = d["source"]
        if p and p not in uniq_paths:
            uniq_paths.append(p)
    id_map: Dict[str, str] = {p: f"L{i+1}" for i, p in enumerate(uniq_paths)}
    state.id_map = id_map
    state.sources = uniq_paths
    state.source_map_str = "\n".join(f"[{id_map[p]}] {p}" for p in uniq_paths) or "(无)"
    return state




def maybe_call_tools_node(state: GraphState) -> GraphState:
    # 可选调用外部工具（如 Crossref 等）
    result = maybe_tool_call(state.question)
    state.tool_result = result or {}

    # 将工具返回的 URL 也加入 sources（不编号，单独显示即可）
    tool_sources: List[str] = []
    if "crossref" in state.tool_result:
        for it in state.tool_result["crossref"].get("items", []):
            url = it.get("url")
            if url:
                tool_sources.append(url)
    # 扩展但不打乱原本本地来源的编号
    for s in tool_sources:
        if s not in state.sources:
            state.sources.append(s)
    return state


def generate(state: GraphState, memory: Memory, strict: bool = False) -> GraphState:
    """
    strict=True 时：不允许 PRIOR（模型外部常识）；不足则说“我不知道”。
    """
    _load_env()

    # 选择 API Key 顺序：环境变量优先 -> SETTINGS
    api_key = os.getenv("DASHSCOPE_API_KEY") or getattr(SETTINGS, "dashscope_api_key", None)
    if not api_key:
        # 保底提示，避免静默 401
        state.answer = (
            "[LOCAL]\n- 未找到相关内容\n\n"
            "[INFERRED]\n- 无\n\n"
            "[PRIOR]\n- 未配置 DASHSCOPE_API_KEY，无法调用模型。请在 .env 设置或导出环境变量。"
        )
        return state

    # base_url：优先 SETTINGS.base_url，其次默认国内站
    base_url = getattr(SETTINGS, "base_url", None) or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = getattr(SETTINGS, "chat_model", None) or "qwen-turbo"

    # === LLM 客户端（国内百炼，OpenAI 兼容） ===
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
    )

    # 组装上下文（仅文本，最多取前 4 条，避免提示过长）
    K = min(len(state.docs), 4)
    context = "\n\n---\n".join(d["text"] for d in state.docs[:K])

    # 三段式强约束提示词
    SYSTEM = (
        "你是一个严格标注信息来源的 MOF RAG 助手。请将回答分为三段并使用下列格式：\n"
        "[LOCAL]\n"
        "- 逐条陈述从<上下文>中直接得到的事实；每条末尾标注来源编号，如 [L1] 或 [L1][L3]。\n\n"
        "[INFERRED]\n"
        "- 仅在可以由 [LOCAL] 条目逻辑推得时给出简短结论；不得引入新事实；每条注明“依据：Lx,Ly”。\n\n"
        "[PRIOR]\n"
        f"- 如必须依赖模型自身常识或外部知识，请在此列出，并说明“不在本地语料中”。{'（严格模式：本段必须留空）' if strict else ''}\n\n"
        "约束：\n"
        "- 绝不胡编；若<上下文>不足以回答，则在 [LOCAL] 列出“未找到相关内容”，并在 [INFERRED]/[PRIOR] 解释原因。\n"
        "- 仅输出结论与引用编号，不输出思维过程。"
    )

    USER_TMPL = (
        "用户问题：{question}\n\n"
        "<上下文>\n{context}\n\n"
        "可用来源（编号→路径）：\n{source_map}\n\n"
        "工具返回（可选，仅做佐证，不计入 Lx 编号）：\n{tool_brief}\n\n"
        "请严格按要求输出三段：[LOCAL]、[INFERRED]、[PRIOR]。"
    )

    # 工具结果简要串（避免把长 JSON 压进提示）
    tool_brief = "(无)"
    if state.tool_result:
        items = state.tool_result.get("crossref", {}).get("items", [])[:3]
        if items:
            lines = []
            for it in items:
                t = it.get("title", [""])[0] if isinstance(it.get("title"), list) else it.get("title", "")
                u = it.get("url", "")
                lines.append(f"- {t} {('(' + u + ')') if u else ''}")
            tool_brief = "\n".join(lines)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("user", USER_TMPL),
    ])

    msgs = prompt.format_messages(
        question=state.question,
        context=context or "(空)",
        source_map=state.source_map_str or "(无)",
        tool_brief=tool_brief,
    )

    resp = llm.invoke(msgs)
    content = getattr(resp, "content", None)
    state.answer = content if isinstance(content, str) else str(resp)

    # 严格模式：强约束 PRIOR 必须为空（再保险）
    if strict and "[PRIOR]" in state.answer:
        # 简单处理：把 PRIOR 段落清空
        parts = state.answer.split("[PRIOR]")
        if len(parts) >= 2:
            left = parts[0]
            # 丢弃 PRIOR 内容，仅保留标题
            state.answer = left.rstrip() + "\n\n[PRIOR]\n- （严格模式：不使用外部知识）"

    # 追加 “来源（编号→路径）” 清单，便于人工核对
    if state.source_map_str:
        state.answer += "\n\n---\n来源（编号→路径）:\n" + state.source_map_str

    # 记忆（不让失败阻塞）
    try:
        memory.add_turn(user=state.question, assistant=state.answer)
    except Exception:
        pass

    return state


# =========================
# Runner builder
# =========================
def make_graph_runner(
    persist_dir: str = "./.chroma_mof",
    top_k: int = 4,
    strict: bool = False,
):
    """
    构建一个带交互方法的 runner：
    - 自动从 persist_dir 构建检索器
    - 严格模式（strict=True）：不允许 PRIOR；不足则“我不知道”
    """
    # 转成布尔（防止外部传了字符串）
    if isinstance(strict, str):
        strict = strict.strip().lower() in {"1", "true", "yes", "on"}

    # 构建检索器 & 记忆
    retriever = build_retriever(persist_dir=persist_dir, top_k=top_k)
    memory = Memory()

    # LangGraph 编排
    g = StateGraph(GraphState)
    g.add_node("parse_query", parse_query)
    g.add_node("retrieve_docs", lambda s: retrieve_docs(s, retriever))
    g.add_node("maybe_tools", maybe_call_tools_node)
    g.add_node("generate", lambda s: generate(s, memory, strict=strict))

    g.set_entry_point("parse_query")
    g.add_edge("parse_query", "retrieve_docs")
    g.add_edge("retrieve_docs", "maybe_tools")
    g.add_edge("maybe_tools", "generate")
    g.add_edge("generate", END)

    compiled = g.compile()

    # 简单交互包装（与 app/cli.py 兼容）
    class Runner:
        def interactive(self):
            print("LangGraph MOF Chatbot (type 'exit' to quit)")
            while True:
                q = input("You: ").strip()
                if q.lower() in ("exit", "quit"):
                    break
                out = compiled.invoke({"question": q})  # 返回 dict
                ans = out.get("answer", str(out))
                print("\nBot:\n" + ans + "\n")

        # 兼容 .run()
        def run(self):
            self.interactive()

        # 允许直接调用：返回 dict，供 CLI 使用 ['answer'] / ['sources']
        def __call__(self, question: str):
            out = compiled.invoke({"question": question})
            return out

    return Runner()
