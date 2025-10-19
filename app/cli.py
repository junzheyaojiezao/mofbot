# app/cli.py
import typer
from rich.console import Console
from .graph import make_graph_runner
from .config import SETTINGS

console = Console()

def _extract_answer(resp):
    # 兼容 dict / pydantic 对象 / LangChain消息 等各种返回
    if isinstance(resp, dict):
        for k in ("answer", "content", "output", "text"):
            if k in resp and resp[k]:
                return resp[k]
        return str(resp)
    # 有 .answer 属性
    ans = getattr(resp, "answer", None)
    if isinstance(ans, str) and ans:
        return ans
    # 有 .content 属性
    cnt = getattr(resp, "content", None)
    if isinstance(cnt, str) and cnt:
        return cnt
    # 兜底
    return str(resp)

def main(
    persist_dir: str = typer.Option(
        "./.chroma_mof", "--persist-dir", "-p",
        help="Path to Chroma vector store directory",
    ),
    top_k: int = typer.Option(
        getattr(SETTINGS, "top_k", 4), "--top-k",
        help="Retriever top-k",
    ),
    strict: bool = typer.Option(
        False, "--strict",
        is_flag=True,
        help="Strict local-only mode (disallow PRIOR/external knowledge)",
    ),
):
    runner = make_graph_runner(persist_dir=persist_dir, top_k=top_k, strict=strict)

    console.print("[bold green]LangGraph MOF Chatbot[/bold green] (type 'exit' to quit)")
    while True:
        q = typer.prompt("You").strip()
        if q.lower() in {"exit", "quit", "q"}:
            console.print("[yellow]Bye![/yellow]")
            break

        resp = runner(q)
        answer = _extract_answer(resp)
        console.print(f"[cyan]Bot:[/cyan]\n{answer}\n")

        # 尝试从各种位置拿 sources
        sources = []
        if isinstance(resp, dict):
            sources = resp.get("sources") or resp.get("refs") or []


if __name__ == "__main__":
    typer.run(main)
