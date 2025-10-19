import os, glob, typer
from typing import List, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from app.config import SETTINGS
from langchain_community.document_loaders import TextLoader

app = typer.Typer()
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

def load_documents(input_dir: str) -> List[Tuple[str, str]]:
    texts = []
    for path in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True):
        if os.path.isdir(path):
            continue
        if path.lower().endswith(".pdf"):
            try:
                reader = PdfReader(path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                texts.append((text, path))
            except Exception:
                pass
        elif any(path.lower().endswith(ext) for ext in [".txt", ".md"]):
            try:
                loader = TextLoader(path, encoding="utf-8")
                text = loader.load()[0].page_content
                texts.append((text, path))
            except Exception:
                pass
    return texts

@app.command()
def main(
    input_dir: str = typer.Option("./data/samples", help="Input folder"),
    persist_dir: str = typer.Option("./.chroma_mof", help="Chroma persist dir"),
    chunk_size: int = typer.Option(600, help="Chunk size"),
    chunk_overlap: int = typer.Option(120, help="Chunk overlap")
):
    os.makedirs(persist_dir, exist_ok=True)
    raw = load_documents(input_dir)
    print(f"[Ingest] Scanning {len(raw)} files under {input_dir} ...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs, metas = [], []
    for text, src in raw:
        for chunk in splitter.split_text(text):
            docs.append(chunk)
            metas.append({"source": src})

    if not docs:
        print("No documents found for ingestion.")
        raise SystemExit(0)

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or SETTINGS.dashscope_api_key,
    )

    vectordb = Chroma.from_texts(
        texts=docs,
        embedding=embeddings,
        metadatas=metas,
        persist_directory=persist_dir,
    )

    vectordb.persist()
    print(f"Ingested {len(docs)} chunks into {persist_dir}")
    print("✅ Ingest done.")

if __name__ == "__main__":
    app()
