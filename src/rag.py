"""Minimal RAG: chunk public documents -> embed -> cosine search.

    python -m src.rag build              # index everything in docs_rag/ (PDF, TXT)
    python -m src.rag ask "What is a record date?"
"""
import json
import os
import re
import sys
from pathlib import Path

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        yield i, page.extract_text() or ""


def load_documents(docs_dir):
    for p in sorted(Path(docs_dir).rglob("*")):
        if p.suffix.lower() == ".pdf":
            for page_no, text in _read_pdf(p):
                yield p.name, page_no, text
        elif p.suffix.lower() == ".txt":
            yield p.name, 1, p.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text, size=1000, overlap=200):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def build_index(docs_dir="docs_rag", out_dir="rag_index", model_name=EMB_MODEL):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks = []
    for source, page, text in load_documents(docs_dir):
        for c in chunk_text(text):
            chunks.append({"source": source, "page": page, "text": c})
    if not chunks:
        sys.exit(f"No PDF/TXT documents found in {docs_dir}/ - see docs_rag/README.md")

    model = SentenceTransformer(model_name)
    emb = model.encode([c["text"] for c in chunks], batch_size=64,
                       normalize_embeddings=True, show_progress_bar=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb.astype("float32"))
    with open(out / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "chunks": chunks}, f)
    n_docs = len({c["source"] for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {n_docs} documents -> {out}/")


class Retriever:
    def __init__(self, index_dir="rag_index"):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        d = Path(index_dir)
        self.emb = np.load(d / "embeddings.npy")
        meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = meta["chunks"]
        self.model = SentenceTransformer(meta["model"])
        self.np = np

    def search(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = self.np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in idx]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        from src.llm import answer_with_context, llm_available
        hits = Retriever().search(sys.argv[2])
        for h in hits:
            print(f"[{h['score']:.2f}] {h['source']} p.{h['page']}: {h['text'][:200]}...")
        if llm_available():
            print("\nANSWER:\n" + answer_with_context(sys.argv[2], hits))
    else:
        print(__doc__)
