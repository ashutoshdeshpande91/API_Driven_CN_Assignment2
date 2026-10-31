"""Core logic behind the REST API (and reusable anywhere): extract, entitlement, ask.

Kept free of web-framework code so it can be tested with fakes.
"""
from pathlib import Path

from src import llm
from src.entitlement import calculate
from src.schema import parse_output
from src.ui_helpers import derived_ex_date, friendly_llm_error, validate

ROOT = Path(__file__).resolve().parent.parent


class NotReady(RuntimeError):
    """The model or the document index has not been built yet."""


class Service:
    def __init__(self, model_dir=None, index_dir=None, extractor=None, retriever=None):
        self.model_dir = Path(model_dir) if model_dir else ROOT / "models" / "ca-extractor"
        self.index_dir = Path(index_dir) if index_dir else ROOT / "rag_index"
        self._extractor, self._retriever = extractor, retriever

    def status(self):
        return {
            "extractor_ready": self._extractor is not None or self.model_dir.exists(),
            "index_ready": self._retriever is not None or self.index_dir.exists(),
            "language_model_configured": llm.llm_available(),
        }

    def _get_extractor(self):
        if self._extractor is None:
            if not self.model_dir.exists():
                raise NotReady("The fine-tuned model is not available. Train it with: python -m src.train")
            from src.extractor import Extractor
            self._extractor = Extractor(str(self.model_dir))
        return self._extractor

    def _get_retriever(self):
        if self._retriever is None:
            if not self.index_dir.exists():
                raise NotReady("The document index is not built. Run: python -m src.rag build")
            from src.rag import Retriever
            self._retriever = Retriever(str(self.index_dir))
        return self._retriever

    def extract(self, text):
        raw = self._get_extractor().extract_raw([text])[0]
        fields = parse_output(raw)
        ex_value, derived = derived_ex_date(fields)
        return {"fields": fields, "raw_output": raw, "warnings": validate(fields),
                "ex_date": {"value": ex_value, "derived_from_record_date": derived}}

    @staticmethod
    def entitlement(fields, holding):
        e = calculate(fields, int(holding))
        return {"supported": e.supported, "headline": e.headline,
                "lines": [{"item": a, "value": b} for a, b in e.lines],
                "warnings": e.warnings, "error": e.error or None,
                "note": "Illustrative arithmetic only; a person must verify before booking."}

    def ask(self, question, k=5):
        hits = self._get_retriever().search(question, k=k)
        answer, note = None, None
        if llm.llm_available():
            try:
                answer = llm.answer_with_context(question, hits)
            except Exception as e:
                note = f"Could not write an answer ({friendly_llm_error(e)}); passages are returned instead."
        else:
            note = "No language-model key is configured; passages are returned without a written answer."
        return {"answer": answer, "note": note,
                "sources": [{"source": h["source"], "page": h["page"], "score": round(h["score"], 4),
                             "text": h["text"]} for h in hits]}
