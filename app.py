"""Corporate Actions Assistant - Streamlit demo.   Run:  streamlit run app.py"""
import html
import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from src import llm
from src.entitlement import calculate
from src.schema import EVENT_TYPES, parse_output
from src.ui_helpers import (EVENT_LABELS, EXAMPLE_QUESTIONS, derived_ex_date, field_rows,
                            friendly_llm_error, key_figures, pretty_date, validate)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "ca-extractor"
INDEX_DIR = ROOT / "rag_index"
TEST_FILE = ROOT / "data" / "test.jsonl"
REPORTS = ROOT / "reports"
DISCLAIMER = ("Demo built on synthetic notices and public documents. Outputs are illustrative and "
              "must be verified by a person before any booking or investment decision.")

st.set_page_config(page_title="Corporate Actions Assistant", page_icon=":bar_chart:", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1180px;}
      .badge {display:inline-block; padding:3px 12px; border-radius:999px; font-size:0.85rem;
              font-weight:600; background:#e8f0fe; color:#174ea6; border:1px solid #c6dafc;}
      .muted {color:#5f6b7a; font-size:0.92rem;}
      .stMetric {background:#f3f6fb; padding:10px 14px; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_extractor():
    from src.extractor import Extractor
    return Extractor(str(MODEL_DIR))


@st.cache_resource(show_spinner=False)
def load_retriever():
    from src.rag import Retriever
    return Retriever(str(INDEX_DIR))


@st.cache_data(show_spinner=False)
def load_samples():
    """Example notices from the held-out test split, grouped by event type."""
    out = {}
    if TEST_FILE.exists():
        with open(TEST_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out.setdefault(r["event_type"], []).append(r["text"])
    return {k: v[:20] for k, v in out.items()}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


st.session_state.setdefault("notice_text", "")
st.session_state.setdefault("messages", [])


def load_example(event_type=None):
    samples = load_samples()
    pool = samples.get(event_type) if event_type else [t for v in samples.values() for t in v]
    if pool:
        st.session_state.notice_text = random.choice(pool)
        st.session_state.pop("result", None)


def clear_notice():
    st.session_state.notice_text = ""
    st.session_state.pop("result", None)


def ask_suggested(question):
    st.session_state.pending_question = question


model_ready, index_ready, token_ready = MODEL_DIR.exists(), INDEX_DIR.exists(), llm.llm_available()
with st.sidebar:
    st.header("System status")

    def status(ok, name, fix):
        st.markdown(f"{':green[Ready]' if ok else ':red[Not ready]'}  \u00b7  {name}")
        if not ok:
            st.caption(f"To fix: {fix}")

    status(model_ready, "Fine-tuned extractor", "run `python -m src.train`")
    status(index_ready, "Document index", "run `python fetch_docs.py` then `python -m src.rag build`")
    if token_ready:
        st.markdown(":green[Ready]  \u00b7  Language model key")
    else:
        st.markdown(":orange[Optional]  \u00b7  Language model key")
        st.caption("No key set, so template summaries are used. See .env.example for HF_TOKEN or another provider.")
    st.divider()
    with st.expander("About this demo"):
        st.write("A prototype for securities operations: it reads a corporate action notice into "
                 "structured fields, and answers rule questions from public SEBI and depository "
                 "documents.")
        st.caption(DISCLAIMER)

st.title("Corporate Actions Assistant")
st.caption("Turn a corporate action notice into structured data, or ask the rulebook a question.")

tab_notice, tab_chat, tab_quality, tab_how = st.tabs(
    ["Process a notice", "Ask the rulebook", "Model quality", "How it works"])


def make_summary(text, fields):
    if token_ready:
        try:
            return llm.summarize_notice(text, fields), "Written by the language model."
        except Exception as e:  # network / quota problems must not break the demo
            return (llm.fallback_summary(fields),
                    f"Template summary ({friendly_llm_error(e)}, so the language model was not used).")
    return llm.fallback_summary(fields), "Template summary (no language-model key is set)."


def render_result(res):
    fields = res["fields"]
    for w in validate(fields):
        st.warning(w)

    event = fields.get("event_type")
    company = html.escape(fields.get("company") or "Company not found")
    isin = html.escape(fields.get("isin") or "ISIN not found")
    st.markdown(
        f'<span class="badge">{html.escape(EVENT_LABELS.get(event, "Unknown event"))}</span>'
        f'&nbsp; <b>{company}</b> &nbsp;<span class="muted">ISIN {isin}</span>',
        unsafe_allow_html=True)
    st.write("")

    figures = key_figures(fields)
    ex_value, ex_derived = derived_ex_date(fields)
    cols = st.columns(2 + len(figures))
    cols[0].metric("Record date", pretty_date(fields["record_date"]) if fields.get("record_date") else "Not found")
    cols[1].metric("Ex-date", pretty_date(ex_value) if ex_value else "Not found")
    if ex_derived:
        cols[1].caption("Not in the notice. Derived as the record date under T+1 (verify).")
    for col, (label, value) in zip(cols[2:], figures):
        col.metric(label, value)

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Extracted fields")
        rows = field_rows(fields)
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=["Field", "Value"]), hide_index=True)
        st.download_button("Download fields as JSON", data=json.dumps(fields, indent=2),
                           file_name="notice_fields.json", mime="application/json")
        with st.expander("Show the model's raw output"):
            st.code(res["raw"], language=None)
    with right:
        st.subheader("What it means")
        st.write(res["summary"])
        st.caption(res["summary_note"])

    with st.container(border=True):
        st.subheader("What does it mean for a holding?")
        st.caption("Illustrative arithmetic from the extracted fields. Not a booking instruction.")
        holding = st.number_input("Shares held on the record date", min_value=0, value=1000,
                                  step=100, key="holding")
        ent = calculate(fields, int(holding))
        if ent.supported:
            st.success(ent.headline)
            st.dataframe(pd.DataFrame(ent.lines, columns=["Item", "Value"]), hide_index=True)
            for w in ent.warnings:
                st.info(w)
        else:
            st.info(ent.error)


with tab_notice:
    st.markdown("##### 1. Choose an example or paste a notice")
    samples = load_samples()
    if samples:
        cols = st.columns(len(EVENT_TYPES) + 1)
        for col, ev in zip(cols, EVENT_TYPES):
            col.button(EVENT_LABELS[ev], key=f"ex_{ev}", on_click=load_example, args=(ev,),
                       help=f"Load a synthetic {EVENT_LABELS[ev].lower()} notice")
        cols[-1].button("Random", key="ex_random", on_click=load_example, help="Load any example")
    else:
        st.caption("Example notices appear here once data/test.jsonl exists (`python -m src.synth`).")

    st.text_area("Notice text", key="notice_text", height=210, label_visibility="collapsed",
                 placeholder="Paste the text of a corporate action notice here...")

    st.markdown("##### 2. Read the notice")
    b1, b2, _ = st.columns([1, 1, 6])
    analyse = b1.button("Analyse notice", type="primary")
    b2.button("Clear", on_click=clear_notice)

    if analyse:
        text = st.session_state.notice_text.strip()
        if len(text) < 40:
            st.warning("Paste a longer notice first (at least a sentence or two).")
        elif not model_ready:
            st.error("The fine-tuned model is not available yet. Train it with `python -m src.train`.")
        else:
            try:
                with st.spinner("Reading the notice (the first run loads the model)..."):
                    raw = load_extractor().extract_raw([text])[0]
                fields = parse_output(raw)
                with st.spinner("Writing the summary..."):
                    summary, note = make_summary(text, fields)
                st.session_state.result = {"text": text, "raw": raw, "fields": fields,
                                           "summary": summary, "summary_note": note}
            except Exception as e:
                st.error(f"Something went wrong while reading the notice: {e}")

    result = st.session_state.get("result")
    if result:
        st.markdown("##### 3. Result")
        if result["text"] != st.session_state.notice_text.strip():
            st.info("The notice text changed after this analysis. Click Analyse notice to refresh.")
        render_result(result)


with tab_chat:
    if not index_ready:
        with st.container(border=True):
            st.subheader("The document index is not built yet")
            st.write("This tab answers questions from public SEBI and depository documents. Set it up once:")
            st.code("python fetch_docs.py\npython -m src.rag build", language="powershell")
    else:
        top = st.columns([6, 1])
        top[0].caption("Answers come from the indexed documents, with the passages shown under each answer.")
        top[1].button("Clear chat", on_click=lambda: st.session_state.update(messages=[]))

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m.get("sources"):
                    with st.expander(f"Sources ({len(m['sources'])} passages)"):
                        for i, s in enumerate(m["sources"], 1):
                            st.markdown(f"**[{i}] {s['source']}, page {s['page']}**  \n"
                                        f"<span class='muted'>match score {s['score']:.2f}</span>",
                                        unsafe_allow_html=True)
                            st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

        if not st.session_state.messages:
            st.markdown("**Try one of these**")
            for q in EXAMPLE_QUESTIONS:
                st.button(q, key=f"q_{q}", on_click=ask_suggested, args=(q,))

        question = st.chat_input("Ask about record dates, bonus, rights, buy-back rules...")
        question = st.session_state.pop("pending_question", None) or question
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            hits, answer = [], ""
            try:
                with st.spinner("Searching the documents..."):
                    hits = load_retriever().search(question, k=5)
            except Exception as e:
                answer = f"Sorry, the document search failed: {e}"
            if hits:
                if token_ready:
                    try:
                        with st.spinner("Writing the answer..."):
                            answer = llm.answer_with_context(question, hits)
                    except Exception as e:
                        answer = (f"I could not write an answer ({friendly_llm_error(e)}), so here are the "
                                  "closest passages from the documents.")
                else:
                    answer = ("No language-model key is set, so I can only show the most relevant passages "
                              "below. Add a key (see .env.example) for written answers.")
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})
            st.rerun()


with tab_quality:
    base, tuned = read_json(REPORTS / "metrics_base.json"), read_json(REPORTS / "metrics_finetuned.json")
    if not tuned:
        with st.container(border=True):
            st.subheader("No evaluation results yet")
            st.write("Score the model on the held-out test notices, then refresh this page:")
            st.code("python -m src.evaluate --model-dir google/flan-t5-base --tag base --limit 100\n"
                    "python -m src.evaluate", language="powershell")
    else:
        st.subheader("Does fine-tuning help?")
        st.caption(f"Exact-match scores on {tuned['n_test']} held-out synthetic notices. "
                   "Real notices are messier, so expect lower scores in practice.")
        c1, c2 = st.columns(2)
        for col, key, title in ((c1, "event_type_accuracy", "Event type correct"),
                                (c2, "full_record_exact_match", "Every field correct")):
            delta = f"{(tuned[key] - base[key]) * 100:+.0f} points vs base model" if base else None
            col.metric(title, f"{tuned[key] * 100:.1f}%", delta)

        fields = pd.DataFrame({"Fine-tuned": tuned["field_exact_match"]})
        if base:
            fields.insert(0, "Base model", pd.Series(base["field_exact_match"]))
        st.markdown("**Accuracy by field**")
        st.bar_chart(fields)

        errors = REPORTS / "errors_finetuned.csv"
        if errors.exists():
            df = pd.read_csv(errors)
            with st.expander(f"Where it went wrong ({len(df)} notices with at least one wrong field)"):
                st.dataframe(df.head(25), hide_index=True)


    st.divider()
    st.subheader("Robustness: hand-written notices")
    hand = read_json(REPORTS / "metrics_handwritten.json")
    if not hand:
        st.caption("13 notices in wording the generator never used (dates like 03.11.2026, ratios in words, "
                   "one-line alerts). Score the model on them:")
        st.code("python -m src.handwritten\n"
                "python -m src.evaluate --data data/handwritten.jsonl --tag handwritten", language="powershell")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric("Event type correct", f"{hand['event_type_accuracy'] * 100:.0f}%")
        h2.metric("Every field correct (exact)", f"{hand['full_record_exact_match'] * 100:.0f}%")
        if "full_record_lenient" in hand:
            h3.metric("Every field correct (ignoring formatting)", f"{hand['full_record_lenient'] * 100:.0f}%",
                      help="Treats 2.50 = 2.5, 1,250 = 1250 and a trailing full stop in a company name as the same.")
        st.caption(f"{hand['n_test']} notices written by hand. Compare with the synthetic test set above: "
                   "the gap shows how much the model depends on the generator's wording.")
        herr = REPORTS / "errors_handwritten.csv"
        if herr.exists():
            hdf = pd.read_csv(herr)
            if len(hdf):
                with st.expander(f"Where it went wrong ({len(hdf)} notices)"):
                    st.dataframe(hdf, hide_index=True)

    st.divider()
    st.subheader("Rulebook retrieval quality")
    rag = read_json(REPORTS / "rag_eval.json")
    if not rag:
        st.caption("Checks whether the right passage comes back in the top results for questions with known answers:")
        st.code("python -m src.rag_eval", language="powershell")
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric(f"Right passage in top {rag['k']}", f"{rag['hit_at_k'] * 100:.0f}%")
        r2.metric("Ranked first", f"{rag['hit_at_1'] * 100:.0f}%")
        r3.metric("Mean reciprocal rank", f"{rag['mrr']:.2f}")
        with st.expander(f"Per question ({rag['questions']} questions)"):
            rag_df = pd.DataFrame(rag["rows"])
            rag_df["rank_of_correct"] = pd.to_numeric(rag_df["rank_of_correct"], errors="coerce")  # blank = no hit
            st.dataframe(rag_df, hide_index=True)


with tab_how:
    st.subheader("How the assistant works")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("**Notice path** (fixed fields, for systems)")
            st.markdown(
                "1. A small language model, fine-tuned on synthetic notices, reads the notice.\n"
                "2. It writes the same set of fields every time: event type, company, ISIN, dates, "
                "ratio, amount, price.\n"
                "3. Plain arithmetic turns the fields into an illustrative entitlement.\n"
                "4. A hosted language model writes the summary and checklist.")
    with b:
        with st.container(border=True):
            st.markdown("**Rulebook path** (free text, for people)")
            st.markdown(
                "1. Your question is matched against passages of SEBI and depository documents.\n"
                "2. The five best passages are sent to a hosted language model.\n"
                "3. It answers from those passages and cites them.\n"
                "4. If the documents do not cover it, it should say so.")
    with st.container(border=True):
        st.markdown("**The same logic as a REST API**")
        st.markdown("Other systems can call it: `POST /extract` (notice in, fields out), `POST /entitlement`, "
                    "`POST /ask` and `GET /health`. Start it with `uvicorn api:app --reload` and open `/docs`.")
    st.caption(DISCLAIMER)
