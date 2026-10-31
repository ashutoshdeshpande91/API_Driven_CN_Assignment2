# Corporate Actions Assistant (AIMLZG549 - Assignment II)

NLP project for the securities middle/back-office domain (India market).
Everything runs on **synthetic notices** + **public regulatory documents** - no workplace data.

| Assignment section | Where |
|---|---|
| I. API-based AI app (3 NLP sub-tasks: classify, extract, summarise) | `app.py`, `src/extractor.py`, `src/llm.py` |
| II. Fine-tuning | `src/synth.py` (dataset), `src/train.py`, `src/evaluate.py` |
| III. RAG chatbot | `src/rag.py`, `docs_rag/`, tab "Ask the rulebook" |
| IV. Situated learning | `report/REPORT_TEMPLATE.md`, `PRIMER.md` |

## Setup (Windows, RTX 4070 Super)
```powershell
cd C:\Soham\BITS\Sem3\APIDrivenCN\ca_assistant
python -m venv .venv
.venv\Scripts\activate
# 1) PyTorch with CUDA: copy the current pip command from https://pytorch.org/get-started/locally/
# 2) then:
pip install -r requirements.txt
copy .env.example .env      # paste your Hugging Face token into .env
```
Check the GPU: `python -c "import torch; print(torch.cuda.is_available())"`

## Run order
```powershell
python -m src.synth --n 3000                                   # 1. synthetic dataset -> data/
python -m src.evaluate --model-dir google/flan-t5-base --tag base --limit 100   # 2. "before" baseline (expect ~0%)
python -m src.train                                            # 3. fine-tune (~10-20 min on a 4070)
python -m src.evaluate                                         # 4. "after" metrics -> reports/
python fetch_docs.py                                           # 5a. download public documents (see docs_rag/README.md)
python -m src.rag build                                        # 5b. build the RAG index
streamlit run app.py                                           # 6. demo
```
If you run out of GPU memory: `python -m src.train --batch 8`. For a larger model:
`python -m src.train --model google/flan-t5-large --lora --batch 8`.

## Things to verify before submitting
- **Ex-date rule.** The generator and app assume ex-date = record date under India's T+1 settlement.
  Confirm against the current NSE/BSE circular and cite it in the report.
- **Language-model route.** Either Google Gemini (`LLM_PROVIDER=gemini` plus `GEMINI_API_KEY`, optional `LLM_MODEL`) or a Hugging Face token (`HF_TOKEN`, optional `HF_LLM_MODEL`). Test with `python check_llm.py`; without any key the app uses template summaries.
- **Document versions.** Download the *current* version of each RAG document and note the download date.
- The synthetic data is template-based, so metrics will look very high. Say so in the report and
  discuss it under "Analyse" (real notices are messier; this is the prototype-vs-production gap).

## RAG documents (added)
`python fetch_docs.py` downloads the public documents in `docs_rag/sources.csv`. See `docs_rag/README.md`.

**Ex-date caveat found while sourcing documents:** most Indian broker FAQs and NSE data summaries say the
ex-date equals the record date under T+1, but at least one site claims it is one trading day *before* the
record date, and a third-party note mentions exchange circulars keeping the ex-date unchanged when a
settlement holiday moved the record date. Treat "same day" as the usual case, not a guarantee, and cite an
official exchange source in the report. The app already labels the derived ex-date as "verify".

## Entitlement calculator and app redesign (added 31 Oct)
- `src/entitlement.py`: plain-Python, illustrative entitlement arithmetic for dividend, bonus, split and
  rights notices (buy-back and merger are deliberately not calculated). No LLM is involved in the numbers.
- `src/ui_helpers.py`: labels, date formatting, headline figures and output validation used by the app
  (no Streamlit import, so it is testable).
- `app.py`: guided flow (choose example -> analyse -> result), headline metrics, validation warnings,
  JSON download, holding calculator, example questions for the chat, a Model quality tab, a How it works tab.
- `.streamlit/config.toml`: light theme.
- Tests for the logic: `python tests/test_logic.py` (the Streamlit UI itself has not been run yet).

## More additions (31 Oct night)
- `RUNBOOK.md`: exact commands and the problems we hit.
- `src/handwritten.py` + `python -m src.evaluate --data data/handwritten.jsonl --tag handwritten`: 13 notices in
  wording the generator never used, to measure how well the model generalises.
- `src/rag_eval.py`: retrieval hit-rate for the rulebook (`python -m src.rag_eval`).
- `api.py` + `src/service.py`: REST API (`uvicorn api:app --reload`, docs at `/docs`): `/health`, `/extract`,
  `/entitlement`, `/ask`.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: CPU container for the API and the app (not tested here).
- `check_llm.py`: tests the language-model connection (`--models` lists usable model names).
- The Model quality tab now also shows the hand-written and retrieval results once those reports exist.

## One-command setup (added 31 Oct night)
`python setup_and_run.py` sets everything up from scratch and runs the whole pipeline (see RUNBOOK.md). It uses only
the Python standard library, skips finished steps, and has `--list`, `--dry-run`, `--only`, `--force`, `--cpu` and
`--launch` options. Its helpers are covered by `python tests/test_logic.py`; the real installs and the training run
were not run by the author in this environment.
