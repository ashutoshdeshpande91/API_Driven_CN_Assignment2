# Corporate Actions Assistant

AIMLZG549 - API-driven Cloud Native Solutions, Assignment II.

An NLP application for securities operations (Indian market) with two parts:

- **Notice extractor:** a fine-tuned `flan-t5-base` model reads a corporate action notice (dividend, stock split, bonus,
  rights, buy-back, merger) and returns its key fields. A hosted language model (Gemini) writes the plain-English summary.
- **Rulebook assistant:** a retrieval-augmented chatbot answers questions from public SEBI and NSDL documents and cites
  the source passages.

The notices used for training are synthetic, and the rulebook documents are public. No company data is used.

| Assignment part | Where |
|---|---|
| I. API-based AI application | `app.py`, `src/extractor.py`, `src/llm.py` |
| II. Fine-tuning | `src/synth.py` (data), `src/train.py`, `src/evaluate.py` |
| III. RAG chatbot | `src/rag.py`, `docs_rag/`, tab "Ask the rulebook" |

## Quick start (recommended)

Install Python 3.10 or newer, then from the project folder:

```bash
python setup_and_run.py
```

The script creates a virtual environment, installs PyTorch and the packages, runs the tests, generates the data, trains
and evaluates the model, builds the document index and offers to start the app. It detects whether an NVIDIA GPU is
available and picks the right PyTorch build and training settings itself, so the same command works with or without a GPU.
Finished steps are skipped when you run it again. Useful options: `--list`, `--dry-run`, `--yes`, `--only`, `--force`,
`--cpu`, `--epochs`, `--launch app`. More detail is in `RUNBOOK.md`.

Open http://localhost:8501 when the app starts.

## Language model key (optional)

The summaries and chat answers use Google Gemini. Set these in your environment, open a new terminal, then run
`python check_llm.py`:

```bash
# Windows (PowerShell)
setx GEMINI_API_KEY "<your key>"
setx LLM_PROVIDER "gemini"
setx LLM_MODEL "gemini-3.6-flash"
setx LLM_REASONING_EFFORT "none"

# macOS / Linux
export GEMINI_API_KEY="<your key>" LLM_PROVIDER=gemini LLM_MODEL=gemini-3.6-flash LLM_REASONING_EFFORT=none
```

`python check_llm.py --models` lists the model names your key can use. Without a key the application still works: it
shows a template summary and returns the retrieved passages instead of a written answer. Never commit your key.

## Hardware: with or without a GPU

| Machine | What happens | Training time |
|---|---|---|
| NVIDIA GPU | CUDA build of PyTorch, 3 epochs | a few minutes |
| No GPU | CPU build of PyTorch, 1 epoch by default | roughly 30 to 90 minutes |

Everything except training runs comfortably on a CPU. If you have no GPU, you can train for free on Google Colab and copy
the result back:

1. In Colab, choose Runtime > Change runtime type > GPU, then run:

   ```
   !git clone https://github.com/soham1601/API_Driven_CN_Assignment2.git
   %cd API_Driven_CN_Assignment2
   !pip install -r requirements.txt
   !python -m src.synth --n 3000
   !python -m src.train
   !cd models && zip -r ca-extractor.zip ca-extractor -x "ca-extractor/checkpoints/*"
   ```

2. Download `models/ca-extractor.zip`, unzip it into the `models/` folder of your copy of the project, and run
   `python setup_and_run.py`. The training step is skipped because `models/ca-extractor` already exists.

## Manual setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
```

Install PyTorch first. With an NVIDIA GPU, use the command from https://pytorch.org/get-started/locally/ (choose a CUDA
version not higher than the one `nvidia-smi` shows). Without a GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"     # True only when a GPU is usable
```

## Run order

```bash
python -m src.synth --n 3000                                   # 1. synthetic dataset -> data/
python -m src.evaluate --model-dir google/flan-t5-base --tag base --limit 100   # 2. baseline before fine-tuning
python -m src.train                                            # 3. fine-tune (add --epochs 1 on a CPU)
python -m src.evaluate                                         # 4. metrics -> reports/
python -m src.handwritten                                      # 5. 13 hand-written test notices
python -m src.evaluate --data data/handwritten.jsonl --tag handwritten
python fetch_docs.py                                           # 6. download the public documents (docs_rag/)
python -m src.rag build                                        # 7. build the retrieval index
python -m src.rag_eval                                         # 8. retrieval evaluation
streamlit run app.py                                           # 9. the application
```

If training runs out of GPU memory, use `python -m src.train --batch 8` (the setup script retries with smaller batches by itself).

## Other components

- **Entitlement calculator** (`src/entitlement.py`): illustrative holding arithmetic for dividends, bonus issues, splits
  and rights issues. Buy-backs and mergers are not calculated.
- **Input checks** (`src/ui_helpers.py`): warnings for missing or malformed fields in the extracted result.
- **REST API** (`api.py`, `src/service.py`): `uvicorn api:app --reload`, documentation at `/docs`. Endpoints:
  `/health`, `/extract`, `/entitlement`, `/ask`.
- **Docker** (`Dockerfile`, `docker-compose.yml`): CPU image for the API and the app. The model and index are mounted
  from your machine (`docker compose up --build`).
- **Tests:** `python tests/test_logic.py` (16 tests, no GPU or key needed).

## Project layout

```
app.py                Streamlit application
api.py                REST API
setup_and_run.py      One-command setup, training, evaluation and launch
check_llm.py          Language-model connection test
fetch_docs.py         Downloads the rulebook documents
src/                  Schema, data generator, training, evaluation, RAG, language-model calls, calculator
tests/                Automated tests
data/                 Synthetic training, validation and test sets, hand-written test notices
docs_rag/             Public SEBI and NSDL documents (see docs_rag/README.md)
reports/              Metrics, error lists and retrieval results
models/, rag_index/   Created by the scripts (not stored in Git)
```

## Notes

- **Synthetic data.** The test-set scores are high because the notices come from the same generator as the training
  data. The 13 hand-written notices give a more realistic view of how the model copes with different wording.
- **Ex-date.** Under T+1 settlement the ex-date is usually the record date. When a notice does not state it, the
  application derives it that way and marks it for verification.
- **Documents.** `python fetch_docs.py --force` downloads fresh copies. Rebuild the index after any change:
  `python -m src.rag build`.
- The outputs are illustrative and must be checked by a person before any booking or investment decision.
