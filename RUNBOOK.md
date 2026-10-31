# Runbook: Corporate Actions Assistant

Run everything from a terminal in the project folder with the virtual environment active. Commands are shown for Windows PowerShell; on macOS or Linux use `source .venv/bin/activate` and `export` instead of `setx`.

## Quickest way: one command (recommended)
Install Python 3.10 or newer from python.org (tick "Add Python to PATH"), open PowerShell in this folder, then:
```powershell
python setup_and_run.py
```
It shows a plan and asks once, then: checks your machine, creates `.venv`, installs PyTorch (a CUDA build when it
finds an NVIDIA GPU, matched to your driver, otherwise the CPU build), installs the packages, runs the tests, optionally stores a language-model
key, generates the data, trains and evaluates the model, downloads the documents, builds the index, checks retrieval,
prints the results and offers to start the app. It is safe to re-run: finished steps are skipped, and the index is
rebuilt only when the documents change.

Useful options:
```powershell
python setup_and_run.py --list                       # the steps
python setup_and_run.py --dry-run                    # show what would run, change nothing
python setup_and_run.py --yes                        # no questions (no key entry either)
python setup_and_run.py --only train,evaluate        # just these steps
python setup_and_run.py --force index                # redo a finished step
python setup_and_run.py --launch both                # start the API and the app at the end
python setup_and_run.py --cpu                        # force the CPU build (training uses 1 epoch unless --epochs is given)
```
It only changes things inside this folder (plus the Hugging Face model cache in your user profile). It never changes
the GPU driver, and it never prints or logs keys.

No GPU: training on a CPU takes roughly 30 to 90 minutes. The README describes how to train on Google Colab instead and
copy the model folder back. A log of each run is written to `logs/`.
The manual steps below are what the script does, in case you want to run one yourself.

## Every time you open a new terminal
```powershell
cd <project folder>
.venv\Scripts\activate
```
(If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.)

## One-time setup
```powershell
python -m venv .venv
.venv\Scripts\activate
# PyTorch first. With an NVIDIA GPU use the command from https://pytorch.org/get-started/locally/
# (choose a CUDA version NOT higher than the one nvidia-smi shows). Without a GPU:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True only with a usable GPU
pip install -r requirements.txt
```

## Language-model key (Gemini route)
```powershell
setx LLM_PROVIDER "gemini"
setx LLM_REASONING_EFFORT "none"
setx LLM_MODEL "<a flash model from: python check_llm.py --models>"
```
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) must already be set. Open a NEW terminal after `setx`, then:
```powershell
python check_llm.py          # expect: Result : OK, the model replied: ready
```
Without any key the app still works: template summaries, and passages instead of written answers.

## Build and evaluate step by step
```powershell
python tests/test_logic.py                                   # 16 tests, no GPU needed
python -m src.synth --n 3000                                 # synthetic data (seed 42)
python -m src.evaluate --model-dir google/flan-t5-base --tag base --limit 100   # "before"
python -m src.train                                          # a few minutes on a GPU; add --epochs 1 on a CPU
python -m src.evaluate                                       # "after" on 300 synthetic test notices
python -m src.handwritten                                    # writes data/handwritten.jsonl (13 notices)
python -m src.evaluate --data data/handwritten.jsonl --tag handwritten
python fetch_docs.py                                         # public documents -> docs_rag/
python -m src.rag build                                      # REBUILD after renaming or adding any PDF
python -m src.rag_eval                                       # retrieval hit-rate
python -m src.rag_eval --with-llm                            # also writes reports/rag_answers.csv
```

## Run it
```powershell
streamlit run app.py                 # the demo, http://localhost:8501
uvicorn api:app --reload             # the REST API, http://127.0.0.1:8000/docs
```
Example API call (PowerShell):
```powershell
$body = @{ text = "Corporate Announcement - Konark Pharma Limited - ISIN: INE3KQ8M2017 The Board has declared an interim dividend of Rs. 5.5 per equity share. Record Date: 20th November 2026." } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/extract -Method Post -ContentType "application/json" -Body $body
```

## Docker (optional)
```powershell
docker compose up --build        # API on :8000, app on :8501; CPU only
```
Needs `models/ca-extractor` and `rag_index/` to exist on this machine. Keys are passed from your environment.

## Common problems
| Symptom | Fix |
|---|---|
| `No module named 'torch'` | Install PyTorch (CUDA build) before `pip install -r requirements.txt` |
| `torch.cuda.is_available()` is False | Expected without an NVIDIA GPU. With a GPU, the CPU build is installed: `pip uninstall torch -y`, reinstall with the CUDA command from pytorch.org |
| Chose a CUDA version higher than the driver | Pick one not higher than `nvidia-smi` shows |
| `SSL: CERTIFICATE_VERIFY_FAILED` on nsdl.com | Download in a browser and save into `docs_rag/` |
| Files saved as `name.pdf.pdf` | Turn on "File name extensions" in File Explorer; rename; rebuild the index |
| Citations show the wrong document name | Rebuild the index after any rename: `python -m src.rag build` |
| 402 from Hugging Face | No credits; use the Gemini route |
| 404 from Gemini | Model name not available: `python check_llm.py --models`, then `setx LLM_MODEL ...` |
| 503 from Gemini | Busy; the app retries 3 times; try again in a minute or pick another model |
| Variables not seen after `setx` | Open a new terminal and activate `.venv` again |
| CUDA out of memory in training | `python -m src.train --batch 8`, and close other GPU programs |

## Do not submit or share
`.env`, `.venv`, `.idea`, any API key, and screenshots that show a key.
