#!/usr/bin/env python3
"""One command to set up and run the Corporate Actions Assistant from scratch.

    python setup_and_run.py              # shows a plan, asks once, then does everything
    python setup_and_run.py --yes        # no questions at all
    python setup_and_run.py --list       # list the steps
    python setup_and_run.py --dry-run    # show what would run, change nothing

Needs only Python 3.10+ (python.org) and an internet connection. Uses the standard library only, so it
runs before anything else is installed. Safe to re-run: finished steps are detected and skipped.
Everything installs into the .venv folder here; the NVIDIA driver and system settings are never touched.
Downloaded models are cached in your user profile (Hugging Face cache). Keys are never printed or logged.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
UA = "Mozilla/5.0 (setup_and_run.py)"
PY_MIN, PY_MAX = (3, 10), (3, 14)
MODEL_DIR = ROOT / "models" / "ca-extractor"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs_rag"
INDEX = ROOT / "rag_index"

STEPS = [
    ("preflight", "Check Python, disk space, internet and GPU", True),
    ("venv", "Create the virtual environment (.venv)", True),
    ("torch", "Install PyTorch (CUDA build when an NVIDIA GPU is found)", True),
    ("deps", "Install the other packages (requirements.txt)", True),
    ("tests", "Run the logic tests", False),
    ("llm", "Language-model key (optional) and connection test", False),
    ("data", "Generate the synthetic dataset", True),
    ("baseline", "Score the untouched base model (the 'before')", False),
    ("train", "Fine-tune the model", True),
    ("evaluate", "Score the fine-tuned model (the 'after')", True),
    ("handwritten", "Score the hand-written notices", False),
    ("docs", "Download the public documents", False),
    ("index", "Build the document index", False),
    ("ragcheck", "Check retrieval quality", False),
    ("summary", "Show the results", False),
    ("launch", "Start the app", False),
]
STEP_IDS = [s[0] for s in STEPS]


def parse_nvidia(smi_text, query_text):
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", smi_text or "")
    info = {"cuda": m.group(1) if m else None, "name": None, "used_mib": None, "total_mib": None}
    first = (query_text or "").strip().splitlines()[:1]
    parts = [p.strip() for p in (first[0] if first else "").split(",")]
    if len(parts) >= 3:
        info["name"] = parts[0]
        try:
            info["used_mib"], info["total_mib"] = int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return info


def nvidia_info():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        smi = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
        q = subprocess.run([exe, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    info = parse_nvidia(smi, q)
    return info if info["name"] else None


def cuda_candidates(driver_cuda, floor=(11, 8)):
    """PyTorch CUDA build tags not newer than the driver supports, newest first: '13.3' -> cu133, cu132, ..."""
    major, minor = [int(x) for x in driver_cuda.split(".")[:2]]
    tags = []
    for M in range(major, floor[0] - 1, -1):
        for m in range(minor if M == major else 9, -1, -1):
            if (M, m) >= floor:
                tags.append("cu%d%d" % (M, m))
    return tags


def torch_index_has(tag):
    """True / False / None (unknown, e.g. network trouble)."""
    url = "https://download.pytorch.org/whl/%s/torch/" % tag
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20)
        r.read(1)
        r.close()
        return True
    except urllib.error.HTTPError as e:
        return False if e.code == 404 else None
    except Exception:
        return None


def is_oom(text):
    return bool(re.search(r"out of memory", text or "", re.I))


def tidy_double_extensions(directory):
    """Fix 'name.pdf.pdf' (Windows hides extensions) without ever overwriting a file."""
    done = []
    for p in sorted(Path(directory).glob("*.pdf.pdf")):
        target = p.with_name(p.name[:-4])
        if not target.exists():
            p.rename(target)
            done.append("%s -> %s" % (p.name, target.name))
    return done


def docs_fingerprint(directory):
    d = Path(directory)
    return {p.name: p.stat().st_size for p in sorted(d.iterdir())
            if p.is_file() and p.suffix.lower() in (".pdf", ".txt")} if d.exists() else {}


def index_is_stale(index_dir, docs_dir):
    """The index is stale when it is missing, or the documents changed (renamed, added, replaced) since it was built."""
    index_dir = Path(index_dir)
    fp = index_dir / ".fingerprint.json"
    if not (index_dir / "chunks.json").exists() or not fp.exists():
        return True
    try:
        return json.loads(fp.read_text(encoding="utf-8")) != docs_fingerprint(docs_dir)
    except Exception:
        return True


def read_env_file(path):
    out = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def update_env_file(path, updates):
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    seen, out = set(), []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in updates:
            out.append("%s=%s" % (key, updates[key]))
            seen.add(key)
        else:
            out.append(line)
    out.extend("%s=%s" % (k, v) for k, v in updates.items() if k not in seen)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def llm_configured(env):
    g = lambda k: (env.get(k) or "").strip()
    if g("HF_TOKEN") or g("HF_KEY"):
        return True
    if g("LLM_PROVIDER").lower() == "gemini" and (g("GEMINI_API_KEY") or g("GOOGLE_API_KEY") or g("LLM_API_KEY")):
        return True
    return bool(g("LLM_BASE_URL") and g("LLM_API_KEY"))


def expected_docs():
    rows = []
    csv_path = DOCS / "sources.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"name": r["file_name"], "url": r["url"], "optional": r["optional"].strip().lower() == "yes"})
    return rows


def model_ready():
    return (MODEL_DIR / "config.json").exists() and any(
        (MODEL_DIR / n).exists() for n in ("model.safetensors", "pytorch_model.bin"))


def newer(a, b):
    """True when file a exists and is newer than file b (or b is missing)."""
    a, b = Path(a), Path(b)
    return a.exists() and (not b.exists() or a.stat().st_mtime >= b.stat().st_mtime)


class Ctx:
    def __init__(self, args):
        self.args = args
        self.dry = args.dry_run
        self.interactive = sys.stdin.isatty() and not args.yes
        self.py = args.python or str(VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        self.gpu = None
        self.log_fh = None
        self.forced = set(x for x in (args.force or "").split(",") if x)
        if not self.dry:
            (ROOT / "logs").mkdir(exist_ok=True)
            self.log_fh = open(ROOT / "logs" / datetime.now().strftime("setup_%Y%m%d_%H%M%S.log"), "w", encoding="utf-8")

    def say(self, msg=""):
        print(msg)
        sys.stdout.flush()
        self.raw_log(msg + "\n")

    def raw_log(self, text):
        if self.log_fh:
            self.log_fh.write(text)
            self.log_fh.flush()

    def force(self, step):
        return step in self.forced


def ask(ctx, question, default=True):
    if not ctx.interactive:
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    ans = input(question + suffix).strip().lower()
    return default if not ans else ans.startswith("y")


def run(ctx, cmd, cwd=None, extra_env=None, quiet=False):
    """Run a command, streaming its output live. Returns (return code, last ~60 KB of output)."""
    cmd = [str(c) for c in cmd]
    if not quiet:
        ctx.say("$ " + " ".join(cmd))
    if ctx.dry:
        return 0, ""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1", TOKENIZERS_PARALLELISM="false")
    env.update(extra_env or {})
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd or ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as e:
        return 127, str(e)
    tail = b""
    while True:
        chunk = proc.stdout.read1(4096)
        if not chunk:
            break
        text = chunk.decode("utf-8", "replace")
        if not quiet:
            sys.stdout.write(text)
            sys.stdout.flush()
        ctx.raw_log(text)
        tail = (tail + chunk)[-60000:]
    return proc.wait(), tail.decode("utf-8", "replace")


def capture(ctx, cmd):
    """Quiet run for checks. In a dry run nothing is executed and the check counts as 'not done'."""
    if ctx.dry:
        return 1, ""
    return run(ctx, cmd, quiet=True)


def pip(ctx, *args):
    return run(ctx, [ctx.py, "-m", "pip"] + list(args))


def torch_state(ctx):
    rc, out = capture(ctx, [ctx.py, "-c", "import torch;print(torch.__version__);print(torch.cuda.is_available())"])
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if rc != 0 or len(lines) < 2:
        return None
    return {"version": lines[-2], "cuda": lines[-1] == "True"}


def can_reach(url):
    try:
        urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=15).close()
        return True, ""
    except urllib.error.HTTPError:
        return True, ""                       # the server answered
    except Exception as e:
        return False, str(e)[:160]


def step_preflight(ctx):
    warn = False
    v = sys.version_info
    ctx.say("Python %d.%d.%d on %s" % (v[0], v[1], v[2], sys.platform))
    if (v[0], v[1]) < PY_MIN:
        return "fail", "Python %d.%d or newer is needed (python.org). Tick 'Add Python to PATH' when installing." % PY_MIN
    if (v[0], v[1]) > PY_MAX:
        ctx.say("  note: Python %d.%d is newer than the versions this was prepared for (3.10-3.14); "
                "if a package fails to install, use Python 3.11 or 3.12." % PY_MAX)
        warn = True
    missing = [p for p in ("src/train.py", "requirements.txt", "api.py", "app.py") if not (ROOT / p).exists()]
    if missing:
        return "fail", "Run this from the project folder; missing: " + ", ".join(missing)

    free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    ctx.say("Free disk space: %.1f GB (about 8-10 GB is needed)" % free_gb)
    if free_gb < 6:
        return "fail", "Not enough free disk space (need roughly 8-10 GB)."
    warn = warn or free_gb < 10

    for name, url in (("PyPI (packages)", "https://pypi.org/simple/"),
                      ("PyTorch downloads", "https://download.pytorch.org/whl/"),
                      ("Hugging Face (models)", "https://huggingface.co/")):
        ok, why = can_reach(url)
        ctx.say("  internet: %-22s %s" % (name, "reachable" if ok else "NOT reachable"))
        if not ok:
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in why:
                hint = " (certificate problem: antivirus or a company proxy may be inspecting HTTPS)"
            if ctx.dry:
                warn = True
                continue
            return "fail", "Cannot reach %s: %s%s" % (name, why, hint)

    ctx.gpu = nvidia_info()
    if ctx.gpu and not ctx.args.cpu:
        ctx.say("GPU: %s, %s of %s MiB in use, driver supports CUDA %s" % (
            ctx.gpu["name"], ctx.gpu["used_mib"], ctx.gpu["total_mib"], ctx.gpu["cuda"]))
        if ctx.gpu["used_mib"] and ctx.gpu["used_mib"] > 4000:
            ctx.say("  WARNING: a lot of GPU memory is already in use. Close games and other GPU programs "
                    "before training.")
            warn = True
    else:
        ctx.say("GPU: none used (%s). Training will run on the CPU, which is much slower."
                % ("--cpu given" if ctx.args.cpu else "no NVIDIA GPU found"))
        warn = warn or not ctx.args.cpu
    return ("warn" if warn else "ok"), ""


def step_venv(ctx):
    if ctx.args.python:
        return "skip", "using the interpreter given with --python"
    if Path(ctx.py).exists() and not ctx.force("venv"):
        return "skip", ".venv already exists"
    rc, _ = run(ctx, [sys.executable, "-m", "venv", str(VENV)])
    if rc != 0:
        return "fail", "Could not create the virtual environment (on Linux: install the python3-venv package)."
    run(ctx, [ctx.py, "-m", "pip", "install", "--upgrade", "pip"])
    return ("dry" if ctx.dry else "ok"), ""


def step_torch(ctx):
    want_cuda = ctx.gpu is not None and not ctx.args.cpu
    st = torch_state(ctx)
    if st and not ctx.force("torch") and (st["cuda"] or not want_cuda):
        return "skip", "PyTorch %s already installed (%s)" % (st["version"], "CUDA works" if st["cuda"] else "CPU")
    if st and want_cuda and not st["cuda"]:
        ctx.say("PyTorch %s is installed but cannot use the GPU (CPU build). Replacing it." % st["version"])
        pip(ctx, "uninstall", "-y", "torch")

    if want_cuda:
        tags = cuda_candidates(ctx.gpu["cuda"]) if ctx.gpu["cuda"] else ["cu126", "cu124", "cu121", "cu118"]
        ctx.say("Driver supports CUDA %s; trying PyTorch builds: %s ..." % (ctx.gpu["cuda"], ", ".join(tags[:4])))
        if ctx.dry:
            return "dry", "would try the newest CUDA build the driver supports"
        tried = 0
        for tag in tags:
            if tried >= 4:
                break
            if torch_index_has(tag) is False:
                continue
            tried += 1
            ctx.say("Installing PyTorch %s build (a large download, several GB) ..." % tag)
            rc, _ = pip(ctx, "install", "torch", "--index-url", "https://download.pytorch.org/whl/" + tag)
            if rc == 0:
                st = torch_state(ctx)
                if st and st["cuda"]:
                    return "ok", "PyTorch %s with CUDA" % st["version"]
                ctx.say("  that build cannot use the GPU; removing it and trying the next one")
                pip(ctx, "uninstall", "-y", "torch")
        ctx.say("Could not get a CUDA build of PyTorch; installing the CPU build instead (training will be slow).")

    cmd = ["install", "torch"] + ([] if sys.platform == "darwin" else ["--index-url", "https://download.pytorch.org/whl/cpu"])
    rc, _ = pip(ctx, *cmd)
    st = torch_state(ctx)
    if ctx.dry:
        return "dry", ""
    if rc != 0 or not st:
        return "fail", "PyTorch did not install. See the log, or use the command from pytorch.org/get-started/locally."
    return "warn", "PyTorch %s installed (CPU only)" % st["version"]


IMPORT_CHECK = ("import transformers, peft, accelerate, sentencepiece, sentence_transformers, huggingface_hub, "
                "pypdf, numpy, pandas, streamlit, dotenv, fastapi, uvicorn")


def step_deps(ctx):
    rc, _ = capture(ctx, [ctx.py, "-c", IMPORT_CHECK])
    if rc == 0 and not ctx.force("deps"):
        return "skip", "all packages already installed"
    rc, tail = pip(ctx, "install", "-r", str(ROOT / "requirements.txt"))
    if ctx.dry:
        return "dry", ""
    if rc != 0:
        hint = ""
        if "sentencepiece" in tail.lower():
            hint = " (sentencepiece often lacks wheels for the newest Python; use Python 3.11 or 3.12)"
        return "fail", "Package installation failed" + hint
    rc, _ = capture(ctx, [ctx.py, "-c", IMPORT_CHECK])
    st = torch_state(ctx)
    if ctx.gpu and not ctx.args.cpu and st and not st["cuda"]:
        return "warn", "packages installed, but PyTorch lost GPU support; rerun with --force torch"
    return ("ok" if rc == 0 else "fail"), ("" if rc == 0 else "some packages still fail to import")


def step_tests(ctx):
    rc, tail = run(ctx, [ctx.py, "tests/test_logic.py"])
    if ctx.dry:
        return "dry", ""
    return ("ok", tail.strip().splitlines()[-1]) if rc == 0 else ("warn", "some logic tests failed; see the output above")


def step_llm(ctx):
    env_file = ROOT / ".env"
    merged = dict(read_env_file(env_file))
    merged.update(os.environ)
    if not llm_configured(merged):
        if not ctx.interactive:
            return "skip", "no language-model key configured (summaries use templates; set a key later, see RUNBOOK.md)"
        ctx.say("A language-model key is optional. Without it the app still works, using template summaries "
                "and showing passages instead of written answers.")
        choice = input("Configure one now? 1 = Google Gemini key, 2 = Hugging Face token, Enter = skip: ").strip()
        import getpass
        if choice == "1":
            key = getpass.getpass("Paste your Gemini API key (not shown): ").strip()
            if key:
                update_env_file(env_file, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": key,
                                           "LLM_REASONING_EFFORT": "none"})
                os.environ.update({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": key})
                ctx.say("Saved to .env (which is git-ignored). Looking up the models your key can use ...")
                rc, out = capture(ctx, [ctx.py, "check_llm.py", "--models"])
                ids = [l.strip() for l in out.splitlines() if l.startswith("   ") and l.strip()]
                flash = [i for i in ids if "flash" in i.lower()
                         and not any(x in i.lower() for x in ("image", "tts", "embed", "live", "audio"))]
                if flash:
                    ctx.say("Models with 'flash' in the name:\n   " + "\n   ".join(flash[:12]))
                name = input("Type the model name to use (Enter = the app's default, gemini-2.5-flash): ").strip()
                if name:
                    update_env_file(env_file, {"LLM_MODEL": name[len("models/"):] if name.startswith("models/") else name})
        elif choice == "2":
            key = getpass.getpass("Paste your Hugging Face token (not shown): ").strip()
            if key:
                update_env_file(env_file, {"HF_TOKEN": key})
        else:
            return "skip", "skipped by choice"
    rc, out = capture(ctx, [ctx.py, "check_llm.py"])
    if ctx.dry:
        return "dry", ""
    result = next((l.strip() for l in out.splitlines() if l.strip().startswith("Result")), "")
    ctx.say("  " + (result or "no result line"))
    if "OK" in result:
        return "ok", "language model reachable"
    return "warn", "language model not working yet (the app falls back to templates); run: python check_llm.py"


def step_data(ctx):
    need = [n for n in ("train", "val", "test") if not (ROOT / "data" / (n + ".jsonl")).exists()]
    if not need and not ctx.force("data"):
        return "skip", "data/ already has train, val and test files"
    rc, _ = run(ctx, [ctx.py, "-m", "src.synth", "--n", "3000"])
    return ("dry" if ctx.dry else ("ok" if rc == 0 else "fail")), ""


def print_metrics(ctx, path, label):
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        ctx.say("  %s: event type %.1f%%, every field %.1f%% (%d notices)" % (
            label, m["event_type_accuracy"] * 100, m["full_record_exact_match"] * 100, m["n_test"]))
    except Exception:
        pass


def step_baseline(ctx):
    path = REPORTS / "metrics_base.json"
    if path.exists() and not ctx.force("baseline"):
        return "skip", "reports/metrics_base.json already exists"
    ctx.say("Downloads the base model (about 1 GB) the first time.")
    rc, _ = run(ctx, [ctx.py, "-m", "src.evaluate", "--model-dir", "google/flan-t5-base", "--tag", "base", "--limit", "100"])
    if ctx.dry:
        return "dry", ""
    print_metrics(ctx, path, "Base model (expected to be near 0%)")
    return ("ok", "") if rc == 0 else ("warn", "baseline failed; it is only needed for the report's 'before' number")


def step_train(ctx):
    if model_ready() and not ctx.force("train"):
        return "skip", "models/ca-extractor already exists"
    if ctx.gpu and ctx.gpu["used_mib"] and ctx.gpu["used_mib"] > 4000 and not ctx.args.cpu:
        if not ask(ctx, "Over 4 GB of GPU memory is already in use (games or other apps?). Continue anyway?", False):
            return "fail", "Close GPU-heavy programs and run again"
    ctx.say("Training takes about 2 minutes on an RTX 4070 SUPER (hours on a CPU).")
    batches = [None, 8, 4]
    for i, b in enumerate(batches):
        cmd = [ctx.py, "-m", "src.train"] + (["--batch", str(b)] if b else [])
        rc, tail = run(ctx, cmd)
        if ctx.dry:
            return "dry", ""
        if rc == 0 and model_ready():
            return "ok", "model saved to models/ca-extractor"
        if is_oom(tail) and i < len(batches) - 1:
            ctx.say("Out of GPU memory: retrying with batch size %d ..." % batches[i + 1])
            continue
        break
    return "fail", "Training failed. See the output above (out-of-memory: close other GPU programs)."


def step_evaluate(ctx):
    path = REPORTS / "metrics_finetuned.json"
    if newer(path, MODEL_DIR / "config.json") and not ctx.force("evaluate"):
        print_metrics(ctx, path, "Fine-tuned model")
        return "skip", "results are up to date"
    rc, _ = run(ctx, [ctx.py, "-m", "src.evaluate"])
    if ctx.dry:
        return "dry", ""
    print_metrics(ctx, path, "Fine-tuned model")
    return ("ok", "") if rc == 0 else ("fail", "evaluation failed")


def step_handwritten(ctx):
    path = REPORTS / "metrics_handwritten.json"
    if newer(path, MODEL_DIR / "config.json") and not ctx.force("handwritten"):
        print_metrics(ctx, path, "Hand-written notices")
        return "skip", "results are up to date"
    run(ctx, [ctx.py, "-m", "src.handwritten"])
    rc, _ = run(ctx, [ctx.py, "-m", "src.evaluate", "--data", "data/handwritten.jsonl", "--tag", "handwritten"])
    if ctx.dry:
        return "dry", ""
    print_metrics(ctx, path, "Hand-written notices")
    return ("ok", "") if rc == 0 else ("warn", "hand-written evaluation failed")


def missing_docs():
    return [d for d in expected_docs() if not d["optional"] and not (DOCS / d["name"]).exists()]


def step_docs(ctx):
    for r in tidy_double_extensions(DOCS) if not ctx.dry else []:
        ctx.say("Renamed " + r)
    if not missing_docs() and not ctx.force("docs"):
        return "skip", "all required documents are already in docs_rag/"
    run(ctx, [ctx.py, "fetch_docs.py"])
    if ctx.dry:
        return "dry", ""
    for r in tidy_double_extensions(DOCS):
        ctx.say("Renamed " + r)
    gone = missing_docs()
    if gone and ctx.interactive:
        ctx.say("\nThese files could not be downloaded automatically (a certificate error on one site is common):")
        for d in gone:
            ctx.say("  %s\n    open in your browser: %s" % (d["name"], d["url"]))
        input("Save them into docs_rag/ under exactly those names (turn on 'File name extensions' in "
              "Explorer), then press Enter to continue (Enter with nothing saved = skip): ")
        for r in tidy_double_extensions(DOCS):
            ctx.say("Renamed " + r)
        gone = missing_docs()
    if gone:
        return "warn", "still missing: " + ", ".join(d["name"] for d in gone)
    return "ok", ""


def step_index(ctx):
    pdfs = docs_fingerprint(DOCS)
    if not pdfs:
        return "warn", "no documents in docs_rag/, so the rulebook tab cannot answer yet"
    if not index_is_stale(INDEX, DOCS) and not ctx.force("index"):
        return "skip", "index is up to date with the documents"
    ctx.say("Builds the index from %d documents (downloads a small embedding model the first time)." % len(pdfs))
    rc, _ = run(ctx, [ctx.py, "-m", "src.rag", "build"])
    if ctx.dry:
        return "dry", ""
    if rc != 0:
        return "warn", "index build failed; the notice features still work"
    (INDEX / ".fingerprint.json").write_text(json.dumps(docs_fingerprint(DOCS)), encoding="utf-8")
    return "ok", ""


def step_ragcheck(ctx):
    if not (INDEX / "chunks.json").exists() and not ctx.dry:
        return "skip", "no index yet"
    path = REPORTS / "rag_eval.json"
    if newer(path, INDEX / "chunks.json") and not ctx.force("ragcheck"):
        return "skip", "results are up to date"
    rc, _ = run(ctx, [ctx.py, "-m", "src.rag_eval"])
    return ("dry" if ctx.dry else ("ok" if rc == 0 else "warn")), ""


def step_summary(ctx):
    ctx.say("")
    print_metrics(ctx, REPORTS / "metrics_base.json", "Base model (before)      ")
    print_metrics(ctx, REPORTS / "metrics_finetuned.json", "Fine-tuned (synthetic)   ")
    print_metrics(ctx, REPORTS / "metrics_handwritten.json", "Hand-written notices    ")
    try:
        r = json.loads((REPORTS / "rag_eval.json").read_text(encoding="utf-8"))
        ctx.say("  Retrieval (%d questions): right passage in top %d: %.0f%%, ranked first: %.0f%%" % (
            r["questions"], r["k"], r["hit_at_k"] * 100, r["hit_at_1"] * 100))
    except Exception:
        pass
    ctx.say("\nStart the demo any time with:   python -m streamlit run app.py   (or: python setup_and_run.py --only launch --launch app)")
    ctx.say("Start the REST API with:        python -m uvicorn api:app --reload   then open http://127.0.0.1:8000/docs")
    return "ok", ""


def step_launch(ctx):
    mode = ctx.args.launch
    if mode is None:
        if not ctx.interactive:
            return "skip", "not launching (use --launch app|api|both)"
        mode = "app" if ask(ctx, "\nStart the app now?", True) else "none"
    if mode == "none":
        return "skip", ""
    api = None
    try:
        if mode in ("api", "both"):
            cmd = [ctx.py, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"]
            if mode == "api":
                if not ctx.dry:
                    threading.Timer(4, webbrowser.open, args=("http://127.0.0.1:8000/docs",)).start()
                ctx.say("API docs: http://127.0.0.1:8000/docs  (press Ctrl+C to stop)")
                run(ctx, cmd)
                return "ok", ""
            ctx.say("$ " + " ".join(cmd))
            api = None if ctx.dry else subprocess.Popen(cmd, cwd=str(ROOT))
            ctx.say("API docs: http://127.0.0.1:8000/docs")
        if not ctx.dry:
            threading.Timer(5, webbrowser.open, args=("http://localhost:8501",)).start()
        ctx.say("App: http://localhost:8501  (press Ctrl+C to stop)")
        run(ctx, [ctx.py, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                  "--browser.gatherUsageStats", "false"])
    except KeyboardInterrupt:
        ctx.say("\nStopped.")
    finally:
        if api:
            api.terminate()
    return "ok", ""


FUNCS = {"preflight": step_preflight, "venv": step_venv, "torch": step_torch, "deps": step_deps,
         "tests": step_tests, "llm": step_llm, "data": step_data, "baseline": step_baseline,
         "train": step_train, "evaluate": step_evaluate, "handwritten": step_handwritten,
         "docs": step_docs, "index": step_index, "ragcheck": step_ragcheck,
         "summary": step_summary, "launch": step_launch}


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Set up and run the Corporate Actions Assistant from scratch.")
    ap.add_argument("--yes", action="store_true", help="assume yes and never prompt (also skips key entry)")
    ap.add_argument("--dry-run", action="store_true", help="show what would run; change nothing")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--only", help="comma-separated step ids to run")
    ap.add_argument("--skip", help="comma-separated step ids to skip")
    ap.add_argument("--from", dest="from_step", help="start at this step id")
    ap.add_argument("--force", help="comma-separated step ids to redo even if already done")
    ap.add_argument("--cpu", action="store_true", help="install and use the CPU build of PyTorch")
    ap.add_argument("--python", help="use this interpreter instead of creating .venv")
    ap.add_argument("--launch", choices=["none", "app", "api", "both"], help="what to start at the end")
    return ap.parse_args(argv)


def select_steps(args):
    ids = list(STEP_IDS)
    if args.from_step:
        if args.from_step not in ids:
            raise SystemExit("Unknown step: " + args.from_step)
        ids = ids[ids.index(args.from_step):]
    if args.only:
        only = args.only.split(",")
        bad = [x for x in only if x not in STEP_IDS]
        if bad:
            raise SystemExit("Unknown step(s): " + ", ".join(bad))
        ids = [i for i in ids if i in only]
    if args.skip:
        ids = [i for i in ids if i not in args.skip.split(",")]
    return [s for s in STEPS if s[0] in ids]


BADGE = {"ok": "[OK]  ", "skip": "[SKIP]", "warn": "[WARN]", "fail": "[FAIL]", "dry": "[DRY] "}


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    args = parse_args(argv)
    if sys.version_info < (3, 8):
        raise SystemExit("Python 3.10 or newer is needed. Install it from python.org and run this again.")
    if args.list:
        for sid, title, critical in STEPS:
            print("  %-12s %s%s" % (sid, title, "" if critical else "  (optional)"))
        return 0

    steps = select_steps(args)
    ctx = Ctx(args)
    if not ctx.interactive and not args.yes and not args.dry_run:
        raise SystemExit("Not running in an interactive terminal. Add --yes to run without questions.")

    ctx.say("Corporate Actions Assistant: setup and run" + ("  [DRY RUN: nothing will be changed]" if ctx.dry else ""))
    ctx.say("Steps: " + ", ".join(s[0] for s in steps))
    ctx.say("")
    ctx.say("What this does:")
    ctx.say("  - installs Python packages and PyTorch into the .venv folder here (the NVIDIA driver is never touched)")
    ctx.say("  - downloads roughly 6-8 GB in total (my estimate): PyTorch and packages, the base model (~1 GB) and")
    ctx.say("    a small embedding model, stored in your user profile's Hugging Face cache, plus public documents")
    ctx.say("  - trains the model (about 2 minutes on a good GPU), evaluates it, builds the document index")
    ctx.say("  - never prints or logs your keys; a log of this run goes to the logs/ folder")
    if not ctx.dry and not ask(ctx, "\nContinue?", True):
        ctx.say("Cancelled.")
        return 0

    results, started = [], time.time()
    try:
        for sid, title, critical in steps:
            ctx.say("\n=== %s: %s ===" % (sid, title))
            t0 = time.time()
            try:
                status, note = FUNCS[sid](ctx)
            except KeyboardInterrupt:
                status, note = "fail", "interrupted"
            except Exception as e:                                     # a bug must not hide behind a stack trace
                status, note = "fail", "unexpected error: %s: %s" % (type(e).__name__, e)
            results.append((sid, status, note, time.time() - t0))
            ctx.say("%s %s%s" % (BADGE[status], sid, (": " + note) if note else ""))
            if status == "fail" and critical:
                ctx.say("\nStopped: '%s' is required. Fix the problem above, then run this script again; "
                        "finished steps are skipped." % sid)
                break
    finally:
        ctx.say("\n=== Summary (%.0f s) ===" % (time.time() - started))
        for sid, status, note, secs in results:
            ctx.say("  %s %-12s %5.0fs  %s" % (BADGE[status], sid, secs, note))
        if ctx.log_fh:
            ctx.say("\nFull log: " + ctx.log_fh.name)
    return 1 if any(s == "fail" and dict((x[0], x[2]) for x in STEPS).get(i) for i, s, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
