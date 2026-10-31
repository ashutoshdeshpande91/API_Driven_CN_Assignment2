"""Test the language-model connection on its own (never prints your key).

    python check_llm.py            # one test call
    python check_llm.py --models   # list model names your key can use (custom endpoint only)
"""
import json
import os
import sys
import urllib.error
import urllib.request

from src import llm
from src.ui_helpers import friendly_llm_error

custom = llm._custom()
if custom:
    route = f"custom OpenAI-compatible endpoint {custom[0]}, model {custom[2]}"
else:
    route = f"Hugging Face Inference API, model {os.getenv('HF_LLM_MODEL', llm.DEFAULT_MODEL)}"
if "--models" in sys.argv:
    custom = llm._custom()
    if not custom:
        raise SystemExit("No custom endpoint is configured (set LLM_PROVIDER=gemini and your key first).")
    req = urllib.request.Request(custom[0].rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {custom[1]}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Could not list models: {e.code} {e.reason}")
    ids = sorted(m.get("id", "") for m in data.get("data", []))
    print(f"{len(ids)} models available to this key:")
    for i in ids:
        print("  ", i)
    print("\nSet LLM_MODEL to one of these names (try it without a leading 'models/' first).")
    raise SystemExit(0)

print("Route         :", route)
print("Key found     :", llm.llm_available())
if not llm.llm_available():
    raise SystemExit("No key found. Set LLM_PROVIDER=gemini with GEMINI_API_KEY, or HF_TOKEN, in .env or the environment.")
try:
    reply = llm.chat("You are terse.", "Reply with the single word: ready", max_tokens=10)
except Exception as e:
    print("Result        : FAILED -", friendly_llm_error(e))
    print("Detail        :", " ".join(str(e).split())[:500])
else:
    print("Result        : OK, the model replied:", reply)
