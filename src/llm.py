"""Language-model calls (Gemini or Hugging Face) with offline fallbacks."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_SUMMARY = (
    "You are an assistant for a securities operations team. Using ONLY the notice text and extracted "
    "fields provided, write (1) a 2-3 sentence plain-English explanation of what the corporate action "
    "means for a shareholder and (2) a short operations checklist of 4-5 bullet points. "
    "Do not invent numbers or dates. If something is not stated, say it is not stated."
)
SYSTEM_RAG = (
    "You answer questions about corporate actions and related market regulations using ONLY the "
    "numbered context passages. Cite passages like [1], [2]. If the answer is not in the passages, "
    "say you could not find it in the provided documents. Do not give investment advice."
)


def _token():
    return os.getenv("HF_TOKEN") or os.getenv("HF_KEY")


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _custom():
    """(base_url, api_key, model) for an OpenAI-compatible endpoint, or None.

    LLM_PROVIDER=gemini is a shortcut: it uses Google's OpenAI-compatible endpoint, reads the key from
    GEMINI_API_KEY (or GOOGLE_API_KEY), and defaults the model to gemini-3.6-flash.
    Otherwise set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL yourself.
    """
    base, key, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
    if (os.getenv("LLM_PROVIDER") or "").strip().lower() == "gemini":
        base = base or GEMINI_BASE
        key = key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model = model or "gemini-3.6-flash"
    if base and key:
        return base, key, model
    return None


def llm_available():
    """True when a custom OpenAI-compatible endpoint with a key, or a Hugging Face token, is configured."""
    return bool(_custom() or _token())


RETRY_CODES = (429, 500, 502, 503, 504)   # temporary errors


def _post_chat(base, key, model, messages, max_tokens, temperature, attempts=3):
    """POST to {base}/chat/completions (OpenAI-compatible) using only the standard library.
    Retries a few times, with a short wait, when the service says it is busy."""
    import json
    import time
    import urllib.error
    import urllib.request
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if os.getenv("LLM_REASONING_EFFORT"):          # e.g. none / low
        body["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT")
    payload = json.dumps(body).encode("utf-8")
    url = base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code in RETRY_CODES and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Client error '{e.code} {e.reason}': {e.read().decode('utf-8', 'ignore')[:300]}")
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not text:
        raise RuntimeError("the model returned no text (try LLM_REASONING_EFFORT=low or a larger max_tokens)")
    return text.strip()


def chat(system, user, max_tokens=500, temperature=0.2):
    """One chat completion: the custom OpenAI-compatible endpoint if configured, else Hugging Face."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    custom = _custom()
    if custom:
        base, key, model = custom
        return _post_chat(base, key, model, messages, max_tokens, temperature)
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=os.getenv("HF_LLM_MODEL", DEFAULT_MODEL), token=_token())
    resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message.content.strip()


def summarize_notice(notice, fields):
    fields_txt = "\n".join(f"- {k}: {v}" for k, v in fields.items())
    return chat(SYSTEM_SUMMARY, f"NOTICE:\n{notice}\n\nEXTRACTED FIELDS:\n{fields_txt}")


def answer_with_context(question, chunks):
    ctx = "\n\n".join(f"[{i}] ({c['source']}, p.{c['page']}) {c['text']}" for i, c in enumerate(chunks, 1))
    return chat(SYSTEM_RAG, f"CONTEXT:\n{ctx}\n\nQUESTION: {question}", max_tokens=600)


def fallback_summary(f):
    """Template summary used when no language model is available."""
    ev = f.get("event_type", "UNKNOWN")
    co = f.get("company", "The company")
    rec = f.get("record_date", "NA")
    if ev == "DIVIDEND":
        what = f"{co} announced a cash dividend of Rs. {f.get('amount', '?')} per share."
    elif ev == "STOCK_SPLIT":
        try:
            mult = int(f["old_fv"]) // int(f["new_fv"])
        except (KeyError, ValueError, ZeroDivisionError):
            mult = "?"
        what = (f"{co} is splitting shares from face value Rs. {f.get('old_fv', '?')} to "
                f"Rs. {f.get('new_fv', '?')}; holdings multiply by {mult} and the price adjusts proportionally.")
    elif ev == "BONUS_ISSUE":
        what = f"{co} is issuing bonus shares in the ratio {f.get('ratio', '?')} (new : held)."
    elif ev == "RIGHTS_ISSUE":
        what = (f"{co} is offering rights shares in the ratio {f.get('ratio', '?')} (new : held) "
                f"at Rs. {f.get('price', '?')} per share.")
    elif ev == "BUYBACK":
        what = (f"{co} is buying back shares at Rs. {f.get('price', '?')} per share, "
                f"up to Rs. {f.get('size_cr', '?')} crore.")
    elif ev == "MERGER":
        what = f"{co} announced a merger/amalgamation with a share exchange ratio of {f.get('ratio', '?')}."
    else:
        what = "Event type could not be determined."
    checklist = [
        f"Confirm positions held on the record date ({rec}) for all affected accounts.",
        "Verify the event details against the exchange / depository announcement before processing.",
        "Update the security master and entitlement calculations; check price/quantity adjustments.",
        "Reconcile post-event positions and cash with custodian / depository statements.",
        "Notify front office and clients where the event needs a decision or affects valuation.",
    ]
    return what + "\n\nOperations checklist:\n" + "\n".join(f"- {c}" for c in checklist)
