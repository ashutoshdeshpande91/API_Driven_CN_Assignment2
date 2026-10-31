"""Plain-assert tests for the pure logic. Run from the project root:  python tests/test_logic.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.entitlement import calculate, indian, inr
from src.schema import format_target, parse_output
from src.ui_helpers import (derived_ex_date, field_rows, friendly_llm_error, key_figures,
                            pretty_date, validate)


def test_formatting():
    assert inr(5500) == "Rs. 5,500"
    assert inr(400000) == "Rs. 4,00,000"
    assert inr("5.5") == "Rs. 5.50"
    assert inr(12345678) == "Rs. 1,23,45,678"
    assert indian(10000) == "10,000"
    assert indian(100) == "100"
    assert pretty_date("2026-11-20") == "20 Nov 2026"
    assert pretty_date("not a date") == "not a date"


def test_dividend():
    r = calculate({"event_type": "DIVIDEND", "amount": "5.5"}, 1000)
    assert r.supported and "Rs. 5,500" in r.headline, r


def test_bonus():
    r = calculate({"event_type": "BONUS_ISSUE", "ratio": "2:4"}, 1000)
    assert r.supported and "500 new shares" in r.headline and "1,500" in r.headline, r
    r = calculate({"event_type": "BONUS_ISSUE", "ratio": "1:2"}, 1001)
    assert "500 new shares" in r.headline and r.warnings, r  # leftover half share is flagged


def test_split():
    r = calculate({"event_type": "STOCK_SPLIT", "old_fv": "10", "new_fv": "1"}, 1000)
    assert r.supported and "10,000 shares" in r.headline and "10x" in r.headline, r
    r = calculate({"event_type": "STOCK_SPLIT", "old_fv": "5", "new_fv": "2"}, 3)
    assert r.supported and len(r.warnings) == 2, r  # 7.5 shares rounded down, flagged


def test_rights():
    r = calculate({"event_type": "RIGHTS_ISSUE", "ratio": "2:5", "price": "1000"}, 1000)
    assert r.supported and "400 rights shares" in r.headline and "Rs. 4,00,000" in r.headline, r
    r = calculate({"event_type": "RIGHTS_ISSUE", "ratio": "1:5", "price": "100"}, 3)
    assert r.supported and any("too small" in w for w in r.warnings), r


def test_unsupported_and_errors():
    assert not calculate({"event_type": "BUYBACK", "price": "500"}, 100).supported
    assert not calculate({"event_type": "MERGER", "ratio": "3:4"}, 100).supported
    assert not calculate({"event_type": "WHATEVER"}, 100).supported
    r = calculate({"event_type": "DIVIDEND"}, 100)
    assert not r.supported and "amount" in r.error, r
    r = calculate({"event_type": "BONUS_ISSUE", "ratio": "abc"}, 100)
    assert not r.supported and "a:b" in r.error, r
    assert not calculate({"event_type": "DIVIDEND", "amount": "5"}, -1).supported


def test_ui_helpers():
    f = parse_output("event_type: DIVIDEND ; company: Konark Pharma Limited ; isin: INE3KQ8M2017 ; "
                     "record_date: 2026-11-20 ; ex_date: NA ; amount: 5.5")
    assert derived_ex_date(f) == ("2026-11-20", True)
    assert derived_ex_date({"ex_date": "2026-11-20", "record_date": "2026-11-20"}) == ("2026-11-20", False)
    assert key_figures(f) == [("Dividend per share", "Rs. 5.50")]
    assert validate(f) == []
    rows = dict(field_rows(f))
    assert rows["Event type"] == "Dividend" and rows["Record date"] == "20 Nov 2026"
    assert rows["Ex-date"] == "Not stated in the notice"
    bad = {"event_type": "DIVIDEND", "isin": "XX1", "record_date": "2026-13-45"}
    w = validate(bad)
    assert any("ISIN" in x for x in w) and any("record date" in x.lower() for x in w) and any("company" in x.lower() for x in w), w
    assert validate({}) and key_figures({"event_type": "RIGHTS_ISSUE", "ratio": "2:5", "price": "1000"}) == [
        ("Rights ratio (new : held)", "2 : 5"), ("Issue price", "Rs. 1,000")]


def test_friendly_errors():
    assert "credits" in friendly_llm_error(Exception("Client error '402 Payment Required' for url x"))
    assert "permission" in friendly_llm_error(Exception("401 Unauthorized"))
    assert "rate limit" in friendly_llm_error(Exception("429 Too Many Requests"))
    assert friendly_llm_error(Exception("something odd\nsecond line")) == "something odd"


def test_llm_routing_and_http(monkeypatch=None):
    import io
    import json as _json
    import os
    import urllib.request
    from src import llm
    keys = ["LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "GEMINI_API_KEY",
            "GOOGLE_API_KEY", "HF_TOKEN", "HF_KEY", "LLM_REASONING_EFFORT"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    real_urlopen = urllib.request.urlopen
    try:
        assert llm._custom() is None and not llm.llm_available()
        os.environ["LLM_PROVIDER"] = "gemini"
        assert llm._custom() is None                       # provider set but no key yet
        os.environ["GEMINI_API_KEY"] = "test-key"
        base, key, model = llm._custom()
        assert base.endswith("/v1beta/openai/") and key == "test-key" and model == "gemini-2.5-flash"
        assert llm.llm_available()

        seen = {}

        class Fake:
            def __init__(self, payload):
                self.buf = io.BytesIO(payload)

            def __enter__(self):
                return self.buf

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            seen["url"], seen["auth"] = req.full_url, req.headers.get("Authorization")
            seen["body"] = _json.loads(req.data.decode("utf-8"))
            return Fake(_json.dumps({"choices": [{"message": {"content": " ready "}}]}).encode("utf-8"))

        urllib.request.urlopen = fake_urlopen
        os.environ["LLM_REASONING_EFFORT"] = "none"
        assert llm.chat("sys", "user") == "ready"
        assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", seen
        assert seen["auth"] == "Bearer test-key"
        assert seen["body"]["model"] == "gemini-2.5-flash" and seen["body"]["reasoning_effort"] == "none"
    finally:
        urllib.request.urlopen = real_urlopen
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


def test_llm_retries_when_busy():
    import io
    import json as _json
    import time
    import urllib.error
    import urllib.request
    from src import llm
    calls = {"n": 0}
    real_urlopen, real_sleep = urllib.request.urlopen, time.sleep

    class Ok:
        def __enter__(self):
            return io.BytesIO(_json.dumps({"choices": [{"message": {"content": "fine"}}]}).encode("utf-8"))

        def __exit__(self, *a):
            return False

    def flaky(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"busy"))
        return Ok()

    try:
        urllib.request.urlopen, time.sleep = flaky, lambda s: None
        assert llm._post_chat("https://x/v1/", "k", "m", [], 10, 0.1) == "fine" and calls["n"] == 3
        calls["n"] = -10                                    # never succeeds within 3 tries
        try:
            llm._post_chat("https://x/v1/", "k", "m", [], 10, 0.1)
            assert False, "should have raised"
        except RuntimeError as e:
            assert "503" in str(e) and "busy" in friendly_llm_error(e)
    finally:
        urllib.request.urlopen, time.sleep = real_urlopen, real_sleep


class _FakeExtractor:
    def extract_raw(self, texts):
        return ["event_type: DIVIDEND ; company: Konark Pharma Limited ; isin: INE3KQ8M2017 ; "
                "record_date: 2026-11-20 ; ex_date: NA ; amount: 5.5"]


class _FakeRetriever:
    def search(self, q, k=5):
        return [{"source": "SEBI_ICDR_Master_Circular_2026-02-09.pdf", "page": 8, "score": 0.6,
                 "text": "The period for advance notice ... shall be at least 3 working days (excluding the date)"}]


def test_service_with_fakes():
    import os
    from src.service import NotReady, Service
    saved = {k: os.environ.pop(k, None) for k in ["HF_TOKEN", "HF_KEY", "LLM_PROVIDER", "LLM_BASE_URL",
                                                  "LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]}
    try:
        svc = Service(model_dir="/nonexistent/m", index_dir="/nonexistent/i",
                      extractor=_FakeExtractor(), retriever=_FakeRetriever())
        out = svc.extract("some notice text that is long enough to count")
        assert out["fields"]["event_type"] == "DIVIDEND" and out["warnings"] == []
        assert out["ex_date"] == {"value": "2026-11-20", "derived_from_record_date": True}
        ent = svc.entitlement(out["fields"], 1000)
        assert ent["supported"] and "Rs. 5,500" in ent["headline"] and ent["error"] is None
        ans = svc.ask("What advance notice is needed?")
        assert ans["answer"] is None and "No language-model key" in ans["note"] and ans["sources"][0]["page"] == 8
        assert svc.status()["extractor_ready"] and not svc.status()["language_model_configured"]
        bare = Service(model_dir="/nonexistent/m", index_dir="/nonexistent/i")
        for call in (lambda: bare.extract("x" * 40), lambda: bare.ask("question here")):
            try:
                call()
                assert False, "should raise NotReady"
            except NotReady:
                pass
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_handwritten_set_and_rag_scoring():
    from src.handwritten import build_rows
    from src.rag_eval import QUESTIONS, phrase_hit, score_question, summarise
    rows = build_rows()
    assert len(rows) == 13 and all(parse_output(r["target"]) == r["fields"] for r in rows)
    item = QUESTIONS[0]
    good = {"source": "SEBI_ICDR_Master_Circular_2026-02-09.pdf", "page": 8, "score": 0.6,
            "text": "advance notice  shall be AT LEAST 3\nworking days"}
    wrong_doc = {"source": "SEBI_BuyBack.pdf", "page": 1, "score": 0.9, "text": "at least 3 working days"}
    assert phrase_hit(good["text"], item["phrases"])
    assert score_question(item, [wrong_doc, good])["rank"] == 2   # right text in the wrong document is ignored
    assert score_question(item, [wrong_doc])["rank"] is None
    s = summarise([{"rank": 1, "source_in_topk": True}, {"rank": 2, "source_in_topk": True},
                   {"rank": None, "source_in_topk": False}])
    assert abs(s["hit_at_k"] - 2 / 3) < 1e-9 and abs(s["hit_at_1"] - 1 / 3) < 1e-9 and abs(s["mrr"] - 0.5) < 1e-9


def test_setup_helpers():
    import json as _j
    import tempfile
    from pathlib import Path as _P
    import setup_and_run as sr
    assert sr.cuda_candidates("13.3")[:4] == ["cu133", "cu132", "cu131", "cu130"]
    assert sr.cuda_candidates("12.4")[0] == "cu124" and sr.cuda_candidates("12.4")[-1] == "cu118"
    assert sr.cuda_candidates("11.8") == ["cu118"] and "cu117" not in sr.cuda_candidates("12.0")
    info = sr.parse_nvidia("| NVIDIA-SMI 610.88  CUDA Version: 13.3 |", "NVIDIA GeForce RTX 4070 SUPER, 834, 12282\n")
    assert info == {"cuda": "13.3", "name": "NVIDIA GeForce RTX 4070 SUPER", "used_mib": 834, "total_mib": 12282}
    assert sr.parse_nvidia("", "")["name"] is None
    assert sr.is_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB")
    assert not sr.is_oom("some other error")
    assert sr.llm_configured({"HF_TOKEN": "x"}) and not sr.llm_configured({})
    assert sr.llm_configured({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"})
    assert not sr.llm_configured({"LLM_PROVIDER": "gemini"})
    assert sr.llm_configured({"LLM_BASE_URL": "https://x/v1", "LLM_API_KEY": "k"})
    with tempfile.TemporaryDirectory() as d:
        d = _P(d)
        (d / "a.pdf.pdf").write_bytes(b"x")
        (d / "b.pdf").write_bytes(b"y")
        (d / "b.pdf.pdf").write_bytes(b"z")
        assert sr.tidy_double_extensions(d) == ["a.pdf.pdf -> a.pdf"]
        assert (d / "a.pdf").exists() and (d / "b.pdf.pdf").exists() and (d / "b.pdf").read_bytes() == b"y"
        env = d / ".env"
        env.write_text("A=1\n# B=2\nHF_TOKEN=old\n", encoding="utf-8")
        sr.update_env_file(env, {"HF_TOKEN": "new", "LLM_PROVIDER": "gemini"})
        assert sr.read_env_file(env) == {"A": "1", "HF_TOKEN": "new", "LLM_PROVIDER": "gemini"}
        assert "# B=2" in env.read_text(encoding="utf-8")
        docs, idx = d / "docs", d / "idx"
        docs.mkdir()
        idx.mkdir()
        (docs / "one.pdf").write_bytes(b"12345")
        assert sr.index_is_stale(idx, docs)                      # nothing built yet
        (idx / "chunks.json").write_text("{}", encoding="utf-8")
        assert sr.index_is_stale(idx, docs)                      # no fingerprint yet
        (idx / ".fingerprint.json").write_text(_j.dumps(sr.docs_fingerprint(docs)), encoding="utf-8")
        assert not sr.index_is_stale(idx, docs)
        (docs / "one.pdf").rename(docs / "renamed.pdf")          # a rename makes the index stale
        assert sr.index_is_stale(idx, docs)


def test_setup_select_and_dry_run():
    import contextlib
    import io
    import setup_and_run as sr
    a = sr.parse_args(["--only", "data,train", "--yes"])
    assert [s[0] for s in sr.select_steps(a)] == ["data", "train"]
    a = sr.parse_args(["--from", "train", "--skip", "launch,summary"])
    ids = [s[0] for s in sr.select_steps(a)]
    assert ids[0] == "train" and "launch" not in ids and "summary" not in ids
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = sr.main(["--dry-run", "--yes", "--only", "venv,train,docs,index", "--python", sys.executable])
    out = buf.getvalue()
    assert code == 0 and "DRY RUN" in out and "[FAIL]" not in out, out[-600:]


def test_lenient_scoring():
    from src.schema import normalize_for_scoring as n
    assert n("amount", "2.50") == n("amount", "2.5") == "2.5"
    assert n("price", "1,250") == n("price", "1250") == "1250"
    assert n("old_fv", "10") == "10" and n("size_cr", "100.0") == "100"
    assert n("company", "Satpura Packaging Ltd.") == n("company", "Satpura Packaging Ltd")
    assert n("company", "X Limited") != n("company", "X Ltd")            # real differences stay different
    assert n("record_date", "2027-02-26") != n("record_date", "2027-02-25")
    assert n("amount", "5.5") != n("amount", "5.6") and n("isin", None) == ""


def test_schema_roundtrip():
    fields = {"event_type": "RIGHTS_ISSUE", "company": "X Ltd", "isin": "INE000000000",
              "record_date": "2026-01-02", "ex_date": "NA", "ratio": "2:5", "price": "1000"}
    assert parse_output(format_target(fields)) == fields


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
