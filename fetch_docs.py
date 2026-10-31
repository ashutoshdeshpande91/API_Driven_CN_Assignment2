"""Download the public documents listed in docs_rag/sources.csv into docs_rag/.

Run this on YOUR machine (needs internet). Standard library only.
    python fetch_docs.py                    # everything not marked optional
    python fetch_docs.py --include-optional # also the legacy depository manuals
    python fetch_docs.py --force            # re-download files that already exist
    python fetch_docs.py --dry-run          # show what would be downloaded

Rows with mode=page point at an HTML page; the script looks for the PDF link on it.
Anything that fails is listed at the end: open the URL in a browser and save the PDF into
docs_rag/ using the file_name from the CSV.
"""
import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE / "docs_rag"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


def get(url, timeout=90, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(str(last))


def find_pdf_link(html, base_url):
    """SEBI pages embed links like .../web/?file=https://www.sebi.gov.in/sebi_data/attachdocs/<mon-yyyy>/<id>.pdf"""
    found = []
    for raw in re.findall(r"""(?:src|href)\s*=\s*['"]([^'"]+)['"]""", html, flags=re.I):
        c = raw.replace("&amp;", "&")
        m = re.search(r"file=([^&#'\"]+\.pdf)", c, flags=re.I)
        if m:
            c = urllib.parse.unquote(m.group(1))
        if re.search(r"\.pdf($|[?#])", c, flags=re.I):
            found.append(urllib.parse.urljoin(base_url, c))
    found.sort(key=lambda u: 0 if "attachdocs" in u else 1)
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(DOCS / "sources.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok, skipped, failed = [], [], []
    for r in rows:
        if r["optional"].strip().lower() == "yes" and not args.include_optional:
            skipped.append((r["file_name"], "optional"))
            continue
        dest = DOCS / r["file_name"]
        if dest.exists() and not args.force:
            skipped.append((r["file_name"], "already downloaded"))
            continue
        if args.dry_run:
            print(f"[dry-run] {r['mode']:4s} {r['url']} -> {dest.name}")
            continue
        print(f"Downloading {r['file_name']} ...")
        try:
            data = get(r["url"])
            if r["mode"] == "page":
                link = find_pdf_link(data.decode("utf-8", "ignore"), r["url"])
                if not link:
                    raise RuntimeError("no PDF link found on the page")
                print(f"   PDF link: {link}")
                data = get(link)
            if not data.startswith(b"%PDF"):
                raise RuntimeError("response is not a PDF (the site may be blocking scripts)")
            dest.write_bytes(data)
            ok.append((r["file_name"], len(data) // 1024))
        except Exception as e:  # keep going; report at the end
            failed.append((r["file_name"], r["url"], str(e)))
        time.sleep(1.5)  # be polite to the servers

    print("\n=== Summary ===")
    for n, kb in ok:
        print(f"OK       {n} ({kb} KB)")
    for n, why in skipped:
        print(f"SKIPPED  {n} ({why})")
    for n, url, why in failed:
        print(f"FAILED   {n}\n         {why}\n         open in browser and save as {n}: {url}")
    if ok:
        print("\nNext: python -m src.rag build")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
