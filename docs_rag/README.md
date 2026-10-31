# Documents for the RAG index

Run from the project root (needs internet):

    python fetch_docs.py          # downloads the files listed in docs_rag/sources.csv
    python -m src.rag build       # then build the index

`sources.csv` lists each document with its URL, whether I opened it while preparing this
(`checked=yes`), and notes. Files that fail to download are listed at the end of the script
output - open the URL in your browser and save the PDF here under the given file name.

## What is included
| Document | Why it matters for corporate actions |
|---|---|
| SEBI LODR Regulations 2015 (amended to 15 Oct 2026) | Disclosure of corporate actions, record-date intimation |
| SEBI Buy-Back Regulations 2018 (amended to 6 Jul 2026) | Tender offer / open market buy-back procedure and timelines |
| SEBI ICDR Master Circular (9 Feb 2026) | Rights issue process, bonus issue T+2 trading, bonus delay fines |
| SEBI ICDR Regulations 2018 (amended to 21 Mar 2026) | Rights and bonus provisions (large file; link found automatically) |
| SEBI T+1 circular (7 Sep 2021) | Settlement cycle behind the ex-date / record-date relationship |
| NSDL circular 2023/0006 | Standard field names for corporate-action data |
| NSDL (1998) and CDSL (2017) operating manuals - optional | Depository-side CA processing concepts; LEGACY, label as such |

## Gaps you should fill by hand
- **An official NSE/BSE circular or FAQ on ex-date vs record date under T+1.** I could not find a
  direct official link. Search nseindia.com / bseindia.com circulars for "ex-date" and "T+1" and
  save the PDF here. Without it, the ex-date rule in the app rests on broker FAQs and NSE data
  quoted by third-party sites.
- Check SEBI > Legal for anything newer than the versions above (amendments appear often).

Always record the download date in your report's references table.
