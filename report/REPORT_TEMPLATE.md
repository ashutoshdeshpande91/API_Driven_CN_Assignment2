# AIMLZG549 - API-driven Cloud Native Solutions: Assignment II
**Project:** Corporate Actions Assistant (NLP)   **Group No:** <fill>   **File name:** <groupid>.docx / .pdf

## Group details
| Sl. No | BITS ID | Name | Contribution (qualitative) | % contribution (out of 100) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |

(Percentages must be consistent with the per-section contributions below.)

## I. API-based AI application (4 marks)
1. **Domain and problem:** securities operations - corporate action notice processing. Why it matters (2-3 lines).
2. **Category:** NLP.
3. **Sub-tasks (3):** (a) event classification, (b) information extraction, (c) summarisation / checklist generation. State the unified objective.
4. **Models:** fine-tuned Flan-T5 (SLM) for (a)+(b); hosted LLM (<model name>, e.g. Google Gemini through its OpenAI-compatible API; a Hugging Face route is also supported) for (c) and the RAG answers, with a template fallback when the API is unavailable. Justify each choice.
5. **Architecture diagram** (notice -> extractor -> summariser; question -> retriever -> LLM).
6. **Screenshots:** app home, "Process a notice" with output, sidebar status.
7. **Code structure and how to run** (short).

## II. Fine-tuning (2 marks)
- **Dataset:** synthetic notices (fictional companies), 2,400 train / 300 validation / 300 test, six balanced event types, one example record. Say plainly why it is synthetic (no confidential data; no labelled public corpus found) and what it does not capture (real notices are messier). Optionally add a small set of real public exchange announcements you label yourselves.
- **Method:** base model, hyperparameters, hardware (RTX 4070 Super 12 GB), training time, training/validation loss plot (`reports/train_log.json`).
- **Results:** table - base model vs fine-tuned: event-type accuracy, per-field exact match, full-record match (`reports/metrics_*.json`). Add the hand-written set (13 notices, different wording): strict vs formatting-tolerant score, and what the failures show (dropped ISINs, abbreviations).
- **Error analysis:** 3-5 examples from `reports/errors_finetuned.csv`.

## III. RAG (4 marks)
- **Documents:** table of sources (title, issuer, version/date downloaded, pages).
- **Pipeline:** chunk size/overlap, embedding model, retrieval (top-k cosine), prompt with citations.
- **Screenshots:** 3+ questions with answers and source expanders, including one question the documents cannot answer (shows refusal).
- **Evaluation:** 13 questions with known answers (`reports/rag_eval.json`): right document in top 5, right passage in top 5, ranked first, and which questions missed and why. Add a few answers you graded by hand (correct / partly / wrong), including one the documents cannot answer.

## IV. Situated Learning - Workplace Application (5 marks)  [INDIVIDUAL - each member writes their own]
**Name / BITS ID:** <fill>   **Industry / role:** <fill>
1. **Anchor (1):** workplace process or problem, organisational context, current workflow, limitations, expected benefit. Use generic, non-confidential descriptions; state that synthetic data stands in for real data.
2. **Apply (2):** data sources / documents / APIs you would use; demonstration with representative inputs and outputs (screenshots); how each sub-task, the fine-tuned model and the RAG bot support the workflow.
3. **Analyse (1):** compare outputs against workplace needs (accuracy, response quality, efficiency, scalability); explain the gap between this prototype and what the organisation would require.
4. **Reflect (1):** constraints - data privacy, security, regulation, infrastructure, cost, integration; name the single most important improvement before deployment and justify it.

## Contribution of each member (detailed)
| Member | Tasks done | Evidence (sections / commits / screenshots) |
|---|---|---|

## Demo video
Google Drive link: <paste, check sharing is set to "anyone with the link">

## References
Regulations, circulars and model/library documentation used (with download dates).

## Extras beyond the brief (mention briefly; they earn no extra marks)
REST API (`api.py`), entitlement calculator, automated tests (`tests/test_logic.py`), Docker files, one-command setup (`setup_and_run.py`).
