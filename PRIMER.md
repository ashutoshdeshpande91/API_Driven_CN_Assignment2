# Primer: corporate actions in securities operations 

## What is a corporate action?
A decision by a listed company that changes what shareholders hold or receive. Examples:

| Event | What happens to a shareholder | Example |
|---|---|---|
| Dividend | Receives cash per share | Rs. 5 per share |
| Stock split | Each share becomes several, price falls proportionally | 1 share of Rs. 10 -> 10 shares of Rs. 1 |
| Bonus issue | Receives free extra shares | 1:2 = 1 new share for every 2 held |
| Rights issue | Right to buy new shares, usually at a discount | 1:5 at Rs. 250 |
| Buyback | Company repurchases shares at a stated price | Rs. 500 per share, up to Rs. 100 crore |
| Merger | Shares exchanged for shares of another company | 7:10 share-swap ratio |

## Key dates
- **Announcement / board meeting date** - when the company decides.
- **Record date** - who is on the shareholder register on this day gets the entitlement.
- **Ex-date** - first day the share trades *without* the entitlement. Buy on or after it and you do not get the benefit. Under India's T+1 settlement the ex-date is generally the same day as the record date (confirm against the current exchange circular).
- **Payment / allotment date** - when cash or shares actually arrive.

## Why operations teams care
Notices arrive as unstructured text (exchange filings, custodian messages, emails). Someone must read
each one, work out the event type and key numbers, and update positions, cash and client records.
A missed date or a mis-read ratio leads to wrong entitlements, breaks in reconciliation and client complaints.
It is repetitive, deadline-driven reading - a good fit for NLP.

## What our project does
1. **Classify + extract** (fine-tuned small model): notice text -> event type, company, ISIN, record date, ex-date, ratio/amount/price.
2. **Summarise** (LLM via API): plain-English impact + operations checklist.
3. **Ask the rulebook** (RAG chatbot over public SEBI / exchange / depository documents): e.g. "When must a record date be notified?"

Data: synthetic notices written in the style of real filings (fictional companies), plus public documents.

## Mapping it to *your* workplace (for Situated Learning, Part 4)
The reusable pattern is **unstructured notice in -> structured fields out -> action checklist -> policy Q&A**.
Examples of close analogues:
- Banking / insurance: customer complaints, claim letters, regulatory circulars.
- HR / admin: policy updates, leave and benefits queries.
- IT / service desk: incident tickets (classify, extract affected system, answer from runbooks).
- Retail / supply chain: supplier notices, invoices, contract clauses.

For each: state the current manual workflow, what you would feed the system, what output you would expect,
how you would judge quality, and what blocks real deployment (privacy, security, integration, cost).
