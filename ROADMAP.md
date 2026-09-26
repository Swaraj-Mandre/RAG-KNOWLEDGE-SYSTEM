# RAG Knowledge System - Roadmap

## What this project is

Ask questions about your own documents and get answers grounded in them - with
honest sourcing. When the documents cannot answer, say so, and (later) go find
the answer on the web and label it as coming from the web.

Two audiences, deliberately kept in one codebase:
- a tool the author actually uses on college material
- a portfolio piece that has to survive technical questioning

Guiding constraint: **simple enough to read and learn from.** No framework is
added unless it earns its place.

---

## Providers (decided 2026-09-26, after comparing 7 options)

| Job | Primary | Fallback 1 | Fallback 2 |
|-----|---------|------------|------------|
| Embeddings | Mistral `mistral-embed-2312` (20M TPM, 1 RPS) | Gemini `gemini-embedding-001` (1K/day) | - |
| Chat | Mistral `mistral-large-2512` (250K TPM, 1 RPS) | Groq `gpt-oss-120b` (1K RPD, 200K TPD) | Gemini model rotation (60/day) |
| OCR / Vision | Mistral OCR API (625 pages/min) | Groq `qwen/qwen3.8-27b` | Gemini model rotation |

Mistral is primary because its per-minute allowance exceeds Groq's entire daily
budget, and its OCR is a purpose-built document API rather than a vision model
used for OCR. One primary provider keeps the code simple; the fallback chain
costs little extra once it exists and protects against the monthly cap.

**Caveat:** Mistral's Limits page shows per-minute limits only. A plan-level
monthly ceiling (~1B tokens/month on the free tier) exists but is not visible
there. Watch Admin > API > Usage.

**Rejected:** Cerebras ($5 credits expiring after 30 days - not a free tier),
OpenRouter (50 requests/day without paying), Cohere (1,000 calls/month),
Jina (1M tokens/month, 7x less than Gemini for this corpus), local models
(author's decision - and Mistral's embedding allowance removes the only strong
reason for local anyway).

**Vector store: keep FAISS.** At hundreds-to-thousands of chunks, brute-force
IndexFlatL2 answers in about a millisecond, and it needs no server or account.
Incremental adds (`add_documents` + `merge_from`) and metadata filtering are
already supported. Revisit only when the demo needs per-session isolated
collections - that is the one case where Chroma's collections would help.

Key facts worth remembering:
- Gemini quotas are **per model**: 2.5 Flash / 2.5 Flash Lite / 3 Flash each get
  their own 20/day, so 60/day on one key. Flash Lite accepts images (confirmed).
- Extra API keys do nothing: Google limits per *project*, Groq per *organization*.
- Surviving a quota wall is an **idempotency** problem, not a key problem.
  `.vision_cache.json` (hash-keyed) is what makes a run resume where it stopped.
- Switching embedding model invalidates the index. mistral-embed is 1024 dims
  vs Gemini's 3072, so the switch costs one full re-ingest - and buys 3x less
  index memory plus effectively unlimited future re-ingests.

---

## Already built

- Markdown answers render correctly in the UI; user input is HTML-escaped;
  the markdown renderer has raw HTML disabled.
- `requirements.txt` cut 170 -> 28 direct dependencies.
- `build_retriever` honours `k`.
- **Eval harness** (`eval/`): hit@1 / hit@3 / hit@k / MRR against an answer key,
  no LLM calls. Splits scores by question style to expose the vocabulary gap.
  `--k` and `--vectorstore` flags for comparing configurations.
- **PPTX diagram reading**: picture-only slides go to a vision model.
  Logo detection by image-hash repetition. Hash-keyed cache; failures not cached.
- **Fixed a real bug**: the ingest batching slept between batches but embedded
  everything in one burst at the end, so it gave zero rate-limit protection.
  It now embeds each batch properly.

## Known gaps in the loaders (found by testing real files)

| File | Problem |
|------|---------|
| `CF FINAL REPORT.docx` | python-docx cannot open it at all. The zip is fine and 28,499 chars + 14 images are recoverable by reading `word/document.xml` directly. Currently the whole report is silently dropped. |
| any scanned PDF | `load_pdf` detects "no text" and returns nothing, even though the vision path exists a few lines above. |
| any `.docx` | tables and embedded images are never read. |
| any `.pptx` | tables are never read. |
| thin-slide detection | tuned to one deck's 86-char header. Must switch to repetition-based boilerplate detection (already proven: finds 3 template lines in the DAA deck, 0 in Arduino PLC, correct on both). |

---

## Plan

### 1. Provider layer
- [x] Add Mistral as the primary chat provider and Groq as fallback 1.
- [x] Simple fallback chain: on 429, move to the next provider/model and retry.
      Copy the LiteLLM *pattern* (cooldown the failed model for the day), not
      the library - roughly 30 lines and it stays readable.
- [x] Switch embeddings to `mistral-embed` (the `-2312` id does not exist on the
      free tier). Gemini stays listed but is never a live fallback: swapping
      embedding models mid-index would corrupt it.
      Requires one full re-ingest (1024 dims vs 3072).
- [x] Route image reading through Groq `qwen/qwen3.8-27b`, with Gemini as fallback.
      Mistral's OCR API was not used - the vision chat models were enough.

### 2. Universal document loading
- [x] Boilerplate detection by repetition, replacing the fixed character threshold.
- [x] docx: raw-XML fallback when python-docx fails.
- [x] docx + pptx: read tables.
- [x] pdf: render pages to images and use vision when a page has no text.
- [x] Run every sample file through the loader and record what each yields.

### 3. Measure
- [x] Finish `eval/questions.json` (25 natural + 12 slide-derived + 5 out-of-doc).
- [x] Baseline against the text-only index.
- [x] Re-run after vision ingest -> before/after number on the same answer key.
- [x] Sweep `chunk_size` and `k`; pick values with evidence, not defaults.

### 4. Citations
- [x] Replace `RetrievalQA` with ~25 explicit lines returning `(answer, sources)`
      and numbering the chunks, so answers cite file + slide.
- [x] Show sources in both the CLI and the web UI.

### 5. Then
- [ ] Conversation memory (follow-ups currently fail).
- [x] Web fallback with honest source labelling - cheap once citations exist.
      DuckDuckGo (no API key), snippets only, answers labelled as web-sourced
      with links. Can be switched off with `ask(..., allow_web=False)`.
- [ ] Cap and isolate the public demo (session-scoped index, rate limit).
- [ ] Incremental ingest: adding one document should not re-embed everything.

---

## Billing safety

No payment method is on file with any provider, so none of them can charge -
they return 429 instead. The only way a bill appears is if a card is added.

- **Gemini**: do NOT link a billing account. Doing so auto-upgrades the project
  to Tier 1 and charging starts.
- **Groq**: stay on the Free plan, no payment method.
- **Mistral**: the only one with real exposure, because pay-as-you-go can extend
  usage beyond the free allowance. Check Admin > Subscriptions > Billing that
  pay-as-you-go is off, and set the Organization monthly spending limit to the
  minimum (hitting it suspends API access, which is the desired hard stop).

## Demo hardening (decided)

A free public URL cannot be made abuse-proof. The goal is to bound exposure and
keep the demo working for the next visitor. Exposure is already capped because
the quota is free and resets - worst case the demo is down for a day, not a bill.

Two modes from one codebase, chosen by a config flag:

- **demo mode** (public): preloaded corpus, **uploads disabled**, global daily
  query cap, per-session cap, input length cap.
- **personal mode** (local): uploads on, no caps, corpus accumulates.

Disabling uploads in demo mode is the single biggest win: it removes the
expensive operations (embedding + OCR), removes the collision where two visitors
overwrite each other's index, and removes the liability of strangers' documents
sitting on a shared container.

Skipped deliberately: IP rate limiting (Streamlit Cloud does not reliably expose
client IP, and NAT means users share them) and CAPTCHA (friction, overkill).
Auth was considered and rejected - a login wall between a recruiter and the demo
is a bad trade.

## Frontend and hosting (decided)

**Keep Streamlit.** The target roles are AI/ML/LLM engineering, where Streamlit
reads as normal tooling rather than a gap. A React rewrite costs a separate API
backend, another host, CORS and auth - weeks of work for no signal in those
roles. Visible citations buy more credibility than a framework change.

| Host | Specs |
|------|-------|
| Streamlit Community Cloud (current) | 1 GB RAM, 1 CPU, ~3 concurrent users, sleeps after 12h |
| Hugging Face Spaces (CPU Basic) | 2 vCPU, 16 GB RAM, 50 GB non-persistent disk, sleeps after 48h |

HF Spaces is better on resources and is where ML practitioners browse, so it is
worth adding as a second deployment from the same repo. Caveat: there is an
active community complaint about recent free-tier SDK restrictions - verify
before relying on it as primary.

## What makes this project not a commodity

A Streamlit + FAISS RAG app on its own is very common. Four things differentiate
it, and all four are on the plan above:

1. An eval number almost no comparable project has.
2. Citations - turns "trust me" into "check it yourself".
3. Handling genuinely messy real documents (a .docx python-docx cannot open, a
   deck where half the content is images). This is the actually hard part of RAG.
4. Multi-provider fallback - operational thinking, not tutorial-following.

Deliberately NOT doing: GraphRAG, reranking, agent frameworks, frontend rewrite.
