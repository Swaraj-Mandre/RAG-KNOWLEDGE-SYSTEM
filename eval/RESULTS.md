# Eval Results

Measured 2026-09-26. 42 questions, 22 answerable from the documents and 20 that
the documents cannot answer. Same questions and the same embedding model
(`mistral-embed`, 1024 dims) in both columns - the only difference is whether
picture-only slides were read by a vision model.

## Does reading the pictures help?

| Metric | Text only (92 chunks) | With vision (210 chunks) | Change |
|--------|----------------------|--------------------------|--------|
| hit@1  | 68.2% | 63.6% | **-4.6** |
| hit@3  | 77.3% | 81.8% | +4.5 |
| hit@5  | 81.8% | **95.5%** | **+13.7** |
| MRR    | 0.739 | 0.742 | +0.003 |
| Never found | 4 questions | 1 question | -3 |

**Honest reading.** Reading the pictures clearly helps us *find* the answer:
hit@5 rose 13.7 points and misses fell from 4 to 1. But hit@1 got slightly
worse and MRR barely moved. Adding 118 chunks means more competition for the
top slot, so the right chunk is found more often but not always ranked first.
Recovering hit@1 is what the chunk-size sweep is for.

## The bigger win: knowing when to say "I don't know"

Search always returns k chunks, so a hit rate means nothing for questions the
documents cannot answer. We compare distances instead (lower = more similar).

| | Text only | With vision |
|---|---|---|
| correct match distance | 0.227 - 0.380 | 0.181 - 0.349 |
| no-answer best distance | 0.375 - 0.613 | 0.372 - 0.613 |
| separable? | **no, ranges overlap** | **yes, clean gap** |

With vision, a cutoff near **0.35** separates answerable from unanswerable
across all 42 questions with no overlap. That turns "I don't have enough
information" from a guess into a measured rule.

## Vocabulary gap

Questions written while reading the slides score higher than questions written
from memory, because they reuse the slides' own wording.

| | Text only | With vision |
|---|---|---|
| written while reading slides (hit@5) | 83.3% | 100.0% |
| written from memory (hit@5) | 80.0% | 90.0% |
| gap | +3.3% | **+10.0%** |

The gap widened. Vision output repeats slide vocabulary, so slide-worded
questions benefit more. Worth remembering: a casual tester who phrases
questions like the source will see a flattering number.

## Caveats

- 22 answerable questions is a small sample. Treat these as directional.
- The ground truth was written by the project author, so there is selection bias.
- One question is still never found; its mapped slide is probably wrong rather
  than retrieval failing. To be corrected during the sweep.

## Reproduce

```bash
python eval/run_eval.py                                   # current index
python eval/run_eval.py --vectorstore vectorstore_textonly  # text-only index
python eval/run_eval.py --k 3                             # try a different k
```

---

## How many chunks should we retrieve? (choosing k)

Run with `--k`. No re-indexing needed, so this costs almost nothing.

| k | hit@k | MRR |
|---|-------|-----|
| 3 | 81.8% | 0.712 |
| **5** | **95.5%** | **0.742** |
| 8 | 95.5% | 0.742 |
| 10 | 95.5% | 0.742 |

**k=5 is the right choice.** Below it we lose 13.7 points of recall; above it we
gain nothing at all. `chunk_size=1000` and `k=5` started as tutorial defaults -
k=5 is now a measured decision.

That k=8 and k=10 change nothing also tells us something useful: the one
question we never find is not hiding at rank 7. It is not retrievable at any
depth, which means its expected slide in questions.json is wrong rather than
retrieval being at fault.

## After fixing the document loaders

Adding the recovered .docx (29,808 characters) and per-page PDF loading took the
index from 210 to 249 chunks. Retrieval scores were **unchanged** (63.6 / 81.8 /
95.5, MRR 0.742).

That is the correct outcome, not a disappointment: the new chunks are a
networking report and PDF pages that have nothing to do with the DAA questions.
39 unrelated chunks were added and displaced no correct answers, which says
retrieval is discriminating well. The benefit of that work would only appear in
questions about the report - which this eval set does not contain yet.

---

## Correction: one answer key was wrong

While wiring up citations I checked the single question that retrieval never
found at any depth:

> "How can you compare two functions such as n, n log n, n^2, and 2^n as input
> size increases?"

The answer key said **slide 22**. Slide 22 is one line defining the *term*
"order of growth" (drop lower-order terms, ignore constants). Slide 23 is a
table titled **ORDERS OF GROWTH** whose columns are literally
`n, log2 n, n log2 n, n^2, n^3, 2^n, n!` - which is what the question asks for.

Retrieval had been returning slide 23 as its **top** result at distance 0.330,
well inside the band where correct answers live (0.18 - 0.35). So retrieval was
right and the answer key was wrong. The key now says slide 23.

**This correction raised the scores, so it is worth being clear about what
changed and why it is not the same as tuning the test until it passes:** the
code was not touched, no threshold was moved, and the fix was decided by reading
the two slides - not by looking at what would score better.

| metric | before the fix | after the fix |
|---|---|---|
| hit@1 | 63.6% | 68.2% |
| hit@3 | 81.8% | 86.4% |
| hit@5 | 95.5% | **100.0%** |
| MRR | 0.742 | **0.787** |
| never found | 1 | **0** |

### The vocabulary gap was partly an illusion

The gap between slide-worded and memory-worded questions dropped from **+10.0%
to +0.0%**. The mislabeled question was sitting in the memory-worded group and
was counted as a miss it never deserved. Once corrected, both groups score the
same.

That is the more useful finding: the earlier "+10% gap" was evidence of a
bookkeeping error, not evidence that retrieval was matching words instead of
meaning. It is a good reminder that an eval is only ever as trustworthy as its
answer key, and that a number moving in the direction you hoped for is a reason
to check it harder, not to celebrate.

---

## The distance threshold does not transfer between documents

The measurements above produced a clean split on the DAA deck: correct answers
sat at 0.18-0.35, unanswerable questions at 0.37 and above. A cutoff of 0.36
separated them perfectly, and the pipeline used it to decide when to say "I
don't have enough information".

While hardening the public demo, that cutoff was tested against a document it
had never seen - a short personal note holding a name, an address, a bank
account number and a medical detail:

| question the note CAN answer | distance | verdict at 0.36 |
|---|---|---|
| What is the bank account number? | 0.533 | refused |
| Who is this record about? | 0.553 | refused |
| What is Priya Sharma allergic to? | 0.405 | refused |
| What is the home address? | 0.589 | refused |

Every single one was refused, with the answer sitting in the retrieved chunk.

### Why

The slides are topical - one slide is about one idea, so a question about that
idea matches the whole chunk closely. The note is the opposite: four unrelated
facts share one chunk, so a question about any one of them matches only a
fraction of it and the distance rises. Nothing was wrong with retrieval; the
right chunk was returned first every time. The cutoff was wrong.

**0.36 was never a property of the embedding model. It was a property of that
deck.** Publishing it as a threshold was the mistake, and it only surfaced
because the demo lets visitors upload documents nobody has seen before.

### What replaced it

- `FAR_LIMIT = 0.75` - loose enough to be a filter for nonsense rather than a
  judge. Past it, nothing retrieved is plausibly related, so we skip the model
  call and go to the web.
- `SPREAD = 0.12` - keep the best chunk and anything nearly as good. Relative to
  whatever came back, so it adapts to the document.
- The model decides the rest. It reads the chunks and replies with a fixed
  sentinel when they do not answer the question, which the pipeline treats
  exactly like finding nothing.

This costs one extra model call for a question the documents cannot answer,
which the old cutoff avoided. That is a fair price for not refusing questions
that the documents clearly do answer.

The retrieval numbers higher up this page are unaffected - they measure which
chunks come back, not what is done with them afterwards.
