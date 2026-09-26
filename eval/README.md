# Retrieval Eval

Answers one question: **when you ask something, does the search find the right chunk?**

No LLM is called - this only embeds the questions, so it is nearly free to run.

## Run it

```bash
python eval/run_eval.py
python eval/run_eval.py --k 3      # try a different number of retrieved chunks
```

## The question file

`questions.json` is a plain list. Each entry:

```json
{
  "q": "What is the time complexity of merge sort?",
  "source": "Updated DAA Unit 1.pptx",
  "slide": 23
}
```

- `source` - the file that actually answers it
- `slide`  - slide number, or `null` if the file has no slides (txt, pdf, image)
- `source: null` means **the documents cannot answer this**. These questions test
  whether the system admits it does not know instead of inventing something.
- `note` is optional and ignored by the script.

## Reading the output

| Metric | Meaning |
|---|---|
| `hit@1` | correct chunk was the top result |
| `hit@3` / `hit@5` | correct chunk was somewhere in the top 3 / top 5 |
| `MRR` | rewards ranking the right chunk high. Rank 1 = 1.0, rank 2 = 0.5, miss = 0 |
| `distance` | how similar the best match was. **Lower = closer.** |

For out-of-document questions a hit rate makes no sense, because search always
returns `k` chunks no matter what. So we compare distances instead: if wrong
answers consistently sit further away than right ones, that gap becomes a
cutoff for answering "I don't have enough information".

## Writing good questions

Write them **without looking at the slides**. If you read the slide first you
will reuse its wording, search will match on those exact words, and the score
will look better than the system really is. Questions phrased the way you would
actually type them are the ones worth measuring.
