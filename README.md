# RAG Knowledge System

Ask questions about your own documents, and get answers that show exactly which slide or page they came from.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-knowledge-system.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Try it live:** [rag-knowledge-system.streamlit.app](https://rag-knowledge-system.streamlit.app)

---

## The problem

You have slide decks, reports and PDFs. Somewhere in them is the answer you need.

Searching by keyword rarely helps, because you remember the *idea*, not the exact words that were used. So you open files one by one and scroll.

Asking a general chatbot does not help either. It has never seen your documents, so it answers from something else and sounds confident either way.

## What this does instead

It reads your documents, then answers from them and only from them.

- **Ask in plain English.** No keywords, no guessing which file it is in.
- **Every answer is checkable.** Each claim carries a number pointing to the real slide or page.
- **It gives you what it found, not a shrug.** If your documents only mention something in passing, you get that detail plus a plain note that there is no more. A half answer from your own file beats a refusal.
- **It admits what it does not know.** If nothing in your documents relates to the question, it says so rather than inventing an answer.
- **It never goes outside your documents.** No web search, no general knowledge. Every sentence traces back to a numbered extract from a file you provided.
- **It reads pictures too.** Slides where the content sits inside a screenshot are read by a vision model, not skipped.

---

## How it works

Two separate jobs. One runs when you add a document, the other when you ask a question.

```
ADDING A DOCUMENT
  file  ->  read text and pictures  ->  cut into pieces
        ->  turn each piece into numbers  ->  save to a local index

ASKING A QUESTION
  question  ->  turn into numbers  ->  find the closest pieces
            ->  anything related?
                  yes  ->  answer from those pieces, with sources
                  a bit ->  give what is there, and say that is all
                  no   ->  say nothing in your documents covers it
```

Turning text into numbers is what makes this work. Text about similar ideas produces similar numbers, even when the words are different. That is why "how long does this take to run" finds a slide about time complexity.

---

## What makes it different

Most small RAG projects stop at "it returns an answer". These are the parts that took the real work.

**It knows when to stay quiet.** A search always returns results, even useless ones. Deciding whether they are good enough is a separate problem, and a fixed threshold does not survive contact with an unfamiliar document.

**It reads slides that look empty.** In the test deck, 32 of 60 slides held only 86 characters of repeated footer text. The actual content was inside pasted screenshots. Today, 141 of 249 indexed pieces come from pictures.

**It is measured, not assumed.** 42 fixed questions, 22 the documents can answer and 20 they cannot. The second group exists to check that wrong matches land measurably further away than right ones.

**It handles follow-up questions.** "Summarize that in three points" has no subject in it. The question is rewritten into one that stands alone before searching.

**Your documents are the only source.** There is no web search and no general knowledge to fall back on. That is a deliberate limit: an answer you cannot trace to your own file is not worth having, and a system with a second source will always be tempted to use it.

---

## Measured results

From `eval/run_eval.py`, run against 42 fixed questions.

| Measure | Result |
|---|---|
| Correct piece ranked first | 68.2% |
| Correct piece in top 3 | 86.4% |
| Correct piece in top 5 | 100% |
| Average ranking score (1.0 perfect) | 0.787 |

Read the first row, not the last. Some questions are answered across a run of slides, so the answer key accepts any one of them, five on average. That makes the top-5 figure easier to reach than it looks. Ranking the right piece first is the number worth judging this on.

This measures retrieval only. It does not test the refusal, because the script never calls the writing model.

Reading pictures raised top-5 accuracy from 81.8% to 95.5% and cut never-found questions from 4 to 1. Full write-up, including a correction that raised the score and why it was legitimate: [eval/RESULTS.md](eval/RESULTS.md)

---

## Built with

| Part | Choice |
|---|---|
| Interface | Streamlit |
| Framework | LangChain |
| Search index | FAISS (local file) |
| Text to numbers | Mistral `mistral-embed` |
| Writing answers | Groq, Mistral, Gemini |
| Reading pictures | Groq vision, Gemini fallback |

Every service used has a free tier and needs no payment card.

Three providers are listed in order. When one runs out of its daily allowance, the next answers. The embedding model is the exception and is never swapped, because different models produce different length number lists and the two cannot be compared.

---

## Run it yourself

```bash
git clone https://github.com/Swaraj-Mandre/RAG-KNOWLEDGE-SYSTEM.git
```

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with any of these keys. One is enough to start, three gives you backups.

```
GROQ_API_KEY=your_key
MISTRAL_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

Put your documents in `data/documents/`, then build the index:

```bash
python app/ingest.py
```

Start the app:

```bash
streamlit run app/streamlit_app.py
```

Adding a document later only reads the new file. Nothing already indexed is processed again.

Deploying your own public copy: [DEPLOY.md](DEPLOY.md)

---

## Project structure

```
app/
  streamlit_app.py    the web interface
  rag_pipeline.py     finding pieces and writing the answer
  ingest.py           reading documents, including pictures
  providers.py        which AI service to call, and what to do when it runs out
  conversation.py     remembering the last few turns
  limits.py           keeping the public demo safe and free
  query.py            the same thing in a terminal
eval/
  run_eval.py         the measuring script
  questions.json      42 test questions
  RESULTS.md          what the numbers showed
```

---

## Supported files

PDF, PPTX, DOCX, TXT, JPG, PNG.

Scanned pages and image-only slides are read by a vision model. Tables in Word and PowerPoint are read as text.

---

## What it cannot do

Worth knowing before you try it.

- **It cannot count across a whole document.** "How many slides mention sorting" needs to read everything at once. It only looks at a handful of pieces.
- **Editing a document rebuilds the index.** Adding a new file is cheap. Changing an existing one is not, by design, because leaving stale text behind is worse than waiting.
- **Reading pictures is not perfect.** A vision model can misread a diagram, which is why anything taken from a picture is labelled in the source list.
- **It only knows what you gave it.** Ask about something your documents do not cover and it will tell you so. That is the design, not a gap to be filled in later.
- **Documents go to outside services to be read.** Fine for coursework and public material. Not suitable as-is for confidential files.

---

## Demo limits

The public demo runs on free hosting and a shared allowance, so it caps usage:

- 3 files per upload, 10 MB each
- 15 questions per visit
- Uploaded files stay in your own session and are deleted automatically

Running it yourself has no limits.

---

## License

MIT. See [LICENSE](LICENSE).
