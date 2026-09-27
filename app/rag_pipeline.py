import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

# providers.py decides WHICH AI service we call and handles falling back
# to another one when a service runs out of its free daily budget.
import providers

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# Prompt Template
PROMPT_TEMPLATE = """You are a helpful AI assistant with access to document chunks below.

Answer the user's question based on the context provided.
- If asked to summarize, summarize everything in the context
- If asked for topics/headings, list them all
- If asked a specific question, answer specifically
- Use bullet points, headings, and examples where helpful
- If the context does not answer the question, reply with exactly NOT_IN_DOCUMENTS
  and nothing else. Do not guess, and do not answer from your own knowledge.
  This is important: it is how the system knows to go and look elsewhere.

Every chunk below is numbered. After each fact you state, put the number of the
chunk it came from in square brackets, like [1] or [2][3]. Do not cite a number
that is not listed below.

Write any mathematics in plain text, for example T(n) = 2T(n/2) + n or O(n log n).
Do not use LaTeX, because it will be shown as raw symbols.

Context:
{context}

Question: {question}

Answer:"""
#We used "if" statement , however it's written in english it mirrors 'Conditional Logic'

# How we decide a question is not covered by the documents.
#
# FAISS gives a distance, where SMALLER means more similar. The first version
# of this used a single measured cutoff of 0.36, and the measurement was real:
# across 42 questions on the slide deck, answers that WERE in the documents sat
# between 0.18 and 0.35, and questions the documents could not answer sat at
# 0.37 and above. A clean gap, so 0.36 separated them perfectly.
#
# That number turned out to describe THAT deck, not documents in general, and
# it is worth understanding why before trusting any such number again.
#
# The slides are topical: one slide is about Big Omega, so a question about Big
# Omega matches it closely. Now upload a one-page note holding a name, an
# address, an account number and a medical detail all together. A question
# about any single field only partly matches the chunk as a whole, and measured
# distances came back between 0.40 and 0.59 - every one of them past 0.36, so
# every one would be refused even though the answer was sitting right there.
#
# Since visitors upload documents we have never seen, a fixed cutoff tuned on
# one deck is the wrong tool. So we use two much weaker rules instead:
#
#   FAR_LIMIT   - past this, nothing retrieved is plausibly related, so we can
#                 skip the model entirely and go straight to the web. It is set
#                 loose on purpose; it is a filter for nonsense, not a judge.
#   SPREAD      - having found the best chunk, keep the others that are nearly
#                 as good and drop the rest. This is relative, so it adapts to
#                 whatever the document happens to be.
#
# Anything in between is decided by the model, which can actually read the text
# and tell whether it answers the question. See NOT_IN_DOCUMENTS below.
FAR_LIMIT = 0.75
SPREAD = 0.12

# The exact words the model is told to reply with when the chunks do not answer
# the question. Asking for a fixed string, rather than trying to guess from
# phrasing, means we can detect it reliably and go and search the web instead.
NOT_IN_DOCUMENTS = "NOT_IN_DOCUMENTS"

# Load Embeddings
def load_embeddings():
    """The model that turns text into numbers so we can search by meaning.
    Whichever service providers.py picks, it must be the SAME one that built
    the vector store - otherwise the numbers do not line up."""
    return providers.build_embeddings()

# Load Vector DB
def load_vector_db(embeddings, vectorstore_path="vectorstore"):
    check_same_embedding_model(vectorstore_path)
    return FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# Safety Check
def check_same_embedding_model(vectorstore_path):
    """Stop early if this vector store was built by a different embedding model.

    Every embedding model produces a different number of values per chunk
    (Mistral gives 1024, Gemini gives 3072). Those numbers are only comparable
    to numbers from the SAME model. So we read the size that is actually stored
    in the FAISS index and compare it with the model we are holding now.
    """
    import faiss

    index_file = Path(vectorstore_path) / "index.faiss"
    if not index_file.exists():
        return  # nothing saved yet, ingest.py will create it

    saved_size = faiss.read_index(str(index_file)).d
    current_size = providers.embedding_size()

    if saved_size != current_size:
        raise ValueError(
            f"\n   This vector store holds {saved_size}-value vectors, but "
            f"'{providers.embedding_name()}' produces {current_size}.\n"
            f"   They came from different embedding models, so search would break.\n"
            f"   Fix: re-run  python app/ingest.py  to rebuild it."
        )

# Build LLM
def build_llm(temperature=0.2):
    """The model that writes the final answer.
    This comes back with backups attached - if the first service is out of
    free budget for the day, LangChain quietly tries the next one."""
    return providers.build_chat_model(temperature)
# 0.0 = The AI is a boring robot that never changes its answer.
# 1.0 = The AI is a poet who might start hallucinating.

# Where A Chunk Came From
def describe_source(chunk):
    """A short, readable label for where a chunk came from, for example
    'Updated DAA Unit 1.pptx, slide 55' or 'DMV theory Assignments.pdf, page 3'."""
    where = chunk.metadata
    name = where.get("source", "unknown file")

    if where.get("slide"):
        return f"{name}, slide {where['slide']}"
    if where.get("page"):
        return f"{name}, page {where['page']}"
    return name


def build_context(results):
    """Turn the search results into two things:

      context - the numbered text we paste into the prompt
      sources - the matching list we show under the answer, so the reader can
                check any claim themselves

    Numbering both the same way is what lets a [2] in the answer line up with
    the second source in the list."""
    blocks, sources = [], []

    for number, (chunk, distance) in enumerate(results, start=1):
        label = describe_source(chunk)
        blocks.append(f"[{number}] from {label}\n{chunk.page_content}")
        sources.append({
            "number": number,
            "label": label,
            "distance": round(float(distance), 3),
            "from_image": bool(chunk.metadata.get("from_image")),
            "kind": "document",   # web_search.py marks its own sources "web"
        })

    return "\n\n".join(blocks), sources


# The Pipeline Object
class RagPipeline:
    """Everything needed to answer a question, built once and reused.

    Holding these three together means query.py and streamlit_app.py do not
    need to know how any of it works - they just call ask()."""

    def __init__(self, vector_db, llm, k):
        self.vector_db = vector_db   # the searchable documents
        self.llm = llm               # the model that writes the answer
        self.k = k                   # how many chunks to look at


# Master Function
def initialize_pipeline(vectorstore_path="vectorstore", k=5):
    """
    Call this once to load everything.
    Returns a ready-to-use RAG pipeline.
    """
    print("\nInitializing RAG Pipeline...")
    embeddings = load_embeddings()

    vector_db = load_vector_db(embeddings, vectorstore_path)
    print("   Vector database loaded")

    llm = build_llm()
    print(f"   Ready to answer using the top {k} chunks\n")

    return RagPipeline(vector_db, llm, k)


# Asking for a summary is not a search, and treating it as one fails badly.
#
# Searching works by finding chunks that resemble the question. "Summarize this
# document" does not resemble any particular slide, so the search returns
# whatever happens to be nearest, decides the document cannot answer, and goes
# off to the web - which is how "summarize this ppt" came back with links to
# online PPT summarising tools instead of a summary.
#
# A summary needs the opposite of a search: a spread of the whole document
# rather than the few parts nearest to a phrase.
SUMMARY_PHRASES = (
    "summar", "overview", "outline", "tl;dr", "tldr", "the gist",
    "key points", "main points", "key topics", "main topics", "main ideas",
    "what is this document", "what is this ppt", "what is this file",
    "what is it about", "what's it about", "what is this about",
    "brief me", "in short", "cover", "contents of",
)

# Words that show the question is about the reader's OWN uploaded files. When
# one of these appears, searching the web instead would be plainly wrong, no
# matter how poorly the search went.
OWN_DOCUMENT_PHRASES = (
    "this ppt", "this pptx", "this document", "this doc", "this file",
    "this pdf", "this deck", "this slide", "these slides", "this presentation",
    "this report", "the document", "the ppt", "the pdf", "the file", "the deck",
    "my document", "my ppt", "my pdf", "my file", "my slides", "my deck",
    "uploaded", "attached",
)

# How many pieces of the document to read when writing a summary. Enough to
# cover the whole thing, small enough to fit comfortably in one prompt.
SUMMARY_CHUNKS = 12


def wants_summary(question):
    """True when the question asks about the document as a whole."""
    text = question.lower()
    return any(phrase in text for phrase in SUMMARY_PHRASES)


def about_own_documents(question):
    """True when the question clearly refers to the reader's own files."""
    text = question.lower()
    return any(phrase in text for phrase in OWN_DOCUMENT_PHRASES)


# Words a person uses to mean a particular kind of file, and the file endings
# they correspond to. Used to narrow a summary when several documents are
# indexed at once: "summarize this ppt" should not summarise your CV as well.
FILE_KINDS = {
    (".pptx", ".ppt"): ("ppt", "pptx", "slide", "deck", "presentation"),
    (".pdf",):         ("pdf",),
    (".docx", ".doc"): ("docx", "word document", "word file", "report"),
    (".txt",):         ("txt", "text file", "notes"),
}


def narrow_to_named_file(chunks, question):
    """Keep only the chunks from the file the question is talking about.

    With one document indexed this changes nothing. With several, it is the
    difference between a useful answer and a muddle: asking about "this ppt"
    while a CV and a PDF are also indexed should not pull those in.

    We look for a file kind ("ppt", "pdf") in the question. If nothing matches,
    or the match would leave us with nothing, we keep everything - guessing
    wrong should never lose the reader their answer.
    """
    text = question.lower()

    for endings, words in FILE_KINDS.items():
        if not any(word in text for word in words):
            continue
        kept = [c for c in chunks
                if str(c.metadata.get("source", "")).lower().endswith(endings)]
        if kept:
            return kept

    return chunks


def spread_of_document(vector_db, limit=SUMMARY_CHUNKS, question=""):
    """Take pieces from across the whole document, evenly spaced.

    Chunks are stored in the order they were read, so walking the list at even
    intervals gives the beginning, the middle and the end rather than whatever
    happened to sit nearest one phrase. That is what a summary needs.

    Returns the same (chunk, distance) shape the search returns, so the rest of
    the pipeline does not need a special case. The distance is recorded as 0
    because nothing was measured - these were chosen by position, not by
    similarity.
    """
    chunks = list(vector_db.docstore._dict.values())
    if not chunks:
        return []

    chunks = narrow_to_named_file(chunks, question)

    if len(chunks) <= limit:
        return [(chunk, 0.0) for chunk in chunks]

    step = len(chunks) / limit
    return [(chunks[int(i * step)], 0.0) for i in range(limit)]


def tidy_citations(text):
    """Rewrite odd bracket styles back to plain [1].

    Some models write citations with full-width CJK brackets - the answer comes
    back saying 'Big O is a bound U+30101U+3011' instead of '[1]'. It is only a
    display problem, but it makes the numbers look broken next to the source
    list, so we normalise them in one place rather than nagging the prompt.
    """
    for left, right in (("【", "】"), ("［", "］")):
        text = text.replace(left, "[").replace(right, "]")

    # Some models also tack an internal reference onto the number, writing
    # "[1†55]" instead of "[1]". Keep the number, drop the rest, so the
    # citation still lines up with the source list underneath.
    text = re.sub(r"\[(\d+)[†‡#][^\]]*\]", r"[\1]", text)
    return text


# ── Query Function ────────────────────────────────────────
def ask(pipeline, question, history=None, allow_web=True):
    """Answer one question and say where the answer came from.

    Returns (answer, sources). `sources` is a list of dicts describing each
    chunk we used, so the UI can print them under the answer.

    The steps are deliberately plain:
      1. find the closest chunks, and how close each one was
      2. if even the best one is far away, the documents cannot answer this.
         Search the web instead - and say clearly that we did.
      3. number the chunks and paste them into the prompt
      4. ask the model, and hand back the answer plus the source list

    `history` is the last few turns of the chat, as (question, answer) pairs.
    When it is given, a follow-up like "summarize that" is first rewritten into
    a question that stands on its own - see conversation.py for why that is
    necessary before searching.

    `allow_web=False` turns step 2 back into a plain "I don't know". That
    matters for the public demo, where we do not want every visitor's question
    triggering an outside search.
    """
    # What the user typed stays untouched for display; `lookup` is the version
    # we actually search with. They differ only for follow-up questions.
    lookup = question
    if history:
        import conversation
        lookup = conversation.standalone_question(pipeline.llm, history, question)

    # Someone asking about "this ppt" wants an answer from their own upload.
    # Answering from the web would be wrong even if the search went badly, so
    # the web door is closed for this question before anything else happens.
    if about_own_documents(question):
        allow_web = False

    try:
        if wants_summary(lookup):
            # A summary needs breadth, not the nearest few chunks. See
            # spread_of_document above for why searching fails here.
            results = spread_of_document(pipeline.vector_db, question=question)
        else:
            # similarity_search_with_score gives us the DISTANCE too, which
            # as_retriever() hides. We need it for the check in step 2.
            results = pipeline.vector_db.similarity_search_with_score(
                lookup, k=pipeline.k
            )
    except Exception as e:
        return friendly_error(e), []

    # A summary is built from the whole document, so the distance checks below
    # do not apply - nothing was measured. Answer straight from what we took.
    if results and wants_summary(lookup):
        context, sources = build_context(results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=lookup)
        try:
            return tidy_citations(pipeline.llm.invoke(prompt).content), sources
        except Exception as e:  # noqa
            return friendly_error(e), sources

    # Nothing even vaguely related came back, so there is no point paying for a
    # model call to confirm it. Go straight to the web.
    if not results or results[0][1] > FAR_LIMIT:
        return elsewhere(pipeline, lookup, allow_web)

    # Keep the best chunk and anything nearly as good, then stop. Measuring
    # from the best result rather than from a fixed number is what lets this
    # work on a document we have never seen: if the best match sits at 0.50,
    # chunks at 0.55 are still worth reading, and chunks at 0.70 are not.
    best = results[0][1]
    close_enough = [row for row in results if row[1] <= best + SPREAD]

    context, sources = build_context(close_enough)
    prompt = PROMPT_TEMPLATE.format(context=context, question=lookup)

    try:
        answer = pipeline.llm.invoke(prompt).content
    except Exception as e:  # noqa
        return friendly_error(e), sources

    # The model has now read the chunks and says they do not answer the
    # question. It is a better judge of that than a distance ever was, because
    # it can actually read the words. Treat it exactly like finding nothing.
    if NOT_IN_DOCUMENTS in answer:
        return elsewhere(pipeline, lookup, allow_web)

    return tidy_citations(answer), sources


def elsewhere(pipeline, question, allow_web):
    """The documents cannot answer this. Search the web, or say so honestly."""
    if allow_web:
        import web_search
        answer, web_sources = web_search.answer_from_web(pipeline.llm, question)
        return tidy_citations(answer), web_sources
    return ("I don't have enough information in your documents to answer that.", [])


def friendly_error(e):
    """Turn a raw API error into something a person can act on."""
    text = str(e).lower()
    if "quota" in text or "429" in text:  # 429 is HTTP Status code (Too Many Requests); just like 404 (not found)
        return "All AI services are out of free quota right now. Wait a minute and try again."
    elif "deadline" in text or "timeout" in text:
        return "Request timed out. Please try again."
    else:
        return f"Error generating answer: {e}"