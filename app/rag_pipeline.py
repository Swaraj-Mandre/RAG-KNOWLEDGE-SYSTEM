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
- If the answer is truly not in the context, say "I don't have enough information"

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

# How far away a chunk is allowed to be before we treat it as unrelated.
#
# FAISS gives a distance, where SMALLER means more similar. We did not guess
# this number - eval/run_eval.py measured it across 42 questions:
#   answers that WERE in the documents     : 0.18 to 0.35
#   questions the documents cannot answer  : 0.37 and above
# So anything past ~0.36 is almost certainly not covered by the documents, and
# we say so instead of letting the model invent an answer.
RELEVANCE_LIMIT = 0.36

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

    try:
        # similarity_search_with_score gives us the DISTANCE too, which
        # as_retriever() hides. We need it for the check in step 2.
        results = pipeline.vector_db.similarity_search_with_score(
            lookup, k=pipeline.k
        )
    except Exception as e:
        return friendly_error(e), []

    # Nothing close enough - the documents simply do not cover this.
    #
    # Deciding this here, from the measured distance, is more reliable than
    # hoping the model admits it. It is also the natural place to look
    # elsewhere: we already know the documents have nothing to offer.
    if not results or results[0][1] > RELEVANCE_LIMIT:
        if allow_web:
            import web_search
            answer, web_sources = web_search.answer_from_web(pipeline.llm, lookup)
            return tidy_citations(answer), web_sources
        return ("I don't have enough information in your documents to answer that.", [])

    # Drop the chunks that came back but are not actually close. Search always
    # returns k results even when only two of them are relevant, and feeding
    # the far ones to the model just invites it to cite something unrelated.
    # Safe to cut here: the eval showed correct chunks never exceeded 0.35.
    close_enough = [row for row in results if row[1] <= RELEVANCE_LIMIT]

    context, sources = build_context(close_enough)
    prompt = PROMPT_TEMPLATE.format(context=context, question=lookup)

    try:
        return tidy_citations(pipeline.llm.invoke(prompt).content), sources
    except Exception as e:  # noqa
        return friendly_error(e), sources


def friendly_error(e):
    """Turn a raw API error into something a person can act on."""
    text = str(e).lower()
    if "quota" in text or "429" in text:  # 429 is HTTP Status code (Too Many Requests); just like 404 (not found)
        return "All AI services are out of free quota right now. Wait a minute and try again."
    elif "deadline" in text or "timeout" in text:
        return "Request timed out. Please try again."
    else:
        return f"Error generating answer: {e}"