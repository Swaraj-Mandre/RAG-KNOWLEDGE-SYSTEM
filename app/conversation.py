"""
conversation.py - remembering what was already said.

The problem this solves
-----------------------

Ask "What is the Master Theorem?" and the system works fine. Then ask
"summarize that in 3 bullet points" and it falls apart.

The reason is worth understanding, because it is not obvious. Searching the
documents works by turning the question into numbers and finding chunks with
similar numbers. But "summarize that in 3 bullet points" contains no subject at
all - the word "that" carries the meaning, and only a human who read the
previous turn knows what "that" refers to. So the search looks for documents
about summarising and bullet points, finds nothing useful, and the answer is
either wrong or a refusal.

The fix: before searching, rewrite the follow-up into a question that stands on
its own. "summarize that in 3 bullet points" becomes "summarize the Master
Theorem in 3 bullet points", which searches perfectly well.

This is the standard solution in RAG systems. It is sometimes called query
rewriting or question condensing.

Two details that matter
-----------------------

1. We do not rewrite every question. Rewriting costs an extra API call, and
   worse, it can damage a question that was already fine. So we first check
   whether the question actually seems to depend on earlier turns.

2. We only keep the last few turns. Old context stops being relevant, and a
   long history makes every prompt bigger and slower for no benefit.
"""

# How many previous question-and-answer pairs to remember.
# Three is enough for "summarize that" and "now explain it simpler" to work,
# without dragging the whole chat into every prompt.
MAX_TURNS = 3

# Answers can be very long. When we put an old answer into a prompt as context
# we only need the gist, so we cut it short.
ANSWER_PREVIEW_CHARS = 400


# Words that usually point back at something said earlier. A question holding
# one of these probably cannot be understood on its own.
REFERRING_WORDS = {
    "it", "its", "that", "this", "these", "those", "them", "they", "their",
    "he", "she", "his", "her", "above", "previous", "earlier", "again",
}

# Openers that continue a previous thought rather than starting a new one.
CONTINUING_STARTS = (
    "and ", "also ", "what about", "how about", "why ", "so ", "then ",
    "more ", "another", "explain", "summarize", "summarise", "simplify",
    "expand", "continue", "elaborate", "give me more", "tell me more",
)


def needs_context(question):
    """Guess whether this question only makes sense after the previous turn.

    This is a cheap check done with plain Python, before spending an API call.
    It is deliberately a little eager: rewriting a question that did not need it
    is usually harmless, while missing one that did need it gives a bad answer.

    Returns True for "summarize that", "explain it simpler", "and the proof?".
    Returns False for "What is the Master Theorem?".
    """
    text = question.lower().strip()

    # A very short question is nearly always a follow-up. On its own,
    # "why?" or "more examples" cannot be searched for.
    if len(text.split()) <= 4:
        return True

    if text.startswith(CONTINUING_STARTS):
        return True

    # Strip punctuation so "that," and "it?" still match.
    words = {w.strip(".,?!:;'\"") for w in text.split()}
    return bool(words & REFERRING_WORDS)


def format_history(turns):
    """Write the remembered turns out as plain text for a prompt."""
    lines = []
    for question, answer in turns:
        short = answer.strip().replace("\n", " ")
        if len(short) > ANSWER_PREVIEW_CHARS:
            short = short[:ANSWER_PREVIEW_CHARS] + "..."
        lines.append(f"User asked: {question}")
        lines.append(f"You answered: {short}")
    return "\n".join(lines)


REWRITE_PROMPT = """Below is a conversation, and then a new question from the user.

The new question may depend on the conversation - for example it may say "that"
or "it" to mean something discussed earlier. Rewrite it as a single question
that makes sense on its own, with the missing subject filled in.

Rules:
- Keep the user's intent exactly. Do not answer the question.
- Keep any instruction about format, such as "in 3 bullet points".
- If the question already stands on its own, repeat it unchanged.
- Reply with the rewritten question only, nothing else.

Conversation so far:
{history}

New question: {question}

Rewritten question:"""


def standalone_question(llm, turns, question):
    """Rewrite a follow-up so it can be searched for on its own.

    If there is no history, or the question already looks self-contained, we
    return it untouched and spend nothing. If the rewrite fails for any reason
    we also return the original - a slightly worse search is much better than
    an error in the user's face.
    """
    if not turns or not needs_context(question):
        return question

    prompt = REWRITE_PROMPT.format(history=format_history(turns), question=question)

    try:
        rewritten = llm.invoke(prompt).content.strip()
    except Exception:
        return question

    # Guard against a model that ignores the instruction and answers instead of
    # rewriting. A rewritten question should be roughly question-sized; if it
    # comes back as a paragraph, something went wrong and the original is safer.
    if not rewritten or len(rewritten) > 300:
        return question

    return rewritten


class Conversation:
    """The last few turns of one chat.

    Kept as a tiny class rather than a loose list so the rest of the app has one
    obvious place to add a turn and one obvious place to read the history.
    """

    def __init__(self, max_turns=MAX_TURNS):
        self.turns = []            # list of (question, answer)
        self.max_turns = max_turns

    def remember(self, question, answer):
        """Store a completed turn, dropping the oldest if we are full."""
        self.turns.append((question, answer))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def clear(self):
        self.turns = []
