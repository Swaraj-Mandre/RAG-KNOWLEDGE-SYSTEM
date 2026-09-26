"""
Retrieval eval - does the search find the right slide?

For each question we search the vector store and check whether the chunk
we expected came back. No LLM is called here, so this is almost free to run
(it only embeds the questions).

Run:
    python eval/run_eval.py
    python eval/run_eval.py --k 3
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "app"))  # lets us reuse rag_pipeline instead of copying it

from rag_pipeline import load_embeddings, load_vector_db

QUESTIONS_FILE = ROOT / "eval" / "questions.json"
VECTORSTORE = ROOT / "vectorstore"


# Load Questions
def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


# Match Check
def is_match(chunk, question):
    """True if this chunk is the one the question expects.

    "slide" may be a single number OR a list of numbers, because some answers
    are spread across a run of slides. Landing on any one of them counts."""
    if chunk.metadata.get("source") != question["source"]:
        return False

    wanted = question.get("slide")
    if wanted is None:          # file has no slides (txt, pdf, image)
        return True
    if isinstance(wanted, list):
        return chunk.metadata.get("slide") in wanted
    return chunk.metadata.get("slide") == wanted


# Search One Question
def check_one(vector_db, question, k):
    """Search, then look for the first correct chunk in the results.

    Returns (rank, distance):
      rank     - position of the correct chunk (1 = top result), None if missed
      distance - how far the BEST result was. Lower means more similar.
    """
    results = vector_db.similarity_search_with_score(question["q"], k=k)

    rank = None
    for position, (chunk, _distance) in enumerate(results, start=1):
        if is_match(chunk, question):
            rank = position
            break

    best_distance = results[0][1] if results else None
    return rank, best_distance


# Scoring Helpers
def hit_rate(ranks, at):
    """Share of questions whose correct chunk landed in the top `at` results."""
    if not ranks:
        return 0.0
    hits = [r for r in ranks if r is not None and r <= at]
    return len(hits) / len(ranks)


def mrr(ranks):
    """Mean Reciprocal Rank. Rank 1 scores 1.0, rank 2 scores 0.5, rank 3 -> 0.33.
    A miss scores 0. Rewards putting the right chunk near the top, not just in the list."""
    if not ranks:
        return 0.0
    total = sum(1 / r if r else 0 for r in ranks)
    return total / len(ranks)


def spread(values):
    """min / middle / max of a list of distances, for a quick look at the range."""
    if not values:
        return None
    ordered = sorted(values)
    middle = ordered[len(ordered) // 2]
    return ordered[0], middle, ordered[-1]


# Report
def score_group(rows, label, k):
    """Print hit rates for one group of in-document questions."""
    if not rows:
        return None
    ranks = [row["rank"] for row in rows]
    print(f"\n{label}  ({len(rows)} questions)")
    print(f"   hit@1   {hit_rate(ranks, 1):6.1%}   correct chunk was the top result")
    print(f"   hit@3   {hit_rate(ranks, 3):6.1%}   correct chunk was in the top 3")
    print(f"   hit@{k}   {hit_rate(ranks, k):6.1%}   correct chunk was in the top {k}")
    print(f"   MRR     {mrr(ranks):6.3f}   higher is better, 1.0 is perfect")
    return hit_rate(ranks, k)


def report(answerable, unanswerable, k):
    print("\n" + "=" * 62)
    print(f"RESULTS  (k={k})")
    print("=" * 62)

    if answerable:
        # Split by how the question was written.
        # Questions written while reading the slides reuse the slide's own words,
        # so search finds them easily. Questions written from memory use different
        # words for the same idea - that is the harder, more realistic case.
        natural = [r for r in answerable if r["style"] == "natural"]
        from_slides = [r for r in answerable if r["style"] == "slide-derived"]

        easy = score_group(from_slides, "B. Written WHILE READING the slides", k)
        hard = score_group(natural, "A. Written FROM MEMORY (real wording)", k)
        score_group(answerable, "ALL in-document questions", k)

        if easy is not None and hard is not None:
            print(f"\n   Vocabulary gap: {easy - hard:+.1%}")
            print("   (how much easier the slide-worded questions were. A big gap means")
            print("    retrieval is matching words, not meaning.)")

        missed = [row for row in answerable if row["rank"] is None]
        if missed:
            print(f"\n   Missed completely ({len(missed)}) - retrieval never found these:")
            for row in missed:
                print(f"      - {row['q'][:70]}")

    # The "not in the documents" questions.
    # Search ALWAYS returns something, so a hit rate is meaningless here.
    # What we want is the distance: if wrong answers sit much further away than
    # right ones, that gap can become an "I don't know" cutoff later.
    if unanswerable:
        good = [row["distance"] for row in answerable if row["rank"] == 1]
        bad = [row["distance"] for row in unanswerable]
        print(f"\nOut-of-document questions: {len(unanswerable)}")
        print("   (search always returns 5 chunks, so we compare distances instead)")

        g, b = spread(good), spread(bad)
        if g:
            print(f"   correct match distance   min {g[0]:.3f} | mid {g[1]:.3f} | max {g[2]:.3f}")
        if b:
            print(f"   no-answer best distance  min {b[0]:.3f} | mid {b[1]:.3f} | max {b[2]:.3f}")
        if g and b:
            if b[0] > g[2]:
                print(f"\n   Clean gap. Anything above {g[2]:.3f} could be answered 'I don't know'.")
            else:
                print("\n   Ranges overlap - distance alone cannot tell these apart yet.")
    print()


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="how many chunks to retrieve")
    parser.add_argument("--vectorstore", default=str(VECTORSTORE),
                        help="which index to test, so you can compare two builds")
    args = parser.parse_args()

    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {QUESTIONS_FILE.name}")

    embeddings = load_embeddings()
    vector_db = load_vector_db(embeddings, args.vectorstore)
    print(f"Vector database loaded from {Path(args.vectorstore).name}\n")

    answerable, unanswerable = [], []

    for number, question in enumerate(questions, start=1):
        rank, distance = check_one(vector_db, question, args.k)
        row = {
            "q": question["q"],
            "rank": rank,
            "distance": distance,
            "style": question.get("style", "natural"),
        }

        if question["source"] is None:
            unanswerable.append(row)
            label = "n/a (not in docs)"
        else:
            answerable.append(row)
            label = f"rank {rank}" if rank else "MISS"

        print(f"{number:3d}. [{label:>17}]  d={distance:.3f}  {question['q'][:52]}")

    report(answerable, unanswerable, args.k)


if __name__ == "__main__":
    main()
