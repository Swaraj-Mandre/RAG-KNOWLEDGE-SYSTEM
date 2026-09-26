import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent)) #relative -> absolute path

from rag_pipeline import initialize_pipeline, ask
from conversation import Conversation

if __name__ == "__main__":
    pipeline = initialize_pipeline()

    # Remembers the last few turns so follow-up questions like
    # "summarize that" know what "that" means.
    chat = Conversation()

    print("Ask questions about your document!") #250 questions/day free (gemini-2.5-flash limit)
    print("   Type 'exit' to quit, or 'new' to start a fresh topic\n")

    while True:
        query = input("You: ").strip() #strip removes the extra space in between sentence from user
        if not query: #avoiding blank API request
            continue
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        if query.lower() in ["new", "reset", "clear"]:
            chat.clear()
            print("Forgot the previous conversation. Ask anything.")
            print()
            continue

        print("\nThinking...")
        answer, sources = ask(pipeline, query, history=chat.turns)

        print("\nAnswer:")
        print(answer)

        # Show where the answer came from, so the numbers like [1] in the
        # answer above can be checked against the real documents.
        if sources:
            # Web sources are printed with their link, so a claim from a
            # stranger's website never looks like a claim from your own slides.
            from_web = sources[0].get("kind") == "web"
            print()
            print("Sources (from the web):" if from_web else "Sources:")
            for s in sources:
                print(f"   [{s['number']}] {s['label']}")
                if s.get("url"):
                    print(f"       {s['url']}")
                elif s["from_image"]:
                    print("       (read from a picture)")

        # Store the turn so the next question can refer back to this one.
        chat.remember(query, answer)


# Conversation memory is handled by conversation.py: the last 3 turns are kept,
# and a follow-up such as "summarize that in 3 bullet points" is rewritten into
# a standalone question before we search the documents. 