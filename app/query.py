import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent)) #relative -> absolute path

from rag_pipeline import initialize_pipeline, ask

if __name__ == "__main__":
    pipeline = initialize_pipeline()

    print("Ask questions about your document!") #250 questions/day free (gemini-2.5-flash limit)
    print("   Type 'exit' to quit\n")

    while True:
        query = input("You: ").strip() #strip removes the extra space in between sentence from user
        if not query: #avoiding blank API request
            continue
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        print("\nThinking...")
        answer, sources = ask(pipeline, query)

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


# Need to add "Conversation Memory" - It can't sense questions like 'Can you summarize that in 3 bullet points' 