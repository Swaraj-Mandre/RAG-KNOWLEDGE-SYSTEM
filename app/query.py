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
            print("\nSources:")
            for s in sources:
                seen_in_picture = "  (read from a picture)" if s["from_image"] else ""
                print(f"   [{s['number']}] {s['label']}{seen_in_picture}")


# Need to add "Conversation Memory" - It can't sense questions like 'Can you summarize that in 3 bullet points' 