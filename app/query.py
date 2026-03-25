import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from rag_pipeline import initialize_pipeline, query_pipeline

if __name__ == "__main__":
    chain = initialize_pipeline()

    print("Ask questions about your document!") #250 questions/day free (gemini-2.5-flash limit)
    print("   Type 'exit' to quit\n")

    while True:
        query = input("You: ").strip() #strip removes the extra space in between sentence from user
        if not query:
            continue
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        print("\nThinking...")
        answer = query_pipeline(chain, query)
        print("\nAnswer:")
        print(answer)