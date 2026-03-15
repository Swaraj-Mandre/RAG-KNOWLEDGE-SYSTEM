import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Initialize new Gemini client
client = genai.Client(api_key=api_key)

# Test query
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what Retrieval Augmented Generation is in 3 lines"
)

print("\nGemini Response:\n")
print(response.text)