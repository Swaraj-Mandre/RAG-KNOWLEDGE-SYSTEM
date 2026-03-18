import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

def ingest_documents():
    print("Loading documents...")

    # 1. Load document
    loader = TextLoader("data/documents/RAG.txt", encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Create embeddings in batches to avoid rate limit
    print("Creating embeddings (this may take a minute)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # 4. Store in FAISS
    print("Storing in vector database...")
    vector_db = FAISS.from_documents(chunks, embeddings)

    # 5. Save locally
    vector_db.save_local("vectorstore")
    print("Vector database created and saved!")

if __name__ == "__main__":
    ingest_documents()