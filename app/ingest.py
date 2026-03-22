import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

#def is function , in which anything created (like 'loader' 'chunks' 'vector_db') is LOCAL. Once function finishes, Python cleans up memory and those variable disapper.
def ingest_documents():
    print("Loading documents...")

    # 1. Load document
    loader = TextLoader("data/documents/RAG.txt", encoding="utf-8") #utf-8 , saves program from crashing for special character or emojis
    documents = loader.load() #yet we got address of document '.load()' plays important role to go to harddrive, opem the file, read the text , and bring back into RAM (memory) 
    print(f"Loaded {len(documents)} pages") #len() = "How many iteams do you contains?" 

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50 #It's like going 50 piece back in last chunk and include that into new chunk . By which no gap , it didnt miss content flow
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