import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# Prompt Template
PROMPT_TEMPLATE = """You are a helpful AI assistant with access to document chunks below.

Answer the user's question based on the context provided.
- If asked to summarize, summarize everything in the context
- If asked for topics/headings, list them all
- If asked a specific question, answer specifically
- Use bullet points, headings, and examples where helpful
- If the answer is truly not in the context, say "I don't have enough information"

Context:
{context}

Question: {question}

Answer:"""

# Load Embeddings
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

# Load Vector DB
def load_vector_db(embeddings, vectorstore_path="vectorstore"):
    return FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

# Build Retriever
def build_retriever(vector_db, k=5):
    return vector_db.as_retriever(search_kwargs={"k": k})

# Build LLM
def build_llm(model="gemini-2.5-flash", temperature=0.2):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature
    )

# Build RAG Chain
def build_rag_chain(retriever, llm):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

# Master Function
def initialize_pipeline(vectorstore_path="vectorstore", k=5):
    """
    Call this once to load everything.
    Returns a ready-to-use RAG chain.
    """
    print("\nInitializing RAG Pipeline...")
    embeddings = load_embeddings()
    print("Embeddings loaded")

    vector_db = load_vector_db(embeddings, vectorstore_path)
    print("Vector database loaded")

    retriever = build_retriever(vector_db, k)
    print("Retriever ready")

    llm = build_llm()
    print("Gemini 2.5 Flash ready")

    chain = build_rag_chain(retriever, llm)
    print("RAG Pipeline ready!\n")

    return chain

# Query Function
def query_pipeline(chain, question):
    """
    Pass the chain and a question, get an answer back.
    """
    response = chain.invoke(question)
    return response['result']