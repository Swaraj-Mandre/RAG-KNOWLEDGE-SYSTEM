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
#We used "if" statement , however it's written in english it mirrors 'Conditional Logic'

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
    return vector_db.as_retriever(search_kwargs={"k": 3})

# Build LLM
def build_llm(model="gemini-2.5-flash", temperature=0.2):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature
    )
# 0.0 = The AI is a boring robot that never changes its answer.
# 1.0 = The AI is a poet who might start hallucinating. 

# Build RAG Chain
def build_rag_chain(retriever, llm):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    ) #telling PromptTemplate exactly which 'holes' it needs to fill
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, #Passing llm (the brain) and retriever (the librarian)
        chain_type_kwargs={"prompt": prompt}
    ) # 'Key : Value'

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

# ── Query Function ────────────────────────────────────────
def query_pipeline(chain, question):
    """
    Pass the chain and a question, get an answer back.
    """
    try:
        response = chain.invoke(question)  # invoke = ask
        return response['result']
    except Exception as e:  # noqa
        if "quota" in str(e).lower() or "429" in str(e):  # 429 is HTTP Status code (Too Many Requests); just like 404 (not found)
            return "API quota exceeded! Free tier limit reached. Wait a minute and try again."
        elif "deadline" in str(e).lower() or "timeout" in str(e).lower():
            return "Request timed out. Please try again."
        else:
            return f"Error generating answer: {str(e)}"