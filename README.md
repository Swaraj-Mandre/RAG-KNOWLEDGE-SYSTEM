# 🧠 RAG Knowledge System

> **An AI-powered system that answers questions using your own documents — powered by Gemini 2.5 Flash & LangChain**

---

## 🚀 What is this?

This project builds a **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload your own documents and ask questions about them. Instead of relying on general AI knowledge, it retrieves exact information from *your* files and generates precise answers.

```
Document → Loader → Chunks → Embeddings → Vector DB → Query → Retrieve → LLM → Answer
            (LangChain)        (FAISS)                          (Gemini 2.5 Flash)
```

---

## ⚙️ Two Pipelines Inside Every RAG System

### 🔵 A. Ingestion Pipeline *(Offline — runs once)*

```
Documents
   ↓
Load Documents       ← LangChain Document Loaders
   ↓
Chunk Documents      ← RecursiveCharacterTextSplitter
   ↓
Generate Embeddings  ← Google Generative AI Embeddings
   ↓
Store in Vector DB   ← FAISS
```

### 🟢 B. Query Pipeline *(Runtime — runs on every question)*

```
User Query
   ↓
Query Embedding         ← Same embedding model
   ↓
Vector Similarity Search ← FAISS retriever
   ↓
Top Relevant Chunks     ← Top-k results
   ↓
Inject Context into Prompt
   ↓
Gemini 2.5 Flash        ← LLM generates answer
   ↓
Final Answer ✅
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemini 2.5 Flash |
| Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | Google Generative AI |
| Language | Python 3.11 |
| Document Loaders | PyPDF, LangChain |

---

## 📁 Project Structure

```
rag-knowledge-system/
│
├── app/
│   ├── main.py           # Entry point & LLM test
│   ├── ingest.py         # Document loading & chunking
│   ├── query.py          # User query interface
│   └── rag_pipeline.py   # Full RAG pipeline
│
├── data/
│   └── documents/        # 📂 Place your documents here
│
├── vectorstore/          # Auto-generated FAISS index
│
├── .env                  # API keys (never commit this!)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Swaraj-Mandre/RAG-KNOWLEDGE-SYSTEM.git
cd RAG-KNOWLEDGE-SYSTEM
```

### 2. Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get your free API key at: https://aistudio.google.com/apikey

### 5. Run
```bash
python app/main.py
```

---

## 🔒 Security Note

> ⚠️ **Never commit your `.env` file!** It contains your API key. The `.gitignore` already excludes it.

---

## 📌 Status

- [x] Project structure setup
- [x] Gemini 2.5 Flash API connected
- [ ] Ingestion pipeline (`ingest.py`)
- [ ] Vector store setup (FAISS)
- [ ] RAG pipeline (`rag_pipeline.py`)
- [ ] Query interface (`query.py`)

---
