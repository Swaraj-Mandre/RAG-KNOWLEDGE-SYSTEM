# RAG Knowledge System

> **An AI-powered system that answers questions using your own documents — powered by Gemini 2.5 Flash & LangChain**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.5Flash-orange)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## What is this?

A **Retrieval-Augmented Generation (RAG)** system that lets you load your own documents and ask questions about them. Instead of relying on general AI knowledge, it retrieves exact information from *your* files and generates precise answers using Gemini 2.5 Flash.

```
Your Document → Chunks → Embeddings → FAISS → Query → Retrieved Chunks → Gemini → Answer
```

---

## Two Pipelines Inside Every RAG System

### A. Ingestion Pipeline *(Offline — runs once)*

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

### B. Query Pipeline *(Runtime — runs on every question)*

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
Final Answer 
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Gemini 2.5 Flash |
| Embeddings | gemini-embedding-001 |
| Framework | LangChain Classic |
| Vector Store | FAISS (local) |
| Language | Python 3.11.9 |

> I used FAISS for this project as "Just pip install" "Works Offline" and most importantly "Free Forever".

## Free Tier Limits (Gemini API)
 
| Resource | Limit |
|----------|-------|
| Questions/day | 500 |
| Questions/min | 10 |
| Embeddings/day | 1,000 |
| Embeddings/min | 100 |

---

## Project Structure

```
rag-knowledge-system/
│
├── app/
│   ├── main.py           # Gemini API connection test
│   ├── ingest.py         # Load → Chunk → Embed → Save to FAISS
│   ├── rag_pipeline.py   # Core RAG logic (embeddings, retriever, LLM)
│   └── query.py          # Interactive Q&A terminal interface
│
├── assets/
│   └── Demo.png
│
├── data/
│   └── documents/        # Drop your files here (.txt .pdf .pptx .docx .jpg .png)
│
├── vectorstore/          # Auto-created FAISS index (git ignored)
│   ├── index.faiss
│   └── index.pkl
│
├── .env                  # Your API key (never commit!)
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Clone & setup
```bash
git clone https://github.com/Swaraj-Mandre/RAG-KNOWLEDGE-SYSTEM.git
cd RAG-KNOWLEDGE-SYSTEM
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add API key
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get your free API key at: https://aistudio.google.com/apikey

## Security Note

> ⚠️ **Never commit your `.env` file!** It contains your API key. The `.gitignore` already excludes it.

### 3. Add your documents
Drop files into `data/documents/` — supports `.txt`, `.pdf`, `.pptx`, `.docx`, `.jpg`, `.png`

### 4. Ingestion Pipeline
Load -> Split/Chunk -> Embed -> Store
```
vectorstore/
├── index.faiss    ← actual vector data (embeddings)
└── index.pkl      ← metadata & original chunk text
```

### 5. Ask questions
```bash
python app/query.py
```

> Man I'm too lazy to work on UI , enjoy the colourful characters for now
