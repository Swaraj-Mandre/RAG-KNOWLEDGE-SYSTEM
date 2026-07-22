import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "documents"


# ── Image Loader ──────────────────────────────────────────
def load_image(file_path):
    """Extract text from image using Gemini Vision — FREE same quota"""
    try:
        import base64
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")  # ← fixed uppercase
        client = genai.Client(api_key=api_key)

        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = Path(file_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"inline_data": {"mime_type": mime, "data": image_data}},
                "Extract ALL text and describe all content visible in this image in detail."
            ]
        )
        return [Document(
            page_content=response.text,
            metadata={"source": Path(file_path).name, "type": "image"}
        )]
    except Exception as e:
        print(f"   Could not process image {Path(file_path).name}: {e}")
        return []


# ── PPTX Loader ───────────────────────────────────────────
def load_pptx(file_path):
    """Extract text from PowerPoint slides — FREE no API"""
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        text_chunks = []
        for i, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            if slide_text:
                content = f"Slide {i+1}:\n" + "\n".join(slide_text)
                text_chunks.append(Document(
                    page_content=content,
                    metadata={"source": Path(file_path).name, "slide": i+1}
                ))
        return text_chunks
    except Exception as e:
        print(f"   Could not process PPTX {Path(file_path).name}: {e}")
        return []


# ── DOCX Loader ───────────────────────────────────────────
def load_docx(file_path):
    """Extract text from Word documents — FREE no API"""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        full_text = "\n".join([
            para.text for para in doc.paragraphs if para.text.strip()
        ])
        if not full_text.strip():
            print(f"   ⚠️ {Path(file_path).name} appears empty")
            return []
        return [Document(
            page_content=full_text,
            metadata={"source": Path(file_path).name, "type": "docx"}
        )]
    except Exception as e:
        print(f"   Could not process DOCX {Path(file_path).name}: {e}")
        return []


# ── PDF Loader ────────────────────────────────────────────
def load_pdf(file_path):
    """Extract text from PDF using PyMuPDF — FREE no API"""
    try:
        import fitz
        pdf_doc = fitz.open(str(file_path))
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        pdf_doc.close()

        if text.strip():
            return [Document(
                page_content=text,
                metadata={"source": Path(file_path).name, "type": "pdf"}
            )]
        else:
            print(f"   ⚠️ {Path(file_path).name} is a scanned PDF — no text found")
            return []
    except Exception as e:
        print(f"   Could not process PDF {Path(file_path).name}: {e}")
        return []


# ── Load All Documents ────────────────────────────────────
def load_all_documents(documents_path=None):
    if documents_path is None:
        documents_path = DOCUMENTS_PATH
    else:
        documents_path = Path(documents_path)

    all_documents = []
    supported = [".txt", ".pdf", ".png", ".jpeg", ".jpg", ".pptx", ".docx"]
    files_found = [f for f in documents_path.iterdir() if f.suffix.lower() in supported]

    if not files_found:
        raise FileNotFoundError(
            "❌ No documents found!\n"
            "   Supported: .txt .pdf .pptx .docx .jpg .png\n"
            "   Place files in data/documents/ folder"
        )

    print(f"Found {len(files_found)} file(s):")

    for file in files_found:
        try:
            print(f"   Loading: {file.name}")
            ext = file.suffix.lower()
            docs = []

            if ext == ".pdf":
                docs = load_pdf(str(file))
            elif ext == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
                docs = loader.load()
            elif ext in [".png", ".jpg", ".jpeg"]:
                docs = load_image(str(file))
            elif ext == ".pptx":
                docs = load_pptx(str(file))
            elif ext == ".docx":
                docs = load_docx(str(file))
            else:
                continue

            for doc in docs:
                doc.metadata["source"] = file.name

            all_documents.extend(docs)
            print(f"      {file.name} — {len(docs)} section(s) loaded")

        except Exception as e:
            print(f"      Skipping {file.name}: {e}")

    return all_documents


# ── Capacity Check ────────────────────────────────────────
def check_capacity(chunks):
    total = len(chunks)
    print(f"\nCapacity Check:")
    print(f"   Total chunks: {total}")

    if total <= 80:
        print("   Safe — will process instantly")
        return False
    elif total <= 500:
        print("   Large — adding delays between batches")
        return True
    else:
        print("   ❌ Too large! Increase chunk_size to reduce chunks")
        print(f"   Try chunk_size=2000 to get ~{total//2} chunks")
        exit(1)


# ── Main Ingest Function ──────────────────────────────────
def ingest_documents(documents_path=None):
    print("\nStarting Ingestion Pipeline...")

    # 1. Load all documents
    documents = load_all_documents(documents_path=documents_path)
    print(f"\nTotal Pages Loaded: {len(documents)}")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Guard — stop if nothing to embed
    if not chunks:
        raise ValueError(
            "❌ No text extracted from documents!\n"
            "   Check if your files are text-based (not scanned images)."
        )

    # 4. Capacity check
    needs_delay = check_capacity(chunks)

    # 5. Create embeddings
    print("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # 6. Store in FAISS with optional batching
    if needs_delay:
        batch_size = 80
        all_docs = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"   Processing batch {i//batch_size + 1} ({len(batch)} chunks)...")
            all_docs.extend(batch)
            if i + batch_size < len(chunks):
                print("   Waiting 60s for rate limit reset...")
                time.sleep(60)
        vector_db = FAISS.from_documents(all_docs, embeddings)
    else:
        vector_db = FAISS.from_documents(chunks, embeddings)

    # 7. Save locally
    vector_db.save_local("vectorstore")
    print("✅ Vector database created and saved!")
    print("   Location: vectorstore/")


if __name__ == "__main__":
    ingest_documents()