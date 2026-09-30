import time
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# providers.py picks which AI service we use for embeddings
import providers

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "documents"

# The file types we know how to read.
SUPPORTED_TYPES = [".txt", ".pdf", ".png", ".jpeg", ".jpg", ".pptx", ".docx"]

# Settings for reading diagram slides (see load_pptx)
THIN_SLIDE_CHARS = 30    # real text shorter than this means the slide is really a picture
THIN_PAGE_CHARS  = 80    # a PDF page with less text than this is probably a scan
TEMPLATE_REPEATS = 3     # a picture used on more slides than this is a logo, not content

VISION_CACHE = Path(__file__).resolve().parent.parent / ".vision_cache.json"


# Vision Cache
# Reading a picture costs an API call, so remember what each one said.
# Keyed by image hash, so the same picture is never sent twice - even if you
# re-run ingest tomorrow. Delete .vision_cache.json to force a fresh read.
def load_vision_cache():
    if VISION_CACHE.exists():
        with open(VISION_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_vision_cache(cache):
    with open(VISION_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)


# Vision Call
# Kept in memory so we only read the cache file from disk once per run.
_cache = None


def vision_cache():
    """The remembered results, loaded from disk the first time we need them."""
    global _cache
    if _cache is None:
        _cache = load_vision_cache()
    return _cache


# How many pictures we are still allowed to read on this run.
# None means "no limit", which is what you want on your own machine.
# The public demo sets a number, so one visitor cannot spend the whole day's
# vision budget on a single 200-slide deck.
_vision_budget = None
_vision_used = 0


def set_vision_budget(limit):
    """Allow only `limit` pictures to be read from here on. Pass None for no
    limit. Call this before ingest_documents()."""
    global _vision_budget, _vision_used
    _vision_budget = limit
    _vision_used = 0


def describe_image(image_bytes, mime="image/png"):
    """Read the text out of a picture and return it.

    The actual reading happens in providers.py, which tries Groq first and
    falls back to Gemini if Groq is out of free budget.

    We remember every result by the picture's fingerprint (an md5 hash of its
    bytes). Identical pictures are therefore read once and only once - even if
    you run this again tomorrow. Failures are deliberately not remembered, so a
    temporary rate limit cannot blank a picture forever."""
    key = hashlib.md5(image_bytes).hexdigest()
    cache = vision_cache()

    if cache.get(key):
        return cache[key]

    # Reading a picture is the most expensive thing this project does, so on the
    # public demo we allow only a fixed number per visitor. A cached picture
    # never counts, because it costs nothing. Running out is not an error: we
    # return empty text and the slide is simply indexed without its picture.
    global _vision_used
    if _vision_budget is not None:
        if _vision_used >= _vision_budget:
            return ""
        _vision_used += 1

    text = providers.read_image(image_bytes, mime)
    if text:
        cache[key] = text
        save_vision_cache(cache)   # save now, so a crash does not lose it
    return text


# Image Loader
def load_image(file_path):
    """Extract text from image using Gemini Vision"""
    try:
        with open(file_path, "rb") as f:  #rb -> read binary | 'with' statement automatically calls f.close(), to avoid memory leak
            image_bytes = f.read()

        suffix = Path(file_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        # nt -> mimetypes.guess_type(file_path)[0]

        # from_image is True because every word of this came out of a vision
        # model looking at a picture. The source list uses it to label the
        # chunk "read from a picture", which warns the reader that it could
        # have been misread. A JPG has no text to read any other way, so it is
        # always True here.
        return [Document(
            page_content=describe_image(image_bytes, mime),
            metadata={"source": Path(file_path).name, "type": "image",
                      "from_image": True}
        )]
    except Exception as e:
        print(f"   Could not process image {Path(file_path).name}: {e}")
        return []

# Picture Helpers (used by the PPTX loader)
def picture_bytes(shape):
    """Return a picture shape's raw bytes, or None if this shape is not a picture."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
        return None
    return shape.image.blob


def find_boilerplate_lines(prs):
    """Find the header and footer lines that repeat across the whole deck.

    Every template stamps the same text on every slide - a department name, a
    unit title, a page marker. If we count those as real content, a slide that
    is nothing but a picture still looks full of words.

    Instead of hard-coding one college's header, we just look for lines that
    appear on more than half the slides. That works for any template, and for a
    deck with no repeated header it simply finds nothing."""
    counts = {}
    slides = list(prs.slides)

    for slide in slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                # set() so a line repeated inside one slide still counts once
                for line in set(shape.text.split("\n")):
                    line = line.strip()
                    if line:
                        counts[line] = counts.get(line, 0) + 1

    appears_on_most = max(2, len(slides) // 2)
    return {line for line, times in counts.items() if times > appears_on_most}


def real_content(slide_text, boilerplate):
    """The slide's text with the repeated template lines taken out."""
    kept = []
    for block in slide_text:
        for line in block.split("\n"):
            line = line.strip()
            if line and line not in boilerplate:
                kept.append(line)
    return " ".join(kept)


def find_template_images(prs):
    """Find logos and background art so we can ignore them.
    They sit on nearly every slide, so we spot them by counting how often the
    exact same image appears. (In the DAA deck one logo repeats 59 times.)"""
    counts = {}
    for slide in prs.slides:
        for shape in slide.shapes:
            data = picture_bytes(shape)
            if data:
                key = hashlib.md5(data).hexdigest()
                counts[key] = counts.get(key, 0) + 1
    return {key for key, times in counts.items() if times > TEMPLATE_REPEATS}


def read_slide_pictures(slide, slide_number, template_images):
    """Read this slide's real pictures and return the text found in them.
    Caching is handled inside describe_image, so this just skips logos."""
    parts = []
    for shape in slide.shapes:
        data = picture_bytes(shape)
        if not data:
            continue

        if hashlib.md5(data).hexdigest() in template_images:
            continue                    # a logo, nothing to learn from it
        if shape.image.ext.lower() not in ("png", "jpg", "jpeg"):
            continue                    # vision models only take normal formats

        try:
            print(f"      slide {slide_number}: reading diagram...")
            text = describe_image(data, f"image/{shape.image.ext.lower()}")
            if text:
                parts.append(text)
        except Exception as e:
            print(f"      slide {slide_number}: could not read it ({type(e).__name__})")

    return "\n".join(parts)


# PPTX Loader
def load_pptx(file_path, use_vision=True):
    """Extract text from PowerPoint slides.

    use_vision=False skips reading pictures. Useful for measuring how much the
    picture-reading actually adds, and for a fast text-only run."""
    try:
        from pptx import Presentation
        prs = Presentation(file_path)

        template_images = find_template_images(prs)   # logos to skip
        boilerplate = find_boilerplate_lines(prs)     # repeated header/footer text

        text_chunks = []
        for i, slide in enumerate(prs.slides): #track both the index and the item
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip(): #hasattr  -> Does this object possess an attribute called 'text'?
                    slide_text.append(shape.text.strip())

                # Tables are a separate kind of shape, not plain text
                if getattr(shape, "has_table", False) and shape.has_table:
                    as_text = table_to_text(shape.table)
                    if as_text:
                        slide_text.append(as_text)

            # Slides that are almost all picture (graphs, tables, worked examples)
            # give us nothing here, because python-pptx cannot read images.
            # So when the real text is thin, ask a vision model what it says.
            # We measure the text AFTER removing the repeated header, otherwise
            # a picture-only slide looks full just because of the template.
            used_vision = False
            if use_vision and len(real_content(slide_text, boilerplate)) < THIN_SLIDE_CHARS:
                seen_text = read_slide_pictures(slide, i + 1, template_images)
                if seen_text:
                    slide_text.append(seen_text)
                    used_vision = True

            if slide_text:
                content = f"Slide {i+1}:\n" + "\n".join(slide_text)
                text_chunks.append(Document(
                    page_content=content,
                    metadata={"source": Path(file_path).name, "slide": i+1,
                              "from_image": used_vision}
                ))
        return text_chunks
    except Exception as e:
        print(f"   Could not process PPTX {Path(file_path).name}: {e}")
        return []


# Table Helper
def table_to_text(table):
    """Turn a table into plain lines so its contents can be searched.

    A table means nothing to a search engine as a grid, so we flatten each row
    into "cell | cell | cell". Works for both Word and PowerPoint tables
    because both expose .rows and .cells with a .text on each cell."""
    lines = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


# DOCX Loader
def read_docx_normally(file_path):
    """The normal way: python-docx gives us paragraphs and tables.
    Returns None if the library refuses to open the file."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)

        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            text = table_to_text(table)
            if text:
                parts.append(text)
        return "\n".join(parts)

    except Exception as e:
        print(f"      python-docx could not open this file ({type(e).__name__}), "
              f"trying the raw XML instead")
        return None


def read_docx_from_xml(file_path):
    """Fallback for broken Word files.

    A .docx is really just a zip folder, and the words live in
    word/document.xml. Some files (often ones made by tools other than Word)
    have one broken internal link that makes python-docx give up on the whole
    document - but the text itself is perfectly fine.

    So we open the zip ourselves and strip the XML tags out. We lose tables and
    formatting this way, but we keep the words, which is what we search on."""
    import zipfile
    import re

    with zipfile.ZipFile(file_path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")

    xml = xml.replace("</w:p>", "\n")        # keep paragraph breaks
    text = re.sub(r"<[^>]+>", " ", xml)      # remove every remaining tag
    return re.sub(r"[ \t]+", " ", text)      # squash runs of spaces


def load_docx(file_path):
    """Extract text from Word documents, including tables."""
    try:
        full_text = read_docx_normally(file_path)
        if full_text is None:
            full_text = read_docx_from_xml(file_path)

        if not full_text.strip():
            print(f"   {Path(file_path).name} appears empty")
            return []

        return [Document(
            page_content=full_text,
            metadata={"source": Path(file_path).name, "type": "docx"}
        )]
    except Exception as e:
        print(f"   Could not process DOCX {Path(file_path).name}: {e}")
        return []


# ── PDF Loader ────────────────────────────────────────────
def read_pdf_page_as_image(page, page_number):
    """Draw one PDF page as a picture and let the vision model read it.

    This is how scanned documents are handled. In a scanned PDF each page is
    really just a photograph of paper, so there is no text to pull out -
    we have to look at it."""
    try:
        print(f"      page {page_number}: no text found, reading it as a picture...")
        picture = page.get_pixmap(dpi=150)      # 150 dpi is plenty to read text
        return describe_image(picture.tobytes("png"), "image/png")
    except Exception as e:
        print(f"      page {page_number}: could not read it ({type(e).__name__})")
        return ""


def load_pdf(file_path, use_vision=True):
    """Extract text from PDF using PyMuPDF""" #10x-50x faster than pure Python libraries like 'pypdf' or 'PyPDF2;
    try:
        import fitz
        pdf_doc = fitz.open(str(file_path)) #use 'with' to avoid Resource Leak on exception

        # One Document per page, not one per file. Two reasons: we keep the
        # page number (useful when we show the user where an answer came from),
        # and we can treat a scanned page differently from a text page.
        pages = []
        for page_number, page in enumerate(pdf_doc, start=1):
            text = page.get_text().strip()
            from_image = False

            # Almost no text usually means the page is a scan or a picture.
            if use_vision and len(text) < THIN_PAGE_CHARS:
                seen = read_pdf_page_as_image(page, page_number)
                if seen:
                    text = seen
                    from_image = True

            if text:
                pages.append(Document(
                    page_content=text,
                    metadata={"source": Path(file_path).name, "type": "pdf",
                              "page": page_number, "from_image": from_image}
                ))

        pdf_doc.close()
        if not pages:
            print(f"   {Path(file_path).name} produced no text at all")
        return pages

    except Exception as e:
        print(f"   Could not process PDF {Path(file_path).name}: {e}")
        return []


# Load All Documents
def find_documents(documents_path=None):
    """List the files in a folder that we know how to read."""
    folder = Path(documents_path) if documents_path else DOCUMENTS_PATH
    return [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_TYPES]


def load_all_documents(documents_path=None, use_vision=True, only_files=None):
    """Read documents into memory.

    only_files lets the caller hand over an exact list instead of reading the
    whole folder. That is what makes adding one new document cheap: we load
    just that file rather than everything sitting beside it.
    """
    if documents_path is None:
        documents_path = DOCUMENTS_PATH
    else:
        documents_path = Path(documents_path)

    all_documents = []
    files_found = list(only_files) if only_files is not None else \
        [f for f in documents_path.iterdir() if f.suffix.lower() in SUPPORTED_TYPES]

    if not files_found:
        raise FileNotFoundError(
            "   No documents found!\n"
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
                docs = load_pdf(str(file), use_vision)
            elif ext == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
                docs = loader.load()
            elif ext in [".png", ".jpg", ".jpeg"]:
                docs = load_image(str(file)) if use_vision else []
            elif ext == ".pptx":
                docs = load_pptx(str(file), use_vision)
            elif ext == ".docx":
                docs = load_docx(str(file))
            else:
                continue

            for doc in docs:
                doc.metadata["source"] = file.name

            all_documents.extend(docs)
            print(f"      {file.name}: {len(docs)} section(s) loaded")

        except Exception as e:
            print(f"      Skipping {file.name}: {e}")

    return all_documents


# Capacity Check 
def check_capacity(chunks):
    total = len(chunks)
    print(f"\nCapacity Check:")
    print(f"   Total chunks: {total}")

    if total <= 80:
        print("   Safe - will process instantly")
        return False
    elif total <= 500:
        print("   Large - adding delays between batches")
        return True
    else:
        # Raise rather than exit(1). exit() throws SystemExit, which does NOT
        # inherit from Exception, so every "except Exception" in the app walks
        # straight past it - including the one wrapped around processing in the
        # web interface. A visitor uploading a very large deck got a page that
        # simply stopped, with no message at all. A ValueError is caught there
        # and shown properly.
        raise ValueError(
            f"That comes to {total} chunks, and the limit is 500. "
            f"Try a smaller document, or raise chunk_size in ingest.py "
            f"(chunk_size=2000 would give roughly {total // 2})."
        )

# Main Ingest Function 
# Remembering what is already in the index
#
# Building the index means one embedding call for every chunk. Doing that again
# for documents that have not changed is pure waste - it costs time and eats
# the free daily budget for no new information.
#
# So alongside the index we keep a small note of which files went into it. Next
# time, anything already listed is left alone and only genuinely new files are
# read and added.
#
# Files are identified by a fingerprint of their contents rather than by name
# or by modification date. Copying a file changes its date but not what is
# inside it, and two different drafts can share a name. The contents are the
# only honest answer to "is this the same document?".
MANIFEST_NAME = "indexed_files.json"


def file_fingerprint(path):
    """A short, stable id for a file's contents."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_manifest(vectorstore_path):
    """Which files are already in this index. Empty when there is no index."""
    note = Path(vectorstore_path) / MANIFEST_NAME
    try:
        return json.loads(note.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_manifest(vectorstore_path, manifest):
    note = Path(vectorstore_path) / MANIFEST_NAME
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def plan_ingest(files, manifest, vectorstore_path):
    """Work out what actually needs doing. Returns (what_to_do, files, why).

    what_to_do is one of:
      "nothing"    - every file is already indexed, exactly as it is now
      "add"        - some new files to append to the existing index
      "rebuild"    - start again from scratch

    A rebuild is needed when a file that was indexed has been edited or
    removed. We could in principle delete just that file's chunks, but the
    index stores them by an internal id we would have to track separately, and
    getting it wrong leaves stale text answering questions - which is worse
    than a slow rebuild. So an edit means starting over, and that is a
    deliberate choice rather than a missing feature.
    """
    index_exists = (Path(vectorstore_path) / "index.faiss").exists()
    if not index_exists or not manifest:
        return "rebuild", files, "no usable index yet"

    now = {f.name: file_fingerprint(f) for f in files}

    gone = [name for name in manifest if name not in now]
    if gone:
        return "rebuild", files, f"removed since last time: {', '.join(gone)}"

    edited = [name for name, mark in now.items()
              if name in manifest and manifest[name] != mark]
    if edited:
        return "rebuild", files, f"edited since last time: {', '.join(edited)}"

    fresh = [f for f in files if f.name not in manifest]
    if not fresh:
        return "nothing", [], "everything is already indexed"

    return "add", fresh, f"new: {', '.join(f.name for f in fresh)}"


def ingest_documents(documents_path=None, vectorstore_path="vectorstore", use_vision=True,
                     incremental=True):
    print("\nStarting Ingestion Pipeline...")

    # 0. Decide how much work is actually needed.
    files = find_documents(documents_path)
    if not files:
        raise FileNotFoundError(
            "   No documents found!\n"
            "   Supported: .txt .pdf .pptx .docx .jpg .png\n"
            "   Place files in data/documents/ folder"
        )

    manifest = load_manifest(vectorstore_path) if incremental else {}
    action, to_load, why = plan_ingest(files, manifest, vectorstore_path) \
        if incremental else ("rebuild", files, "incremental turned off")

    print(f"   {action.title()}: {why}")

    if action == "nothing":
        print("   Nothing to do - the index already matches your documents.")
        return

    # 1. Load only the documents we decided to read
    documents = load_all_documents(documents_path=documents_path,
                                   use_vision=use_vision, only_files=to_load)
    print(f"\nTotal Pages Loaded: {len(documents)}")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    ) # ["\n\n", "\n", " ", ""]
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Guard : stop if nothing to embed
    if not chunks:
        raise ValueError(
            "   No text extracted from documents!\n"
            "   Check if your files are text-based (not scanned images)."
        )

    # 4. Capacity check
    needs_delay = check_capacity(chunks)

    # 5. Create embeddings
    print("Creating embeddings...")
    embeddings = providers.build_embeddings()

    # 6. Store in FAISS with optional batching
    # The free tier allows 100 embedding requests per minute, and one chunk is
    # one request. So we must actually embed a batch, wait, then embed the next.
    # (Collecting batches in a list and embedding them all at the end does not
    #  help - that is still one giant burst and still hits the 429.)
    if needs_delay:
        batch_size = 80
        pause = providers.embedding_batch_pause()   # depends on the service we are using
        vector_db = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"   Processing batch {i//batch_size + 1} ({len(batch)} chunks)...")

            if vector_db is None:
                vector_db = FAISS.from_documents(batch, embeddings)  # first batch creates it
            else:
                vector_db.add_documents(batch)                        # later batches append

            if i + batch_size < len(chunks):
                print(f"   Waiting {pause}s to stay inside the rate limit...")
                time.sleep(pause)
    else:
        vector_db = FAISS.from_documents(chunks, embeddings)

    # 6b. Adding to what is already there, rather than replacing it.
    #
    # We built the new chunks above exactly as a rebuild would, then open the
    # existing index and append them. If opening it fails - a different
    # embedding model, a damaged file - we keep what we just built instead of
    # crashing, which turns a bad day into a plain rebuild.
    merge_failed = False
    if action == "add":
        try:
            existing = FAISS.load_local(vectorstore_path, embeddings,
                                        allow_dangerous_deserialization=True)
            existing.merge_from(vector_db)
            vector_db = existing
            print(f"   Added to the existing index ({len(chunks)} new chunks)")
        except Exception as e:
            print(f"   Could not open the existing index ({type(e).__name__}), "
                  f"writing a fresh one instead")
            merge_failed = True

    # 7. Save locally
    vector_db.save_local(vectorstore_path)

    # 7b. Write down what is now in the index, so the next run can skip it.
    #
    # The note must describe what is REALLY in the index, never what we meant
    # to put there. That matters in one case: the merge just above failing. We
    # then save only the new chunks, so writing down every file would be untrue
    # - the next run would read the note, decide there was nothing left to do,
    # and the older documents would be gone with nothing said about it. Writing
    # down only what we actually saved means the next run notices the rest are
    # missing and puts them back.
    if action == "add" and not merge_failed:
        manifest.update({f.name: file_fingerprint(f) for f in to_load})
    elif merge_failed:
        manifest = {f.name: file_fingerprint(f) for f in to_load}
    else:
        manifest = {f.name: file_fingerprint(f) for f in files}
    save_manifest(vectorstore_path, manifest)
    print("   Vector database created and saved!")
    print(f"   Location: {vectorstore_path}/  (built with {providers.embedding_name()})")


if __name__ == "__main__":
    ingest_documents()