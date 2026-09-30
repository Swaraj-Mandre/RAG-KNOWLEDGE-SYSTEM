import os
import sys
import shutil
import tempfile
import html
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from markdown_it import MarkdownIt

sys.path.append(str(Path(__file__).resolve().parent))
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

st.set_page_config(
    page_title="RAG Knowledge System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: #0D1117; color: #E6EDF3; }
#MainMenu, footer, header { visibility: hidden; }
/* ...but not the arrow that re-opens the sidebar. It lives inside that same
   header, so hiding the header hid it too: collapsing the sidebar left no way
   to bring it back, and reloading did not help because the browser remembers
   the collapsed state. Visibility is inherited, so it has to be turned back on
   explicitly on the button itself. */
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] * { visibility: visible !important; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #21262D !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 24px 20px !important; }

.sidebar-logo { font-size: 15px; font-weight: 600; color: #E6EDF3; letter-spacing: -0.2px; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
.sidebar-logo-dot { width: 8px; height: 8px; background: #7C3AED; border-radius: 50%; display: inline-block; }
.sidebar-version { font-size: 11px; color: #484F58; margin-bottom: 20px; }
.sidebar-section { font-size: 10px; font-weight: 600; color: #484F58; text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 8px 0; }
.sidebar-hint { font-size: 11px; color: #484F58; margin-bottom: 10px; line-height: 1.5; }
.sidebar-divider { height: 1px; background: #21262D; margin: 16px 0; border: none; }
.file-chip { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: #0D1117; border: 1px solid #21262D; border-left: 2px solid #7C3AED; border-radius: 4px; margin-bottom: 6px; font-size: 12px; color: #C9D1D9; word-break: break-all; }
.file-chip-dot { width: 6px; height: 6px; background: #238636; border-radius: 50%; flex-shrink: 0; }
.sidebar-footer { font-size: 10px; color: #30363D; margin-top: 8px; line-height: 1.6; }

[data-testid="stFileUploader"] { background: #0D1117; border: 1px dashed #30363D; border-radius: 8px; padding: 4px; }
[data-testid="stFileUploader"]:hover { border-color: #7C3AED; }

.stButton > button { font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500; height: 36px; border-radius: 6px; width: 100%; transition: all 0.15s ease; }
div[data-testid="stButton"]:nth-of-type(1) > button { background-color: #7C3AED; color: #FFFFFF; border: 1px solid #7C3AED; }
div[data-testid="stButton"]:nth-of-type(1) > button:hover:not(:disabled) { background-color: #6D28D9; box-shadow: 0 4px 12px rgba(124,58,237,0.3); }
div[data-testid="stButton"]:nth-of-type(1) > button:disabled { background-color: #21262D; color: #484F58; border-color: #21262D; cursor: not-allowed; }
div[data-testid="stButton"]:nth-of-type(2) > button { background-color: transparent; color: #7D8590; border: 1px solid #30363D; }
div[data-testid="stButton"]:nth-of-type(2) > button:hover:not(:disabled) { color: #E6EDF3; border-color: #7D8590; background-color: #21262D; }
div[data-testid="stButton"]:nth-of-type(2) > button:disabled { color: #30363D; border-color: #21262D; cursor: not-allowed; }

.main-title { font-size: 20px; font-weight: 600; color: #E6EDF3; letter-spacing: -0.3px; margin: 20px 0 4px 0; }
.main-subtitle { font-size: 13px; color: #7D8590; margin: 0 0 20px 0; }
.main-divider { height: 1px; background: #21262D; border: none; margin: 0 0 24px 0; }

.welcome-container { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 45vh; text-align: center; padding: 40px 20px; }
.welcome-icon { width: 48px; height: 48px; background: #161B22; border: 1px solid #21262D; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; font-size: 20px; color: #7C3AED; }
.welcome-title { font-size: 16px; font-weight: 500; color: #E6EDF3; margin: 0 0 8px 0; }
.welcome-subtitle { font-size: 13px; color: #7D8590; margin: 0; max-width: 320px; line-height: 1.6; }
.welcome-steps { display: flex; gap: 16px; margin-top: 32px; flex-wrap: wrap; justify-content: center; }
.welcome-step { background: #161B22; border: 1px solid #21262D; border-radius: 8px; padding: 16px 20px; text-align: left; width: 180px; }
.step-number { font-size: 10px; font-weight: 600; color: #7C3AED; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.step-text { font-size: 12px; color: #7D8590; line-height: 1.5; }

[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stChatMessageAvatar"] { display: none !important; }
.msg-user { display: flex; justify-content: flex-end; margin: 8px 0; }
.msg-user-bubble { background: #21262D; border: 1px solid #30363D; border-radius: 12px 12px 2px 12px; padding: 10px 16px; font-size: 14px; color: #E6EDF3; max-width: 70%; line-height: 1.6; word-wrap: break-word; }
.msg-ai { display: flex; justify-content: flex-start; margin: 8px 0; align-items: flex-start; }
.msg-ai-avatar { width: 28px; height: 28px; background: #7C3AED; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; color: white; flex-shrink: 0; margin-right: 10px; margin-top: 2px; }
.msg-ai-bubble { background: #161B22; border: 1px solid #21262D; border-radius: 2px 12px 12px 12px; padding: 12px 16px; font-size: 14px; color: #C9D1D9; line-height: 1.7; word-wrap: break-word; max-width: 75%; }

/* keeps converted markdown (headings, bullets, code) tight inside the bubble */
.msg-ai-bubble p:first-child { margin-top: 0; }
.msg-ai-bubble p:last-child { margin-bottom: 0; }
.msg-ai-bubble ul, .msg-ai-bubble ol { margin: 8px 0; padding-left: 22px; }
.msg-ai-bubble li { margin: 3px 0; }
.msg-ai-bubble h1, .msg-ai-bubble h2, .msg-ai-bubble h3 { font-size: 14px; font-weight: 600; color: #E6EDF3; margin: 12px 0 6px 0; }
.msg-ai-bubble code { background: #0D1117; border: 1px solid #21262D; border-radius: 4px; padding: 1px 5px; font-size: 12px; }
.msg-ai-bubble pre { background: #0D1117; border: 1px solid #21262D; border-radius: 6px; padding: 10px; overflow-x: auto; }
.msg-ai-bubble pre code { border: none; padding: 0; }
.msg-ai-bubble table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
.msg-ai-bubble th, .msg-ai-bubble td { border: 1px solid #30363D; padding: 5px 9px; text-align: left; }

/* the "where this came from" list under an answer */
.sources { margin-top: 12px; padding-top: 10px; border-top: 1px solid #21262D; }
.sources-title { font-size: 10px; font-weight: 600; color: #484F58; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.source-item { font-size: 12px; color: #7D8590; margin: 3px 0; line-height: 1.5; }
.source-num { color: #7C3AED; font-weight: 600; margin-right: 6px; }
.source-tag { font-size: 10px; color: #484F58; margin-left: 6px; }

@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse-dot { 0%,100%{opacity:.3;transform:scale(.8)} 50%{opacity:1;transform:scale(1)} }
.thinking-wrapper { display: flex; align-items: center; gap: 10px; padding: 12px 0; }
.spinner-ring { width: 16px; height: 16px; border: 2px solid #21262D; border-top: 2px solid #7C3AED; border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0; }
.thinking-dots { display: flex; gap: 4px; align-items: center; }
.thinking-dots span { width: 5px; height: 5px; background: #7C3AED; border-radius: 50%; animation: pulse-dot 1.2s ease-in-out infinite; }
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
.thinking-label { font-size: 12px; color: #7D8590; }

.status-processing { display: inline-flex; align-items: center; gap: 6px; background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.3); border-radius: 20px; padding: 4px 12px; font-size: 12px; color: #A78BFA; margin-top: 8px; }
.status-dot { width: 6px; height: 6px; background: #7C3AED; border-radius: 50%; animation: pulse-dot 1s ease-in-out infinite; }
.status-ready { display: inline-flex; align-items: center; gap: 6px; background: rgba(35,134,54,0.1); border: 1px solid rgba(35,134,54,0.3); border-radius: 20px; padding: 4px 12px; font-size: 12px; color: #3FB950; margin-top: 8px; }

[data-testid="stChatInput"] { background: #161B22 !important; border: 1px solid #30363D !important; border-radius: 10px !important; }
[data-testid="stChatInput"]:focus-within { border-color: #7C3AED !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: #E6EDF3 !important; font-size: 15px !important; font-family: 'Inter', sans-serif !important; padding: 14px 16px !important; min-height: 52px !important; border: none !important; outline: none !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #484F58 !important; font-size: 14px !important; }
[data-testid="stChatInput"] button { background: #7C3AED !important; border-radius: 6px !important; color: white !important; border: none !important; margin: 8px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

import limits

# Imported early, and deliberately so. Importing providers is what copies the
# keys out of Streamlit Cloud's secrets box into the environment, and the
# sidebar below reports which keys were found. If this import happened later -
# inside the Process button, where it used to - the sidebar would say no keys
# exist on a perfectly working deployment. The module itself is cheap: it pulls
# in langchain only inside the functions that need it.
import providers  # noqa: F401  (imported for its startup side effect)

DEMO = limits.demo_mode()

# Where this browser tab's searchable index lives.
#
# On your own machine there is one index, kept next to the project, and it is
# reused every time you open the app.
#
# On the public demo that would be wrong and unsafe: everyone would share one
# index, so a question typed by one visitor would search a document uploaded by
# a different visitor. Their CV, their report, their invoice. So in demo mode
# every browser tab gets its own private folder with a random name, and stale
# folders are swept away after a couple of hours.
if DEMO:
    if "session_id" not in st.session_state:
        st.session_state.session_id = limits.new_session_id()
        limits.cleanup_old_sessions()     # tidy up whoever left before us
    VECTORSTORE = limits.session_dir(st.session_state.session_id)
else:
    VECTORSTORE = Path(__file__).resolve().parent.parent / "vectorstore"


def load_existing_index():
    """If documents were already indexed on a previous run, reuse that index.

    Building the index is the slow and expensive part - it calls the embedding
    service for every chunk, and the vision model for every picture. Throwing
    that away each time the page reloads would mean re-uploading and paying for
    it again before you could ask a single question. So we just open it.

    Returns None if there is nothing saved yet, or if what is saved was built
    by a different embedding model (initialize_pipeline refuses that case)."""
    # Never on the public demo. There, an existing index belongs either to the
    # owner of the project or to another visitor, and neither is ours to open.
    if DEMO:
        return None
    if not (VECTORSTORE / "index.faiss").exists():
        return None
    try:
        from rag_pipeline import initialize_pipeline
        return initialize_pipeline(str(VECTORSTORE))
    except Exception:
        return None   # a stale or mismatched index - the user can rebuild it


if st.session_state.chain is None and "tried_existing" not in st.session_state:
    st.session_state.tried_existing = True   # only attempt this once per session
    st.session_state.chain = load_existing_index()
    if st.session_state.chain:
        st.session_state.uploaded_files = ["(previously indexed documents)"]


# Message Rendering
# html=False -> any raw HTML the model writes gets escaped, not executed.
# enable("table") -> markdown tables still render.
md = MarkdownIt("commonmark", {"html": False}).enable("table")


def render_user(text):
    """Draw a user bubble. html.escape() so typing < or <b> shows as plain text."""
    safe = html.escape(text)
    st.markdown(
        f'<div class="msg-user"><div class="msg-user-bubble">{safe}</div></div>',
        unsafe_allow_html=True
    )


def render_sources(sources):
    """Build the small "where this came from" list that sits under an answer.

    The numbers here match the [1] and [2] the model wrote inside the answer,
    so a reader can check any claim against the real slide or page instead of
    taking the answer on trust. That is the whole point of citations."""
    if not sources:
        return ""   # never show an empty "Sources" heading

    rows = []
    for s in sources:
        # Tell the reader when the text was read out of a picture by the vision
        # model, because that can misread and is worth checking.
        tag = ('<span class="source-tag">read from a picture</span>'
               if s.get("from_image") else "")
        rows.append(
            f'<div class="source-item">'
            f'<span class="source-num">[{s["number"]}]</span>'
            f'{html.escape(s["label"])}{tag}</div>'
        )

    return ('<div class="sources">'
            '<div class="sources-title">Sources</div>'
            + "".join(rows) + '</div>')


def render_ai(text, sources=None):
    """Draw an assistant bubble, with its sources underneath.

    The answer arrives as markdown (bullets, headings), but an HTML <div> ignores
    markdown - so convert to HTML first, else the whole answer shows as one line.
    `sources` is optional because "I don't have enough information" has none."""
    body = md.render(text)
    if sources:
        body += render_sources(sources)
    st.markdown(
        f'<div class="msg-ai"><div class="msg-ai-avatar">R</div><div class="msg-ai-bubble">{body}</div></div>',
        unsafe_allow_html=True
    )


with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo"><span class="sidebar-logo-dot"></span>RAG Knowledge System</div>
    <div class="sidebar-version">Groq · Mistral · Gemini · FAISS</div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Show which API keys the app can actually see. Without this, a missing key
    # only shows up as a failure in the middle of processing a document, long
    # after you have waited - and it looks like the document was the problem.
    import os as _os
    found = [name for name in ("GROQ_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY")
             if _os.getenv(name)]
    if not found:
        st.error(
            "No API keys found.\n\n"
            "Hosted on Streamlit Cloud: open Manage app - Settings - Secrets and "
            'add them as KEY = "value", then reboot.\n\n'
            "Running locally: put them in a .env file in the project root."
        )
    elif len(found) < 3:
        st.caption(f"{len(found)} of 3 providers available - "
                   f"fewer backups when one runs out of free quota.")

    st.markdown('<div class="sidebar-section">Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-hint">PDF · TXT · PPTX · DOCX · JPG · PNG</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "pptx", "docx", "jpg", "png", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("Process documents", disabled=not bool(uploaded), key="process_btn"):
        # On the public demo, check the upload before spending anything on it.
        # Refusing a 200 MB file costs nothing; embedding one costs the whole
        # day's budget.
        allowed, why_not = (True, "") if not DEMO else limits.check_upload(uploaded)
        if not allowed:
            st.warning(why_not)
            st.stop()

        status_placeholder = st.empty()
        status_placeholder.markdown('<div class="status-processing"><span class="status-dot"></span>Ingesting documents...</div>', unsafe_allow_html=True)
        inbox = None          # set inside the try, cleaned up in the finally
        try:
            # Cap how many pictures the vision model may read for this visitor.
            # None means no cap, which is what you get on your own machine.
            from ingest import set_vision_budget
            set_vision_budget(limits.MAX_VISION_IMAGES if DEMO else None)

            # Where the uploaded files are written before they are read.
            #
            # On the public demo they go to a scratch folder inside the demo's
            # own temp area, and are deleted the moment indexing is finished.
            # They were previously written to a folder nobody ever cleaned up,
            # while the sidebar told visitors their files were removed
            # automatically. The sidebar was wrong, so the code is now right.
            #
            # On your own machine they join the rest of your material in
            # data/documents, because that is what processing a document should
            # mean locally: the file is yours, and it is still there tomorrow.
            if DEMO:
                inbox = Path(tempfile.mkdtemp(dir=limits.base_dir()))
            else:
                inbox = Path(__file__).resolve().parent.parent / "data" / "documents"
                inbox.mkdir(parents=True, exist_ok=True)

            saved_files = []
            for file in uploaded:
                with open(inbox / file.name, "wb") as f:
                    f.write(file.getbuffer())
                saved_files.append(file.name)

            from ingest import ingest_documents, load_manifest

            if DEMO:
                # Build the new index NEXT TO the old one, then swap.
                #
                # The old code deleted the index first and then started
                # indexing. If indexing then failed halfway - a quota error is
                # the usual reason - the working index was already gone and
                # there was no way back. Building somewhere else first means a
                # failure costs nothing. Named after this visitor's own folder,
                # not a fixed name: two people uploading at the same moment
                # must not share a staging directory, or each would overwrite
                # the other's half-built index.
                staging = VECTORSTORE.parent / (VECTORSTORE.name + "_building")
                if staging.exists():
                    shutil.rmtree(staging)

                ingest_documents(documents_path=str(inbox),
                                 vectorstore_path=str(staging))

                # Indexing worked, so now it is safe to replace the old index.
                if VECTORSTORE.exists():
                    shutil.rmtree(VECTORSTORE)
                staging.rename(VECTORSTORE)
            else:
                # Add to the index that is already there, rather than building
                # a new one from the uploaded files alone.
                #
                # Building a separate index and swapping it in is right for the
                # demo, where each visitor should only ever see their own
                # upload. Locally it was quietly destructive: processing one
                # new file threw away everything else you had already indexed,
                # because the fresh folder had no record of it. Pointing the
                # ingest at data/documents instead means it sees the whole
                # collection, skips what it has already read, and adds only
                # what is new. Exactly what running ingest.py does.
                ingest_documents(documents_path=str(inbox),
                                 vectorstore_path=str(VECTORSTORE))

            from rag_pipeline import initialize_pipeline
            st.session_state.chain = initialize_pipeline(str(VECTORSTORE))

            # Locally the index may now hold more than was just uploaded, so
            # list what is actually in it rather than only the new arrivals.
            indexed = sorted(load_manifest(str(VECTORSTORE))) if not DEMO else []
            st.session_state.uploaded_files = indexed or saved_files
            st.session_state.messages = []
            status_placeholder.markdown(f'<div class="status-ready">{len(saved_files)} file(s) ready</div>', unsafe_allow_html=True)
        except Exception as e:
            status_placeholder.empty()
            st.error(f"Error: {str(e)}")
        finally:
            # The raw uploads have been read into the index by now, so the copy
            # is no longer needed. Deleting it here rather than leaving it for
            # the two-hourly sweep means a visitor's file is gone within
            # seconds, which is what the sidebar promises them.
            if DEMO and inbox is not None:
                shutil.rmtree(inbox, ignore_errors=True)

    if st.session_state.uploaded_files:
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Loaded</div>', unsafe_allow_html=True)
        for f in st.session_state.uploaded_files:
            # escape() because a file name is text somebody else chose. Naming
            # a file with a tag in it should show that tag, not run it.
            st.markdown(f'<div class="file-chip"><span class="file-chip-dot"></span>'
                        f'{html.escape(f)}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    if st.button("Clear chat", disabled=len(st.session_state.messages) == 0, key="clear_btn"):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if DEMO:
        # Say the limits out loud. A visitor who knows there are 15 questions
        # spends them well; one who finds out by being cut off just leaves.
        #
        # Reserve the spot now, fill it in later.
        #
        # Streamlit runs this file from top to bottom on every interaction. The
        # sidebar is near the top, but the question is handled at the bottom -
        # so anything drawn here shows the counts from BEFORE the question was
        # asked, and nothing redraws it afterwards. Keeping an empty slot lets
        # us write the real numbers once the question has been counted.
        demo_slot = st.empty()
    else:
        st.markdown('<div class="sidebar-footer">Free tier · answers cite their source · local vector storage</div>', unsafe_allow_html=True)

def show_demo_limits():
    """Write the current demo counts into the slot kept in the sidebar.

    Called once now, so the numbers appear immediately, and again after a
    question has been answered so the counts are not one question behind."""
    asked_now = sum(1 for m in st.session_state.messages if m["role"] == "user")
    left = max(0, limits.MAX_QUESTIONS_PER_SESSION - asked_now)
    used_today, per_day = limits.questions_today()
    demo_slot.markdown(
        f'<div class="sidebar-section">Demo limits</div>'
        f'<div class="sidebar-footer">'
        f'{left} of {limits.MAX_QUESTIONS_PER_SESSION} questions left this visit<br>'
        f'up to {limits.MAX_FILES} files, {limits.MAX_FILE_MB} MB each<br>'
        f'{used_today} of {per_day} questions used today<br><br>'
        f'Your documents stay in your own session and are deleted '
        f'automatically. Run it yourself with your own keys - see the README.'
        f'</div>',
        unsafe_allow_html=True)


if DEMO:
    show_demo_limits()

st.markdown('<p class="main-title">RAG Knowledge System</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Upload your documents and ask anything about them.</p>', unsafe_allow_html=True)
st.markdown('<hr class="main-divider">', unsafe_allow_html=True)

if not st.session_state.chain and not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">◈</div>
        <p class="welcome-title">No documents loaded</p>
        <p class="welcome-subtitle">Upload a document from the sidebar and click Process to get started.</p>
        <div class="welcome-steps">
            <div class="welcome-step"><div class="step-number">Step 1</div><div class="step-text">Upload a PDF, PPTX, DOCX, TXT, or image file</div></div>
            <div class="welcome-step"><div class="step-number">Step 2</div><div class="step-text">Click Process documents to build the knowledge base</div></div>
            <div class="welcome-step"><div class="step-number">Step 3</div><div class="step-text">Ask any question about your document</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        render_user(message["content"])
    else:
        # .get() not [...] - older messages in a running session may predate sources
        render_ai(message["content"], message.get("sources"))

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.chain:
        st.warning("Upload and process your documents first.")
        st.stop()

    # On the public demo, check the caps before anything is spent. asked is the
    # number of questions this visitor has already put to the app.
    if DEMO:
        asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
        allowed, why_not = limits.check_question(prompt, asked)
        if not allowed:
            st.warning(why_not)
            st.stop()
        limits.record_question()   # counts towards today's shared total
        limits.touch_session(st.session_state.session_id)   # still in use

    render_user(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    thinking = st.empty()
    thinking.markdown("""
    <div class="thinking-wrapper">
        <div class="spinner-ring"></div>
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        <span class="thinking-label">Searching documents...</span>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Turn the stored chat into (question, answer) pairs so a follow-up
        # like "summarize that" knows what it refers to. We only send the last
        # few turns - conversation.py trims them.
        from conversation import MAX_TURNS
        past = st.session_state.messages[:-1]          # drop the question just asked
        history = [
            (past[i]["content"], past[i + 1]["content"])
            for i in range(0, len(past) - 1, 2)
            if past[i]["role"] == "user" and past[i + 1]["role"] == "assistant"
        ][-MAX_TURNS:]

        # ask() hands back two things: the answer, and the chunks it used.
        from rag_pipeline import ask
        answer, sources = ask(st.session_state.chain, prompt, history=history)
        thinking.empty()
        render_ai(answer, sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

        # Redraw the sidebar counts now that this question has been counted,
        # so they are not showing the state from before it was asked.
        if DEMO:
            show_demo_limits()
    except Exception as e:
        thinking.empty()
        st.error(f"Error: {str(e)}")