"""
limits.py - the rules that make a public demo safe to run.

Running this project on your own laptop has no limits: it is your machine, your
API keys and your documents. Putting the same app on a public link is a very
different situation, and three things change.

1. Other people's documents are not yours to keep or to show.
   A visitor may upload a CV, an invoice, a college report. If everybody shared
   one search index, the next visitor's question would search the previous
   visitor's private file. So each visitor gets their own index, in their own
   folder, and the folder is deleted once it goes stale.

2. The free tier is a shared bucket.
   Every upload spends embedding calls, and every picture spends a vision call.
   A handful of visitors uploading big slide decks can empty the daily budget
   in minutes and the demo is dead for everyone else. So uploads are capped.

3. Someone will try to break it.
   Not always out of malice - a curious person will happily drop a 200 MB file
   in to see what happens. Small, boring limits prevent most of that.

Everything below is a number in one place, on purpose. When the demo runs out
of budget too quickly, or is too stingy, this is the only file to edit.
"""

import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path


# Demo mode is off unless the environment says otherwise, so running the app
# locally behaves exactly as it always has: no caps, your own documents loaded.
# On the public deployment we set DEMO_MODE=1.
def demo_mode():
    setting = os.getenv("DEMO_MODE", "")

    # On Streamlit Community Cloud there is no .env file and no environment
    # variable: settings are pasted into a box and arrive as st.secrets. This
    # check has to work before providers.py has been imported, so it looks for
    # itself rather than relying on that module to have copied things across.
    if not setting:
        try:
            import streamlit as st
            setting = str(st.secrets.get("DEMO_MODE", ""))
        except Exception:
            setting = ""      # not hosted, or no secrets set - plainly not a demo

    return setting.strip().lower() in ("1", "true", "yes", "on")


# ---- What a visitor may upload -----------------------------------------
MAX_FILES = 3               # per upload
MAX_FILE_MB = 10            # each; Streamlit's own default of 200 is far too big
MAX_VISION_IMAGES = 12      # pictures sent to the vision model, per visitor

# ---- What a visitor may ask --------------------------------------------
MAX_QUESTIONS_PER_SESSION = 15
MAX_QUESTION_CHARS = 500

# ---- What everybody together may use in one day ------------------------
# This protects the free tier itself. Without it, one busy afternoon empties
# the daily quota and the demo is broken for the rest of the day.
MAX_QUESTIONS_PER_DAY = 250

# ---- Housekeeping ------------------------------------------------------
# How long an abandoned visitor folder is kept before being deleted.
SESSION_MAX_AGE_HOURS = 2


# Everything the demo writes lives under one folder in the system temp area,
# so it never touches the project folder and never ends up in git.
def base_dir():
    path = Path(tempfile.gettempdir()) / "rag_demo"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- One folder per visitor --------------------------------------------
def new_session_id():
    """A random id for one visitor. uuid4 is random, not guessable, so one
    visitor cannot reach another visitor's folder by guessing a number."""
    return uuid.uuid4().hex[:16]


def session_dir(session_id):
    """Where this visitor's search index lives."""
    return base_dir() / f"session_{session_id}"


def touch_session(session_id):
    """Say that this visitor is still here.

    The sweep below decides what is stale from when a folder was last changed.
    Asking a question never changes the folder, it only reads from it - so
    somebody who uploads a document and then spends a long afternoon reading
    answers would have their own index swept away underneath them. Marking the
    folder on every question keeps it alive for as long as it is being used.
    """
    try:
        os.utime(session_dir(session_id), None)
    except OSError:
        pass        # folder already gone, which the caller will notice anyway


def cleanup_old_sessions(max_age_hours=SESSION_MAX_AGE_HOURS):
    """Delete visitor folders that have not been touched for a while.

    A web app never reliably learns that someone closed the tab, so there is no
    moment where we can say "this visitor has left, clean up". Instead we sweep
    on the way in: whenever somebody new arrives, anything stale goes.

    Returns how many folders were removed, which is handy in the logs.
    """
    cutoff = time.time() - max_age_hours * 3600
    removed = 0

    for folder in base_dir().glob("session_*"):
        try:
            if folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        except OSError:
            pass        # already gone, or in use - not worth crashing over
    return removed


# ---- Checking an upload before we spend anything on it ------------------
def check_upload(files):
    """Decide whether this set of uploaded files is allowed.

    Returns (ok, message). The message is written for the visitor to read, so
    it says what the limit is rather than just refusing.
    """
    if not files:
        return False, "Choose at least one file first."

    if len(files) > MAX_FILES:
        return False, (f"This demo accepts up to {MAX_FILES} files at a time. "
                       f"You selected {len(files)}.")

    for f in files:
        size_mb = getattr(f, "size", 0) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            return False, (f"'{f.name}' is {size_mb:.1f} MB. The limit in this "
                           f"demo is {MAX_FILE_MB} MB per file.")

    return True, ""


def check_question(question, asked_so_far):
    """Decide whether this visitor may ask another question.

    Checked in order from cheapest to most annoying: an over-long question is
    the visitor's own doing and easy to fix, running out of turns is not.
    """
    if len(question) > MAX_QUESTION_CHARS:
        return False, (f"That question is {len(question)} characters. Please keep "
                       f"it under {MAX_QUESTION_CHARS} in this demo.")

    if asked_so_far >= MAX_QUESTIONS_PER_SESSION:
        return False, (f"This demo allows {MAX_QUESTIONS_PER_SESSION} questions per "
                       f"visit, to keep it free for everyone. Reload the page to "
                       f"start again with a new document.")

    used, allowed = questions_today()
    if used >= allowed:
        return False, ("The demo has used up today's free quota. Please try again "
                       "tomorrow, or run the project yourself with your own API "
                       "keys - the setup is in the README.")

    return True, ""


# ---- The shared daily counter ------------------------------------------
# One small file holding today's date and how many questions have been asked.
# It is deliberately simple. If the server restarts the count resets, which
# means the cap is a safety net rather than an exact accountant - and that is
# an honest trade for something this size.
def _counter_file():
    return base_dir() / "questions_today.json"


def questions_today():
    """Returns (used_today, allowed_per_day)."""
    try:
        saved = json.loads(_counter_file().read_text(encoding="utf-8"))
        if saved.get("date") == date.today().isoformat():
            return int(saved.get("count", 0)), MAX_QUESTIONS_PER_DAY
    except (OSError, ValueError, TypeError):
        pass            # missing or damaged file - treat as a fresh day
    return 0, MAX_QUESTIONS_PER_DAY


def record_question():
    """Add one to today's count. Never raises: a broken counter must not stop
    somebody from using the demo."""
    used, _ = questions_today()
    try:
        _counter_file().write_text(
            json.dumps({"date": date.today().isoformat(), "count": used + 1}),
            encoding="utf-8",
        )
    except OSError:
        pass
