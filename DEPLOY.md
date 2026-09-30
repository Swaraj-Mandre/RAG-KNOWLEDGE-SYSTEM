# Putting this online, for free

The app runs on **Streamlit Community Cloud**, which is free for public apps
built from a public GitHub repository. No card, no trial, no expiry.

The whole thing takes about ten minutes, and most of that is waiting for the
container to install packages.

---

## Before you start

You need three API keys. All three are free and none of them asks for a card.

| Service | Where | Used for |
|---|---|---|
| Groq | console.groq.com | main chat model, and reading pictures |
| Mistral | console.mistral.ai | backup chat, and all embeddings |
| Google AI Studio | aistudio.google.com | second backup, and vision fallback |

You do **not** need all three for the app to start - `providers.py` quietly uses
whichever keys are present. But with only one, a busy day empties its quota and
the demo stops answering.

---

## Step 1 - push everything to GitHub

The repository must be public for the free tier.

```bash
git push origin main
```

Check that `.env` is **not** there. It is listed in `.gitignore`, so it should
never have been committed, but it costs nothing to look:

```bash
git ls-files | Select-String -Pattern "\.env"
```

Silence is the correct answer.

---

## Step 2 - create the app

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Choose **Create app** -> **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `Swaraj-Mandre/RAG-KNOWLEDGE-SYSTEM`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`

Do not deploy yet - the keys go in first.

---

## Step 3 - paste the secrets

Open **Advanced settings** and put this in the Secrets box, with your own keys:

```toml
GROQ_API_KEY = "gsk_..."
MISTRAL_API_KEY = "..."
GOOGLE_API_KEY = "AIza..."
DEMO_MODE = "1"
```

`DEMO_MODE = "1"` is the important line. It is what switches on every
protection in `app/limits.py`:

- each visitor gets their own private index folder, swept after two hours
- at most 3 files, 10 MB each
- at most 12 pictures sent to the vision model per visitor
- 15 questions per visit, 250 per day across everyone
- questions capped at 500 characters

Without it the app runs in local mode: no caps, and it would try to load the
index sitting in the repository. Do not forget this line.

### Why secrets and not a .env file

On your own machine the keys are read from `.env`. Streamlit Cloud has no such
file - it hands the values over as `st.secrets` instead, which
`os.getenv` cannot see. `providers.load_hosted_secrets()` copies them into the
environment at import time so the rest of the code does not have to care which
of the two it is running on.

---

## Step 4 - deploy, then check these four things

Click **Deploy** and wait for the build.

1. **It starts.** If it crashes, open the logs from the menu in the bottom
   right. A missing package is the usual cause.
2. **The sidebar shows a "Demo limits" section.** If it does not, `DEMO_MODE`
   did not arrive and every cap is off. Fix that before sharing the link.
3. **It does not already contain documents.** A fresh visit must say "No
   documents loaded". If it shows your DAA slides, demo mode is off.
4. **Upload something and ask a question.** The first question is slow, because
   the container is waking up.

---

## Things worth knowing

**Reboot after changing anything outside `streamlit_app.py`.** This one costs
real time if you do not know it. Pushing new code does not restart the Python
process. `streamlit_app.py` is run again from the top on every interaction, so
your changes to it appear straight away, but `import limits` and `import ingest`
hand back the copies already loaded in memory from the last boot. So you get the
new file calling the old module, which shows up as a puzzling error like
`module 'limits' has no attribute 'touch_session'` pointing at a line that is
plainly correct in front of you.

Open **Manage app**, click the three dots at the bottom right of the log panel,
and choose **Reboot app**. Everything is re-imported and the error disappears.
Rule of thumb: touched only `streamlit_app.py`, just push. Touched anything else
in `app/`, push and then reboot.

**The app sleeps.** A free app with no visitors goes to sleep and takes about
thirty seconds to wake. That is normal. If you are sending the link to someone
who matters, open it yourself a minute beforehand.

**Storage is temporary.** Uploaded documents and their indexes live in the
container's temp folder and vanish when it restarts. For a demo that is exactly
what you want - nobody's document is kept.

**The daily counter resets on restart.** It is a file in that same temp folder,
so a restart forgets the count. It is a safety net, not an accountant.

**Quota is shared by every visitor.** The free tiers are per account, not per
person using the demo. That is what the caps in `limits.py` are protecting. If
the demo runs dry too often, lower `MAX_QUESTIONS_PER_SESSION` before raising
`MAX_QUESTIONS_PER_DAY`.

---

## Running it yourself instead

```bash
git clone https://github.com/Swaraj-Mandre/RAG-KNOWLEDGE-SYSTEM.git
```
```bash
pip install -r requirements.txt
```

Put your keys in a `.env` file at the project root:

```
GROQ_API_KEY=...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
```

Then index your documents and start the app:

```bash
python app/ingest.py
```
```bash
streamlit run app/streamlit_app.py
```

With no `DEMO_MODE` there are no caps, and the app reuses the index it built
last time instead of asking you to upload again.
