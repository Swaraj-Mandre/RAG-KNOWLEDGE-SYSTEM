"""
providers.py - which AI service we call, and what happens when one runs out.

We use three services and all of them are free: Mistral, Groq and Google Gemini.
Each one has a daily or per-minute limit. When a service is out of budget it
replies with a "429" error instead of an answer.

Rather than crashing, we keep a list of services in order and move down the list.

Two rules worth understanding, because they are not the same:

  CHAT can fall back freely.
      Any chat model can answer any question, so if Mistral is busy, Groq
      answering instead is perfectly fine.

  EMBEDDINGS must NOT fall back.
      An embedding model turns text into a list of numbers, and each model uses
      a different length (Mistral = 1024 numbers, Gemini = 3072). Those numbers
      are only comparable to each other if they came from the SAME model. If
      half our document was embedded by Mistral and half by Gemini, search would
      return nonsense. So we pick one embedding model at the start and keep it.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


def load_hosted_secrets():
    """Copy Streamlit Cloud's secrets into the environment.

    On your own machine the API keys come from the .env file, read above. When
    the app is hosted on Streamlit Community Cloud there is no .env - you paste
    the keys into a settings box instead, and Streamlit hands them over as
    st.secrets rather than as environment variables. os.getenv would simply
    return None and every provider would look unavailable.

    So we copy them across once, at import time. Everything below this line can
    then keep using os.getenv and does not need to know where it is running.

    Wrapped in try/except because streamlit is not installed when this module is
    used from the command line, and there is nothing to copy in that case.
    """
    try:
        import streamlit as st
        for name in ("GROQ_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY", "DEMO_MODE"):
            if name not in os.environ and name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except Exception:
        pass    # not running under Streamlit, or no secrets configured


load_hosted_secrets()


# Chat models, in the order we try them.
#
# The order is deliberate: strongest model first, most generous model last.
# We spend the good-but-limited model while we have it, then fall back to the
# one we can always reach. Every ID below was tested and works on the free tier.
CHAT_MODELS = [
    # 120B model - the best answers, but only ~110 real questions a day
    {"service": "groq",    "model": "openai/gpt-oss-120b",      "key": "GROQ_API_KEY",    "note": "1K req/day, 200K tokens/day"},
    # 14B model - smaller, but a huge per-minute budget, so it rarely runs out
    {"service": "mistral", "model": "ministral-14b-latest",     "key": "MISTRAL_API_KEY", "note": "937K tokens/min"},
    # Gemini gets its own 20/day PER MODEL, so three models = 60 more questions
    {"service": "google",  "model": "gemini-2.5-flash",         "key": "GOOGLE_API_KEY",  "note": "20 req/day"},
    {"service": "google",  "model": "gemini-3.8-flash",         "key": "GOOGLE_API_KEY",  "note": "20 req/day"},
    {"service": "google",  "model": "gemini-flash-lite-latest", "key": "GOOGLE_API_KEY",  "note": "20 req/day"},
]

# Embedding models, best first. Only ONE of these is ever used (see note above).
#   size        - how many numbers the model produces for each chunk of text
#   batch_pause - seconds to wait between batches while indexing, so we stay
#                 inside the service's per-minute limit. Mistral allows 20M
#                 tokens a minute so it barely needs a pause; Google's free
#                 tier allows only 100 embeddings a minute, so it needs a full
#                 minute between batches.
EMBEDDING_MODELS = [
    {"service": "mistral", "model": "mistral-embed",              "key": "MISTRAL_API_KEY", "size": 1024, "batch_pause": 2},
    {"service": "google",  "model": "models/gemini-embedding-001", "key": "GOOGLE_API_KEY", "size": 3072, "batch_pause": 60},
]


# Helpers
def has_key(spec):
    """True if the API key this model needs is present in .env."""
    return bool(os.getenv(spec["key"]))


def usable(model_list):
    """Keep only the models whose API key we actually have."""
    return [spec for spec in model_list if has_key(spec)]


# Builders
def make_chat_model(spec, temperature):
    """Create one chat model object from a row of CHAT_MODELS."""
    if spec["service"] == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=spec["model"], temperature=temperature)

    if spec["service"] == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=spec["model"], temperature=temperature)

    if spec["service"] == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=spec["model"], temperature=temperature)

    raise ValueError(f"Unknown service: {spec['service']}")


def make_embeddings(spec):
    """Create one embedding model object from a row of EMBEDDING_MODELS."""
    if spec["service"] == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(model=spec["model"])

    if spec["service"] == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=spec["model"])

    raise ValueError(f"Unknown service: {spec['service']}")


# What the rest of the app calls
def build_chat_model(temperature=0.2):
    """Build the chat model, with every other model in the list as a backup.

    .with_fallbacks() is LangChain's built-in safety net. If the first model
    raises any error, LangChain quietly tries the next one instead of failing.
    We let it catch every error, not just 429, so a bad key or a network blip
    also rolls over to the next service.
    """
    choices = usable(CHAT_MODELS)
    if not choices:
        raise ValueError("No chat API key found in .env "
                         "(need MISTRAL_API_KEY, GROQ_API_KEY or GOOGLE_API_KEY)")

    models = [make_chat_model(spec, temperature) for spec in choices]
    first = choices[0]
    print(f"   Chat: {first['service']}/{first['model']}  "
          f"({len(models) - 1} backup(s) ready)")

    if len(models) == 1:
        return models[0]
    return models[0].with_fallbacks(models[1:])


def build_embeddings():
    """Pick ONE embedding model and stay with it for the whole vector store.

    There is deliberately no fallback here - mixing embedding models would make
    the saved vectors unusable. See the explanation at the top of this file.
    """
    choices = usable(EMBEDDING_MODELS)
    if not choices:
        raise ValueError("No embedding API key found in .env "
                         "(need MISTRAL_API_KEY or GOOGLE_API_KEY)")

    spec = choices[0]
    print(f"   Embeddings: {spec['service']}/{spec['model']}  "
          f"({spec['size']} numbers per chunk)")
    return make_embeddings(spec)


def embedding_name():
    """Which embedding model we are using, as a short readable string."""
    choices = usable(EMBEDDING_MODELS)
    return f"{choices[0]['service']}/{choices[0]['model']}" if choices else "unknown"


def embedding_size():
    """How many numbers our embedding model produces per chunk.
    Used to check a saved vector store still matches the model we have now."""
    choices = usable(EMBEDDING_MODELS)
    return choices[0]["size"] if choices else 0


def embedding_batch_pause():
    """Seconds to wait between batches while indexing, for this service."""
    choices = usable(EMBEDDING_MODELS)
    return choices[0]["batch_pause"] if choices else 60


# Vision (reading text out of a picture)
#
# Same idea as chat: try the best one, fall back when it runs out.
# Falling back is safe here - any vision model can read any picture.
VISION_MODELS = [
    {"service": "groq",   "model": "qwen/qwen3.8-27b",  "key": "GROQ_API_KEY",   "note": "200K tokens/day"},
    {"service": "google", "model": "gemini-2.5-flash",  "key": "GOOGLE_API_KEY", "note": "20 req/day"},
    {"service": "google", "model": "gemini-3.8-flash",  "key": "GOOGLE_API_KEY", "note": "20 req/day"},
]

READ_IMAGE_PROMPT = (
    "Extract ALL text and describe all content visible in this image in detail."
)


def read_image(image_bytes, mime="image/png", prompt=READ_IMAGE_PROMPT):
    """Ask a vision model what it can see in a picture. Returns the text it read.

    We hand the picture over as a "data URL", which is just the raw bytes
    written out as text so they can travel inside a normal JSON request.

    All our chat models understand the same message format, so one piece of
    code works for every service. We try each model in turn and return the
    first answer we get. If every model fails, we raise the last error.
    """
    import base64
    from langchain_core.messages import HumanMessage

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ])

    choices = usable(VISION_MODELS)
    if not choices:
        raise ValueError("No vision API key found in .env")

    last_error = None
    for spec in choices:
        try:
            model = make_chat_model(spec, temperature=0.0)
            return model.invoke([message]).content
        except Exception as e:
            last_error = e
            print(f"      ({spec['service']} could not read it: {type(e).__name__}, trying next)")

    raise last_error
