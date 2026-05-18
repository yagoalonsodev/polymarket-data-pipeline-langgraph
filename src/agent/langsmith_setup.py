"""Variables de entorno para trazas en LangSmith (smith.langchain.com)."""
from __future__ import annotations

import os


def configure_langsmith() -> bool:
    """
    Activa trazado hacia LangSmith si hay API key y LANGSMITH_TRACING=true.
    Idempotente: se llama al arrancar Streamlit.
    """
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)

    tracing_on = (os.getenv("LANGSMITH_TRACING") or "").strip().lower() in ("1", "true", "yes", "on")
    if tracing_on and api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_TRACING", "true")

    project = (os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "").strip()
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
        os.environ.setdefault("LANGSMITH_PROJECT", project)

    endpoint = (os.getenv("LANGSMITH_ENDPOINT") or "").strip()
    if endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)

    return bool(api_key and tracing_on)
