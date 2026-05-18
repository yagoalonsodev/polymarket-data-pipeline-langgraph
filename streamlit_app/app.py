import json
import os
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.agent.langgraph_agent import AgentConfig, build_graph
from src.agent.langsmith_setup import configure_langsmith

# Las 3 preguntas del enunciado (Parte 2 / demo), siempre disponibles.
DEFAULT_SAVED_QUERIES: list[str] = [
    "¿Qué mercados han tenido mayor volumen esta semana?",
    "¿Qué mercado cambió más de probabilidad en las últimas 24 horas?",
    "¿Qué mercados son los más activos actualmente?",
]

_SAVE_FILE = Path(__file__).resolve().parent / "saved_queries_user.json"

# Clave del cuadro de texto inferior (sidebar la rellena aquí; tú pulsas Enviar).
_DRAFT_KEY = "draft_query"
# No tocar draft_query tras instanciar el widget; vaciar en el siguiente run.
_CLEAR_DRAFT_FLAG = "_clear_draft_next"


def _load_user_saved() -> list[str]:
    if not _SAVE_FILE.exists():
        return []
    try:
        data = json.loads(_SAVE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def _save_user_saved(queries: list[str]) -> None:
    _SAVE_FILE.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")


def _merged_saved_queries(user_extra: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in DEFAULT_SAVED_QUERIES + user_extra:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


_root_env = Path(__file__).resolve().parent.parent / ".env"
if _root_env.exists():
    load_dotenv(_root_env, override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

st.set_page_config(page_title="Polymarket CSGO Chatbot", layout="centered")
st.title("Chatbot Polymarket (CSGO) · LangGraph + Neon")

_langsmith_on = configure_langsmith()

if "history" not in st.session_state:
    st.session_state.history = []
if "user_saved_queries" not in st.session_state:
    st.session_state.user_saved_queries = _load_user_saved()
if _DRAFT_KEY not in st.session_state:
    st.session_state[_DRAFT_KEY] = ""
if st.session_state.pop(_CLEAR_DRAFT_FLAG, False):
    st.session_state[_DRAFT_KEY] = ""

neon_url = _env("NEON_DATABASE_URL")
llm_provider = (_env("LLM_PROVIDER", "ollama") or "ollama").strip().lower()
openai_key = _env("OPENAI_API_KEY")
openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
ollama_base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
ollama_model = _env("OLLAMA_MODEL", "deepseek-coder:6.7b") or "deepseek-coder:6.7b"

if not neon_url:
    st.error("Falta `NEON_DATABASE_URL` en el entorno.")
    st.stop()

if llm_provider == "openai" and not openai_key:
    st.error("LLM_PROVIDER=openai pero falta `OPENAI_API_KEY` en el entorno.")
    st.stop()

agent = build_graph(
    AgentConfig(
        neon_database_url=neon_url,
        llm_provider=llm_provider,
        openai_api_key=openai_key,
        openai_model=openai_model,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        enable_news_tool=True,
    )
)

with st.sidebar:
    st.markdown("### Consultas guardadas")
    st.caption("Pulsa una: el texto baja al cuadro de abajo; luego **Enviar**.")
    if st.button("Borrar conversación", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    all_q = _merged_saved_queries(st.session_state.user_saved_queries)
    for i, q in enumerate(all_q):
        label = q if len(q) <= 56 else q[:53] + "…"
        if st.button(label, key=f"load_saved_{i}", use_container_width=True):
            st.session_state[_DRAFT_KEY] = q
            st.rerun()

show_sql = st.toggle("Mostrar SQL / datos", value=True)
if _langsmith_on:
    st.caption(
        "Trazas en LangSmith: [smith.langchain.com](https://smith.langchain.com) "
        f"→ proyecto `{os.environ.get('LANGSMITH_PROJECT', 'proyecto3-polymarket-csgo')}`"
    )
st.caption("Bonus: si preguntas por noticias, usa HLTV (CSGO/CS2).")

# Historial tipo chat
for item in st.session_state.history:
    if not isinstance(item, dict):
        continue
    q = item.get("question") or ""
    a = item.get("answer") or ""
    if q:
        with st.chat_message("user"):
            st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        if item.get("error"):
            st.error(item.get("error"))
        news = item.get("news") or []
        if news:
            with st.expander("Noticias HLTV (raw)", expanded=False):
                st.json(news)
        if show_sql and not item.get("news_only") and not item.get("chitchat_only"):
            sql = item.get("sql") or ""
            if sql:
                st.code(sql, language="sql")
            rows = item.get("rows", [])
            if rows:
                st.dataframe(rows, use_container_width=True)

st.divider()
st.markdown("**Tu pregunta** (edita si quieres y pulsa Enviar)")
col_a, col_b = st.columns([4, 1])
with col_a:
    st.text_input(
        "Pregunta",
        key=_DRAFT_KEY,
        placeholder="Escribe tu pregunta o elige una en la barra lateral…",
        label_visibility="collapsed",
    )
with col_b:
    send = st.button("Enviar", type="primary", use_container_width=True)

if send:
    q_text = (st.session_state.get(_DRAFT_KEY) or "").strip()
    if q_text:
        state = {"question": q_text}
        with st.spinner("Pensando… (Ollama puede tardar un poco)"):
            try:
                out = agent.invoke(
                    state,
                    config={
                        "run_name": "streamlit_query",
                        "tags": ["streamlit", "polymarket-csgo"],
                    },
                )
            except Exception as e:  # noqa: BLE001
                out = {"question": q_text, "error": str(e), "answer": ""}
        if out is None:
            out = {"question": q_text, "error": "El agente devolvió None.", "answer": ""}
        st.session_state.history.append(out)
        st.session_state[_CLEAR_DRAFT_FLAG] = True
        st.rerun()
    else:
        st.warning("Escribe una pregunta o elige una guardada.")

if (st.session_state.get(_DRAFT_KEY) or "").strip():
    if st.button("Guardar la consulta del cuadro de texto", use_container_width=True):
        q_text = st.session_state[_DRAFT_KEY].strip()
        merged = _merged_saved_queries(st.session_state.user_saved_queries)
        if q_text not in merged:
            st.session_state.user_saved_queries.append(q_text)
            _save_user_saved(st.session_state.user_saved_queries)
            st.success("Consulta guardada (aparece en la barra lateral).")
            st.rerun()
        else:
            st.info("Esa consulta ya está en la lista.")
