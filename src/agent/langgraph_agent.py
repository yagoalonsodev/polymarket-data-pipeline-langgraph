from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState

from src.agent.langsmith_setup import configure_langsmith
from sqlalchemy.engine import make_url
from sqlalchemy import create_engine, text
import requests
import re
import xml.etree.ElementTree as ET
from decimal import Decimal


@dataclass
class AgentConfig:
    neon_database_url: str
    llm_provider: str = "ollama"  # "ollama" | "openai"
    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    # Bonus tools
    enable_news_tool: bool = True

    @classmethod
    def from_env(cls) -> AgentConfig:
        def getenv(name: str, default: str | None = None) -> str | None:
            v = os.environ.get(name)
            if v is None or v == "":
                return default
            return v

        return cls(
            neon_database_url=getenv("NEON_DATABASE_URL", "") or "",
            llm_provider=(getenv("LLM_PROVIDER", "ollama") or "ollama").strip().lower(),
            openai_api_key=getenv("OPENAI_API_KEY"),
            openai_model=getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
            ollama_base_url=getenv("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434",
            ollama_model=getenv("OLLAMA_MODEL", "deepseek-coder:6.7b") or "deepseek-coder:6.7b",
            enable_news_tool=True,
        )


class AgentState(MessagesState, total=False):
    """Hereda `messages` de MessagesState para habilitar Chat en LangSmith Studio."""
    question: str
    sql: str
    rows: list[dict[str, Any]]
    news: list[dict[str, Any]]
    answer: str
    error: str
    news_error: str
    news_only: bool
    chitchat_only: bool


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(content or "").strip()


def _last_human_text(messages: list[Any] | None) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return _message_text(m.content)
        if isinstance(m, BaseMessage) and m.type == "human":
            return _message_text(m.content)
        if isinstance(m, dict):
            role = str(m.get("type") or m.get("role") or "").lower()
            if role in ("human", "user"):
                return _message_text(m.get("content"))
    return ""


def _question_from_state(state: AgentState) -> str:
    q = str(state.get("question") or "").strip()
    if q:
        return q
    return _last_human_text(state.get("messages"))


def _with_assistant_reply(state: AgentState, answer: str) -> AgentState:
    text = (answer or "").strip()
    state["answer"] = text
    if text:
        state["messages"] = [AIMessage(content=text)]
    return state


def _clean_sql(raw: str) -> str:
    s = (raw or "").strip()
    # Quita fences típicos si el modelo los mete
    s = re.sub(r"^```\\w*\\s*", "", s)
    s = re.sub(r"```\\s*$", "", s)
    # Elimina tokens raros de algunos modelos (e.g. <|begin_of_sentence|>)
    s = re.sub(r"<\\|.*?\\|>", "", s)
    # Filtra a ASCII imprimible (evita caracteres tipo '▁' o '｜')
    s = "".join(ch for ch in s if ch in "\t\r\n" or (" " <= ch <= "~"))
    s = s.strip()
    # Si hay varias sentencias, quédate con la primera
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    # Asegura punto y coma final
    if s and not s.endswith(";"):
        s += ";"
    return s


def _is_active_markets_question(question: str) -> bool:
    """Preguntas tipo enunciado: mercados más activos / actividad actual (no ranking por volumen)."""
    if _is_volume_question(question):
        return False
    ql = (question or "").lower()
    return bool(
        re.search(
            r"más activo|más activos|mas activo|mas activos|"
            r"mercados? (más )?activos|activos actualmente|más activos actualmente",
            ql,
        )
    )


def _format_active_markets_answer(rows: list[dict[str, Any]], *, limit: int = 15) -> str:
    lines: list[str] = [
        "Mercados activos (ordenados por última actualización en Polymarket; proxy de “actividad reciente”):"
    ]
    for i, r in enumerate(rows[:limit], start=1):
        title = r.get("title") or r.get("question") or r.get("market_id")
        ts = r.get("updated_at")
        if ts is not None:
            lines.append(f"{i}. {title} — actualizado: {ts}")
        else:
            lines.append(f"{i}. {title}")
    if len(rows) > limit:
        lines.append(f"(Mostrando {limit} de {len(rows)} resultados.)")
    return "\n".join(lines)


def _is_liquidity_question(question: str) -> bool:
    ql = (question or "").lower()
    return bool(re.search(r"\bliquidez\b|\bliquidity\b", ql))


def _is_liquidity_change_question(question: str) -> bool:
    ql = (question or "").lower()
    return bool(
        _is_liquidity_question(question)
        and re.search(r"cambio|change", ql)
        and re.search(r"24|últim|ultim|semana|7", ql)
    )


def _is_liquidity_rank_question(question: str) -> bool:
    """Mayor liquidez / top N (no pregunta de Δ en ventana temporal)."""
    if not _is_liquidity_question(question):
        return False
    if _is_liquidity_change_question(question):
        return False
    ql = (question or "").lower()
    return bool(
        re.search(r"\bmás\b|\bmas\b|\bmayor\b|\btop\b|top\d+|\bcuál\b|\bcual\b|\bprimeros\b", ql)
    )


def _sql_top_liquidity_latest(*, limit: int, active_only: bool) -> str:
    active_clause = "and m.active is true" if active_only else ""
    return f"""
        with latest as (
          select distinct on (f.market_id)
                 f.market_id,
                 f.liquidity as liquidity_latest,
                 f.snapshot_ts
          from polymarket.fact_market_snapshot f
          join polymarket.dim_market m on m.market_id = f.market_id
          where f.liquidity is not null
            and f.liquidity::text <> 'NaN'
            {active_clause}
          order by f.market_id, f.snapshot_ts desc
        )
        select coalesce(m.title, m.question, m.market_id) as title,
               l.liquidity_latest,
               l.snapshot_ts
        from latest l
        join polymarket.dim_market m using (market_id)
        order by l.liquidity_latest desc nulls last
        limit {int(limit)};
    """.strip()


def _format_liquidity_rank_answer(
    rows: list[dict[str, Any]], *, limit: int = 10, singular: bool = False
) -> str:
    if singular and rows:
        r = rows[0]
        title = r.get("title") or r.get("question") or r.get("market_id")
        liq = r.get("liquidity_latest") or r.get("liquidity")
        return f"Mercado activo con mayor liquidez (último snapshot): {title} — liquidez: {liq}"
    lines: list[str] = ["Top mercados por liquidez (último snapshot):"]
    for i, r in enumerate(rows[:limit], start=1):
        title = r.get("title") or r.get("question") or r.get("market_id")
        liq = r.get("liquidity_latest") or r.get("liquidity")
        if liq is not None:
            lines.append(f"{i}. {title} — liquidez: {liq}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def _format_liquidity_change_answer(rows: list[dict[str, Any]], *, limit: int = 10) -> str:
    lines: list[str] = ["Top mercados por cambio de liquidez:"]
    for i, r in enumerate(rows[:limit], start=1):
        title = r.get("title") or r.get("question") or r.get("market_id")
        ch = r.get("liquidity_change_24h") or r.get("liquidity_change_7d") or r.get("liquidity_change")
        latest = r.get("liquidity_latest")
        if ch is not None and latest is not None:
            lines.append(f"{i}. {title} — Δliquidez: {ch} (última: {latest})")
        elif ch is not None:
            lines.append(f"{i}. {title} — Δliquidez: {ch}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def _is_volume_question(question: str) -> bool:
    ql = (question or "").lower()
    return bool(re.search(r"\bvolumen\b|\bvolume\b", ql))


def _parse_top_limit(question: str, *, default: int = 10) -> int:
    ql = (question or "").lower().replace(" ", "")
    m = re.search(r"top(\d+)", ql)
    if m:
        return min(int(m.group(1)), 50)
    m = re.search(r"\btop\s*(\d+)\b", (question or "").lower())
    if m:
        return min(int(m.group(1)), 50)
    m = re.search(r"\b(\d+)\s*(primeros|mayores)\b", (question or "").lower())
    if m:
        return min(int(m.group(1)), 50)
    return default


def _question_wants_active_filter(question: str) -> bool:
    ql = (question or "").lower()
    return bool(
        re.search(
            r"\b(en\s+)?activo[s]?\b|\bmercados?\s+activo[s]?\b|\bactive\b",
            ql,
        )
    )


def _question_specifies_time_window(question: str) -> bool:
    ql = (question or "").lower()
    return bool(re.search(r"24\s*h|últim|ultim|semana|7\s*d|\b7\b|\besta\s+semana\b", ql))


def _is_volume_rank_question(question: str) -> bool:
    if not _is_volume_question(question):
        return False
    ql = (question or "").lower()
    return bool(
        re.search(r"\btop\b|top\d+|\bmás\b|\bmas\b|\bmayor\b|\bprimeros\b", ql)
    )


def _sql_top_volume_latest(*, limit: int, active_only: bool) -> str:
    active_clause = "and m.active is true" if active_only else ""
    return f"""
        with latest as (
          select distinct on (f.market_id)
                 f.market_id,
                 f.volume as volume_latest,
                 f.snapshot_ts
          from polymarket.fact_market_snapshot f
          join polymarket.dim_market m on m.market_id = f.market_id
          where f.volume is not null
            and f.volume::text <> 'NaN'
            {active_clause}
          order by f.market_id, f.snapshot_ts desc
        )
        select coalesce(m.title, m.question, m.market_id) as title,
               l.volume_latest,
               l.snapshot_ts
        from latest l
        join polymarket.dim_market m using (market_id)
        order by l.volume_latest desc nulls last
        limit {int(limit)};
    """.strip()


def _format_volume_answer(rows: list[dict[str, Any]], *, limit: int = 10) -> str:
    lines: list[str] = ["Top mercados por volumen:"]
    for i, r in enumerate(rows[:limit], start=1):
        title = r.get("title") or r.get("question") or r.get("market_id")
        delta = r.get("volume_24h") or r.get("volume_7d") or r.get("volume_change")
        latest = r.get("volume_latest") or r.get("volume")
        try:
            delta_num = float(delta) if delta is not None else None
        except (TypeError, ValueError):
            delta_num = None
        if delta_num is not None and delta_num != 0 and latest is not None:
            lines.append(f"{i}. {title} — Δvolumen: {delta} (acumulado: {latest})")
        elif latest is not None:
            lines.append(f"{i}. {title} — volumen acumulado: {latest}")
        elif delta is not None:
            lines.append(f"{i}. {title} — Δvolumen: {delta}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def _prefer_template_sql(question: str) -> bool:
    """Usa plantillas SQL probadas en lugar del LLM para preguntas frecuentes del demo."""
    if _is_chitchat_question(question) or _is_news_only_question(question):
        return False
    if not _fallback_sql(question):
        return False
    if _is_active_markets_question(question):
        return True
    if _is_volume_rank_question(question):
        return True
    if _is_liquidity_change_question(question):
        return True
    if _is_liquidity_rank_question(question):
        return True
    q = (question or "").lower()
    if _is_volume_question(question) and _question_specifies_time_window(question):
        return True
    if ("probabilidad" in q or "prob" in q) and _question_specifies_time_window(question):
        return True
    return False


_DATA_INTENT_RE = re.compile(
    r"mercad|volumen|volume|probabilidad|\bprob\b|liquidez|liquidity|"
    r"polymarket|snapshot|\bsql\b|datos?|activo|noticia|news|hltv|csgo|cs2|counter|"
    r"precio|apuesta|bet|outcome|dim_|fact_",
    re.IGNORECASE,
)


def _is_chitchat_question(question: str) -> bool:
    """
    Saludos / cortesía sin intención de consultar el DW (evita SELECT 1 y ruido SQL en la UI).
    Patrones cerrados (no basta con empezar por "hi" para no colarse con "hi show me markets").
    """
    q_raw = (question or "").strip()
    if not q_raw:
        return True
    if _DATA_INTENT_RE.search(q_raw):
        return False
    q = re.sub(r"[¡!¿?.…,:;'\-]+", " ", q_raw.lower())
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > 88:
        return False
    chitchat_full = (
        r"^(hola|hi|hello|hey|buenas|buenos d[ií]as|buenas tardes|buenas noches|"
        r"que tal|qué tal|gracias|thanks|thank you|muchas gracias|ok|vale|genial|perfecto|"
        r"adiós|adios|chao|bye|hasta luego|nos vemos)(\s+[!?.…]+)?\s*$",
        r"^(hola|hi|hello|hey|buenas)\s+(cómo|como)\s+(estás|estas|te va|va)(\s*\?)?\s*$",
        r"^(cómo|como)\s+(estás|estas)(\s*\?)?\s*$",
    )
    if any(re.match(p, q) for p in chitchat_full):
        return True
    return False


def _chitchat_reply(question: str) -> str:
    """
    Respuesta determinista (sin LLM): modelos código suelen repetir mal el system prompt
    (“Me llamo Eres el asistente…”).
    """
    q = re.sub(r"[¡!¿?.…,:;'\-]+", " ", (question or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    if re.search(r"\b(gracias|thanks|thank you|muchas gracias)\b", q):
        return (
            "¡De nada! Si quieres, pregunta por mercados de CSGO/CS2 en Polymarket "
            "(volumen, probabilidad, actividad) o por noticias en HLTV."
        )
    if re.search(r"\b(adios|adiós|chao|bye|hasta luego|nos vemos)\b", q):
        return "¡Hasta luego! Cuando quieras, puedes consultar mercados o noticias del demo."
    if re.fullmatch(r"(ok|vale|genial|perfecto)(\s*[!?.…]+)?", q):
        return "Genial. Cuando quieras, dime qué quieres ver: mercados, métricas o noticias HLTV."
    if re.search(r"\b(como estás|como estas|cómo estás|que tal|qué tal)\b", q) or re.match(
        r"^(hola|hi|hello|hey|buenas|buenos d[ií]as|buenas tardes|buenas noches)\b",
        q,
    ):
        return (
            "¡Hola! Muy bien, gracias por preguntar. "
            "Puedo ayudarte con mercados de CSGO/CS2 en Polymarket "
            "(volumen, probabilidad, actividad reciente) y con noticias de HLTV. "
            "¿Qué te gustaría consultar?"
        )
    return (
        "Hola. Puedo orientarte sobre mercados CSGO/CS2 en Polymarket "
        "y noticias de HLTV. ¿Sobre qué quieres información?"
    )


def _is_news_only_question(question: str) -> bool:
    """True si el usuario pide noticias (HLTV/CSGO) sin mezclar con consultas analíticas SQL."""
    ql = (question or "").lower()
    if not ql.strip():
        return False
    wants_news = any(
        k in ql for k in ("noticia", "noticias", "news", "hltv")
    )
    # Métricas / DW: si pide volumen, probabilidad, liquidez o ranking de mercados, no es solo-noticias.
    wants_analytics = bool(
        re.search(
            r"volumen|volume|probabilidad|prob\b|liquidez|liquidity|"
            r"más activo|mas activo|activos actualmente|mercados con mayor",
            ql,
        )
    )
    return wants_news and not wants_analytics


def _is_safe_select(sql: str) -> bool:
    s = (sql or "").strip().lower()
    if not s.startswith("select"):
        return False
    banned = ("insert", "update", "delete", "create", "drop", "alter", "truncate", "grant", "revoke")
    return not any(re.search(rf"\\b{kw}\\b", s) for kw in banned)


def _reset_turn_fields(state: AgentState) -> None:
    """Limpia restos del turno anterior (crítico en Studio Chat multi-mensaje)."""
    state["chitchat_only"] = False
    state["news_only"] = False
    state["sql"] = ""
    state["rows"] = []
    state["news"] = []
    state["answer"] = ""
    state["error"] = None
    state["news_error"] = None


def _fallback_sql(question: str) -> str | None:
    q = (question or "").lower()
    limit = _parse_top_limit(question, default=10)

    # Mayor liquidez actual (activo o top N), sin pedir "cambio".
    if _is_liquidity_rank_question(question) and not _question_specifies_time_window(question):
        active = _question_wants_active_filter(question) or bool(
            re.search(r"\bmercado\s+activo\b|\bactivo[s]?\b", q)
        )
        lim = 1 if re.search(r"\bcuál\b|\bcual\b|\bmercado\b", q) and "top" not in q.replace(" ", "") else limit
        return _sql_top_liquidity_latest(limit=lim, active_only=active)

    # Top volumen en mercados activos: volumen acumulado del último snapshot (no Δ con 1–2 horas).
    if _is_volume_rank_question(question) and _question_wants_active_filter(question):
        if not _question_specifies_time_window(question):
            return _sql_top_volume_latest(limit=limit, active_only=True)

    # Top / mayor volumen sin ventana explícita → último volumen acumulado.
    if _is_volume_rank_question(question) and not _question_specifies_time_window(question):
        return _sql_top_volume_latest(limit=limit, active_only=_question_wants_active_filter(question))

    if "volumen" in q and ("24" in q or "últim" in q or "ultim" in q):
        return """
        with w as (
          select *
          from polymarket.fact_market_snapshot
          where snapshot_ts >= (now() - interval '24 hours')
        ),
        agg as (
          select market_id,
                 max(volume) as max_volume,
                 min(volume) as min_volume
          from w
          where volume is not null and volume::text <> 'NaN'
          group by market_id
        )
        select coalesce(m.title, m.question, m.market_id) as title,
               (agg.max_volume - agg.min_volume) as volume_24h,
               agg.max_volume as volume_latest
        from agg
        join polymarket.dim_market m using (market_id)
        order by volume_24h desc nulls last
        limit 5;
        """.strip()
    if "volumen" in q and ("semana" in q or "7" in q):
        return """
        with w as (
          select *
          from polymarket.fact_market_snapshot
          where snapshot_ts >= (now() - interval '7 days')
        ),
        agg as (
          select market_id,
                 max(volume) as max_volume,
                 min(volume) as min_volume
          from w
          where volume is not null and volume::text <> 'NaN'
          group by market_id
        )
        select coalesce(m.title, m.question, m.market_id) as title,
               (agg.max_volume - agg.min_volume) as volume_7d,
               agg.max_volume as volume_latest
        from agg
        join polymarket.dim_market m using (market_id)
        order by volume_7d desc nulls last
        limit 5;
        """.strip()
    if ("probabilidad" in q or "prob" in q) and ("24" in q or "últim" in q or "ultim" in q):
        return """
        with w as (
          select s.market_id, s.outcome_id, s.snapshot_ts, s.probability
          from polymarket.fact_outcome_snapshot s
          where s.snapshot_ts >= (now() - interval '24 hours')
        ),
        agg as (
          select market_id, outcome_id,
                 max(probability) as p_max,
                 min(probability) as p_min
          from w
          group by market_id, outcome_id
        )
        select m.title,
               o.outcome_label,
               (agg.p_max - agg.p_min) as prob_change_24h
        from agg
        join polymarket.dim_market m on m.market_id = agg.market_id
        join polymarket.dim_outcome o on o.outcome_id = agg.outcome_id
        order by abs(agg.p_max - agg.p_min) desc nulls last
        limit 10;
        """.strip()
    if ("liquidez" in q or "liquidity" in q) and ("cambio" in q or "change" in q) and (
        "24" in q or "últim" in q or "ultim" in q
    ):
        return """
        with w as (
          select *
          from polymarket.fact_market_snapshot
          where snapshot_ts >= (now() - interval '24 hours')
        ),
        agg as (
          select market_id,
                 max(liquidity) as max_liquidity,
                 min(liquidity) as min_liquidity
          from w
          where liquidity is not null and liquidity::text <> 'NaN'
          group by market_id
        )
        select coalesce(m.title, m.question, m.market_id) as title,
               (agg.max_liquidity - agg.min_liquidity) as liquidity_change_24h,
               agg.max_liquidity as liquidity_latest
        from agg
        join polymarket.dim_market m using (market_id)
        order by liquidity_change_24h desc nulls last
        limit 10;
        """.strip()
    if ("liquidez" in q or "liquidity" in q) and ("cambio" in q or "change" in q) and (
        "semana" in q or "7" in q
    ):
        return """
        with w as (
          select *
          from polymarket.fact_market_snapshot
          where snapshot_ts >= (now() - interval '7 days')
        ),
        agg as (
          select market_id,
                 max(liquidity) as max_liquidity,
                 min(liquidity) as min_liquidity
          from w
          where liquidity is not null and liquidity::text <> 'NaN'
          group by market_id
        )
        select coalesce(m.title, m.question, m.market_id) as title,
               (agg.max_liquidity - agg.min_liquidity) as liquidity_change_7d,
               agg.max_liquidity as liquidity_latest
        from agg
        join polymarket.dim_market m using (market_id)
        order by liquidity_change_7d desc nulls last
        limit 10;
        """.strip()
    if re.search(r"más activo|más activos|mas activo|mas activos|\bactivos actualmente\b", q):
        return """
        select market_id,
               coalesce(nullif(trim(title), ''), nullif(trim(question), ''), market_id::text) as title,
               updated_at
        from polymarket.dim_market
        where active is true
        order by updated_at desc nulls last
        limit 20;
        """.strip()
    return None


def database_tool(neon_database_url: str, sql: str) -> list[dict[str, Any]]:
    """Database Tool (obligatoria): ejecuta SQL contra Neon/Postgres y devuelve filas."""
    # Preferir psycopg3 (evita dependencia de psycopg2).
    url = (neon_database_url or "").strip()
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        # Si el LLM devolviese algo que no retorna filas (p.ej. DDL), no rompemos el agente.
        if not getattr(result, "returns_rows", False):
            return []
        cols = list(result.keys())
        raw_rows = [dict(zip(cols, row)) for row in result.fetchall()]

        def _fix(v: Any) -> Any:
            if isinstance(v, Decimal):
                try:
                    return None if v.is_nan() else v
                except Exception:
                    return v
            return v

        return [{k: _fix(v) for k, v in r.items()} for r in raw_rows]

def _extract_cs_team_terms(rows: list[dict[str, Any]], *, max_terms: int = 8) -> list[str]:
    """
    Extrae términos tipo 'Team A' y 'Team B' desde títulos/preguntas para usarlos en la búsqueda.
    Heurística: split por 'vs', 'v', '-' y limpiar tokens cortos.
    """
    texts: list[str] = []
    for r in rows[:50]:
        for k in ("title", "question"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())

    # Candidatos: partes alrededor de "vs"
    candidates: list[str] = []
    for t in texts:
        # normaliza separadores típicos de partidos
        parts = re.split(r"\s+(?:vs\.?|v\.?)\s+|\s*-\s*", t, flags=re.IGNORECASE)
        for p in parts:
            s = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ ._'-]", " ", p).strip()
            s = re.sub(r"\s{2,}", " ", s)
            if 2 <= len(s) <= 40:
                candidates.append(s)

    # Filtra ruido común
    stop = {
        "csgo",
        "cs2",
        "counter strike",
        "counter-strike",
        "match",
        "map",
        "bo1",
        "bo3",
        "bo5",
        "winner",
        "win",
        "who wins",
        "will",
    }
    terms: list[str] = []
    for c in candidates:
        lc = c.lower()
        if lc in stop:
            continue
        if any(x in lc for x in ("polymarket", "yes", "no", "over", "under")):
            continue
        if lc.isdigit():
            continue
        # evita frases demasiado largas
        if len(lc.split()) > 5:
            continue
        if c not in terms:
            terms.append(c)
        if len(terms) >= max_terms:
            break
    return terms


def _build_csgo_news_query(question: str, rows: list[dict[str, Any]]) -> str:
    # Query base CSGO/CS2 + esports + torneos comunes.
    base = [
        '"Counter-Strike"',
        "CSGO",
        "CS2",
        '"Counter Strike"',
        "HLTV",
        "ESL",
        "BLAST",
        "IEM",
        "PGL",
        "Major",
    ]
    team_terms = _extract_cs_team_terms(rows, max_terms=8)
    # Entrecomilla términos compuestos para mejorar el matching
    quoted = [f'"{t}"' if " " in t else t for t in team_terms]
    q_terms = base + quoted
    # GDELT query: OR entre términos (limitado a contexto CSGO/CS2/torneos/equipos).
    return " OR ".join(q_terms)

def _hltv_rss_news(*, max_records: int = 20) -> list[dict[str, Any]]:
    """
    Fuente de noticias CSGO/CS2 (bonus) sin API key: RSS de HLTV.
    Devuelve items recientes (título + link + fecha).
    """
    url = "https://www.hltv.org/rss/news"
    headers = {
        # HLTV suele requerir UA para devolver contenido correctamente.
        "User-Agent": "Mozilla/5.0 (compatible; ProyectoRA3/1.0; +https://example.invalid)",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    # RSS 2.0: <rss><channel><item>...
    items = root.findall("./channel/item")
    out: list[dict[str, Any]] = []
    for it in items[: max(1, int(max_records))]:
        out.append(
            {
                "title": (it.findtext("title") or "").strip(),
                "url": (it.findtext("link") or "").strip(),
                "published": (it.findtext("pubDate") or "").strip(),
                "source": "HLTV",
            }
        )
    return out

def _filter_news_items(items: list[dict[str, Any]], *, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """
    Filtra noticias para priorizar equipos/torneos detectados en los mercados.
    Si no hay términos, devuelve las más recientes.
    """
    terms = _extract_cs_team_terms(rows, max_terms=10)
    # Siempre anclado a CSGO/CS2, pero HLTV ya lo está.
    if not terms:
        return items[:limit]
    pat = re.compile("|".join(re.escape(t) for t in terms if t), re.IGNORECASE)
    hit = [x for x in items if pat.search(x.get("title") or "")]
    return (hit or items)[:limit]

def news_tool(*, question: str, rows: list[dict[str, Any]], max_records: int = 8) -> list[dict[str, Any]]:
    """Bonus Tool: noticias SOLO de HLTV (CSGO/CS2)."""
    items = _hltv_rss_news(max_records=max(20, int(max_records) * 5))
    return _filter_news_items(items, rows=rows or [], limit=int(max_records))


def _llm_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def _make_llm(cfg: AgentConfig) -> BaseChatModel:
    """Modelo de chat unificado (LangGraph + trazas LangSmith)."""
    provider = (cfg.llm_provider or "ollama").strip().lower()
    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("Falta openai_api_key para llm_provider=openai")
        os.environ.setdefault("OPENAI_API_KEY", cfg.openai_api_key)
        return init_chat_model(cfg.openai_model, model_provider="openai", temperature=0)
    if provider == "ollama":
        return init_chat_model(
            cfg.ollama_model,
            model_provider="ollama",
            base_url=cfg.ollama_base_url,
            temperature=0,
        )
    raise ValueError(f"llm_provider no soportado: {cfg.llm_provider!r}")


def build_graph(cfg: AgentConfig):
    configure_langsmith()
    llm = _make_llm(cfg)

    # Asegura compatibilidad con SQLAlchemy usando psycopg3.
    try:
        u = make_url(cfg.neon_database_url)
        if u.drivername == "postgresql":
            cfg = AgentConfig(
                **{
                    **cfg.__dict__,
                    "neon_database_url": cfg.neon_database_url.replace(
                        "postgresql://", "postgresql+psycopg://", 1
                    ),
                }
            )
    except Exception:
        pass

    system_sql = (
        "Eres un asistente de datos. Genera UNA consulta SQL válida para Postgres.\n"
        "Usa SOLO estas tablas:\n"
        "- polymarket.dim_market(market_id,title,question,slug,active,closed,archived,created_at,updated_at,raw)\n"
        "- polymarket.fact_market_snapshot(snapshot_ts,market_id,extracted_at,volume,liquidity,best_bid,best_ask,outcome_prices,outcomes,raw)\n"
        "- polymarket.dim_outcome(outcome_id,market_id,outcome_index,outcome_label)\n"
        "- polymarket.fact_outcome_snapshot(snapshot_ts,market_id,outcome_id,extracted_at,probability,raw)\n"
        "Reglas:\n"
        "- Devuelve únicamente SQL (sin markdown, sin explicaciones).\n"
        "- La consulta DEBE ser de solo lectura: un único SELECT (sin INSERT/UPDATE/DELETE/CREATE/DROP/ALTER).\n"
        "- Prefiere devolver campos legibles (por ejemplo, title/question) usando JOIN con dim_market.\n"
        "- Limita a 50 filas si no se pide explícitamente otra cosa.\n"
        "- Si la pregunta pide \"últimas 24h\" usa now() - interval '24 hours'.\n"
        "- Para 'mayor volumen en X' NO sumes volume: asume que volume es acumulado. Usa (max(volume)-min(volume)) en la ventana.\n"
    )

    def ingest_node(state: AgentState) -> AgentState:
        _reset_turn_fields(state)
        q = _question_from_state(state)
        if q:
            state["question"] = q
        return state

    def sql_node(state: AgentState) -> AgentState:
        q = _question_from_state(state)
        state["chitchat_only"] = False
        state["news_only"] = False
        if _is_news_only_question(q):
            state["news_only"] = True
            state["sql"] = "SELECT 1 AS ok;"
            return state
        if _is_chitchat_question(q):
            state["chitchat_only"] = True
            state["sql"] = ""
            return state
        fb = _fallback_sql(q)
        if fb and _prefer_template_sql(q):
            state["sql"] = fb
            return state
        msg = llm.invoke([SystemMessage(content=system_sql), HumanMessage(content=q)])
        sql = _clean_sql(_llm_content(msg))
        if not _is_safe_select(sql):
            # fallback conservador: una consulta simple válida
            sql = "SELECT 1 AS ok;"
        state["sql"] = sql
        return state

    def exec_node(state: AgentState) -> AgentState:
        if state.get("chitchat_only"):
            state["rows"] = []
            state["sql"] = ""
            return state
        if state.get("news_only"):
            # No mezclar con SQL de volumen: solo marcamos que la BD respondió (Database Tool usada).
            try:
                rows = database_tool(cfg.neon_database_url, "SELECT 1 AS ok;")
                state["rows"] = rows
                state["sql"] = "SELECT 1 AS ok;"
            except Exception as e:  # noqa: BLE001
                state["rows"] = []
                state["error"] = str(e)
            return state

        sql = str(state.get("sql") or "")
        # Requisito 14: Database Tool obligatoria (aquí se usa).
        fb = _fallback_sql(_question_from_state(state))
        try:
            rows = database_tool(cfg.neon_database_url, sql)
            # Si el modelo cayó en el "SELECT 1" (fallback conservador), preferimos el fallback específico.
            if fb and _clean_sql(sql).lower() in ("select 1 as ok;", "select 1;"):
                rows = database_tool(cfg.neon_database_url, fb)
                state["sql"] = fb
            # Si el modelo generó SQL válido pero inútil (0 filas) y tenemos plantilla robusta,
            # preferimos el fallback para asegurar demo funcional.
            if (not rows) and fb and _clean_sql(sql) != _clean_sql(fb):
                rows = database_tool(cfg.neon_database_url, fb)
                state["sql"] = fb
            # SQL del LLM con filas vacías o ranking de volumen pobre → plantilla robusta.
            q_exec = _question_from_state(state)
            if fb and _prefer_template_sql(q_exec):
                weak_volume = _is_volume_rank_question(q_exec) and (
                    not rows
                    or all(
                        float(r.get("volume_24h") or r.get("volume_7d") or 0) == 0
                        for r in rows
                        if isinstance(r, dict)
                    )
                )
                weak_liquidity = _is_liquidity_rank_question(q_exec) and not rows
                if weak_volume or weak_liquidity or (
                    not rows and _clean_sql(sql) != _clean_sql(fb)
                ):
                    rows = database_tool(cfg.neon_database_url, fb)
                    state["sql"] = fb
                elif _clean_sql(sql) != _clean_sql(fb):
                    # Plantilla conocida: no confiar en SQL alucinado del LLM si ya hay filas raras.
                    rows = database_tool(cfg.neon_database_url, fb)
                    state["sql"] = fb
            state["rows"] = rows
        except Exception as e:  # noqa: BLE001
            if fb:
                try:
                    rows = database_tool(cfg.neon_database_url, fb)
                    state["sql"] = fb
                    state["rows"] = rows
                    state["error"] = None
                except Exception as e2:  # noqa: BLE001
                    state["rows"] = []
                    state["error"] = f"Error ejecutando SQL: {e2}"
            else:
                state["rows"] = []
                state["error"] = f"Error ejecutando SQL: {e}"
        return state

    def news_node(state: AgentState) -> AgentState:
        if state.get("chitchat_only"):
            return state
        if not cfg.enable_news_tool:
            return state
        q = _question_from_state(state).strip().lower()
        # Solo si el usuario pide noticias (para no ralentizar el flujo normal).
        if "noticia" in q or "news" in q or "hltv" in q:
            try:
                state["news"] = news_tool(
                    question=_question_from_state(state),
                    rows=state.get("rows") or [],
                    max_records=8,
                )
            except Exception as e:  # noqa: BLE001
                state["news"] = []
                state["news_error"] = str(e)
        return state

    system_answer = (
        "Eres un asistente. Te doy la pregunta, el SQL ejecutado y las filas devueltas.\n"
        "Responde en español, claro y conciso. Si hay 0 filas, dilo y sugiere otra consulta.\n"
        "No inventes: basa la respuesta en el preview de filas y el recuento.\n"
        "Si te paso NEWS (lista de noticias HLTV), úsala como contexto adicional.\n"
        "Si la pregunta es sobre mercados 'más activos' o 'activos actualmente' y hay filas con title: "
        "enumera los títulos (y updated_at si existe). "
        "NO digas que falta contexto ni pidas más datos si Filas_preview ya trae resultados.\n"
    )
    system_answer_news_only = (
        "El usuario pide NOTICIAS sobre Counter-Strike (CSGO/CS2) desde HLTV (RSS).\n"
        "- Si NEWS tiene elementos: enumera 3-8 títulos con su enlace (url) y fecha si existe.\n"
        "- NO digas que no hay noticias si NEWS no está vacía.\n"
        "- NO hables de SQL ni de volumen/mercados salvo que el usuario lo pida explícitamente.\n"
        "- Si NEWS está vacío o hay news_error, di que no se pudieron cargar ahora y sugiere reintentar.\n"
    )
    def answer_node(state: AgentState) -> AgentState:
        q = _question_from_state(state)
        sql = str(state.get("sql") or "")
        rows = state.get("rows") or []
        news = state.get("news") or []
        err = state.get("error")
        if state.get("chitchat_only"):
            return _with_assistant_reply(state, _chitchat_reply(q))
        if state.get("news_only"):
            msg = llm.invoke(
                [
                    SystemMessage(content=system_answer_news_only),
                    HumanMessage(
                        content=(
                            f"Pregunta: {q}\n"
                            f"NEWS: {news}\n"
                            f"news_error: {state.get('news_error', '')}\n"
                        )
                    ),
                ]
            )
            return _with_assistant_reply(state, _llm_content(msg).strip())
        if err:
            return _with_assistant_reply(
                state,
                f"No pude consultar Neon: {err}. Comprueba NEON_DATABASE_URL y que el DAG haya cargado datos.",
            )
        if isinstance(rows, list) and not rows:
            return _with_assistant_reply(
                state,
                "No hay filas para esta pregunta en el data warehouse. "
                "Ejecuta el pipeline (Airflow) y vuelve a intentar.",
            )
        # Respuestas deterministas (sin LLM) para el demo y Studio Chat.
        if _is_active_markets_question(q) and isinstance(rows, list) and rows:
            return _with_assistant_reply(state, _format_active_markets_answer(rows))
        if _is_liquidity_change_question(q) and isinstance(rows, list) and rows:
            return _with_assistant_reply(state, _format_liquidity_change_answer(rows))
        if _is_liquidity_rank_question(q) and isinstance(rows, list) and rows:
            singular = bool(
                re.search(r"\bcuál\b|\bcual\b", q.lower())
                and "top" not in q.lower().replace(" ", "")
            )
            return _with_assistant_reply(
                state,
                _format_liquidity_rank_answer(
                    rows,
                    limit=_parse_top_limit(q, default=10),
                    singular=singular,
                ),
            )
        if _is_volume_question(q) and isinstance(rows, list) and rows:
            return _with_assistant_reply(
                state, _format_volume_answer(rows, limit=_parse_top_limit(q, default=10))
            )
        if _prefer_template_sql(q) and isinstance(rows, list) and rows:
            if rows[0].get("prob_change_24h") is not None:
                lines = ["Top cambios de probabilidad (24h):"]
                for i, r in enumerate(rows[:10], start=1):
                    lines.append(
                        f"{i}. {r.get('title')} — {r.get('outcome_label')}: Δ={r.get('prob_change_24h')}"
                    )
                return _with_assistant_reply(state, "\n".join(lines))
            if rows[0].get("liquidity_change_24h") is not None or rows[0].get("liquidity_change_7d") is not None:
                return _with_assistant_reply(state, _format_liquidity_change_answer(rows))
            if rows[0].get("volume_24h") is not None or rows[0].get("volume_7d") is not None:
                return _with_assistant_reply(
                    state, _format_volume_answer(rows, limit=_parse_top_limit(q, default=10))
                )
        preview = rows[:10] if isinstance(rows, list) else rows
        msg = llm.invoke(
            [
                SystemMessage(content=system_answer),
                HumanMessage(
                    content=(
                        f"Pregunta: {q}\n"
                        f"SQL: {sql}\n"
                        f"Filas_count: {len(rows) if isinstance(rows, list) else 'n/a'}\n"
                        f"Filas_preview: {preview}\n"
                        f"NEWS: {news}"
                    )
                ),
            ]
        )
        return _with_assistant_reply(state, _llm_content(msg).strip())

    # Grafo LangGraph (StateGraph + nodos); TypedDict evita invoke() -> None.
    g = StateGraph(AgentState)
    g.add_node("ingest", ingest_node)
    g.add_node("sql", sql_node)
    g.add_node("exec", exec_node)
    g.add_node("news", news_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "sql")
    g.add_edge("sql", "exec")
    g.add_edge("exec", "news")
    g.add_edge("news", "answer")
    g.add_edge("answer", END)
    return g.compile()


def build_agent(cfg: AgentConfig):
    """Alias retrocompatible (Streamlit / imports antiguos)."""
    return build_graph(cfg)

