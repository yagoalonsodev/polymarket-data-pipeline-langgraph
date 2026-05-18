"""
Grafo compilado para LangGraph Studio (`langgraph dev`).

Antes de abrir smith.langchain.com/studio, arranca el servidor local:
  uv sync --extra studio && langgraph dev
"""
from pathlib import Path

from dotenv import load_dotenv

root = Path(__file__).resolve().parents[2]
load_dotenv(root / ".env")

from src.agent.langgraph_agent import AgentConfig, build_graph  # noqa: E402

graph = build_graph(AgentConfig.from_env())
