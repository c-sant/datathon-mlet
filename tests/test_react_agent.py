import sys
import types
from importlib.machinery import ModuleSpec


class _FakeIndexFlatL2:
    def __init__(self, dim):
        self.dim = dim
        self._embeddings = None

    def add(self, embeddings):
        self._embeddings = embeddings

    def search(self, query_embedding, top_k):
        # Retorno mínimo para manter compatibilidade quando chamado em testes.
        return [[0.0] * top_k], [[0] * top_k]


fake_faiss = types.ModuleType("faiss")
fake_faiss.__spec__ = ModuleSpec("faiss", loader=None)
fake_faiss.IndexFlatL2 = _FakeIndexFlatL2
sys.modules.setdefault("faiss", fake_faiss)

import agent.react_agent as react_agent


def test_parse_agent_output_parses_json_action_input():
    raw = (
        "Thought: Vou buscar contexto\n"
        "Action: search_documents\n"
        'Action Input: {"query": "petroleo", "top_k": 2}'
    )

    parsed = react_agent._parse_agent_output(raw)

    assert parsed["thought"] == "Vou buscar contexto"
    assert parsed["action"] == "search_documents"
    assert parsed["action_input"] == {"query": "petroleo", "top_k": 2}
    assert parsed["final_answer"] is None


def test_execute_tool_returns_unknown_tool_message():
    result = react_agent._execute_tool("nao_existe", {})

    assert result == "Ferramenta desconhecida: nao_existe."


def test_run_agent_returns_final_answer_directly(monkeypatch):
    monkeypatch.setattr(react_agent, "index", object())
    monkeypatch.setattr(react_agent, "all_chunks", ["chunk"])

    def fake_generate_text(*args, **kwargs):
        return "Final Answer: Resposta final do agente"

    monkeypatch.setattr(react_agent, "generate_text", fake_generate_text)

    result = react_agent.run_agent("Qual o cenario para 2026?")

    assert result["answer"] == "Resposta final do agente"
    assert len(result["trace"]) == 1
    assert result["trace"][0]["step"] == 1


def test_run_agent_executes_search_with_top_k(monkeypatch):
    monkeypatch.setattr(react_agent, "index", object())
    monkeypatch.setattr(react_agent, "all_chunks", ["chunk"])

    calls = {"count": 0}

    def fake_generate_text(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                "Thought: Vou procurar no indice\n"
                "Action: search_documents\n"
                'Action Input: {"query": "dolar"}'
            )
        return "Final Answer: Concluido"

    captured = {}

    def fake_execute_tool(action, action_input):
        captured["action"] = action
        captured["action_input"] = action_input
        return "Observacao fake"

    monkeypatch.setattr(react_agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(react_agent, "_execute_tool", fake_execute_tool)

    result = react_agent.run_agent("Quero noticias", top_k=7, max_steps=3)

    assert captured["action"] == "search_documents"
    assert captured["action_input"] == {"query": "dolar", "top_k": 7}
    assert result["answer"] == "Concluido"
    assert len(result["trace"]) == 2
    assert result["trace"][0]["observation"] == "Observacao fake"


def test_run_agent_fallbacks_to_generate_answer_on_error(monkeypatch):
    monkeypatch.setattr(react_agent, "index", object())
    monkeypatch.setattr(react_agent, "all_chunks", ["chunk"])

    def fake_generate_text(*args, **kwargs):
        raise RuntimeError("modelo indisponivel")

    def fake_generate_answer(query, observation):
        return f"fallback::{query}::{observation}"

    monkeypatch.setattr(react_agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(react_agent, "generate_answer", fake_generate_answer)

    result = react_agent.run_agent("Teste erro")

    assert result["answer"].startswith("fallback::Teste erro::")
    assert "Falha na geração de texto" in result["trace"][0]["error"]
