import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from rag.data_loader import load_news
from rag.generator import generate_answer, generate_text
from rag.embedding import ingest_documents, embedder, index, all_chunks, metadata
from rag.retriever import retrieve


@dataclass
class AgentTool:
    name: str
    description: str
    func: Callable[[Any], str]


def _format_search_results(results: list[dict]) -> str:
    if not results:
        return "Nenhum documento relevante foi encontrado."

    lines = []
    for item in results:
        metadata = item.get("metadata", {})
        lines.append(
            f"Rank {item['rank']} | Distância: {item['distance']:.4f} | Doc: {metadata.get('doc_id', 'n/a')} - {metadata.get('title', '')}\n{item['text']}"
        )
    return "\n\n".join(lines)


def tool_search_documents(input_data: Any) -> str:
    if index is None or len(all_chunks) == 0:
        return "O índice de busca não está disponível. Execute uma ingestão antes de usar esta ferramenta."

    if isinstance(input_data, str):
        payload = {"query": input_data, "top_k": 3}
    elif isinstance(input_data, dict):
        payload = input_data
    else:
        return "Formato de entrada inválido para search_documents. Use uma string ou um objeto JSON."

    query = str(payload.get("query", "")).strip()
    top_k = int(payload.get("top_k", 3))
    if not query:
        return "A ferramenta search_documents requer o campo query."

    results = retrieve(query, embedder, index, all_chunks, metadata, top_k=top_k)
    return _format_search_results(results)


def tool_fetch_news(_: Any) -> str:
    news_docs = load_news()
    if not news_docs:
        return "Não foi possível carregar notícias financeiras no momento."

    stats = ingest_documents(news_docs, overwrite=False, log_run=False)
    return (
        f"Notícias carregadas e indexadas. Documents atuais: {stats['doc_count']}. "
        f"Chunks atuais: {stats['chunk_count']}."
    )


def tool_summarize_context(input_data: Any) -> str:
    if isinstance(input_data, dict):
        context = str(input_data.get("context", "")).strip()
    else:
        context = str(input_data).strip()

    if not context:
        return "A ferramenta summarize_context requer um campo context com texto para resumir."

    prompt = (
        "Resuma o seguinte contexto em português de forma objetiva e concisa:\n\n"
        f"{context}\n\n"
        "Resumo:" 
    )
    raw = generate_text(prompt, max_new_tokens=128, temperature=0.5)
    return raw.strip()


TOOLS = [
    AgentTool(
        name="search_documents",
        description=(
            "Busca trechos relevantes na base de conhecimento usando RAG. "
            "Entrada: JSON com {\"query\": string, \"top_k\": int opcional}."
        ),
        func=tool_search_documents,
    ),
    AgentTool(
        name="fetch_news",
        description=(
            "Atualiza a base de conhecimento com notícias financeiras recentes das fontes padrão. "
            "Não requer entrada adicional."
        ),
        func=tool_fetch_news,
    ),
    AgentTool(
        name="summarize_context",
        description=(
            "Resume um texto em português de forma clara e objetiva. "
            "Entrada: JSON com {\"context\": string}."
        ),
        func=tool_summarize_context,
    ),
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}


def _format_tool_descriptions() -> str:
    lines = [
        f"{tool.name}: {tool.description}" for tool in TOOLS
    ]
    return "\n".join(lines)


def _parse_agent_output(text: str) -> dict[str, Any]:
    thought = ""
    action = None
    action_input = None
    final_answer = None

    thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\nFinal Answer:|$)", text, re.S)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*([A-Za-z0-9_]+)", text)
    if action_match:
        action = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(.*?)(?:\nThought:|\nFinal Answer:|$)", text, re.S)
    if input_match:
        action_input = input_match.group(1).strip()

    final_match = re.search(r"Final Answer:\s*(.*)", text, re.S)
    if final_match:
        final_answer = final_match.group(1).strip()

    if action_input:
        try:
            action_input = json.loads(action_input)
        except json.JSONDecodeError:
            action_input = action_input.strip('"')

    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "final_answer": final_answer,
        "raw": text.strip(),
    }


def _build_agent_prompt(query: str, history: list[dict[str, Any]], observation: str) -> str:
    prompt = (
        "Você é um agente ReAct especializado em finanças e mercado financeiro. Use apenas as ferramentas listadas abaixo para responder. "
        "Se o usuário fizer uma pergunta sobre investimentos, análise de mercado, notícias financeiras ou resumo de contexto, escolha a ferramenta apropriada.\n\n"
        "Ferramentas disponíveis:\n"
        f"{_format_tool_descriptions()}\n\n"
        "Formato de resposta esperado:\n"
        "Thought: <seu raciocínio aqui>\n"
        "Action: <nome_da_ferramenta>\n"
        "Action Input: <entrada_em_JSON_ou_texto>\n\n"
        "Se você já tem informação suficiente para responder de forma final, responda apenas com:\n"
        "Final Answer: <resposta em português>\n\n"
    )

    if history:
        prompt += "Histórico:\n"
        for step in history:
            prompt += (
                f"Thought: {step.get('thought', '')}\n"
                f"Action: {step.get('action', '')}\n"
                f"Action Input: {json.dumps(step.get('action_input', ''), ensure_ascii=False)}\n"
                f"Observation: {step.get('observation', '')}\n\n"
            )

    prompt += f"Usuário: {query}\n"
    if observation:
        prompt += f"Observation: {observation}\n"
    prompt += "Thought:"
    return prompt


def _execute_tool(action: str, action_input: Any) -> str:
    tool = TOOL_MAP.get(action)
    if not tool:
        return f"Ferramenta desconhecida: {action}."
    return tool.func(action_input)


def run_agent(query: str, top_k: int = 3, max_steps: int = 3) -> dict[str, Any]:
    history = []
    observation = ""

    if index is None or len(all_chunks) == 0:
        observation = "O índice de busca está vazio ou indisponível."

    for step in range(max_steps):
        prompt = _build_agent_prompt(query, history, observation)
        try:
            raw_output = generate_text(prompt, max_new_tokens=256, temperature=0.5)
        except Exception as exc:
            return {
                "query": query,
                "answer": generate_answer(query, observation),
                "trace": [
                    {
                        "step": step + 1,
                        "error": f"Falha na geração de texto: {exc}",
                    }
                ],
            }

        parsed = _parse_agent_output(raw_output)
        step_record = {
            "step": step + 1,
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "raw_output": parsed["raw"],
            "observation": "",
        }

        if parsed["final_answer"]:
            return {
                "query": query,
                "answer": parsed["final_answer"],
                "trace": history + [step_record],
            }

        if not parsed["action"]:
            return {
                "query": query,
                "answer": generate_answer(query, observation),
                "trace": history + [step_record],
            }

        action_input = parsed["action_input"]
        if parsed["action"] == "search_documents" and isinstance(action_input, dict):
            action_input.setdefault("top_k", top_k)

        observation = _execute_tool(parsed["action"], action_input)
        step_record["observation"] = observation
        history.append(step_record)

        if parsed["action"] == "fetch_news":
            query = query

    return {
        "query": query,
        "answer": generate_answer(query, observation),
        "trace": history,
    }
