import json
import re
from typing import Any

from agent.tools import TOOL_MAP, TOOLS
import rag.embedding as _emb
from rag.generator import generate_answer, generate_text
from utils.config_loader import load_config

_cfg = load_config()["agent"]
_INFERENCE_MAX_NEW_TOKENS: int = _cfg["inference"]["max_new_tokens"]
_INFERENCE_TEMPERATURE: float = _cfg["inference"]["temperature"]
_DEFAULT_MAX_STEPS: int = _cfg["max_steps"]
_DEFAULT_TOP_K: int = _cfg["top_k"]


def _format_tool_descriptions() -> str:
    lines = [f"{tool.name}: {tool.description}" for tool in TOOLS]
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


def run_agent(
    query: str, top_k: int = _DEFAULT_TOP_K, max_steps: int = _DEFAULT_MAX_STEPS
) -> dict[str, Any]:
    history = []
    observation = ""

    if _emb.index is None or len(_emb.all_chunks) == 0:
        observation = "O índice de busca está vazio ou indisponível."

    for step in range(max_steps):
        prompt = _build_agent_prompt(query, history, observation)
        try:
            raw_output = generate_text(
                prompt, max_new_tokens=_INFERENCE_MAX_NEW_TOKENS, temperature=_INFERENCE_TEMPERATURE
            )
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
