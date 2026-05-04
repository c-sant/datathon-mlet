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


def _synthesize_final_answer(query: str, observation: str) -> str:
    obs_for_answer = observation[:1800] + "..." if len(observation) > 1800 else observation
    synthesis_prompt = (
        "Com base no contexto abaixo, responda de forma objetiva e em português:\n"
        f"Pergunta: {query}\n\nContexto:\n{obs_for_answer}\n\nResposta:"
    )
    try:
        return generate_text(
            synthesis_prompt,
            max_new_tokens=_INFERENCE_MAX_NEW_TOKENS,
            temperature=0.3,
        )
    except Exception:
        return generate_answer(query, obs_for_answer)


def _format_tool_descriptions() -> str:
    lines = [f"{tool.name}: {tool.description}" for tool in TOOLS]
    return "\n".join(lines)


def _parse_agent_output(text: str) -> dict[str, Any]:
    thought = ""
    action = None
    action_input = None
    final_answer = None

    # Preferred contract: strict JSON payload from the model.
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            action = payload.get("action")
            if isinstance(action, str) and action.strip().lower() in {"", "null", "none"}:
                action = None

            action_input = payload.get("action_input")
            if isinstance(action_input, str) and action_input.strip().lower() in {"", "null", "none"}:
                action_input = None

            final_answer = payload.get("final_answer")
            if isinstance(final_answer, str) and final_answer.strip().lower() in {"", "null", "none"}:
                final_answer = None

            return {
                "thought": str(payload.get("thought", "")).strip(),
                "action": action,
                "action_input": action_input,
                "final_answer": final_answer,
                "raw": text.strip(),
            }
        except json.JSONDecodeError:
            # Fallback to legacy parser if model emits invalid JSON.
            pass

    thought_match = re.search(r"(?:Thought|Pensamento):\s*(.*?)(?:\n(?:Action|Ação|Acao):|\n(?:Final Answer|Resposta Final):|$)", text, re.S | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"(?:Action|Ação|Acao):\s*([A-Za-z0-9_]+)", text, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip()

    input_match = re.search(r"(?:Action Input|Entrada|Parâmetros):\s*(.*?)(?:\n(?:Thought|Pensamento):|\n(?:Final Answer|Resposta Final):|$)", text, re.S | re.IGNORECASE)
    if input_match:
        action_input = input_match.group(1).strip()

    final_match = re.search(r"(?:Final Answer|Resposta Final):\s*(.*)", text, re.S | re.IGNORECASE)
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
        "Você é um agente ReAct de finanças.\n"
        "Responda SOMENTE com JSON válido (sem markdown e sem texto fora do JSON).\n\n"
        "Ferramentas disponíveis:\n"
        f"{_format_tool_descriptions()}\n\n"
        "Contrato obrigatório de resposta (JSON):\n"
        '{"thought":"...","action":"search_documents|fetch_news|summarize_context|null","action_input":{},"final_answer":"...|null"}\n\n'
        "Regras:\n"
        "1) Se faltar dado, escolha action (não responda final imediatamente).\n"
        "2) Quando já houver contexto suficiente, use action=null e preencha final_answer.\n"
        "3) action_input deve ser objeto JSON (ou null quando action=null).\n\n"
        "4) Se o usuário pedir atualizar notícias/recentes, use fetch_news antes de responder.\n"
        "5) Se o usuário pedir resumo/bullets, use summarize_context antes de responder.\n\n"
    )

    if history:
        prompt += "Histórico:\n"
        for step in history:
            prompt += (
                f"step={step.get('step', '')}\n"
                f"thought={step.get('thought', '')}\n"
                f"action={step.get('action', '')}\n"
                f"action_input={json.dumps(step.get('action_input', ''), ensure_ascii=False)}\n"
                f"observation={step.get('observation', '')}\n\n"
            )

    prompt += f"Usuário: {query}\n"
    if observation:
        # Truncate observation to avoid small models echoing the full context back
        obs_truncated = observation[:600] + "..." if len(observation) > 600 else observation
        prompt += f"Observation: {obs_truncated}\n"
    prompt += "Retorne apenas o JSON do contrato obrigatório."
    return prompt


def _execute_tool(action: str, action_input: Any) -> str:
    tool = TOOL_MAP.get(action)
    if not tool:
        return f"Ferramenta desconhecida: {action}."
    return tool.func(action_input)


def _normalize_action(action: Any) -> str | None:
    if not isinstance(action, str):
        return None

    raw = action.strip()
    if not raw:
        return None

    if raw in TOOL_MAP:
        return raw

    # Handle malformed outputs like "search_documents|fetch_news".
    for token in re.split(r"[|,/;\s]+", raw):
        token = token.strip()
        if token in TOOL_MAP:
            return token

    return None


def _infer_forced_action(
    query: str, history: list[dict[str, Any]], observation: str
) -> tuple[str | None, Any]:
    q = (query or "").lower()
    actions_done = {str(item.get("action") or "") for item in history}

    wants_news_update = any(
        token in q for token in ["atualize", "atualizar", "recentes", "notícias", "noticias"]
    )
    wants_summary = any(
        token in q for token in ["resuma", "resumo", "bullet", "bullets", "sumarize", "summarize"]
    )

    if wants_news_update and "fetch_news" not in actions_done:
        return "fetch_news", {}

    if wants_summary and "summarize_context" not in actions_done:
        return "summarize_context", {"context": observation}

    return None, None


def run_agent(
    query: str, top_k: int = _DEFAULT_TOP_K, max_steps: int = _DEFAULT_MAX_STEPS
) -> dict[str, Any]:
    history = []
    trace = []
    observation = ""

    # Pre-step: always search documents first so the LLM always has real context,
    # regardless of whether it emits an Action: or jumps straight to an answer.
    if _emb.index is not None and len(_emb.all_chunks) > 0:
        observation = _execute_tool("search_documents", {"query": query, "top_k": top_k})
        trace.append(
            {
                "step": 1,
                "thought": "Pré-busca automática para garantir contexto inicial.",
                "action": "search_documents",
                "action_input": {"query": query, "top_k": top_k},
                "raw_output": "",
                "observation": observation,
            }
        )
    else:
        observation = "O índice de busca está vazio ou indisponível."
        trace.append(
            {
                "step": 1,
                "thought": "Índice indisponível na pré-busca.",
                "action": "search_documents",
                "action_input": {"query": query, "top_k": top_k},
                "raw_output": "",
                "observation": observation,
            }
        )

    for step in range(max_steps):
        prompt = _build_agent_prompt(query, history, observation)
        try:
            raw_output = generate_text(
                prompt, max_new_tokens=_INFERENCE_MAX_NEW_TOKENS, temperature=_INFERENCE_TEMPERATURE
            )
        except Exception as exc:
            return {
                "query": query,
                "answer": _synthesize_final_answer(query, observation),
                "trace": trace + [
                    {
                        "step": step + 2,
                        "error": f"Falha na geração de texto: {exc}",
                    }
                ],
            }

        parsed = _parse_agent_output(raw_output)
        parsed["action"] = _normalize_action(parsed.get("action"))

        # If the model tries to return both action and final answer in the same step,
        # enforce ReAct behavior: execute the action first.
        if parsed.get("action") and parsed.get("final_answer"):
            parsed["final_answer"] = None

        forced_action, forced_input = _infer_forced_action(query, history, observation)
        if forced_action:
            parsed["action"] = forced_action
            parsed["action_input"] = forced_input
            parsed["final_answer"] = None

        step_record = {
            "step": step + 2,
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
                "trace": trace + history + [step_record],
            }

        if not parsed["action"]:
            # Do not terminate early: keep ReAct loop running until max_steps.
            step_record["observation"] = "Nenhuma ação emitida pelo modelo neste passo."
            history.append(step_record)
            continue

        action_input = parsed["action_input"]
        if action_input is None:
            action_input = {}
        if parsed["action"] == "search_documents" and isinstance(action_input, dict):
            action_input.setdefault("query", query)
            action_input.setdefault("top_k", top_k)
        if parsed["action"] == "summarize_context" and isinstance(action_input, dict):
            action_input.setdefault("context", observation)

        observation = _execute_tool(parsed["action"], action_input)
        step_record["observation"] = observation
        history.append(step_record)

        if parsed["action"] == "fetch_news":
            query = query

    # Synthesise final answer directly via vLLM using truncated observation.
    # Avoids token overflow from passing the full observation to generate_answer.
    return {
        "query": query,
        "answer": _synthesize_final_answer(query, observation),
        "trace": trace + history,
    }
