from dataclasses import dataclass
from typing import Any, Callable

from rag.data_loader import load_news
import rag.embedding as _emb
from rag.embedding import ingest_documents
from rag.generator import generate_text
from rag.retriever import retrieve
from utils.config_loader import load_config

_cfg_summarize = load_config()["agent"]["tool_summarize"]


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
        item_metadata = item.get("metadata", {})
        lines.append(
            f"Rank {item['rank']} | Distância: {item['distance']:.4f} | Doc: {item_metadata.get('doc_id', 'n/a')} - {item_metadata.get('title', '')}\n{item['text']}"
        )
    return "\n\n".join(lines)


def tool_search_documents(input_data: Any) -> str:
    if _emb.index is None or len(_emb.all_chunks) == 0:
        return "O índice de busca não está disponível. Execute uma ingestão antes de usar esta ferramenta."

    if isinstance(input_data, str):
        payload = {"query": input_data, "top_k": 3}
    elif isinstance(input_data, dict):
        payload = input_data
    else:
        return (
            "Formato de entrada inválido para search_documents. Use uma string ou um objeto JSON."
        )

    query = str(payload.get("query", "")).strip()
    top_k = int(payload.get("top_k", 3))
    if not query:
        return "A ferramenta search_documents requer o campo query."

    results = retrieve(query, _emb.embedder, _emb.index, _emb.all_chunks, _emb.metadata, top_k=top_k)
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
    raw = generate_text(
        prompt,
        max_new_tokens=_cfg_summarize["max_new_tokens"],
        temperature=_cfg_summarize["temperature"],
    )
    return raw.strip()


TOOLS = [
    AgentTool(
        name="search_documents",
        description=(
            "Busca trechos relevantes na base de conhecimento usando RAG. "
            'Entrada: JSON com {"query": string, "top_k": int opcional}.'
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
            'Entrada: JSON com {"context": string}.'
        ),
        func=tool_summarize_context,
    ),
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}
