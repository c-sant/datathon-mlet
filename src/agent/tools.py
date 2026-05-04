from dataclasses import dataclass
import re
from typing import Any, Callable

from rag.data_loader import load_news
import rag.embedding as _emb
from rag.embedding import ingest_documents
from rag.generator import generate_text
from rag.retriever import retrieve
from utils.config_loader import load_config

_cfg_summarize = load_config()["agent"]["tool_summarize"]


def _extract_b3_tickers(text: str) -> list[str]:
    if not text:
        return []
    # Ex.: ITUB4, BBDC4, BBAS3, PETR4
    return sorted({m.group(0).upper() for m in re.finditer(r"\b[A-Za-z]{4}\d{1,2}\b", text)})


def _is_ticker_in_corpus(ticker: str) -> bool:
    t = (ticker or "").upper()
    if not t:
        return False
    t_lower = t.lower()
    for chunk in _emb.all_chunks or []:
        try:
            if t in chunk or t_lower in chunk.lower():
                return True
        except Exception:
            continue
    return False


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


def _rerank_results_for_tickers(results: list[dict], requested_tickers: list[str]) -> list[dict]:
    if not results or not requested_tickers:
        return results

    requested = {ticker.upper() for ticker in requested_tickers}

    def _classify(item: dict) -> tuple[int, int, float]:
        metadata = item.get("metadata", {})
        haystack = " ".join(
            [
                str(item.get("text", "")),
                str(metadata.get("title", "")),
                str(metadata.get("doc_id", "")),
            ]
        ).upper()
        mentioned = set(_extract_b3_tickers(haystack))
        requested_hits = len(requested & mentioned)
        other_hits = len(mentioned - requested)
        return (requested_hits, other_hits, mentioned)

    scored = []
    for item in results:
        requested_hits, other_hits, mentioned = _classify(item)
        scored.append((item, requested_hits, other_hits, mentioned))

    # Sort: prefer requested_hits desc, then other_hits asc, then distance asc
    scored.sort(key=lambda x: (x[1], -x[2], -float(x[0].get("distance", 9999.0))), reverse=True)

    # Filter: remove chunks that mention ONLY other tickers (not the queried one)
    filtered = [
        item for item, req_hits, other_hits, mentioned in scored
        if req_hits > 0 or len(mentioned) == 0
    ]

    # Fall back to full reranked list if filtering removed everything
    final = filtered if filtered else [x[0] for x in scored]

    for rank, item in enumerate(final, start=1):
        item["rank"] = rank
    return final


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

    # Dynamic ingestion: if the query references unseen tickers, ingest on demand.
    tickers = _extract_b3_tickers(query)
    missing_tickers = [t for t in tickers if not _is_ticker_in_corpus(t)]
    if missing_tickers:
        try:
            new_docs = load_news(tickers=missing_tickers, include_ticker_pages=True)
            if new_docs:
                ingest_documents(new_docs, overwrite=False, log_run=False)
        except Exception as exc:
            # Keep search resilient even if on-demand ingestion fails.
            print(f"Aviso: ingestão dinâmica falhou para {missing_tickers}: {exc}")

    results = retrieve(query, _emb.embedder, _emb.index, _emb.all_chunks, _emb.metadata, top_k=top_k)
    results = _rerank_results_for_tickers(results, tickers)
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
