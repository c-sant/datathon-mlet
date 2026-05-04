import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.react_agent import run_agent
import rag.embedding as _emb
from rag.embedding import ingest_documents
from rag.generator import generate_answer
from rag.mlflow_loader import load_mlflow_docs
from rag.retriever import retrieve
from security.guardrails import InputGuardrail, OutputGuardrail

app = FastAPI(
    title="Datathon RAG API",
    description=(
        "API de Recuperação e Geração Aumentada (RAG) para consultas financeiras e de modelos ML.\n\n"
        "**Fluxo recomendado:**\n"
        "1. `POST /ingest` — indexa notícias/documentos externos\n"
        "2. `POST /ingest_mlflow` — indexa métricas e parâmetros do modelo treinado\n"
        "3. `GET /query` — consulta direta via RAG\n"
        "4. `POST /agent` — consulta via agente ReAct (multi-etapas)"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class Document(BaseModel):
    id: str | None = Field(None, description="Identificador único do documento")
    title: str | None = Field(None, description="Título do documento")
    text: str = Field(..., description="Conteúdo em texto do documento")


class IngestRequest(BaseModel):
    docs: list[Document] = Field(..., description="Lista de documentos a serem ingeridos")
    overwrite: bool = Field(True, description="Substituir a base existente se True")


@lru_cache
def _get_input_guardrail() -> InputGuardrail:
    return InputGuardrail()


@lru_cache
def _get_output_guardrail() -> OutputGuardrail:
    return OutputGuardrail(language="pt")


@lru_cache
def _get_input_guardrail() -> InputGuardrail:
    return InputGuardrail()


@lru_cache
def _get_output_guardrail() -> OutputGuardrail:
    return OutputGuardrail(language="pt")


def _fix_mojibake(text: str) -> str:
    if not text:
        return text
    if any(token in text for token in ("Ã", "â", "€", "™")):
        try:
            fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if fixed:
                return fixed
        except Exception:
            pass
    return text


def _normalize_text(text: str) -> str:
    text = _fix_mojibake(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _validate_user_query(query: str) -> str:
    clean_query = _normalize_text(query)
    if not clean_query:
        raise HTTPException(status_code=400, detail="Input bloqueado: consulta vazia.")

    is_valid, reason = _get_input_guardrail().validate(clean_query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    return clean_query


def _sanitize_public_value(value: Any) -> Any:
    if isinstance(value, str):
        if not value:
            return value
        return _get_output_guardrail().sanitize(value)

    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _sanitize_public_value(item) for key, item in value.items()}

    return value


def _validate_user_query(query: str) -> str:
    clean_query = _normalize_text(query)
    if not clean_query:
        raise HTTPException(status_code=400, detail="Input bloqueado: consulta vazia.")

    is_valid, reason = _get_input_guardrail().validate(clean_query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    return clean_query


def _sanitize_public_value(value: Any) -> Any:
    if isinstance(value, str):
        if not value:
            return value
        return _get_output_guardrail().sanitize(value)

    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _sanitize_public_value(item) for key, item in value.items()}

    return value


def _build_context(results: list[dict], max_chars: int = 1800) -> str:
    """Monta contexto estruturado, limpo e com menor redundância para geração."""
    selected: list[str] = []
    seen_texts: list[str] = []
    total_chars = 0

    for i, item in enumerate(results, start=1):
        raw_text = item.get("text", "")
        text = _normalize_text(raw_text)
        if not text:
            continue

        # Evita contexto redundante por similaridade muito alta entre chunks.
        is_duplicate = any(
            SequenceMatcher(None, text.lower(), prev.lower()).ratio() > 0.9 for prev in seen_texts
        )
        if is_duplicate:
            continue

        meta = item.get("metadata", {}) or {}
        title = _normalize_text(meta.get("title", ""))
        fetched_at = _normalize_text(meta.get("fetched_at", ""))

        excerpt = text[:520]
        source_block = f"Fonte {i}: {title or 'Sem título'}"
        if fetched_at:
            source_block += f" ({fetched_at})"
        source_block += f"\nTrecho: {excerpt}"

        if total_chars + len(source_block) > max_chars and selected:
            break

        selected.append(source_block)
        seen_texts.append(text)
        total_chars += len(source_block)

    return "\n\n".join(selected)


def _is_model_query(query: str) -> bool:
    q = (query or "").lower()
    terms = (
        "modelo",
        "modelos",
        "treino",
        "treinamento",
        "mlflow",
        "métrica",
        "métricas",
        "metricas",
        "mae",
        "rmse",
        "mape",
        "baseline",
        "pytorch",
        "sklearn",
        "keras",
        "feature",
        "pca",
        "janela",
        "epoch",
    )
    return any(t in q for t in terms)


def _score_model_result(item: dict) -> int:
    meta = item.get("metadata", {}) or {}
    title = _normalize_text(meta.get("title", "")).lower()
    text = _normalize_text(item.get("text", "")).lower()

    score = 0
    high = ("mlflow", "métricas", "metricas", "parâmetros", "parametros", "treinamento", "treino")
    mid = ("pca", "feature", "train", "mae", "rmse", "mape", "janela", "epoch", "batch")

    for token in high:
        if token in title:
            score += 6
        if token in text:
            score += 3

    for token in mid:
        if token in title:
            score += 3
        if token in text:
            score += 1

    return score


def _rerank_for_model_query(query: str, results: list[dict], top_k: int) -> list[dict]:
    if not _is_model_query(query):
        return results[:top_k]

    ranked = sorted(results, key=_score_model_result, reverse=True)
    return ranked[:top_k]


def _query_with_rag(q: str, top_k: int) -> dict:
    if _emb.index is None or len(_emb.all_chunks) == 0:
        raise HTTPException(status_code=503, detail="Índice de busca não está disponível.")

    candidate_k = max(top_k, 8) if _is_model_query(q) else top_k
    raw_results = retrieve(
        q,
        _emb.embedder,
        _emb.index,
        _emb.all_chunks,
        _emb.metadata,
        top_k=candidate_k,
    )
    results = _rerank_for_model_query(q, raw_results, top_k)
    context = _build_context(results)
    if not context:
        context = " ".join([_normalize_text(r.get("text", "")) for r in results]).strip()

    answer = generate_answer(q, context)
    return {"query": q, "top_k": top_k, "context": context, "answer": answer}


@app.post(
    "/ingest",
    tags=["1. Ingestão"],
    summary="Ingerir documentos externos (notícias, textos livres)",
)
def ingest_rag(payload: IngestRequest):
    """
    Adiciona ou substitui documentos no índice FAISS.

    - **overwrite=true**: apaga o índice atual e reindexar tudo
    - **overwrite=false**: adiciona apenas documentos novos (upsert)
    """
    try:
        stats = ingest_documents(
            [doc.model_dump() for doc in payload.docs],
            overwrite=payload.overwrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "doc_count": stats["doc_count"],
        "chunk_count": stats["chunk_count"],
        "overwrite": payload.overwrite,
    }


@app.post(
    "/ingest_mlflow",
    tags=["1. Ingestão"],
    summary="Ingerir métricas e parâmetros do modelo treinado (MLflow + DVC)",
)
def ingest_mlflow():
    """
    Lê automaticamente os resultados do pipeline DVC/MLflow e os indexa no FAISS:

    - Runs do MLflow com métricas (MAE, RMSE, MAPE por modelo)
    - `reports/metrics.json` — comparação consolidada dos modelos
    - `reports/selected_features.csv` + `pca_explained_variance.csv`
    - `params.yaml` — hiperparâmetros de treinamento

    Não apaga documentos já existentes no índice.
    """
    docs = load_mlflow_docs()
    if not docs:
        return {
            "status": "ok",
            "message": "Nenhum documento MLflow encontrado.",
            "doc_count": 0,
            "chunk_count": 0,
        }

    stats = ingest_documents(docs, overwrite=False)
    return {
        "status": "ok",
        "doc_count": stats["doc_count"],
        "chunk_count": stats["chunk_count"],
        "sources": [d["title"] for d in docs],
    }


@app.get(
    "/query",
    tags=["2. Consulta"],
    summary="Consulta direta via RAG (recuperação + geração)",
)
def query_rag(q: str, top_k: int = 3):
    """
    Recupera os `top_k` trechos mais relevantes do índice e gera uma resposta.

    Para perguntas sobre métricas/modelos, a resposta é determinística (sem LLM).
    Para perguntas gerais, usa o modelo flan-t5-base.

    **Exemplos de perguntas sobre o modelo:**
    - `compare os modelos treinados`
    - `qual modelo teve menor MAE`
    - `qual o ranking por MAPE`
    - `quais os parâmetros de treinamento usados`
    """
    safe_query = _validate_user_query(q)
    result = _query_with_rag(safe_query, top_k)
    return _sanitize_public_value(result)
    safe_query = _validate_user_query(q)
    result = _query_with_rag(safe_query, top_k)
    return _sanitize_public_value(result)


class AgentRequest(BaseModel):
    query: str = Field(..., description="Consulta do usuário para o agente ReAct")
    top_k: int = Field(3, description="Número de documentos a recuperar durante a busca")


@app.post(
    "/agent",
    tags=["2. Consulta"],
    summary="Consulta via agente ReAct (raciocínio multi-etapas)",
)
def agent_rag(payload: AgentRequest):
    """
    Executa um loop ReAct (Reasoning + Acting): o agente decide quais ferramentas
    usar e itera até formular uma resposta final.

    Mais adequado para perguntas complexas que exigem múltiplas buscas.
    Para perguntas simples, prefira `GET /query`.

    Para consultas de métricas/modelo, aplica caminho rápido via RAG direto
    para reduzir latência sem perder precisão.
    """
    safe_query = _validate_user_query(payload.query)

    if _is_model_query(safe_query):
        fast = _query_with_rag(safe_query, payload.top_k)
        response = {
            "query": safe_query,
    safe_query = _validate_user_query(payload.query)

    if _is_model_query(safe_query):
        fast = _query_with_rag(safe_query, payload.top_k)
        response = {
            "query": safe_query,
            "answer": fast["answer"],
            "trace": [
                {
                    "step": 1,
                    "thought": "Consulta de modelo detectada; aplicado caminho rápido sem loop ReAct.",
                    "action": "query_rag_fast_path",
                    "action_input": {"top_k": payload.top_k},
                    "observation": "Resposta gerada via recuperação direta e fallback analítico.",
                }
            ],
        }
        return _sanitize_public_value(response)
        return _sanitize_public_value(response)

    try:
        result = run_agent(safe_query, top_k=payload.top_k)
        result = run_agent(safe_query, top_k=payload.top_k)
    except Exception as exc:
        fast = _query_with_rag(safe_query, payload.top_k)
        response = {
            "query": safe_query,
        fast = _query_with_rag(safe_query, payload.top_k)
        response = {
            "query": safe_query,
            "answer": fast["answer"],
            "trace": [
                {
                    "step": 1,
                    "thought": "Execução ReAct falhou; aplicado fallback resiliente via RAG direto.",
                    "action": "query_rag_exception_fallback",
                    "action_input": {"top_k": payload.top_k},
                    "observation": f"Erro original do agente: {type(exc).__name__}",
                }
            ],
        }
        return _sanitize_public_value(response)
        return _sanitize_public_value(response)

    # Detecta resposta vazia/template gerada quando o FLAN não consegue seguir
    # o formato ReAct — fallback para caminho RAG direto com contexto real.
    answer = (result.get("answer") or "").strip()
    _bad = {"reposta objetiva:", "resposta objetiva:", "reposta objetiva", "resposta objetiva", ""}
    if answer.lower().rstrip(":").strip() in _bad or answer.lower().startswith("reposta objetiv"):
        fast = _query_with_rag(safe_query, payload.top_k)
        fast = _query_with_rag(safe_query, payload.top_k)
        result["answer"] = fast["answer"]
        result.setdefault("trace", []).append(
            {
                "step": len(result.get("trace", [])) + 1,
                "thought": "Resposta do agente era inválida; aplicado fallback via RAG direto.",
                "action": "query_rag_fallback",
                "action_input": {"top_k": payload.top_k},
                "observation": "Resposta substituída pelo caminho analítico/RAG.",
            }
        )

    return _sanitize_public_value(result)
    return _sanitize_public_value(result)
