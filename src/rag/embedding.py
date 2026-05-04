import os
from pathlib import Path

import faiss
import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.data_loader import load_news


# 🔹 Função para dividir documentos em chunks
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)


def _init_embedder(model_name="all-MiniLM-L6-v2"):
    """Inicializa o modelo de embeddings com token HF opcional."""
    if not HF_TOKEN:
        return SentenceTransformer(model_name)

    try:
        return SentenceTransformer(model_name, token=HF_TOKEN)
    except TypeError:
        # Compatibilidade com versoes antigas do sentence-transformers.
        return SentenceTransformer(model_name, use_auth_token=HF_TOKEN)


# 🔹 Inicializa modelo de embeddings
embedder = _init_embedder("all-MiniLM-L6-v2")

docs = []
all_chunks = []
metadata = []
index = None
# Conjunto de IDs já ingeridos — garante upsert incremental sem duplicatas
_ingested_doc_ids: set = set()


def build_index(documents):
    """Cria um índice FAISS a partir de documentos já normalizados."""
    chunks = []
    metadata_entries = []
    for doc in documents:
        text = doc.get("text", "").strip()
        if not text:
            continue
        chunks_for_doc = chunk_text(text)
        for i, chunk in enumerate(chunks_for_doc):
            chunks.append(chunk)
            metadata_entries.append(
                {
                    "doc_id": doc.get("id") or f"doc_{len(chunks) - 1}",
                    "chunk_id": i,
                    "title": doc.get("title", ""),
                    "fetched_at": doc.get("fetched_at", ""),
                }
            )

    if not chunks:
        raise ValueError("Nenhum texto válido foi informado para ingestão.")

    embeddings = embedder.encode(chunks)
    vector_index = faiss.IndexFlatL2(embeddings.shape[1])
    vector_index.add(np.array(embeddings, dtype=np.float32))

    return vector_index, chunks, metadata_entries


def ingest_documents(documents, overwrite=False, log_run=True):
    """Ingesta documentos no pipeline RAG com estratégia incremental (upsert).

    A atualização é NUNCA destrutiva: novos documentos são adicionados ao índice
    existente via upsert por ID. O store permanece disponível durante toda a
    atualização — o swap é atômico ao final.

    Args:
        documents (list[dict]): Lista de documentos com chaves `id`, `title`, `text`.
        overwrite (bool): Se True, força rebuild completo (usar apenas em inicialização).
                          Por padrão False — comportamento incremental.
        log_run (bool): Se True, cria um run MLflow para esta ingestão.

    Returns:
        dict: Estatísticas da ingestão (total, novos, ignorados).
    """
    global docs, all_chunks, metadata, index, _ingested_doc_ids

    normalized_docs = []
    for i, doc in enumerate(documents):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        normalized_docs.append(
            {
                "id": doc.get("id") or f"doc_{i}",
                "title": doc.get("title", ""),
                "text": text,
                "fetched_at": doc.get("fetched_at", ""),
            }
        )

    if not normalized_docs:
        raise ValueError("Nenhum documento válido encontrado para ingestão.")

    if overwrite:
        # Rebuild completo — usado apenas na inicialização.
        # O novo índice é construído ANTES de substituir o atual (atomic swap).
        new_index, new_chunks, new_metadata = build_index(normalized_docs)
        new_ids = {d["id"] for d in normalized_docs}
        # Atomic swap: o store antigo permanece disponível até este ponto
        docs, all_chunks, metadata, index, _ingested_doc_ids = (
            normalized_docs, new_chunks, new_metadata, new_index, new_ids
        )
        skipped = 0
        added = len(normalized_docs)
    else:
        # Upsert incremental: filtra apenas docs com IDs novos ou atualizados.
        # Docs com mesmo ID são ignorados (sem janela de store vazio).
        new_docs = [d for d in normalized_docs if d["id"] not in _ingested_doc_ids]
        skipped = len(normalized_docs) - len(new_docs)

        if not new_docs:
            return {
                "doc_count": len(docs),
                "chunk_count": len(all_chunks),
                "added": 0,
                "skipped": skipped,
            }

        # Gera embeddings apenas dos novos docs
        new_chunks_list = []
        new_metadata_list = []
        for doc in new_docs:
            chunks_for_doc = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks_for_doc):
                new_chunks_list.append(chunk)
                new_metadata_list.append(
                    {
                        "doc_id": doc["id"],
                        "chunk_id": i,
                        "title": doc["title"],
                        "fetched_at": doc["fetched_at"],
                    }
                )

        new_embeddings = np.array(
            embedder.encode(new_chunks_list), dtype=np.float32
        )

        if index is None:
            # Primeira carga — cria o índice
            new_index = faiss.IndexFlatL2(new_embeddings.shape[1])
            new_index.add(new_embeddings)
            index = new_index
        else:
            # Adiciona vetores ao índice existente — store nunca fica vazio
            index.add(new_embeddings)

        all_chunks.extend(new_chunks_list)
        metadata.extend(new_metadata_list)
        docs.extend(new_docs)
        _ingested_doc_ids.update(d["id"] for d in new_docs)
        added = len(new_docs)

    if log_run:
        with mlflow.start_run(run_name="RAG_ingest"):
            mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
            mlflow.log_param("vector_store", "FAISS")
            mlflow.log_param("num_docs", len(docs))
            mlflow.log_param("num_chunks", len(all_chunks))
            mlflow.log_param("added_docs", added)
            mlflow.log_param("skipped_docs", skipped)
            try:
                artifact_path = Path(__file__).resolve()
                if artifact_path.is_file():
                    mlflow.log_artifact(str(artifact_path))
            except OSError as exc:
                print(f"Aviso: não foi possível registrar artifact no MLflow: {exc}")

    print(
        f"Ingestão concluída. Adicionados: {added} docs | "
        f"Ignorados (já existentes): {skipped} | "
        f"Total no store: {len(docs)} docs / {len(all_chunks)} chunks"
    )

    return {
        "doc_count": len(docs),
        "chunk_count": len(all_chunks),
        "added": added,
        "skipped": skipped,
    }


# Ingestão inicial com notícias padrão (overwrite=True apenas aqui — rebuild inicial).
# Evita ruído no startup da API caso o backend de tracking não esteja acessível.
try:
    ingest_documents(load_news(), overwrite=True, log_run=False)
except Exception as exc:
    print(f"Falha na ingestão inicial de notícias: {exc}")
