import mlflow
import faiss
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

# 🔹 Inicializa modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
all_chunks = []
metadata = []
index = None


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
            metadata_entries.append({
                "doc_id": doc.get("id") or f"doc_{len(chunks) - 1}",
                "chunk_id": i,
                "title": doc.get("title", "")
            })

    if not chunks:
        raise ValueError("Nenhum texto válido foi informado para ingestão.")

    embeddings = embedder.encode(chunks)
    vector_index = faiss.IndexFlatL2(embeddings.shape[1])
    vector_index.add(np.array(embeddings))

    return vector_index, chunks, metadata_entries


def ingest_documents(documents, overwrite=True, log_run=True):
    """Ingesta uma lista de documentos no pipeline RAG.

    Args:
        documents (list[dict]): Lista de documentos com chaves `id`, `title`, `text`.
        overwrite (bool): Se True, substitui a base existente; senão, acumula.
        log_run (bool): Se True, cria um run MLflow para esta ingestão.

    Returns:
        dict: Estatísticas da ingestão.
    """
    global docs, all_chunks, metadata, index

    normalized_docs = []
    for i, doc in enumerate(documents):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        normalized_docs.append({
            "id": doc.get("id") or f"doc_{i}",
            "title": doc.get("title", ""),
            "text": text,
        })

    if not normalized_docs:
        raise ValueError("Nenhum documento válido encontrado para ingestão.")

    if overwrite:
        docs = normalized_docs
    else:
        docs.extend(normalized_docs)

    index, all_chunks, metadata = build_index(docs)

    if log_run:
        with mlflow.start_run(run_name="RAG_ingest"):
            mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
            mlflow.log_param("vector_store", "FAISS")
            mlflow.log_param("num_docs", len(docs))
            mlflow.log_param("num_chunks", len(all_chunks))
            mlflow.log_artifact(__file__)

    print(
        "Ingestão concluída. Documentos coletados, chunkados e embeddings armazenados no FAISS."
    )

    return {
        "doc_count": len(docs),
        "chunk_count": len(all_chunks),
    }


# Ingestão inicial com notícias padrão
try:
    ingest_documents(load_news(), overwrite=True, log_run=True)
except Exception as exc:
    print(f"Falha na ingestão inicial de notícias: {exc}")
