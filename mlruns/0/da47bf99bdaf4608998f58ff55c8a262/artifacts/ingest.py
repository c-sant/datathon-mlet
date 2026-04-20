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

# 🔹 Carrega documentos
# Se não passar nada, usa as URLs default do data_loader.py
# Se quiser sobrescrever, basta passar uma lista de URLs: load_news(["url1", "url2"])
docs = load_news()

# 🔹 Inicializa modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Pré-processa e gera embeddings
all_chunks = []
metadata = []
for doc in docs:
    chunks = chunk_text(doc["text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata.append({
            "doc_id": doc["id"],
            "chunk_id": i,
            "title": doc.get("title", "")
        })

embeddings = embedder.encode(all_chunks)

# 🔹 Cria índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 🔹 Log no MLflow
with mlflow.start_run(run_name="RAG_ingest"):
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("vector_store", "FAISS")
    mlflow.log_param("num_docs", len(docs))
    mlflow.log_param("num_chunks", len(all_chunks))
    mlflow.log_artifact(__file__)

print("✅ Ingestão concluída. Documentos coletados, chunkados e embeddings armazenados no FAISS.")