import numpy as np
import faiss

def retrieve(query, embedder, index, chunks, metadata, top_k=3):
    """
    Busca os top-k chunks mais relevantes para a query.
    
    Args:
        query (str): Pergunta do usuário.
        embedder: Modelo de embeddings (SentenceTransformer).
        index: Índice FAISS já carregado.
        chunks (list): Lista de textos chunkados.
        metadata (list): Metadados correspondentes aos chunks.
        top_k (int): Número de resultados a retornar.
    
    Returns:
        list: Lista de dicionários com texto e metadados.
    """
    # Gera embedding da query
    q_emb = embedder.encode([query])
    
    # Busca no índice de vetores
    q_emb = np.array(q_emb, dtype=np.float32)
    if hasattr(index, "search"):
        distances, indices = index.search(q_emb, top_k)
    else:
        distances, indices = index.kneighbors(q_emb, n_neighbors=top_k)

    # Monta resultados
    results = []
    for rank, idx in enumerate(indices[0]):
        result = {
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "text": chunks[idx],
            "metadata": metadata[idx]
        }
        results.append(result)
    
    return results