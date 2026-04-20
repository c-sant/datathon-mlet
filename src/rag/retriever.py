import numpy as np


def retrieve(query, embedder, index, chunks, metadata, top_k=3):
    """Busca os top-k chunks mais relevantes para a query."""
    q_emb = embedder.encode([query])

    q_emb = np.array(q_emb, dtype=np.float32)
    if hasattr(index, "search"):
        distances, indices = index.search(q_emb, top_k)
    else:
        distances, indices = index.kneighbors(q_emb, n_neighbors=top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        result = {
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "text": chunks[idx],
            "metadata": metadata[idx],
        }
        results.append(result)

    return results
