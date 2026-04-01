import sys
import os

# 🔹 Garante que a raiz do projeto está no PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
for path in (src_path, project_root):
    if path not in sys.path:
        sys.path.insert(0, path)

from rag.ingest import embedder, index, all_chunks, metadata
from rag.retriever import retrieve
from rag.generator import generate_answer

def run_local_test(query="Quais ações estão recomendadas para 2026?", top_k=3):
    print("🔹 Query:", query)

    # Passo 1: Busca chunks relevantes
    results = retrieve(query, embedder, index, all_chunks, metadata, top_k=top_k)
    print("\n--- Resultados do Retriever ---")
    for r in results:
        print(f"Rank {r['rank']} | Distância: {r['distance']}")
        print(f"Doc: {r['metadata']['doc_id']} - {r['metadata'].get('title','')}")
        print(f"Texto: {r['text']}\n")

    # Passo 2: Concatena contexto
    context = " ".join([r["text"] for r in results])

    # Passo 3: Gera resposta final
    answer = generate_answer(query, context)
    print("\n--- Resposta Final ---")
    print(answer)

if __name__ == "__main__":
    run_local_test()