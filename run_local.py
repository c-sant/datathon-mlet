import os
import sys


def _ensure_project_paths() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")

    for path in (src_path, project_root):
        if path not in sys.path:
            sys.path.insert(0, path)


def run_local_test(query: str = "Quais ações estão recomendadas para 2026?", top_k: int = 3):
    _ensure_project_paths()

    from rag.embedding import all_chunks, embedder, index, metadata
    from rag.generator import generate_answer
    from rag.retriever import retrieve

    print("Query:", query)

    results = retrieve(query, embedder, index, all_chunks, metadata, top_k=top_k)
    print("\n--- Resultados do Retriever ---")
    for result in results:
        print(f"Rank {result['rank']} | Distância: {result['distance']}")
        print(f"Doc: {result['metadata']['doc_id']} - {result['metadata'].get('title', '')}")
        print(f"Texto: {result['text']}\n")

    context = " ".join([result["text"] for result in results])
    answer = generate_answer(query, context)

    print("\n--- Resposta Final ---")
    print(answer)


if __name__ == "__main__":
    run_local_test()
