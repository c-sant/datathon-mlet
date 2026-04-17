#!/usr/bin/env python3
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "src"


@lru_cache(maxsize=1)
def _load_runtime_dependencies():
    for path in (project_root, src_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import uvicorn

    from agent.react_agent import run_agent
    from rag.data_loader import load_news
    from rag.embedding import all_chunks, embedder, index, ingest_documents, metadata
    from rag.generator import generate_answer
    from rag.retriever import retrieve

    return {
        "all_chunks": all_chunks,
        "embedder": embedder,
        "generate_answer": generate_answer,
        "index": index,
        "ingest_documents": ingest_documents,
        "load_news": load_news,
        "metadata": metadata,
        "retrieve": retrieve,
        "run_agent": run_agent,
        "uvicorn": uvicorn,
    }


def load_documents_from_file(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, dict):
        docs = payload.get("docs") or payload.get("documents") or payload.get("data")
        if docs is None:
            raise ValueError(
                "O JSON deve conter uma lista sob a chave 'docs', 'documents' ou 'data'."
            )
        if not isinstance(docs, list):
            raise ValueError("O campo de documentos deve ser uma lista.")
        return docs

    if isinstance(payload, list):
        return payload

    raise ValueError("O conteúdo do arquivo JSON deve ser uma lista ou um objeto com chave 'docs'.")


def run_offline(query: str, top_k: int = 3):
    deps = _load_runtime_dependencies()

    if deps["index"] is None or len(deps["all_chunks"]) == 0:
        print(
            "O índice RAG não está disponível. Execute uma ingestão antes de usar o modo offline."
        )
        return

    print(f"🔍 Executando pipeline offline para: {query}")
    results = deps["retrieve"](
        query,
        deps["embedder"],
        deps["index"],
        deps["all_chunks"],
        deps["metadata"],
        top_k=top_k,
    )
    for item in results:
        print(f"\nRank {item['rank']} | Distância: {item['distance']:.4f}")
        print(f"Doc: {item['metadata'].get('doc_id', 'n/a')} - {item['metadata'].get('title', '')}")
        print(item["text"])

    context = "\n\n".join([item["text"] for item in results])
    answer = deps["generate_answer"](query, context)

    print("\n--- Resposta Final ---")
    print(answer)


def run_ingest(file_path: Path | None, overwrite: bool = True):
    deps = _load_runtime_dependencies()

    if file_path:
        docs = load_documents_from_file(file_path)
    else:
        print("Carregando notícias financeiras padrão...")
        docs = deps["load_news"]()

    stats = deps["ingest_documents"](docs, overwrite=overwrite, log_run=True)
    print(f"Ingestão concluída: {stats['doc_count']} docs, {stats['chunk_count']} chunks.")


def run_api(host: str, port: int, reload: bool):
    deps = _load_runtime_dependencies()
    print(f"Iniciando API RAG em http://{host}:{port}")
    deps["uvicorn"].run(
        "serving.app:app", host=host, port=port, reload=reload, app_dir=str(src_root)
    )


def run_agent_query(query: str, top_k: int = 3):
    deps = _load_runtime_dependencies()
    result = deps["run_agent"](query, top_k=top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline RAG multi-modo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser(
        "offline", help="Executa o pipeline RAG local sem servidor HTTP"
    )
    offline.add_argument(
        "--query",
        default="Quais ações estão recomendadas para 2026?",
        help="Pergunta de teste para o pipeline",
    )
    offline.add_argument("--top-k", type=int, default=3, help="Número de chunks retornados")

    api = subparsers.add_parser("api", help="Inicia a API FastAPI para o pipeline RAG")
    api.add_argument("--host", default="0.0.0.0", help="Host onde a API será exposta")
    api.add_argument("--port", type=int, default=8000, help="Porta da API")
    api.add_argument("--reload", action="store_true", help="Ativa recarga automática do servidor")

    ingest = subparsers.add_parser("ingest", help="Ingesta documentos no índice RAG")
    ingest.add_argument("--file", type=Path, help="Arquivo JSON com documentos para ingestão")
    ingest.add_argument("--overwrite", action="store_true", help="Sobrescreve o índice existente")
    ingest.add_argument(
        "--append", action="store_true", help="Acrescenta documentos ao índice existente"
    )

    agent = subparsers.add_parser("agent", help="Executa o agente ReAct para uma consulta")
    agent.add_argument("--query", required=True, help="Pergunta a ser enviada ao agente")
    agent.add_argument(
        "--top-k", type=int, default=3, help="Número de chunks recuperados pelo agente"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "offline":
        run_offline(args.query, top_k=args.top_k)
    elif args.command == "api":
        run_api(args.host, args.port, args.reload)
    elif args.command == "ingest":
        run_ingest(args.file, overwrite=args.overwrite and not args.append)
    elif args.command == "agent":
        run_agent_query(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
