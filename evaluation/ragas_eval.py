"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.
Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval
            Augmented Generation. https://arxiv.org/abs/2309.15217
"""
import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")
RESULTS_PATH = Path("data/golden_set/ragas_results.json")


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    """Carrega o golden set de pares Q&A do domínio financeiro B3.

    Args:
        path: Caminho para o arquivo JSON do golden set.

    Returns:
        Lista de dicionários com query, expected_answer e contexts.

    Raises:
        FileNotFoundError: Se o golden set não existir.
        ValueError: Se o golden set tiver menos de 20 pares.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden set não encontrado em: {path}")

    with open(path, encoding="utf-8") as f:
        golden_set = json.load(f)

    if len(golden_set) < 20:
        raise ValueError(
            f"Golden set deve ter ≥ 20 pares. Encontrados: {len(golden_set)}"
        )

    logger.info("Golden set carregado: %d pares", len(golden_set))
    return golden_set


def evaluate_rag_pipeline(
    rag_fn: Callable[[str], tuple[str, list[str]]],
    golden_set_path: Path = GOLDEN_SET_PATH,
    save_results: bool = True,
) -> dict[str, float]:
    """Avalia o pipeline RAG de dados financeiros contra o golden set.

    Args:
        rag_fn: Função que recebe uma query e retorna (answer, contexts).
                Exemplo: lambda q: agent.query(q)
        golden_set_path: Caminho para o golden set JSON.
        save_results: Se True, salva resultados em disco.

    Returns:
        Dicionário com as 4 métricas RAGAS obrigatórias:
        - faithfulness: resposta baseada nos contextos recuperados
        - answer_relevancy: relevância da resposta para a pergunta
        - context_precision: proporção de contextos relevantes recuperados
        - context_recall: cobertura do contexto sobre a resposta esperada
    """
    golden_set = load_golden_set(golden_set_path)

    logger.info("Gerando respostas do pipeline RAG para %d queries...", len(golden_set))

    results = []
    for i, item in enumerate(golden_set):
        try:
            answer, contexts = rag_fn(item["query"])
            results.append({
                "question": item["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
            })
            logger.debug("Query %d/%d processada: %s", i + 1, len(golden_set), item["query"][:60])
        except Exception as e:
            logger.error("Erro na query %d: %s — %s", i + 1, item["query"][:60], e)
            results.append({
                "question": item["query"],
                "answer": "Erro ao processar query.",
                "contexts": [],
                "ground_truth": item["expected_answer"],
            })

    dataset = Dataset.from_list(results)

    logger.info("Executando avaliação RAGAS (4 métricas obrigatórias)...")
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    metrics = {
        "faithfulness": round(float(scores["faithfulness"]), 4),
        "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
        "context_precision": round(float(scores["context_precision"]), 4),
        "context_recall": round(float(scores["context_recall"]), 4),
    }

    ragas_score = sum(metrics.values()) / len(metrics)
    metrics["ragas_score_mean"] = round(ragas_score, 4)

    logger.info("RAGAS scores: %s", metrics)
    _log_interpretation(metrics)

    if save_results:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "metrics": metrics,
            "n_samples": len(golden_set),
            "details": results,
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Resultados salvos em: %s", RESULTS_PATH)

    return metrics


def _log_interpretation(metrics: dict[str, float]) -> None:
    """Loga interpretação das métricas para o domínio financeiro."""
    thresholds = {
        "faithfulness": (0.85, "respostas devem ser baseadas nos dados da B3 recuperados"),
        "answer_relevancy": (0.80, "respostas devem responder diretamente perguntas sobre ações"),
        "context_precision": (0.75, "recuperador deve trazer contextos relevantes sobre o ativo"),
        "context_recall": (0.75, "contextos recuperados devem cobrir a resposta esperada"),
    }

    logger.info("=== Interpretação das métricas RAGAS ===")
    for metric, (threshold, description) in thresholds.items():
        value = metrics.get(metric, 0)
        status = "OK" if value >= threshold else "ABAIXO DO THRESHOLD"
        logger.info(
            "%s: %.4f (threshold: %.2f) [%s] — %s",
            metric, value, threshold, status, description,
        )


def generate_ragas_report(results_path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Gera DataFrame com detalhes por query para análise.

    Args:
        results_path: Caminho para o JSON de resultados RAGAS.

    Returns:
        DataFrame com métricas por query.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data["details"])
    df["n_contexts"] = df["contexts"].apply(len)
    df["answer_len"] = df["answer"].apply(len)

    logger.info("Relatório RAGAS gerado: %d queries analisadas", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Mock RAG function para testes sem API
    def mock_rag_fn(query: str) -> tuple[str, list[str]]:
        """Mock do pipeline RAG para testes locais."""
        golden_set = load_golden_set()
        for item in golden_set:
            if item["query"] == query:
                return item["expected_answer"], item["contexts"]
        return "Informação não encontrada.", []

    metrics = evaluate_rag_pipeline(rag_fn=mock_rag_fn)
    print("\n=== RAGAS Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")