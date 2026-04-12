"""A/B Test de prompts para o agente RAG de dados financeiros B3.

Compara variantes de prompt em termos de qualidade de resposta,
usando RAGAS e LLM-as-judge como métricas de avaliação.
"""
import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

AB_RESULTS_PATH = Path("data/golden_set/ab_test_results.json")

#Variantes de prompt

PROMPT_VARIANT_A = """Você é um assistente especializado em mercado financeiro brasileiro.
Responda perguntas sobre ações, índices e dados da B3 (Bovespa) de forma direta e precisa.

Use APENAS as informações fornecidas no contexto abaixo para responder.
Se a informação não estiver no contexto, diga claramente que não possui esse dado.

Contexto:
{context}

Pergunta: {query}

Resposta:"""

PROMPT_VARIANT_B = """Você é um analista financeiro experiente especializado na B3.
Sua função é fornecer análises precisas sobre ações e índices brasileiros.

IMPORTANTE:
- Baseie-se EXCLUSIVAMENTE nos dados do contexto fornecido
- Cite valores numéricos com precisão (preços em R$, variações em %)
- Se não houver dados suficientes, informe explicitamente
- Para questões de investimento, adicione contexto de risco quando relevante

Dados disponíveis:
{context}

Consulta do investidor: {query}

Análise:"""

PROMPT_VARIANT_C = """Contexto financeiro: {context}

Pergunta sobre B3: {query}

Responda de forma concisa e precisa com os dados do contexto acima."""

VARIANTS = {
    "A": PROMPT_VARIANT_A,
    "B": PROMPT_VARIANT_B,
    "C": PROMPT_VARIANT_C,
}


@dataclass
class VariantResult:
    """Resultado de uma variante de prompt em uma query."""
    variant_name: str
    query: str
    answer: str
    contexts: list[str]
    latency_ms: float = 0.0
    judge_score: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0


@dataclass
class ABTestReport:
    """Relatório completo do A/B test."""
    variants_tested: list[str]
    n_queries: int
    results_by_variant: dict[str, list[VariantResult]] = field(default_factory=dict)

    def summary(self) -> dict:
        """Retorna sumário estatístico por variante."""
        summary = {}
        for variant, results in self.results_by_variant.items():
            scores = [r.judge_score for r in results if r.judge_score > 0]
            latencies = [r.latency_ms for r in results if r.latency_ms > 0]
            summary[variant] = {
                "avg_judge_score": round(statistics.mean(scores), 3) if scores else 0,
                "std_judge_score": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0,
                "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
                "n_evaluated": len(scores),
            }
        return summary

    def winner(self) -> str:
        """Retorna a variante com maior score médio."""
        s = self.summary()
        return max(s, key=lambda v: s[v]["avg_judge_score"])


def build_rag_fn_with_prompt(
    base_rag_fn: Callable[[str], tuple[str, list[str]]],
    prompt_template: str,
    llm_fn: Callable[[str], str],
) -> Callable[[str], tuple[str, list[str]]]:
    """Constrói função RAG com prompt específico para A/B test.

    Args:
        base_rag_fn: Função de recuperação de contextos (retriever).
        prompt_template: Template do prompt com {query} e {context}.
        llm_fn: Função que chama o LLM com o prompt completo.

    Returns:
        Função RAG que usa o prompt da variante.
    """
    def rag_with_prompt(query: str) -> tuple[str, list[str]]:
        _, contexts = base_rag_fn(query)
        context_str = "\n".join(contexts)
        prompt = prompt_template.format(query=query, context=context_str)
        answer = llm_fn(prompt)
        return answer, contexts

    return rag_with_prompt


def run_ab_test(
    retriever_fn: Callable[[str], tuple[str, list[str]]],
    llm_fn: Callable[[str], str],
    golden_set_path: Path = Path("data/golden_set/golden_set.json"),
    variants: dict[str, str] | None = None,
    n_samples: int | None = None,
    save_results: bool = True,
) -> ABTestReport:
    """Executa A/B test comparando variantes de prompt no domínio financeiro B3.

    Args:
        retriever_fn: Função que recupera contextos dado uma query.
        llm_fn: Função que gera resposta dado um prompt completo.
        golden_set_path: Caminho para o golden set.
        variants: Dict de variantes {nome: template}. Usa padrão se None.
        n_samples: Número de queries para testar (None = todas).
        save_results: Se True, salva resultados em disco.

    Returns:
        ABTestReport com resultados por variante.
    """
    import time
    from evaluation.llm_judge import evaluate_with_llm_judge

    if variants is None:
        variants = VARIANTS

    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    if n_samples:
        golden_set = golden_set[:n_samples]

    logger.info(
        "Iniciando A/B test: %d variantes × %d queries = %d avaliações",
        len(variants), len(golden_set), len(variants) * len(golden_set),
    )

    report = ABTestReport(
        variants_tested=list(variants.keys()),
        n_queries=len(golden_set),
    )

    for variant_name, prompt_template in variants.items():
        logger.info("Testando variante %s...", variant_name)
        rag_fn = build_rag_fn_with_prompt(retriever_fn, prompt_template, llm_fn)
        variant_results = []

        for item in golden_set:
            t0 = time.time()
            answer, contexts = rag_fn(item["query"])
            latency_ms = (time.time() - t0) * 1000

            judge = evaluate_with_llm_judge(
                query=item["query"],
                answer=answer,
                ground_truth=item["expected_answer"],
                contexts=contexts,
            )

            variant_results.append(VariantResult(
                variant_name=variant_name,
                query=item["query"],
                answer=answer,
                contexts=contexts,
                latency_ms=latency_ms,
                judge_score=judge.overall_score,
            ))

        report.results_by_variant[variant_name] = variant_results
        avg = statistics.mean(r.judge_score for r in variant_results if r.judge_score > 0)
        logger.info("Variante %s concluída | Score médio: %.2f/5.0", variant_name, avg)

    winner = report.winner()
    logger.info("Variante vencedora: %s | Score: %.2f", winner, report.summary()[winner]["avg_judge_score"])

    if save_results:
        AB_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "summary": report.summary(),
            "winner": winner,
            "n_queries": report.n_queries,
            "variants_tested": report.variants_tested,
            "details": {
                v: [
                    {
                        "query": r.query,
                        "answer": r.answer,
                        "judge_score": r.judge_score,
                        "latency_ms": r.latency_ms,
                    }
                    for r in results
                ]
                for v, results in report.results_by_variant.items()
            },
        }
        with open(AB_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Resultados A/B test salvos em: %s", AB_RESULTS_PATH)

    return report


def print_ab_summary(report: ABTestReport) -> None:
    """Imprime sumário formatado do A/B test."""
    summary = report.summary()
    winner = report.winner()

    print("\n" + "=" * 60)
    print("A/B TEST — Variantes de Prompt (Domínio: B3 Financeiro)")
    print("=" * 60)
    print(f"{'Variante':<12} {'Score Médio':>12} {'Desvio Padrão':>14} {'Latência (ms)':>14}")
    print("-" * 60)
    for variant, stats in summary.items():
        marker = " ← VENCEDOR" if variant == winner else ""
        print(
            f"{variant:<12} {stats['avg_judge_score']:>12.3f} "
            f"{stats['std_judge_score']:>14.3f} "
            f"{stats['avg_latency_ms']:>14.1f}"
            f"{marker}"
        )
    print("=" * 60)
    print(f"\nPrompt recomendado para produção: Variante {winner}")
    print(f"Template:\n{VARIANTS[winner][:200]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Mock para teste local
    def mock_retriever(query: str) -> tuple[str, list[str]]:
        return "", ["Contexto financeiro simulado para: " + query]

    def mock_llm(prompt: str) -> str:
        return "Resposta simulada baseada no contexto financeiro disponível."

    report = run_ab_test(
        retriever_fn=mock_retriever,
        llm_fn=mock_llm,
        n_samples=3,
        variants={"A": PROMPT_VARIANT_A, "B": PROMPT_VARIANT_B},
    )
    print_ab_summary(report)