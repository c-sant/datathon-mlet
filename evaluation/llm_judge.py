"""LLM-as-judge para avaliação de respostas sobre dados financeiros da B3.

Critérios obrigatórios (≥ 3, incluindo critério de negócio):
  1. Coerência factual     — dados numéricos corretos (preços, %)
  2. Relevância temática   — responde sobre o ativo/índice perguntado
  3. Critério de negócio   — adequação para decisão de investimento
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JUDGE_RESULTS_PATH = Path("data/golden_set/llm_judge_results.json")

JUDGE_SYSTEM_PROMPT = """Você é um especialista em mercado financeiro brasileiro e avaliador de sistemas de IA.
Sua tarefa é avaliar respostas de um sistema RAG que responde perguntas sobre ações e índices da B3 (Bovespa).

Você deve avaliar a resposta em 4 critérios, dando uma nota de 1 a 5 para cada um:

CRITÉRIO 1 — Coerência Factual (financial_accuracy):
- 5: Todos os dados numéricos (preços, percentuais, datas) são corretos e precisos
- 4: Dados corretos com pequenas imprecisões não críticas
- 3: Dados majoritariamente corretos, mas com 1 erro relevante
- 2: Múltiplos erros numéricos ou confusão entre ativos
- 1: Dados completamente incorretos ou inventados (alucinação)

CRITÉRIO 2 — Relevância Temática (topic_relevance):
- 5: Responde exatamente o que foi perguntado sobre o ativo/índice
- 4: Responde à pergunta com pequenos desvios
- 3: Parcialmente relevante, mas mistura informações não solicitadas
- 2: Resposta tangencial, não aborda o ponto principal
- 1: Completamente fora do tema

CRITÉRIO 3 — Critério de Negócio: Utilidade para Decisão de Investimento (investment_utility):
- 5: Resposta clara e acionável, permite decisão de investimento fundamentada
- 4: Informação útil com ressalvas menores
- 3: Informação parcialmente útil, falta contexto para decisão
- 2: Informação insuficiente ou ambígua para tomada de decisão
- 1: Resposta pode induzir a erro em decisão de investimento

CRITÉRIO 4 — Clareza e Completude (clarity_completeness):
- 5: Resposta clara, bem estruturada e completa
- 4: Clara e quase completa
- 3: Compreensível mas faltam informações importantes
- 2: Confusa ou incompleta de forma relevante
- 1: Incompreensível ou trivialmente incompleta

Responda APENAS com um JSON válido no seguinte formato:
{
  "financial_accuracy": <1-5>,
  "topic_relevance": <1-5>,
  "investment_utility": <1-5>,
  "clarity_completeness": <1-5>,
  "overall_score": <média das 4 notas, 1 casa decimal>,
  "justification": "<justificativa em 2-3 frases>",
  "critical_issues": "<liste problemas críticos ou 'Nenhum'>"
}"""


@dataclass
class JudgeResult:
    """Resultado da avaliação do LLM-as-judge."""
    query: str
    answer: str
    ground_truth: str
    financial_accuracy: int
    topic_relevance: int
    investment_utility: int
    clarity_completeness: int
    overall_score: float
    justification: str
    critical_issues: str

    def passed(self, threshold: float = 3.5) -> bool:
        """Retorna True se a resposta passou no threshold mínimo."""
        return self.overall_score >= threshold


def evaluate_with_llm_judge(
    query: str,
    answer: str,
    ground_truth: str,
    contexts: Optional[list[str]] = None,
    model: str = "claude-sonnet-4-20250514",
) -> JudgeResult:
    """Avalia uma resposta usando LLM-as-judge com critérios financeiros.

    Args:
        query: Pergunta original do usuário sobre ações/índices B3.
        answer: Resposta gerada pelo pipeline RAG.
        ground_truth: Resposta esperada do golden set.
        contexts: Contextos recuperados pelo RAG (opcional).
        model: Modelo LLM usado como juiz.

    Returns:
        JudgeResult com notas e justificativa por critério.
    """
    import anthropic

    client = anthropic.Anthropic()

    context_str = ""
    if contexts:
        context_str = "\nContextos recuperados pelo RAG:\n" + "\n".join(
            f"  [{i+1}] {ctx}" for i, ctx in enumerate(contexts)
        )

    user_message = f"""Avalie a seguinte resposta de um sistema RAG financeiro sobre dados da B3:

PERGUNTA: {query}

RESPOSTA DO SISTEMA: {answer}

RESPOSTA ESPERADA (ground truth): {ground_truth}
{context_str}

Avalie os 4 critérios e retorne APENAS o JSON."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)

        result = JudgeResult(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            financial_accuracy=scores["financial_accuracy"],
            topic_relevance=scores["topic_relevance"],
            investment_utility=scores["investment_utility"],
            clarity_completeness=scores["clarity_completeness"],
            overall_score=scores["overall_score"],
            justification=scores["justification"],
            critical_issues=scores["critical_issues"],
        )

        status = "APROVADO" if result.passed() else "REPROVADO"
        logger.info(
            "Judge [%s] %.1f/5.0 — %s",
            status, result.overall_score, query[:60],
        )
        return result

    except Exception as e:
        logger.error("Erro no LLM-as-judge para query '%s': %s", query[:60], e)
        return JudgeResult(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            financial_accuracy=0,
            topic_relevance=0,
            investment_utility=0,
            clarity_completeness=0,
            overall_score=0.0,
            justification=f"Erro na avaliação: {e}",
            critical_issues="Falha no judge",
        )


def run_judge_on_golden_set(
    rag_fn,
    golden_set_path: Path = Path("data/golden_set/golden_set.json"),
    save_results: bool = True,
    pass_threshold: float = 3.5,
) -> dict:
    """Executa o LLM-as-judge em todo o golden set.

    Args:
        rag_fn: Função RAG que recebe query e retorna (answer, contexts).
        golden_set_path: Caminho para o golden set.
        save_results: Se True, salva resultados em disco.
        pass_threshold: Nota mínima para aprovação (default: 3.5/5.0).

    Returns:
        Dicionário com métricas agregadas e detalhes por query.
    """
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    results = []
    for i, item in enumerate(golden_set):
        logger.info("Avaliando query %d/%d...", i + 1, len(golden_set))
        answer, contexts = rag_fn(item["query"])
        result = evaluate_with_llm_judge(
            query=item["query"],
            answer=answer,
            ground_truth=item["expected_answer"],
            contexts=contexts,
        )
        results.append(result)

    scores_list = [r for r in results if r.overall_score > 0]

    summary = {
        "n_evaluated": len(golden_set),
        "n_passed": sum(1 for r in results if r.passed(pass_threshold)),
        "pass_rate": round(sum(1 for r in results if r.passed(pass_threshold)) / len(results), 3),
        "avg_financial_accuracy": round(
            sum(r.financial_accuracy for r in scores_list) / max(len(scores_list), 1), 2
        ),
        "avg_topic_relevance": round(
            sum(r.topic_relevance for r in scores_list) / max(len(scores_list), 1), 2
        ),
        "avg_investment_utility": round(
            sum(r.investment_utility for r in scores_list) / max(len(scores_list), 1), 2
        ),
        "avg_clarity_completeness": round(
            sum(r.clarity_completeness for r in scores_list) / max(len(scores_list), 1), 2
        ),
        "avg_overall_score": round(
            sum(r.overall_score for r in scores_list) / max(len(scores_list), 1), 2
        ),
        "details": [
            {
                "query": r.query,
                "overall_score": r.overall_score,
                "financial_accuracy": r.financial_accuracy,
                "topic_relevance": r.topic_relevance,
                "investment_utility": r.investment_utility,
                "clarity_completeness": r.clarity_completeness,
                "passed": r.passed(pass_threshold),
                "justification": r.justification,
                "critical_issues": r.critical_issues,
            }
            for r in results
        ],
    }

    logger.info(
        "Judge concluído: %d/%d aprovados (%.1f%%) | Score médio: %.2f/5.0",
        summary["n_passed"], summary["n_evaluated"],
        summary["pass_rate"] * 100, summary["avg_overall_score"],
    )

    if save_results:
        JUDGE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JUDGE_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Resultados do judge salvos em: %s", JUDGE_RESULTS_PATH)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Mock para teste local sem API de RAG
    def mock_rag_fn(query: str) -> tuple[str, list[str]]:
        golden_set_path = Path("data/golden_set/golden_set.json")
        with open(golden_set_path) as f:
            gs = json.load(f)
        for item in gs:
            if item["query"] == query:
                return item["expected_answer"], item["contexts"]
        return "Informação não disponível.", []

    # Testar com 1 query
    result = evaluate_with_llm_judge(
        query="Qual foi o preço de fechamento do PETR4 em 2024-01-15?",
        answer="O PETR4 fechou em R$ 37,42 em 15 de janeiro de 2024.",
        ground_truth="O preço de fechamento do PETR4 em 2024-01-15 foi de R$ 37,42.",
        contexts=["PETR4 registrou fechamento em R$ 37,42 no pregão de 15 de janeiro de 2024."],
    )
    print(f"\nTeste individual:")
    print(f"  Overall: {result.overall_score}/5.0 | Passou: {result.passed()}")
    print(f"  Justificativa: {result.justification}")