"""Telemetria do agente RAG financeiro — Prometheus + Langfuse.

Métricas operacionais (Prometheus):
  - Latência de resposta
  - Throughput de requests
  - Taxa de erros
  - Score de qualidade RAGAS

Métricas de qualidade LLM (Langfuse):
  - Faithfulness por ticker
  - Relevância por categoria de query
  - Score do LLM-as-judge
"""
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
)

logger = logging.getLogger(__name__)

# Métricas Prometheus

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latência das requisições ao pipeline RAG financeiro",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    labelnames=["ticker", "query_category"],
)

REQUEST_COUNT = Counter(
    "rag_request_total",
    "Total de requisições ao pipeline RAG",
    labelnames=["ticker", "query_category", "status"],
)

ERROR_COUNT = Counter(
    "rag_error_total",
    "Total de erros no pipeline RAG",
    labelnames=["error_type"],
)

LLM_TOKENS_USED = Counter(
    "llm_tokens_total",
    "Total de tokens consumidos pelo LLM",
    labelnames=["model", "direction"],  # direction: input | output
)

RAGAS_FAITHFULNESS = Gauge(
    "ragas_faithfulness_score",
    "Score de faithfulness RAGAS (última avaliação)",
    labelnames=["ticker"],
)

RAGAS_RELEVANCY = Gauge(
    "ragas_answer_relevancy_score",
    "Score de answer_relevancy RAGAS (última avaliação)",
    labelnames=["ticker"],
)

RAGAS_CONTEXT_PRECISION = Gauge(
    "ragas_context_precision_score",
    "Score de context_precision RAGAS (última avaliação)",
    labelnames=["ticker"],
)

RAGAS_CONTEXT_RECALL = Gauge(
    "ragas_context_recall_score",
    "Score de context_recall RAGAS (última avaliação)",
    labelnames=["ticker"],
)

LLM_JUDGE_SCORE = Gauge(
    "llm_judge_score",
    "Score do LLM-as-judge (última avaliação)",
    labelnames=["ticker", "criterion"],
)

DRIFT_PSI_SCORE = Gauge(
    "drift_psi_score",
    "PSI score de drift por feature financeira",
    labelnames=["ticker", "feature"],
)

DRIFT_STATUS = Gauge(
    "drift_status",
    "Status de drift (0=stable, 1=warning, 2=critical)",
    labelnames=["ticker"],
)

ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Número de requisições ativas no momento",
)

RETRIEVAL_LATENCY = Summary(
    "rag_retrieval_latency_seconds",
    "Latência da etapa de recuperação (retriever)",
    labelnames=["ticker"],
)

GENERATION_LATENCY = Summary(
    "rag_generation_latency_seconds",
    "Latência da etapa de geração (LLM)",
    labelnames=["ticker"],
)

DRIFT_STATUS_MAP = {"stable": 0, "warning": 1, "critical": 2}


#Decoradores e Context Managers

@contextmanager
def track_request(ticker: str = "unknown", category: str = "general"):
    """Context manager para rastrear métricas de uma requisição RAG.

    Uso:
        with track_request(ticker="PETR4", category="price_query"):
            answer = rag_pipeline.query(user_query)
    """
    ACTIVE_REQUESTS.inc()
    start = time.time()
    status = "success"
    try:
        yield
    except Exception as e:
        status = "error"
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        duration = time.time() - start
        REQUEST_LATENCY.labels(ticker=ticker, query_category=category).observe(duration)
        REQUEST_COUNT.labels(ticker=ticker, query_category=category, status=status).inc()
        ACTIVE_REQUESTS.dec()
        logger.debug(
            "Request rastreada: ticker=%s cat=%s status=%s lat=%.3fs",
            ticker, category, status, duration,
        )


def track_rag_fn(ticker: str = "unknown", category: str = "general"):
    """Decorador para funções RAG — adiciona tracking automático de métricas.

    Uso:
        @track_rag_fn(ticker="VALE3", category="technical_indicator")
        def get_moving_average(query: str) -> tuple[str, list[str]]:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with track_request(ticker=ticker, category=category):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# Langfuse Integration

class FinancialRAGTracer:
    """Integração com Langfuse para telemetria de qualidade LLM.

    Rastreia traces completos de requests incluindo retrieval,
    generation e scores de qualidade para o domínio financeiro B3.
    """

    def __init__(self, public_key: Optional[str] = None, secret_key: Optional[str] = None):
        """Inicializa o tracer Langfuse.

        Args:
            public_key: Chave pública Langfuse (ou via LANGFUSE_PUBLIC_KEY env).
            secret_key: Chave secreta Langfuse (ou via LANGFUSE_SECRET_KEY env).
        """
        try:
            from langfuse import Langfuse
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
            )
            self._enabled = True
            logger.info("Langfuse inicializado com sucesso")
        except Exception as e:
            logger.warning("Langfuse não disponível: %s | Continuando sem telemetria LLM", e)
            self._enabled = False
            self.langfuse = None

    def trace_rag_query(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ticker: str,
        category: str,
        latency_ms: float,
        ragas_scores: Optional[dict[str, float]] = None,
        judge_score: Optional[float] = None,
    ) -> Optional[str]:
        """Registra um trace completo de query RAG no Langfuse.

        Args:
            query: Pergunta do usuário sobre ações B3.
            answer: Resposta gerada pelo pipeline.
            contexts: Contextos recuperados pelo retriever.
            ticker: Ativo financeiro relacionado à query.
            category: Categoria da query (ex: "price_query", "comparison").
            latency_ms: Latência total em milissegundos.
            ragas_scores: Scores RAGAS (opcional).
            judge_score: Score do LLM-as-judge (opcional).

        Returns:
            trace_id do registro Langfuse, ou None se desabilitado.
        """
        if not self._enabled:
            return None

        try:
            trace = self.langfuse.trace(
                name="rag-financial-b3",
                input=query,
                output=answer,
                metadata={
                    "ticker": ticker,
                    "query_category": category,
                    "latency_ms": latency_ms,
                    "n_contexts": len(contexts),
                    "ragas_scores": ragas_scores or {},
                    "judge_score": judge_score,
                    "domain": "bovespa_b3",
                },
                tags=[ticker, category, "datathon-fase05"],
            )

            # Span de retrieval
            trace.span(
                name="retrieval",
                input=query,
                output={"contexts": contexts, "n_retrieved": len(contexts)},
                metadata={"ticker": ticker},
            )

            # Span de generation
            trace.span(
                name="generation",
                input={"query": query, "contexts": contexts},
                output=answer,
                metadata={"ticker": ticker},
            )

            # Scores de qualidade como scores no trace
            if ragas_scores:
                for metric, value in ragas_scores.items():
                    trace.score(
                        name=f"ragas_{metric}",
                        value=value,
                        comment=f"RAGAS {metric} para ticker {ticker}",
                    )

            if judge_score is not None:
                trace.score(
                    name="llm_judge_overall",
                    value=judge_score / 5.0,  # normalizar para [0, 1]
                    comment=f"LLM-as-judge score para query financeira — {ticker}",
                )

            logger.debug("Trace Langfuse registrado: %s", trace.id)
            return trace.id

        except Exception as e:
            logger.warning("Falha ao registrar trace Langfuse: %s", e)
            return None

    def flush(self) -> None:
        """Força envio de traces pendentes ao Langfuse."""
        if self._enabled and self.langfuse:
            self.langfuse.flush()


# Helpers de Update de Métricas

def update_ragas_metrics(ticker: str, scores: dict[str, float]) -> None:
    """Atualiza Gauges Prometheus com scores RAGAS mais recentes.

    Args:
        ticker: Código do ativo (ex: "PETR4", "IBOV").
        scores: Dicionário com métricas RAGAS.
    """
    metric_map = {
        "faithfulness": RAGAS_FAITHFULNESS,
        "answer_relevancy": RAGAS_RELEVANCY,
        "context_precision": RAGAS_CONTEXT_PRECISION,
        "context_recall": RAGAS_CONTEXT_RECALL,
    }
    for key, gauge in metric_map.items():
        if key in scores:
            gauge.labels(ticker=ticker).set(scores[key])

    logger.info("Métricas RAGAS atualizadas para %s: %s", ticker, scores)


def update_drift_metrics(ticker: str, drift_result) -> None:
    """Atualiza Gauges Prometheus com resultados de drift.

    Args:
        ticker: Código do ativo monitorado.
        drift_result: DriftResult do módulo drift.py.
    """
    status_value = DRIFT_STATUS_MAP.get(drift_result.status, 0)
    DRIFT_STATUS.labels(ticker=ticker).set(status_value)

    for feature, psi in drift_result.psi_scores.items():
        DRIFT_PSI_SCORE.labels(ticker=ticker, feature=feature).set(psi)

    logger.info(
        "Métricas de drift atualizadas para %s: status=%s",
        ticker, drift_result.status,
    )


def update_judge_metrics(ticker: str, judge_result) -> None:
    """Atualiza Gauges Prometheus com scores do LLM-as-judge.

    Args:
        ticker: Código do ativo avaliado.
        judge_result: JudgeResult do módulo llm_judge.py.
    """
    criteria = {
        "financial_accuracy": judge_result.financial_accuracy,
        "topic_relevance": judge_result.topic_relevance,
        "investment_utility": judge_result.investment_utility,
        "clarity_completeness": judge_result.clarity_completeness,
        "overall": judge_result.overall_score,
    }
    for criterion, value in criteria.items():
        LLM_JUDGE_SCORE.labels(ticker=ticker, criterion=criterion).set(value)


def start_metrics_server(port: int = 8001) -> None:
    """Inicia servidor HTTP de métricas Prometheus.

    Args:
        port: Porta do servidor (default: 8001).
              Grafana fará scraping nesta porta.
    """
    start_http_server(port)
    logger.info("Servidor de métricas Prometheus iniciado na porta %d", port)
    logger.info("Acesse: http://localhost:%d/metrics", port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    start_metrics_server(port=8001)

    logger.info("Simulando requests RAG para geração de métricas...")
    import random
    import time

    tickers = ["PETR4", "VALE3", "ITUB4", "BBAS3", "IBOV"]
    categories = ["price_query", "performance", "technical_indicator", "comparison", "factual"]

    for i in range(10):
        ticker = random.choice(tickers)
        category = random.choice(categories)

        with track_request(ticker=ticker, category=category):
            time.sleep(random.uniform(0.1, 2.0))

        update_ragas_metrics(ticker, {
            "faithfulness": random.uniform(0.7, 0.95),
            "answer_relevancy": random.uniform(0.75, 0.98),
            "context_precision": random.uniform(0.65, 0.90),
            "context_recall": random.uniform(0.60, 0.88),
        })

        logger.info("Request %d simulada: %s / %s", i + 1, ticker, category)

    logger.info("Métricas disponíveis em http://localhost:8001/metrics")
    logger.info("Pressione Ctrl+C para encerrar.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Servidor encerrado.")