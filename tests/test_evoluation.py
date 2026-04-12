"""Cobre:
  - Golden set (schema, contagem, integridade)
  - RAGAS evaluation (mock sem API)
  - LLM-as-judge (parsing, critérios, threshold)
  - Drift detection (PSI, thresholds, trigger)
  - A/B test (variantes, comparação)
  - Métricas Prometheus (tipos, labels)
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

#Fixtures

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")


@pytest.fixture
def golden_set() -> list[dict]:
    """Carrega o golden set do domínio financeiro B3."""
    if not GOLDEN_SET_PATH.exists():
        pytest.skip("Golden set não encontrado — execute a geração primeiro")
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_financial_df() -> pd.DataFrame:
    """DataFrame OHLCV sintético para testes de features e drift."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 252
    close = 37.5 * (1 + rng.normal(0.0003, 0.018, n)).cumprod()
    return pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.005, n)),
        "high":   close * (1 + rng.uniform(0.002, 0.015, n)),
        "low":    close * (1 - rng.uniform(0.002, 0.015, n)),
        "close":  close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        "ticker": "PETR4",
    })


@pytest.fixture
def mock_rag_fn():
    """Mock do pipeline RAG que retorna respostas do golden set."""
    def _fn(query: str) -> tuple[str, list[str]]:
        if not GOLDEN_SET_PATH.exists():
            return "Resposta simulada.", ["Contexto financeiro simulado."]
        with open(GOLDEN_SET_PATH) as f:
            gs = json.load(f)
        for item in gs:
            if item["query"] == query:
                return item["expected_answer"], item["contexts"]
        return "Informação não encontrada.", []
    return _fn


# Golden Set

class TestGoldenSet:
    """Testes de integridade do golden set financeiro."""

    def test_minimum_20_pairs(self, golden_set):
        """Golden set deve ter no mínimo 20 pares (critério do datathon)."""
        assert len(golden_set) >= 20, (
            f"Golden set tem apenas {len(golden_set)} pares. Mínimo: 20"
        )

    def test_required_fields_present(self, golden_set):
        """Todos os itens devem ter os campos obrigatórios."""
        required = {"query", "expected_answer", "contexts"}
        for i, item in enumerate(golden_set):
            missing = required - set(item.keys())
            assert not missing, f"Item {i} faltando campos: {missing}"

    def test_no_empty_queries(self, golden_set):
        """Nenhuma query pode estar vazia."""
        for i, item in enumerate(golden_set):
            assert item["query"].strip(), f"Query vazia no item {i}"

    def test_no_empty_expected_answers(self, golden_set):
        """Nenhuma resposta esperada pode estar vazia."""
        for i, item in enumerate(golden_set):
            assert item["expected_answer"].strip(), (
                f"expected_answer vazio no item {i}"
            )

    def test_contexts_are_lists(self, golden_set):
        """Contexts deve ser uma lista (pode ser vazia em casos extremos)."""
        for i, item in enumerate(golden_set):
            assert isinstance(item["contexts"], list), (
                f"Contexts no item {i} não é lista: {type(item['contexts'])}"
            )

    def test_minimum_one_context_per_item(self, golden_set):
        """Cada item deve ter pelo menos 1 contexto."""
        for i, item in enumerate(golden_set):
            assert len(item["contexts"]) >= 1, (
                f"Item {i} sem contextos: '{item['query'][:50]}'"
            )

    def test_financial_domain_coverage(self, golden_set):
        """Golden set deve cobrir diferentes categorias do domínio financeiro."""
        categories = [item.get("metadata", {}).get("category", "") for item in golden_set]
        unique_categories = set(c for c in categories if c)
        assert len(unique_categories) >= 3, (
            f"Golden set cobre apenas {len(unique_categories)} categorias. "
            f"Mínimo esperado: 3. Categorias: {unique_categories}"
        )

    def test_tickers_referenced(self, golden_set):
        """Golden set deve referenciar múltiplos ativos da B3."""
        tickers = [item.get("metadata", {}).get("ticker", "") for item in golden_set]
        unique_tickers = set(t for t in tickers if t and t not in ("MULTI", "GENERIC", "SECTOR", "MACRO", "CDI"))
        assert len(unique_tickers) >= 3, (
            f"Golden set cobre apenas {len(unique_tickers)} tickers. Esperado ≥ 3."
        )


#RAGAS Evaluation

class TestRAGASEvaluation:
    """Testes do módulo de avaliação RAGAS."""

    def test_load_golden_set_exists(self, tmp_path):
        """load_golden_set deve levantar FileNotFoundError se arquivo ausente."""
        from evaluation.ragas_eval import load_golden_set
        with pytest.raises(FileNotFoundError):
            load_golden_set(tmp_path / "inexistente.json")

    def test_load_golden_set_minimum_pairs(self, tmp_path):
        """load_golden_set deve levantar ValueError se menos de 20 pares."""
        from evaluation.ragas_eval import load_golden_set
        small_set = [{"query": "q", "expected_answer": "a", "contexts": ["c"]}] * 5
        path = tmp_path / "small.json"
        with open(path, "w") as f:
            json.dump(small_set, f)
        with pytest.raises(ValueError, match="≥ 20"):
            load_golden_set(path)

    def test_evaluate_returns_4_metrics(self, tmp_path, mock_rag_fn):
        """evaluate_rag_pipeline deve retornar exatamente 4 métricas RAGAS."""
        # Cria golden set mínimo para teste
        gs = [
            {
                "query": f"Pergunta {i} sobre PETR4?",
                "expected_answer": f"Resposta {i} sobre preço da ação.",
                "contexts": [f"Contexto financeiro {i} sobre PETR4."],
            }
            for i in range(20)
        ]
        gs_path = tmp_path / "gs.json"
        with open(gs_path, "w") as f:
            json.dump(gs, f)

        with patch("evaluation.ragas_eval.evaluate") as mock_eval:
            mock_eval.return_value = {
                "faithfulness": 0.88,
                "answer_relevancy": 0.85,
                "context_precision": 0.79,
                "context_recall": 0.82,
            }

            from evaluation.ragas_eval import evaluate_rag_pipeline
            metrics = evaluate_rag_pipeline(
                rag_fn=lambda q: ("resp", ["ctx"]),
                golden_set_path=gs_path,
                save_results=False,
            )

        required_metrics = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        assert required_metrics.issubset(set(metrics.keys())), (
            f"Métricas faltando: {required_metrics - set(metrics.keys())}"
        )

    def test_metrics_in_valid_range(self, tmp_path):
        """Todas as métricas RAGAS devem estar no intervalo [0, 1]."""
        with patch("evaluation.ragas_eval.load_golden_set") as mock_load, \
             patch("evaluation.ragas_eval.evaluate") as mock_eval:
            mock_load.return_value = [
                {"query": "q", "expected_answer": "a", "contexts": ["c"]}
            ] * 20
            mock_eval.return_value = {
                "faithfulness": 0.92,
                "answer_relevancy": 0.87,
                "context_precision": 0.81,
                "context_recall": 0.78,
            }

            from evaluation.ragas_eval import evaluate_rag_pipeline
            metrics = evaluate_rag_pipeline(
                rag_fn=lambda q: ("resp", ["ctx"]),
                save_results=False,
            )

        for metric, value in metrics.items():
            if metric != "ragas_score_mean":
                assert 0.0 <= value <= 1.0, (
                    f"Métrica {metric}={value} fora do intervalo [0, 1]"
                )


# LLM-as-judge

class TestLLMJudge:
    """Testes do módulo LLM-as-judge."""

    def test_judge_result_has_4_criteria(self):
        """JudgeResult deve ter os 4 critérios obrigatórios."""
        from evaluation.llm_judge import JudgeResult
        result = JudgeResult(
            query="Qual o preço do PETR4?",
            answer="R$ 37,42",
            ground_truth="R$ 37,42",
            financial_accuracy=5,
            topic_relevance=5,
            investment_utility=4,
            clarity_completeness=5,
            overall_score=4.75,
            justification="Resposta precisa e relevante.",
            critical_issues="Nenhum",
        )
        assert result.financial_accuracy == 5
        assert result.topic_relevance == 5
        assert result.investment_utility == 4
        assert result.clarity_completeness == 5

    def test_judge_result_passed_threshold(self):
        """JudgeResult.passed() deve respeitar o threshold configurado."""
        from evaluation.llm_judge import JudgeResult
        good = JudgeResult("q", "a", "gt", 4, 4, 4, 4, 4.0, "ok", "Nenhum")
        bad = JudgeResult("q", "a", "gt", 2, 2, 2, 2, 2.0, "ruim", "Vários")

        assert good.passed(threshold=3.5) is True
        assert bad.passed(threshold=3.5) is False

    @patch("evaluation.llm_judge.anthropic")
    def test_evaluate_parses_json_response(self, mock_anthropic):
        """evaluate_with_llm_judge deve parsear resposta JSON corretamente."""
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(
            text=json.dumps({
                "financial_accuracy": 5,
                "topic_relevance": 4,
                "investment_utility": 4,
                "clarity_completeness": 5,
                "overall_score": 4.5,
                "justification": "Resposta financeira precisa.",
                "critical_issues": "Nenhum",
            })
        )]

        from evaluation.llm_judge import evaluate_with_llm_judge
        result = evaluate_with_llm_judge(
            query="Qual o P/L do IBOV?",
            answer="O P/L médio do IBOV é 8.5x.",
            ground_truth="O P/L do Ibovespa é aproximadamente 8.5x.",
        )

        assert result.overall_score == 4.5
        assert result.financial_accuracy == 5
        assert result.passed(threshold=3.5) is True

    def test_business_criterion_present(self):
        """investment_utility é o critério de negócio obrigatório pelo datathon."""
        from evaluation.llm_judge import JUDGE_SYSTEM_PROMPT
        assert "investment_utility" in JUDGE_SYSTEM_PROMPT, (
            "Critério de negócio 'investment_utility' deve estar no system prompt do judge"
        )
        assert "decisão de investimento" in JUDGE_SYSTEM_PROMPT.lower() or \
               "investimento" in JUDGE_SYSTEM_PROMPT.lower(), (
            "System prompt deve mencionar critério de negócio financeiro"
        )


# Drift Detection

class TestDriftDetection:
    """Testes do módulo de drift detection financeiro."""

    def test_compute_financial_features_output_shape(self, sample_financial_df):
        """compute_financial_features deve retornar todas as features esperadas."""
        from src.monitoring.drift import FINANCIAL_FEATURES, compute_financial_features
        features = compute_financial_features(sample_financial_df)

        for col in FINANCIAL_FEATURES:
            assert col in features.columns, f"Feature '{col}' ausente no output"

    def test_compute_financial_features_no_nulls(self, sample_financial_df):
        """Features financeiras não podem ter null após computação."""
        from src.monitoring.drift import compute_financial_features
        features = compute_financial_features(sample_financial_df)
        null_counts = features.isnull().sum()
        assert null_counts.sum() == 0, (
            f"Nulls encontrados nas features: {null_counts[null_counts > 0].to_dict()}"
        )

    def test_compute_financial_features_missing_column(self, sample_financial_df):
        """compute_financial_features deve levantar ValueError se coluna OHLCV ausente."""
        from src.monitoring.drift import compute_financial_features
        df_sem_close = sample_financial_df.drop(columns=["close"])
        with pytest.raises(ValueError, match="Colunas ausentes"):
            compute_financial_features(df_sem_close)

    def test_psi_thresholds_constants(self):
        """Thresholds PSI devem estar nos valores documentados no datathon."""
        from src.monitoring.drift import PSI_CRITICAL, PSI_WARNING
        assert PSI_WARNING == 0.10, f"PSI_WARNING deve ser 0.10, mas é {PSI_WARNING}"
        assert PSI_CRITICAL == 0.20, f"PSI_CRITICAL deve ser 0.20, mas é {PSI_CRITICAL}"

    def test_synthetic_data_generation(self):
        """create_synthetic_financial_data deve gerar dados válidos para B3."""
        from src.monitoring.drift import create_synthetic_financial_data
        df = create_synthetic_financial_data(n_samples=60, drift_factor=0.3, ticker="PETR4")

        assert len(df) == 60
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
        assert (df["close"] > 0).all(), "Preços de fechamento devem ser positivos"
        assert (df["volume"] > 0).all(), "Volume deve ser positivo"
        assert (df["high"] >= df["low"]).all(), "Máxima deve ser ≥ mínima"

    @patch("src.monitoring.drift.Report")
    def test_run_drift_detection_returns_result(self, mock_report_cls, sample_financial_df, tmp_path):
        """run_drift_detection deve retornar DriftResult com status válido."""
        # Mock do Evidently Report
        mock_report = MagicMock()
        mock_report.as_dict.return_value = {
            "metrics": [
                {"result": {"dataset_drift": False, "share_of_drifted_columns": 0.14}},
                {"result": {"target_drift": False}},
                *[
                    {
                        "metric": "ColumnDriftMetric",
                        "result": {"column_name": f"feature_{i}", "drift_score": 0.05},
                    }
                    for i in range(7)
                ],
            ]
        }
        mock_report_cls.return_value = mock_report

        from src.monitoring.drift import FINANCIAL_FEATURES, run_drift_detection

        cur_df = sample_financial_df.head(60)
        result = run_drift_detection(
            reference_df=sample_financial_df,
            current_df=cur_df,
            ticker="PETR4",
            log_to_mlflow=False,
            save_report=False,
        )

        assert result.status in ("stable", "warning", "critical")
        assert isinstance(result.trigger_retrain, bool)
        assert isinstance(result.psi_scores, dict)


#Métricas Prometheus

class TestPrometheusMetrics:
    """Testes das métricas Prometheus do pipeline RAG financeiro."""

    def test_metrics_are_importable(self):
        """Todas as métricas Prometheus devem ser importáveis."""
        from src.monitoring.metrics import (
            ACTIVE_REQUESTS,
            DRIFT_PSI_SCORE,
            DRIFT_STATUS,
            ERROR_COUNT,
            LLM_JUDGE_SCORE,
            LLM_TOKENS_USED,
            RAGAS_FAITHFULNESS,
            REQUEST_COUNT,
            REQUEST_LATENCY,
        )
        assert REQUEST_LATENCY is not None
        assert REQUEST_COUNT is not None
        assert RAGAS_FAITHFULNESS is not None
        assert DRIFT_PSI_SCORE is not None

    def test_track_request_increments_counter(self):
        """track_request deve incrementar REQUEST_COUNT ao finalizar."""
        from src.monitoring.metrics import REQUEST_COUNT, track_request

        before = REQUEST_COUNT.labels(ticker="VALE3", query_category="test", status="success")._value.get()

        with track_request(ticker="VALE3", category="test"):
            time.sleep(0.01)

        after = REQUEST_COUNT.labels(ticker="VALE3", query_category="test", status="success")._value.get()
        assert after > before

    def test_track_request_records_error_on_exception(self):
        """track_request deve registrar erro em ERROR_COUNT se exceção ocorrer."""
        from src.monitoring.metrics import ERROR_COUNT, track_request

        before = ERROR_COUNT.labels(error_type="ValueError")._value.get()

        with pytest.raises(ValueError):
            with track_request(ticker="IBOV", category="test"):
                raise ValueError("Erro simulado no pipeline RAG")

        after = ERROR_COUNT.labels(error_type="ValueError")._value.get()
        assert after > before

    def test_update_ragas_metrics(self):
        """update_ragas_metrics deve atualizar Gauges corretamente."""
        from src.monitoring.metrics import RAGAS_FAITHFULNESS, update_ragas_metrics

        scores = {
            "faithfulness": 0.91,
            "answer_relevancy": 0.87,
            "context_precision": 0.80,
            "context_recall": 0.75,
        }
        update_ragas_metrics("PETR4", scores)

        assert RAGAS_FAITHFULNESS.labels(ticker="PETR4")._value.get() == 0.91

    def test_drift_status_map_coverage(self):
        """DRIFT_STATUS_MAP deve cobrir todos os estados possíveis de drift."""
        from src.monitoring.metrics import DRIFT_STATUS_MAP
        assert "stable" in DRIFT_STATUS_MAP
        assert "warning" in DRIFT_STATUS_MAP
        assert "critical" in DRIFT_STATUS_MAP
        assert DRIFT_STATUS_MAP["stable"] < DRIFT_STATUS_MAP["warning"] < DRIFT_STATUS_MAP["critical"]


#A/B Test

class TestABTest:
    """Testes do módulo de A/B test de prompts."""

    def test_prompt_variants_exist(self):
        """Deve haver pelo menos 2 variantes de prompt definidas."""
        from evaluation.ab_test_prompts import VARIANTS
        assert len(VARIANTS) >= 2, (
            f"Apenas {len(VARIANTS)} variante(s) definida(s). Mínimo: 2"
        )

    def test_prompt_templates_have_placeholders(self):
        """Todos os templates devem ter {query} e {context}."""
        from evaluation.ab_test_prompts import VARIANTS
        for name, template in VARIANTS.items():
            assert "{query}" in template, f"Variante {name} sem placeholder {{query}}"
            assert "{context}" in template, f"Variante {name} sem placeholder {{context}}"

    def test_ab_report_winner_selection(self):
        """ABTestReport.winner() deve retornar variante com maior score."""
        from evaluation.ab_test_prompts import ABTestReport, VariantResult

        report = ABTestReport(variants_tested=["A", "B"], n_queries=5)
        report.results_by_variant = {
            "A": [VariantResult("A", "q", "a", [], judge_score=3.2) for _ in range(5)],
            "B": [VariantResult("B", "q", "a", [], judge_score=4.5) for _ in range(5)],
        }

        assert report.winner() == "B"

    def test_build_rag_fn_with_prompt(self):
        """build_rag_fn_with_prompt deve formatar query e context corretamente."""
        from evaluation.ab_test_prompts import build_rag_fn_with_prompt

        captured_prompts = []
        def mock_retriever(q): return "", ["contexto B3 simulado"]
        def mock_llm(prompt):
            captured_prompts.append(prompt)
            return "resposta simulada"

        template = "Context: {context}\nQuery: {query}\nAnswer:"
        fn = build_rag_fn_with_prompt(mock_retriever, template, mock_llm)

        answer, contexts = fn("Qual o preço do VALE3?")

        assert len(captured_prompts) == 1
        assert "Qual o preço do VALE3?" in captured_prompts[0]
        assert "contexto B3 simulado" in captured_prompts[0]
        assert answer == "resposta simulada"