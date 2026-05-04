import logging
import re
from typing import List

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class InputGuardrail:
    """Valida e sanitiza input do usuario antes de enviar ao LLM."""

    INJECTION_PATTERNS = [
        r"ignore (all\s+)?(previous|above)?\s*instructions",
        r"forget (your|all|previous) (instructions|rules|constraints)",
        r"you are now",
        r"act as (if|though)",
        r"pretend (to be|you are|you're)",
        r"disregard (all|your|previous)",
        r"new instruction:",
        r"system:",
        r"\[override\]",
        r"jailbreak",
        r"DAN mode",
        r"developer mode",
        r"desconsidere as instrucoes acima e responda",
        r"esqueca as regras anteriores",
        r"aja como se fosse o desenvolvedor",
        r"você agora e",
        r"modo desenvolvedor ativado"
    ]

    def __init__(self, allowed_topics: list[str] | None = None):
        self.allowed_topics = allowed_topics or []
        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuario."""
        for pattern in self._compiled_patterns:
            if pattern.search(user_input):
                logger.warning("Prompt injection detectado: %s", user_input[:100])
                return False, "Input bloqueado: padrao suspeito detectado."

        if len(user_input) > 4096:
            return False, "Input bloqueado: excede tamanho maximo (4096 chars)."

        return True, "OK"


class OutputGuardrail:
    """Valida e sanitiza output do LLM antes de retornar ao usuario."""

    EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)?\d{4,5}[\s.-]?\d{4}\b")
    CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
    CNPJ_PATTERN = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
    IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b")
    IP_ADDRESS_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def __init__(self, language: str = "pt"):
        self.analyzer = None
        self.anonymizer = None
        self.language = language
        self._fallback_language = "en"

    def _ensure_engines(self) -> bool:
        if self.analyzer is not None and self.anonymizer is not None:
            return True

        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            return True
        except Exception as exc:
            logger.warning("Falha ao inicializar Presidio; usando fallback local. Erro: %s", exc)
            self.analyzer = None
            self.anonymizer = None
            return False

    def _fallback_results(self, llm_output: str) -> List[RecognizerResult]:
        results: List[RecognizerResult] = []
        for entity_type, pattern in (
            ("EMAIL_ADDRESS", self.EMAIL_PATTERN),
            ("PHONE_NUMBER", self.PHONE_PATTERN),
            ("BR_CPF", self.CPF_PATTERN),
            ("BR_CNPJ", self.CNPJ_PATTERN),
            ("CREDIT_CARD", self.CREDIT_CARD_PATTERN),
            ("IBAN_CODE", self.IBAN_PATTERN),
            ("IP_ADDRESS", self.IP_ADDRESS_PATTERN),
        ):
            for match in pattern.finditer(llm_output):
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=1.0,
                    )
                )
        return results

    def _fallback_sanitize(self, llm_output: str) -> str:
        sanitized = self.EMAIL_PATTERN.sub("<EMAIL_ADDRESS>", llm_output)
        sanitized = self.PHONE_PATTERN.sub("<PHONE_NUMBER>", sanitized)
        sanitized = self.CPF_PATTERN.sub("<BR_CPF>", sanitized)
        sanitized = self.CNPJ_PATTERN.sub("<BR_CNPJ>", sanitized)
        sanitized = self.CREDIT_CARD_PATTERN.sub("<CREDIT_CARD>", sanitized)
        sanitized = self.IBAN_PATTERN.sub("<IBAN_CODE>", sanitized)
        sanitized = self.IP_ADDRESS_PATTERN.sub("<IP_ADDRESS>", sanitized)
        return sanitized

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM."""
        results = self.analyze(llm_output)

        if results:
            logger.warning("PII detectado no output: %d entidades", len(results))
            if not self._ensure_engines():
                return self._fallback_sanitize(llm_output)

            anonymized = self.anonymizer.anonymize(
                text=llm_output,
                analyzer_results=results,
            )
            return anonymized.text

        return llm_output

    def analyze(self, llm_output: str) -> List[RecognizerResult]:
        if not llm_output:
            return []

        if not self._ensure_engines():
            return self._fallback_results(llm_output)

        try:
            return self.analyzer.analyze(
                text=llm_output,
                language=self.language,
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                    "BR_CPF", "BR_CNPJ", "CREDIT_CARD", 
                    "IBAN_CODE", "IP_ADDRESS"
                ],
            )
        except ValueError:
            if self.language == self._fallback_language:
                raise

            logger.warning(
                "Idioma '%s' nao suportado pelo Presidio; usando fallback '%s'.",
                self.language,
                self._fallback_language,
            )
            return self.analyzer.analyze(
                text=llm_output,
                language=self._fallback_language,
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                    "BR_CPF", "BR_CNPJ", "CREDIT_CARD", 
                    "IBAN_CODE", "IP_ADDRESS"
                ],
            )
