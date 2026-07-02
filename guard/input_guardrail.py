import json
from typing import Optional

from groq import Groq
from guardrails import Guard
from guardrails.validator_base import FailResult, PassResult, Validator, register_validator
from presidio_analyzer import AnalyzerEngine

from guard.messages import (
    ADVICE_REQUEST_MESSAGE,
    JAILBREAK_MESSAGE,
    JUDGE_ERROR_MESSAGE,
    OFFENSIVE_MESSAGE,
    PII_MESSAGE,
)
from guard.prompts import COMBINED_CLASSIFIER_PROMPT
from guard.result import GuardrailResult
from utils.helpers import get_logger, load_config

logger = get_logger(__name__)

JUDGE_MODEL = "llama-3.1-8b-instant"

# Restricted to actually privacy-sensitive identifiers. Presidio's unfiltered
# default (all entity types) flags DATE_TIME/LOCATION/PERSON too, which
# false-positives constantly on ordinary stock questions ("last quarter",
# "Waaree Energies").
PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "US_BANK_NUMBER",
    "IP_ADDRESS",
    "IN_PAN",
    "IN_AADHAAR",
    "IN_VEHICLE_REGISTRATION",
    "IN_PASSPORT",
    "IN_VOTER",
]

_analyzer: Optional[AnalyzerEngine] = None
_groq_client: Optional[Groq] = None


def _get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    return _analyzer


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=load_config()["GROQ_API_KEY"])
    return _groq_client


@register_validator(name="presidio-pii", data_type="string")
class PresidioPII(Validator):
    """Custom GuardrailsAI validator wrapping Presidio directly (no Hub account needed)."""

    def validate(self, value, metadata):
        if _get_analyzer().analyze(text=value, language="en", entities=PII_ENTITIES):
            return FailResult(error_message="PII detected in message")
        return PassResult()


def _call_judge_llm(prompt: str) -> dict:
    """Combined LLM classification call: advice-request, jailbreak, offensive.

    Raises on any API/parsing failure — caller is responsible for failing closed.
    """
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


class InputGuardrail:
    def __init__(self):
        self._pii_guard = Guard().use(PresidioPII(on_fail="exception"))

    def validate(
        self,
        user_message: str,
        ticker: str,
        company_name: str,
        chat_history: str = "",
    ) -> GuardrailResult:
        try:
            self._pii_guard.validate(user_message)
        except Exception:
            return GuardrailResult(False, PII_MESSAGE, ["pii"], "PII detected in user message")

        prompt = COMBINED_CLASSIFIER_PROMPT.format(
            user_message=user_message,
            ticker=ticker,
            company_name=company_name,
            chat_history=chat_history or "(none)",
        )
        try:
            classification = _call_judge_llm(prompt)
        except Exception as exc:
            logger.warning("Guardrail judge call failed, failing closed: %s", exc)
            return GuardrailResult(False, JUDGE_ERROR_MESSAGE, ["judge_error"], str(exc))

        reason = classification.get("reason")
        if classification.get("is_offensive"):
            return GuardrailResult(False, OFFENSIVE_MESSAGE, ["offensive"], reason)
        if classification.get("is_jailbreak"):
            return GuardrailResult(False, JAILBREAK_MESSAGE, ["jailbreak"], reason)
        if classification.get("is_advice_request"):
            return GuardrailResult(False, ADVICE_REQUEST_MESSAGE, ["advice_request"], reason)

        return GuardrailResult(True, user_message, [], None)
