"""Session-only conversation memory: per-turn 2-3 line summaries + a
verbatim window of the last N ok turns. Nothing persisted to disk."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from groq import Groq

from utils.helpers import get_logger, load_config

logger = get_logger(__name__)

SUMMARY_MODEL = "llama-3.1-8b-instant"
SUMMARY_PROMPT = (
    "Summarize this chatbot exchange in 2-3 short lines. Keep tickers, "
    "numbers and dates exact. No preamble.\n\n"
    "User: {question}\n\nAssistant: {answer}"
)

_groq_client: Optional[Groq] = None


def _summary_llm_call(prompt: str) -> str:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=load_config()["GROQ_API_KEY"])
    response = _groq_client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def summarize_turn(question: str, answer: str) -> str:
    """2-3 line turn summary; truncation fallback so a summarizer failure
    never blocks the turn."""
    try:
        return _summary_llm_call(
            SUMMARY_PROMPT.format(question=question, answer=answer))
    except Exception as exc:
        logger.warning("Turn summarizer failed, using fallback: %s", exc)
        return f"Q: {question[:150]} | A: {answer[:150]}"


@dataclass
class Turn:
    user: str
    assistant: str
    summary: str
    status: str = "ok"  # ok | blocked | error


class ConversationMemory:
    def __init__(self, verbatim_turns: int = 6):
        self.turns: List[Turn] = []
        self._n = verbatim_turns

    def add_turn(self, user: str, assistant: str, summary: str,
                 status: str = "ok") -> None:
        self.turns.append(Turn(user, assistant, summary, status))

    def context_messages(self) -> List[Tuple[str, str]]:
        """Older/non-ok turns as one system summary note + last-N ok turns
        verbatim. Blocked/error turns never appear verbatim."""
        window = self.turns[-self._n:]
        verbatim = [t for t in window if t.status == "ok"]
        summarized = [t for t in self.turns if t not in verbatim]

        messages: List[Tuple[str, str]] = []
        if summarized:
            note = "Summary of earlier conversation turns:\n" + "\n".join(
                f"- {t.summary}" for t in summarized)
            messages.append(("system", note))
        for t in verbatim:
            messages.append(("user", t.user))
            messages.append(("assistant", t.assistant))
        return messages

    def history_text(self) -> str:
        """Compact bullet list of recent turn summaries for the guardrail judge."""
        return "\n".join(f"- {t.summary}" for t in self.turns[-10:])
