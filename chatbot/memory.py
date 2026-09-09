"""A in-memory conversation history for the chatbot agent, with turn summaries and a context window."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from groq import Groq

from utils.helpers import get_logger, load_config

logger = get_logger(__name__)

SUMMARY_MODEL = "openai/gpt-oss-20b"
SUMMARY_PROMPT = (
    "Summarize this chatbot exchange in 2-3 short lines. Keep tickers, "
    "numbers and dates exact. No preamble.\n\n"
    "User: {question}\n\nAssistant: {answer}"
)

_groq_client: Optional[Groq] = None

# to summarize the turn 
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
    """Summarize a single turn, with fallback to a truncated Q&A if the LLM fails."""
    try:
        return _summary_llm_call(
            SUMMARY_PROMPT.format(question=question, answer=answer))
    except Exception as exc:
        logger.warning("Turn summarizer failed, using fallback: %s", exc)
        return f"Q: {question[:150]} | A: {answer[:150]}"

# a dataclass to store a single turn of conversation (user input, assistant response, summary, and status)
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

    # Add a new turn to the conversation memory
    def add_turn(self, user: str, assistant: str, summary: str,
                 status: str = "ok") -> None:
        self.turns.append(Turn(user, assistant, summary, status))

    def context_messages(self) -> List[Tuple[str, str]]:
        """Return a list of (role, content) tuples for the last N turns(as it is), with earlier turns summarized."""
        window = self.turns[-self._n:]
        verbatim = [t for t in window if t.status == "ok"]
        verbatim_ids = {id(t) for t in verbatim}
        summarized = [t for t in self.turns if id(t) not in verbatim_ids]

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
        """Return a text summary of the last 10 turns, for guardrail checks."""
        return "\n".join(f"- {t.summary}" for t in self.turns[-10:])
