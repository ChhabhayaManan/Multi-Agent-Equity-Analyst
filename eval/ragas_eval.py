"""RAGAS evaluation of the research chatbot over the golden set.

Metrics: faithfulness + context precision (LLM judge = Gemini). Run sparingly
- each question drives real LLM/retrieval calls and the judge adds more."""

from eval.golden import load_golden_set
from tools.pinecone_tools import namespace_exists
from utils.helpers import get_logger
from utils.tracing import init_tracing, set_run_metadata, traceable

logger = get_logger(__name__)


def build_dataset(items, ask_fn):
    """Pure: map golden items to RAGAS rows via ask_fn(question)->(answer, contexts)."""
    rows = []
    for it in items:
        answer, contexts = ask_fn(it.question)
        rows.append({
            "user_input": it.question,
            "response": answer,
            "retrieved_contexts": list(contexts),
            "reference": it.ground_truth,
        })
    return rows


def _ensure_indexed(ticker: str) -> None:
    if not namespace_exists(ticker):
        raise SystemExit(
            f"No indexed namespace for {ticker}. Generate its report first: "
            f'python -m workflow.graph {ticker} "<Company Name>"')


def _session_ask_fn(ticker: str):
    """Return ask(question)->(answer, contexts), fresh ChatSession per question."""
    from chatbot.chatbot_agent import ChatSession

    # company_name is only used for tool prompts; ticker drives retrieval.
    def ask(question: str):
        session = ChatSession(ticker, ticker)
        resp = session.ask(question)
        return resp.answer, resp.retrieved_contexts

    return ask


def _judge():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0))


def _evaluate(rows):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
    judge = _judge()
    metrics = [Faithfulness(llm=judge),
               LLMContextPrecisionWithReference(llm=judge)]
    dataset = EvaluationDataset.from_list(rows)
    return evaluate(dataset, metrics=metrics, llm=judge)


@traceable(run_type="chain", name="ragas_eval")
def run_eval(path: str = "data/golden_set.json") -> dict:
    init_tracing()
    ticker, items = load_golden_set(path)
    _ensure_indexed(ticker)
    rows = build_dataset(items, _session_ask_fn(ticker))
    result = _evaluate(rows)
    means = result.to_pandas().mean(numeric_only=True)
    scores = {k: float(v) for k, v in dict(means).items()}
    logger.info("RAGAS scores: %s", scores)
    set_run_metadata({"run_type": "ragas_eval", "ticker": ticker, **scores})
    return scores


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/golden_set.json"
    for name, val in run_eval(p).items():
        print(f"{name:45s} {val:.3f}")
