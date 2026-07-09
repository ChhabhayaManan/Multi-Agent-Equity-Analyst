"""Blocking CLI for the research chatbot. The input() loop guarantees the
user cannot submit a second question until the previous answer prints."""

import sys
import webbrowser
from pathlib import Path

from chatbot.chatbot_agent import ChatSession
from tools.market_tools import search_ticker


def _pick_ticker() -> tuple:
    query = input("Company or ticker: ").strip()
    if not query:
        return None, None
    try:
        matches = search_ticker(query)[:5]
    except Exception as exc:
        print(f"Ticker search failed: {exc}")
        return None, None
    if not matches:
        print("No matches found.")
        return None, None
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m['ticker']} — {m['name']} ({m['exchange']})")
    choice = input("Pick a number (or Enter for 1): ").strip() or "1"
    try:
        picked = matches[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None, None
    return picked["ticker"], picked["name"]


def main() -> int:
    print("=== Stock Research Chatbot ===")
    try:
        ticker, name = _pick_ticker()
        if not ticker:
            return 1
        try:
            session = ChatSession(ticker, name,
                                  status_callback=lambda s: print(f"  ~ {s}"))
        except ValueError as exc:
            print(f"\n{exc}")
            return 1
        print(f"\nChatting about {name} ({ticker}). "
              "Type 'exit' to quit.\n")
        while True:
            question = input(f"[{ticker}] You: ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Bye.")
                return 0
            response = session.ask(question)
            print(f"\nBot: {response.answer}\n")
            if response.sources_used:
                print(f"  (sources: {', '.join(response.sources_used)})")
            for chart in response.charts:
                path = Path(chart).resolve()
                if path.exists():
                    webbrowser.open(path.as_uri())
                    print(f"  (chart opened: {path.name})")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
