ADVICE_WORDS = ("buy", "sell", "hold", "recommend", "invest", "accumulate")


def _render(template, **vars):
    defaults = dict(ticker="HDFCBANK.NS", company_name="HDFC Bank Ltd",
                    retry_feedback="")
    defaults.update(vars)
    return template.invoke(defaults).to_string()


def test_all_templates_render_with_ticker():
    from templates.prompts.fundamentals_agent import FUNDAMENTALS_PROMPT
    from templates.prompts.news_analysis_generator import NEWS_PROMPT
    from templates.prompts.event_timeline_creator import EVENTS_PROMPT
    from templates.prompts.financial_docs_analyzer import DOCS_PROMPT
    from templates.prompts.competitor_intelligence_agent import COMPETITOR_STRUCT_PROMPT
    from templates.prompts.synthesis_agent import SYNTHESIS_PROMPT

    rendered = [
        _render(FUNDAMENTALS_PROMPT, context="{}"),
        _render(NEWS_PROMPT, articles="[]"),
        _render(EVENTS_PROMPT, announcements="[]"),
        _render(DOCS_PROMPT, contexts="{}"),
        _render(COMPETITOR_STRUCT_PROMPT, transcript="..."),
        _render(SYNTHESIS_PROMPT, specialist_outputs="{}", missing="[]"),
    ]
    for text in rendered:
        assert "HDFCBANK.NS" in text
        assert "NEVER use the words" in text  # common header present


def test_relevance_rules_in_specialist_prompts():
    from templates.prompts.news_analysis_generator import NEWS_PROMPT
    text = _render(NEWS_PROMPT, articles="[]")
    assert "Waaree" in text                # lookalike-name guard
    assert "subsidiar" in text.lower()     # subsidiary handling


def test_retry_feedback_injected():
    from templates.prompts.news_analysis_generator import NEWS_PROMPT
    text = _render(NEWS_PROMPT, articles="[]",
                   retry_feedback="Previous attempt was rejected: too short.")
    assert "too short" in text


def test_competitor_system_prompt_is_string():
    from templates.prompts.competitor_intelligence_agent import COMPETITOR_SYSTEM
    text = COMPETITOR_SYSTEM.format(ticker="HDFCBANK.NS", company_name="HDFC Bank Ltd")
    assert "3-5 true competitors" in text


def test_classifier_prompt_allows_any_stock():
    from guard.prompts import COMBINED_CLASSIFIER_PROMPT

    text = COMBINED_CLASSIFIER_PROMPT.format(
        company_name="HDFC Bank Ltd", ticker="HDFCBANK.NS",
        chat_history="(none)", user_message="What is the P/E of TCS?")
    # widened scope must be stated to the judge
    assert "any NSE/BSE-listed stock" in text
    # single-company-only phrasing must be gone
    assert "may only answer factual questions about" not in text
    # advice/jailbreak/offensive axes unchanged
    for key in ("is_advice_request", "is_jailbreak", "is_offensive"):
        assert key in text


def test_chatbot_system_prompt():
    from templates.prompts.chatbot import CHATBOT_SYSTEM

    text = CHATBOT_SYSTEM.format(ticker="HDFCBANK.NS",
                                 company_name="HDFC Bank Ltd")
    assert "HDFCBANK.NS" in text
    assert "search_research" in text            # RAG tool named
    assert "I don't have that information" in text
    assert "any NSE/BSE-listed stock" in text   # other stocks allowed
    for banned in ("buy", "sell", "hold"):
        assert banned in text.lower()           # the never-say list is present
