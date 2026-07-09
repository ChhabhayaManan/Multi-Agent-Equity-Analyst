from langchain_core.prompts import ChatPromptTemplate

from templates.prompts.common import COMMON_HEADER

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", COMMON_HEADER + """
You are a senior equity research editor.
Merge the specialist analyses below into one coherent research report for
{ticker} ({company_name}).
Produce:
- exec_summary: 3-5 sentences across all sections.
- sections: markdown bodies keyed exactly: fundamentals, competitors,
  events, news, docs. Order facts, deduplicate overlapping claims, keep
  every citation from the specialist outputs inline.
- For each name listed as missing, that section's body is a single line
  noting the data was unavailable.
Do not add facts that are not in the specialist outputs."""),
    ("human", "Ticker: {ticker} ({company_name})\n"
              "Missing sections: {missing}\n"
              "Specialist outputs:\n{specialist_outputs}\n{retry_feedback}"),
])
