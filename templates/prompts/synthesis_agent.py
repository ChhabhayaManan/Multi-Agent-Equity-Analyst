from langchain_core.prompts import ChatPromptTemplate
from templates.prompts.common import COMMON_HEADER

#synthesis agent prompt
SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", COMMON_HEADER + """
You are a senior equity research editor.
Read the specialist analyses below for {ticker} ({company_name}) and write
one thing only:
- exec_summary: a cross-agent overview, about two short paragraphs (6-9
  sentences). Connect the fundamentals to the peer comparison, the recent
  news and corporate events, and what management guides in official
  documents. Name the specific numbers, peers, events and guidance you are
  drawing on.
- Agents listed as missing produced no data: do not speculate about them,
  and do not mention their absence more than once.
Do not add facts that are not in the specialist outputs. The detailed
sections are rendered from the structured data directly, so do not write
them."""),
    ("human", "Ticker: {ticker} ({company_name})\n"
              "Missing sections: {missing}\n"
              "Specialist outputs:\n{specialist_outputs}\n{retry_feedback}"),
])
