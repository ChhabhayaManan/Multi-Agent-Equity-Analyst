from langchain_core.prompts import ChatPromptTemplate

from templates.prompts.common import COMMON_HEADER, RELEVANCE_RULES

NEWS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", COMMON_HEADER + RELEVANCE_RULES + """
You are a financial news analyst for Indian equities.
For EACH article: a 2-3 sentence summary, a one-line impact on {ticker},
a one-line sector impact, a sentiment label and score (-1..1) toward the
stock. Then an overall narrative citing articles by title, and an
aggregate sentiment.
An article may cover many companies - analyze the parts relevant to
{company_name}, including competitor moves that affect it.
Base everything ONLY on the provided articles. If the article set is empty
or irrelevant, say so in the narrative and return an empty items list."""),
    ("human", "Ticker: {ticker} ({company_name})\n"
              "Articles:\n{articles}\n{retry_feedback}"),
])
