import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch


load_dotenv()

@tool
def triple(num:float) -> float:
    """
    param num: a num to triple
    returns: the triple of the input number
    """
    return float(num * 3)

tools = [triple, TavilySearch(max_results=1)]

llm = ChatAnthropic(
    base_url=os.environ["FCC_BASE_URL"].removesuffix('/v1'),
    model_name=os.environ["FCC_MODEL"],
    api_key=os.environ["FCC_API_KEY"],
    temperature=0
)

llm_with_tools = llm.bind_tools(tools=tools)



