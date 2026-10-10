# langgraph nodes

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import tools, llm_with_tools

load_dotenv()

SYSTEM_MESSAGE = '''
You are an helpful assistant that can use tools to answer questions.
'''

#defining all the nodes
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
        Run the agent reasoning node
    """

    response = llm_with_tools.invoke(
        [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            *state["messages"]
        ]
    )
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)