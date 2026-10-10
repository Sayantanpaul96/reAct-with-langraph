from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON) # setting the start to the agent_reason
flow.add_node(ACT, tool_node)

#                       |-.-.-.-.-.-.-.-> END
# start --------> agent_reason -.-.-.-.-> Act
#                       |<-----------------|

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END, # defines conditional move from agent_reason to END
    ACT: ACT  # defines conditional move from agent_reason to ACT
})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


def main():
    print("Hello from react-with-langgraph!")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="WHat is the temperature in tokyo? List it and then triple it"
                )
            ]
        }
    )
    print(res["messages"][LAST].content)


if __name__ == "__main__":
    main()
