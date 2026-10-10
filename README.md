# LangGraph Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
   - [Nodes](#nodes)
   - [Edges](#edges)
   - [State Management](#state-management)
   - [Conditional Edges](#conditional-edges)
3. [Building Your First LangGraph Agent](#building-your-first-langgraph-agent)
   - [Setting up the Environment](#setting-up-the-environment)
   - [Defining Nodes and Edges](#defining-nodes-and-edges)
   - [Compiling and Running the Graph](#compiling-and-running-the-graph)
4. [Advanced Topics](#advanced-topics)
   - [Persistence](#persistence)
   - [Human-in-the-loop](#human-in-the-loop)
   - [Streaming](#streaming)
   - [Integration with LangChain and LangSmith](#integration-with-langchain-and-langsmith)
5. [Common Patterns](#common-patterns)
   - [ReAct Agent](#react-agent)
   - [Tool Usage](#tool-usage)
   - [Multi-agent Systems](#multi-agent-systems)
6. [Best Practices](#best-practices)
7. [25 Common Interview Questions](#25-common-interview-questions)
8. [Resources](#resources)
9. [My 2 cents](#2-cents)

---

## Introduction
LangGraph is a library for building stateful, multi-actor applications with LLMs, extending the LangChain framework. It allows developers to define complex workflows as graphs, where nodes represent functions or agents and edges represent transitions between them. LangGraph provides control over the flow of execution, enabling deterministic and controllable AI applications.

### Simple LangGraph Diagram
```mermaid
graph TD
    A[Start] --> B[LLM Node]
    B --> C{Decision}
    C -->|Yes| D[Tool Node]
    C -->|No| E[End]
    D --> B
    E --> F[End]
```

In the above diagram:
- **Start**: Entry point of the graph.
- **LLM Node**: A node that calls a language model.
- **Decision**: A conditional edge that checks the LLM's output.
- **Tool Node**: A node that executes a tool based on the decision.
- **End**: The termination node.

This example shows a simple ReAct-like loop where the LLM decides whether to use a tool or finish.

---

## Core Concepts

### Nodes
Nodes are the fundamental building blocks of a LangGraph. Each node is a Python function that takes the current state as input and returns an updated state. Nodes can perform various operations, such as:
- Calling an LLM
- Executing tools
- Performing deterministic computations
- Acting as entry or exit points (Start and End nodes)

#### Example: LLM Node
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

def llm_node(state):
    """A node that uses an LLM to generate a response."""
    response = llm.invoke(state["input"])
    return {"output": response.content}
```

#### Example: Tool Node
```python
from langchain.tools import Tool

def search_tool(query):
    # Mock search function
    return f"Results for: {query}"

def tool_node(state):
    """A node that executes a search tool."""
    query = state["input"]
    result = search_tool(query)
    return {"tool_output": result}
```

### Edges
Edges define the transitions between nodes in the graph. They dictate the flow of execution from one node to another. There are two types of edges:
- **Regular Edges**: Unconditional transitions that always lead from one node to another.
- **Conditional Edges**: Transitions that depend on the current state, allowing for dynamic routing based on conditions evaluated at runtime.

#### Example: Regular Edge
```python
workflow.add_edge("llm_node", "tool_node")
```
This edge always moves from `llm_node` to `tool_node`.

#### Example: Conditional Edge
```python
def should_continue(state):
    """Determine whether to continue with a tool or end."""
    if state.get("need_tool"):
        return "tool_node"
    else:
        return "end"

workflow.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        "tool_node": "tool_node",
        "end": END
    }
)
```
This conditional edge checks the state to decide the next node.

#### Diagram: Regular vs Conditional Edges
```mermaid
graph LR
    A[Node A] -->|Regular Edge| B[Node B]
    A -->|Conditional Edge| C[Node C]
    A -->|Conditional Edge| D[Node D]
```

### State Management
The state in LangGraph is a persistent dictionary that flows through the graph. Each node reads from and writes to this state. State management includes:
- **State Schema**: Defines the structure of the state (keys and their types).
- **State Reducers**: Functions that specify how to update the state when a node returns an update. Reducers can append, average, or custom-merge values.
- **Persistence**: The state can be saved to and loaded from storage (e.g., memory, database) to enable pausing and resuming graph execution.

#### Example: State Schema
```python
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    tool_output: Optional[str]
    final_answer: Optional[str]
```

#### Example: State Reducer
```python
# Reducer for chat_history: append new messages
def append_history(left: List[str], right: List[str]) -> List[str]:
    return left + right

# When defining the graph, you can specify reducers for state keys
# (This is done implicitly when using TypedDict with certain annotations, 
#  but LangGraph also allows explicit reducers via checkpointers or custom logic.)
```

#### Diagram: State Flow
```mermaid
graph LR
    A[Initial State] --> B[Node 1: Updates chat_history]
    B --> C[Node 2: Reads chat_history, adds tool_output]
    C --> D[Node 3: Reads tool_output, produces final_answer]
    D --> E[Final State]
```

### Conditional Edges
Conditional edges enable dynamic decision-making in the graph. They are defined by a condition function that evaluates the current state and returns the next node to visit. This allows for branching and looping based on runtime conditions.

#### Example: Conditional Edge for ReAct Loop
```python
def react_router(state):
    """Route based on whether the agent thinks it's done."""
    if state.get("final_answer"):
        return "end"
    elif state.get("tool_output"):
        return "llm_node"  # Go back to LLM to reason on tool output
    else:
        return "tool_node"  # Initial step: use a tool

workflow.add_conditional_edges(
    "llm_node",
    react_router,
    {
        "llm_node": "llm_node",
        "tool_node": "tool_node",
        "end": END
    }
)
```

#### Diagram: Conditional Edge Routing
```mermaid
graph TD
    A[LLM Node] -->|Has final answer| B[End]
    A -->|Has tool output| A
    A -->|Needs tool| C[Tool Node]
    C --> A
```

---

## Building Your First LangGraph Agent

### Setting up the Environment
To get started with LangGraph, install the required packages:
```bash
pip install langgraph
# Optionally install LangChain for LLM integrations
pip install langchain
```

### Defining Nodes and Edges
Here's a complete example of a simple agent that uses an LLM and a tool, with a conditional edge to decide when to stop.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain.chat_models import ChatOpenAI

# Define the state schema
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    tool_output: Optional[str]
    final_answer: Optional[str]

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Define node functions
def llm_node(state: AgentState) -> AgentState:
    """Node that uses the LLM to generate a response or decide on tool use."""
    # In a real scenario, you would prompt the LLM to decide whether to use a tool
    # For simplicity, we'll just generate a response based on input and history
    history = "\n".join(state["chat_history"])
    prompt = f"History: {history}\nInput: {state['input']}\nResponse:"
    response = llm.invoke(prompt)
    # Assume the LLM output is a string that we treat as the final answer
    # In practice, you might parse the output to decide on tool use
    return {
        "chat_history": state["chat_history"] + [response.content],
        "final_answer": response.content
    }

def tool_node(state: AgentState) -> AgentState:
    """Node that executes a tool (e.g., search)."""
    # Mock tool: just echoes the input
    tool_result = f"Tool result for: {state['input']}"
    return {
        "chat_history": state["chat_history"] + [tool_result],
        "tool_output": tool_result
    }

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("llm", llm_node)
workflow.add_node("tool", tool_node)

# Add edges
workflow.add_edge("llm", "tool")  # After LLM, always go to tool (for this example)
workflow.add_conditional_edges(
    "tool",
    # After tool, decide whether to go back to LLM or end
    lambda state: "llm" if state.get("tool_output") else "end",
    {
        "llm": "llm",
        "end": END
    }
)

# Set entry point
workflow.set_entry_point("llm")

# Compile the graph
app = workflow.compile()

# Run the graph
result = app.invoke({
    "input": "What is the capital of France?",
    "chat_history": [],
    "tool_output": None,
    "final_answer": None
})
print(result)
```

### Compiling and Running the Graph
After defining the nodes and edges, compile the graph using `workflow.compile()`. The compiled graph can be invoked with an initial state using `.invoke()` or `.stream()` for streaming outputs.

#### Diagram: Example Graph Flow
```mermaid
graph TD
    A[Start: LLM Node] --> B[Tool Node]
    B -->|Has output| C[LLM Node]
    B -->|No output| D[End]
    C --> B
```

---

## Advanced Topics

### Persistence
LangGraph supports persistence through checkpointers. By providing a checkpointer (e.g., `MemorySaver`), the graph's state can be saved at each step, allowing for:
- Pausing and resuming execution
- Fault tolerance
- Human-in-the-loop interventions

#### Example: Using MemorySaver
```python
from langgraph.checkpoint.memory import MemorySaver

# Create a checkpointer
memory = MemorySaver()

# Compile the graph with the checkpointer
app = workflow.compile(checkpointer=memory)

# Now you can run the graph and it will save state after each step
# To resume, you would need to provide the thread_id (more on that below)
```

#### Example: Persisting to a Database (e.g., PostgreSQL)
```python
# For persistent storage, you can use PostgreSaver
from langgraph.checkpoint.postgres import PostgreSaver

# Connection string for PostgreSQL
connection_string = "postgresql://user:password@localhost:5432/langgraph"
checkpointer = PostgreSaver(connection_string)

app = workflow.compile(checkpointer=checkpointer)
```

#### Diagram: Persistence Workflow
```mermaid
graph LR
    A[Invoke Graph] --> B[Checkpointer Saves State]
    B --> C[Node Execution]
    C --> D[Checkpointer Saves Updated State]
    D --> E[Next Node]
    E --> F[Checkpointer Saves State]
    F --> G[Continue Until End]
```

### Human-in-the-loop
Human-in-the-loop (HITL) allows pausing the graph execution to await human input. This is achieved by breaking the graph at a node and resuming after human intervention.

#### Example: HITL with Interrupt
```python
from langgraph.graph import Interrupt

def human_input_node(state: AgentState) -> AgentState:
    """Node that waits for human input."""
    # In a real application, you might return an Interrupt to pause the graph
    # Here we simulate by checking if human input is provided in state
    if state.get("human_input"):
        # Process the human input
        return {"chat_history": state["chat_history"] + [state["human_input"]]}
    else:
        # Request human input by raising an Interrupt
        raise Interrupt("Waiting for human input...")

# In the graph, you would add this node and then resume after providing input
```

#### Diagram: HITL Flow
```mermaid
graph TD
    A[Node Before HITL] --> B[Human Input Node]
    B -->|Interrupt| C[Wait for Human]
    C -->|Human Provides Input| B
    B -->|Processed| D[Node After HITL]
```

### Streaming
LangGraph supports streaming of intermediate results, enabling real-time feedback during graph execution. Use the `.stream()` method to receive updates as the graph progresses.

#### Example: Streaming Output
```python
# Stream the graph execution
for chunk in app.stream({
    "input": "Explain quantum computing in simple terms.",
    "chat_history": [],
    "tool_output": None,
    "final_answer": None
}):
    print(chunk)
    # Each chunk represents the state after a node executes
```

#### Diagram: Streaming Process
```mermaid
graph TD
    A[Start Stream] --> B[Node 1 Executes]
    B --> C[Emit State Chunk 1]
    C --> D[Node 2 Executes]
    D --> E[Emit State Chunk 2]
    E --> F[Node 3 Executes]
    F --> G[Emit Final State Chunk]
```

### Integration with LangChain and LangSmith
LangGraph works seamlessly with LangChain components (LLMs, tools, chains) and can be monitored using LangSmith for tracing and debugging.

#### Example: Using LangChain Tools and LLMs
```python
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)

# Define a LangChain tool
search = Tool(
    name="Search",
    func=lambda q: f"Mock search result for {q}",
    description="Useful for answering questions about current events"
)

# Initialize an agent with LangChain (optional, but shows integration)
agent_chain = initialize_agent(
    [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Use the agent chain inside a LangGraph node
def langchain_agent_node(state):
    """Node that uses a LangChain agent chain."""
    result = agent_chain.run(state["input"])
    return {"output": result}
```

#### Example: Enabling LangSmith Tracing
```python
# Set environment variables for LangSmith (or pass in config)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"

# Then when you run the graph, it will automatically trace to LangSmith
```

#### Diagram: LangSmith Integration
```mermaid
graph LR
    A[LangGraph Execution] --> B[LangSmith Tracer]
    B --> C[Trace Storage]
    C --> D[LangSmith Dashboard]
```

---

## Common Patterns

### ReAct Agent
The ReAct (Reasoning and Acting) pattern combines reasoning and acting in iterative steps. In LangGraph, this can be implemented with a loop of nodes: thought, action, and observation.

#### Example: ReAct Implementation
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List

class ReActState(TypedDict):
    input: str
    thoughts: List[str]
    actions: List[str]
    observations: List[str]
    final_answer: Optional[str]

def thought_node(state):
    """Generate a thought based on current state."""
    # Use LLM to think about what to do next
    thought = "I need to search for information."
    return {"thoughts": state["thoughts"] + [thought]}

def action_node(state):
    """Decide on an action (e.g., use a tool)."""
    action = "search"
    return {"actions": state["actions"] + [action]}

def observation_node(state):
    """Execute the action and record observation."""
    # Mock observation
    observation = "Found relevant information."
    return {"observations": state["observations"] + [observation]}

def router(state):
    """Decide next step based on observations."""
    if state.get("final_answer"):
        return "end"
    elif len(state["observations"]) < 2:  # Need more observations
        return "thought"
    else:
        return "act"  # Or go to final answer based on observations

# Build the graph with the ReAct pattern
workflow = StateGraph(ReActState)
workflow.add_node("thought", thought_node)
workflow.add_node("act", action_node)
workflow.add_node("observe", observation_node)

workflow.add_edge("thought", "act")
workflow.add_edge("act", "observe")
workflow.add_conditional_edges(
    "observe",
    router,
    {
        "thought": "thought",
        "act": "act",
        "end": END
    }
)

workflow.set_entry_point("thought")
app = workflow.compile()
```

#### Diagram: ReAct Loop
```mermaid
graph TD
    A[Thought] --> B[Action]
    B --> C[Observation]
    C -->|Need more| A
    C -->|Done| D[Final Answer]
```

### Tool Usage
Tools can be integrated as nodes or invoked within nodes. LangGraph allows defining custom tools that can be called by agents.

#### Example: Custom Tool as a Node
```python
def calculator_tool(state):
    """A tool that performs calculations."""
    expression = state["input"]
    try:
        result = eval(expression)  # In practice, use a safe evaluator
        return {"tool_output": str(result)}
    except Exception as e:
        return {"tool_output": f"Error: {e}"}

# Use this tool in a node
def tool_node(state):
    return calculator_tool(state)
```

#### Example: Tool Invoked Within a Node
```python
from langchain.tools import Tool

def search_node(state):
    """Node that invokes a search tool within its logic."""
    search_tool = Tool(
        name="Search",
        func=lambda q: f"Results for {q}",
        description="A search tool"
    )
    result = search_tool.run(state["query"])
    return {"search_result": result}
```

### Multi-agent Systems
LangGraph facilitates multi-agent systems by defining multiple agent nodes that communicate via the shared state or through message passing.

#### Example: Two-Agent Collaboration
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class MultiAgentState(TypedDict):
    input: str
    agent1_output: Optional[str]
    agent2_output: Optional[str]
    final_result: Optional[str]

def agent1_node(state):
    """First agent processes the input."""
    # Agent 1 does some processing
    output = f"Agent 1 processed: {state['input']}"
    return {"agent1_output": output}

def agent2_node(state):
    """Second agent uses the output of agent 1."""
    # Agent 2 builds on agent 1's work
    output = f"Agent 2 refined: {state['agent1_output']}"
    return {"agent2_output": output}

def combiner_node(state):
    """Combines outputs from both agents."""
    result = f"{state['agent1_output']} + {state['agent2_output']}"
    return {"final_result": result}

# Build the graph
workflow = StateGraph(MultiAgentState)
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)
workflow.add_node("combiner", combiner_node)

workflow.add_edge("agent1", "agent2")
workflow.add_edge("agent2", "combiner")
workflow.add_edge("combiner", END)

workflow.set_entry_point("agent1")
app = workflow.compile()
```

#### Diagram: Multi-agent Flow
```mermaid
graph LR
    A[Agent 1] --> B[Agent 2]
    B --> C[Combiner]
    C --> D[Final Result]
```

---

## Best Practices
- Keep nodes focused on a single responsibility.
- Use conditional edges for dynamic routing instead of complex node logic.
- Leverage persistence for long-running workflows.
- Test individual nodes before integrating them into the graph.
- Use LangSmith for debugging and monitoring complex graphs.
- Design state schemas carefully to avoid unnecessary data duplication.
- Use interrupts judiciously for human-in-the-loop to avoid deadlocks.
- When using asynchronous execution, ensure proper error handling in nodes.
- Keep the graph acyclic unless cycles are intentionally designed (e.g., for ReAct).
- Use checkpointers that match your deployment environment (in-memory for dev, persistent for prod).

---

## 25 Common Interview Questions
1. **What is LangGraph and how does it differ from LangChain?**
   LangGraph is an extension of LangChain focused on building stateful, multi-actor applications with explicit control flow via graphs, whereas LangChain provides components for LLM applications without built-in graph-based control.

2. **Explain the core components of a LangGraph.**
   The core components are nodes (functions that process state), edges (transitions between nodes), and state (a persistent dictionary that flows through the graph).

3. **How do you define a node in LangGraph?**
   A node is defined as a Python function that takes the current state as input and returns an updated state dictionary.

4. **What are the types of edges in LangGraph?**
   The two main types are regular edges (unconditional transitions) and conditional edges (transitions based on state conditions).

5. **How does state management work in LangGraph?**
   State is managed as a TypedDict that flows through the graph. Each node reads from and writes to the state, with reducers defining how updates are applied.

6. **What is the purpose of a checkpointer in LangGraph?**
   A checkpointer (e.g., MemorySaver) saves the graph's state at each step, enabling persistence, fault tolerance, and human-in-the-loop capabilities.

7. **How can you implement a human-in-the-loop workflow in LangGraph?**
   By using interrupts or breaking the graph at specific nodes to await human input, then resuming execution after the input is provided.

8. **Explain the difference between `.invoke()` and `.stream()` in LangGraph.**
   `.invoke()` runs the graph to completion and returns the final state, while `.stream()` yields intermediate states as the graph executes, allowing for real-time updates.

9. **How do you handle cycles in a LangGraph?**
   Cycles are allowed in LangGraph and can be implemented using conditional edges that loop back to previous nodes based on state conditions.

10. **What is the ReAct pattern and how can it be implemented in LangGraph?**
    The ReAct pattern alternates between reasoning (thinking) and acting (using tools). In LangGraph, this is implemented as a cycle of nodes for thought, action, and observation.

11. **How do you integrate LangChain components (like LLMs) into a LangGraph node?**
    You can use LangChain LLMs, tools, or chains within node functions to process state and generate outputs.

12. **What are state reducers and why are they important?**
    State reducers specify how to merge node outputs into the current state (e.g., appending to a list, averaging values), ensuring consistent state updates.

13. **How can you persist LangGraph state to a database?**
    By using a checkpointer that supports persistent storage (e.g., PostgreSaver, MongoDBSaver) instead of in-memory storage.

14. **What is the role of the START and END nodes in LangGraph?**
    The START node is the entry point of the graph, and the END node signifies completion. They are optional but help define clear boundaries.

15. **How do you debug a LangGraph?**
    Use LangSmith for tracing, or add logging within nodes to inspect state transitions and outputs.

16. **Explain how conditional edges enable dynamic workflows.**
    Conditional edges evaluate the current state to determine the next node, allowing the graph to adapt its flow based on runtime conditions.

17. **What are the benefits of using LangGraph for agentic applications?**
    LangGraph provides control over agent behavior, enables complex workflows with loops and branching, and supports persistence and human intervention.

18. **How do you handle errors in LangGraph nodes?**
    Errors can be caught within nodes and stored in the state, or the graph can be designed to transition to an error-handling node via conditional edges.

19. **Can LangGraph work without LangChain?**
    Yes, LangGraph can be used independently, though it integrates smoothly with LangChain components.

20. **What is the difference between a synchronous and asynchronous LangGraph execution?**
    Synchronous execution blocks until completion, while asynchronous execution (using `.astream()` or `.ainvoke()`) allows for non-blocking operation in async contexts.

21. **How do you scale a LangGraph application for production?**
    Use persistent checkpointers, monitor with LangSmith, deploy as a service, and consider horizontal scaling for stateless components.

22. **Explain the concept of "thread" in LangGraph persistence.**
    A thread is a unique identifier for a graph execution instance, allowing multiple concurrent executions to have isolated states.

23. **How do you integrate external APIs or tools into a LangGraph node?**
    External APIs can be called directly within node functions, or wrapped as LangChain tools for reuse.

24. **What are some common use cases for LangGraph?**
    Common use cases include multi-step reasoning agents, workflow automation, decision trees, chatbots with complex logic, and simulation systems.

25. **How does LangGraph compare to other state machine or workflow engines?**
    LangGraph is specifically designed for LLM-agent applications, offering tight integration with LLMs and flexible state management, whereas generic workflow engines may lack LLM-centric features.

---

## Resources
- Official Documentation: https://langchain-ai.github.io/langgraph/
- LangGraph GitHub: https://github.com/langchain-ai/langgraph
- LangSmith for Tracing: https://smith.langchain.com/
- LangChain Hub: https://smith.langchain.com/hub

---

## 2-cents


# LangGraph an Introduction

# Introduction to LangGraph

            Decide output step |  Decide which steps to take | Decide what steps are available to take
Human                🧑                         🧑                              🧑
LLM Call             🤖                         🧑                              🧑
Chaining (RAG)       🤖                         🧑                              🧑
Router(No cycle)     🤖                         🧑                              🧑 <Langchain above ^>
------------------------------------------------------------------------------------------------------
State Executed       🤖                         🤖                              🧑 <Langraph Located>
Agents Autonomous    🤖                         🤖                              🤖


# Issues with the reAct loop:

    We see in the react loop we have the following:
        
        - LLMs
        - Agents (which do the work)
        - Actions (taken by the LLMs)
        - Tools (which the agent can access)
        - Observations (which the LLM can act upon)

    But this creates a massive issue:

        - One thing is that the decisions are up to the LLM
        - An LLM might keep calling a tool and go into an infinite loop
        - this will never give me the required output

    Now by taking away one layer of autonomy from the agents we can gain a lot of reliability and resilience to the system

# What is a LangGraph
    the core pillar it supports
        - controllability
            - Developers define the control flow
            - LLM makes decisions inside the flow
            - Represent this as nodes and edges

        - persistence
        - Human in the loop
        - Streaming
    Works perfectly with or without LangChain
        - Integrates with LangSmith

# Graphs
    - Consists of nodes (vertices)
    - Edges (Connecting nodes)

# State machines:
    - Represented as graphs where the states are nodes
    - Edges are the transitions

# Definition
    - LangChain is a powerful library built on LangChain
    - We can build very powerful flows using those nodes
        and edges.
    - We can build very powerful applications using it
    - Agentic applications

# Flow engineering
    - It is a systematic and strategic approach for developing a software that will incorporate AI driven decision-making processes.
    - The entential goal of flow engineering is how AI systems with LLMs handle tasks by defining a clear flow or sequence of operations.
    - these flows are not linear and may involve complex decision making flows. where the AI may generate multiple outputs which are often assesed and refined in an iterative cycle.

    - flow engineering guides the AI though a well defined set of steps in order to improve the output quality of the AI systems.

    - developers want to define the flow and we define the scope and what needs to be done.

    - but we can incorparate an LLM to decide which flow to take in the defined params based on the inputs

    - 60% will go into archtechture. 35% will go for fine-tuning. 5% will go to the prompt

# Core componenets of LangGraph

    - With LangGraph We are going to define a well defined control flow, state machine or graph all the same.
    - Within this flow we are going to use LLMs and define where we are going to go next.
    - Basically we need 3 things
            - Nodes
            - Edges
            - Conditional Edges
    
## Nodes
    - Basically python functions you can put any code you want here. It can be deterministic code, python code, python code that calls an LLM or can even call an LLM agent.

    - Two types are node are always present:
        - Start Node: Start Node is our entry point to the graph Execution. So we start from there.
        - End Node: End Node is going to be our last node.
        - these are build in nodes. they dont really do any thing just define the start and end of the function (No-Op)


## Edges and Conditional Edges:
    - Edges connect those nodes within the graph execution.
    - Conditional Edges help to make decisions wheather to go to Node A or Node B and this is dynamic and very very flexible

# State Management:
    -   COntains state schema and state reducers
    -   Bassically contains a dict that maintains some val related to the node.
        - Reducers define how to apply the Node Output to the current state.
        - Kind of think of that as a collector
            - Eg: add new dialogue to the conversation history
            - Eg: Avg together multiple agent nodes.
        - It can be anything for example the result of the LLM everytime.
        - It can be very complex customising it to anything we want 
        - It is local to the graph i.e. it can be accessed by every node during the graph's execution and it is also available on wach edge can it can be local.
        - We can also persist it into persistant storage 
            - So if you want to stop for some reason the flow execution of the graph and want to continue from the same point with that given state we can do it with persistance.

# Nodes more info:
    -  Nodes are special functions that always recieve as an input the current graph state, which is going to have all the infromation that the nodes need in order for it to do its work

   - ANd what the node function basicall returns is the one that updates this state for it. We always return the keys for the dictionary as to what keys we want to update in the state.

   - So nodes updates the state and the edges and conditional edges relie on this state to deicde what need to be done in the next steps of the flow exectution.

# SOME Other concepts:
    - Cyclic Graph
    - Human in the loop
    - Persistance
*This guide aims to provide a comprehensive overview of LangGraph for implementation and interview preparation.*