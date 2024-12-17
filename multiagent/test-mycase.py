import streamlit as st
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from decision_tree.decision_tree_node import decision_tree_node  # Import your Decision Tree Agent
from qa_agent.browsing_node import qa_agent_node  # Import your Q&A Agent
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_core.tools import tool
from typing_extensions import TypedDict

# Define the Agent State
class AgentState(TypedDict):
    messages: list
    current_route: str
    next_agent: str
    shared_memory: dict  # Shared memory for continuity

# Define Routing Functions
def pre_greeting_routing(default_route: str):
    def routing(state: AgentState) -> str:
        return state.get("current_route", default_route) or default_route
    return routing

def post_greeting_routing(default_route: str):
    def routing(state: AgentState) -> str:
        if "next_agent" in state:
            return state["next_agent"]
        return default_route
    return routing

# Define the Redirect Tool
@tool
def redirect_tool(next_agent: str) -> dict:
    if next_agent not in ["Q&A Agent", "Decision Tree Agent"]:
        raise ValueError(f"Invalid next_agent: {next_agent}")
    return {"next_agent": next_agent}

# Initialize LLM
llm = ChatVertexAI(model="gemini-1.5-flash-002")

# Supervisor Prompt
greeting_agent_prompt = """
You are the supervisor agent for visa inquiries. Your tasks:
- Greet the user.
- Identify their needs.
- Redirect them to the appropriate agent:
  - Q&A Agent: For general inquiries.
  - Decision Tree Agent: For eligibility assessments and structured workflows.
"""

# Build StateGraph
builder = StateGraph(AgentState)

builder.add_conditional_edges(
    START,
    pre_greeting_routing("Greeting Agent"),
)

# Add Nodes
builder.add_node(
    "Greeting Agent", 
    lambda state, config: redirect_tool("Q&A Agent" if "?" in state["messages"][-1].content else "Decision Tree Agent")
)
builder.add_node("Q&A Agent", qa_agent_node)
builder.add_node("Decision Tree Agent", decision_tree_node)

# Add Routing Logic
builder.add_conditional_edges(
    "Greeting Agent",
    {
        "Q&A Agent": lambda state: state["next_agent"] == "Q&A Agent",
        "Decision Tree Agent": lambda state: state["next_agent"] == "Decision Tree Agent",
    },
)
builder.add_conditional_edges(
    "Q&A Agent",
    {
        "Greeting Agent": post_greeting_routing("Greeting Agent"),
        END: lambda state: state.get("next_agent") == END,
    },
)
builder.add_conditional_edges(
    "Decision Tree Agent",
    {
        "Greeting Agent": post_greeting_routing("Greeting Agent"),
        END: lambda state: state.get("next_agent") == END,
    },
)

# Compile Graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Streamlit Integration
st.set_page_config(page_title="AVISA Chatbot")

# Initialize Session State
if "state" not in st.session_state:
    st.session_state.state = AgentState(
        messages=[],
        current_route="Greeting Agent",
        next_agent="",
        shared_memory={}
    )

st.title("AVISA Chatbot")

# Display Chat History
for msg in st.session_state.state["messages"]:
    role = "user" if "HumanMessage" in str(type(msg)) else "assistant"
    st.chat_message(role).write(msg.content)

# User Input
if user_input := st.chat_input("Enter your message"):
    # Append user message to state
    st.session_state.state["messages"].append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Process input through LangGraph
    with st.spinner("Processing..."):
        st.session_state.state = graph.invoke(st.session_state.state, {})

    # Display updated conversation
    for msg in st.session_state.state["messages"]:
        if "assistant" in str(type(msg)):
            st.chat_message("assistant").write(msg.content)
