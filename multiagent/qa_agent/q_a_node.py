from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage

# Import your agent initialization logic
from .q_a_agent import LLM_init, generate_chat_history, sanitize_output

def qa_agent_node(state, config):
    """Node function for the Q&A Agent."""
    # Initialize the LLM Chain for the Q&A Agent
    llm_chain = LLM_init()

    # Generate chat history
    chat_history = generate_chat_history(state["messages"])

    # Get the last message from the user
    last_message = state["messages"][-1].content

    # Generate a response
    llm_response = llm_chain.predict(human_input=last_message, chat_history=chat_history)

    # Sanitize the output
    sanitized_response = sanitize_output(llm_response)

    # Append the agent's response to the state
    state["messages"].append(
        BaseMessage(
            content=sanitized_response,
            name="Q&A Agent",
        )
    )

    return state
