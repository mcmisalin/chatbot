from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from .q_a_agent import LLM_init, generate_chat_history, sanitize_output

def qa_agent_node(state, config):
    """Node function for the Q&A Agent."""
    # Initialize the LLM Chain for the Q&A Agent
    llm_chain = LLM_init()

    # Extract chat history from the state
    chat_history = generate_chat_history(state["messages"])

    # Get the user's latest message
    user_message = next((msg for msg in state["messages"] if isinstance(msg, HumanMessage)), None)
    if not user_message:
        raise ValueError("No user message found in the state.")

    # Generate the LLM response
    llm_response = llm_chain.predict(human_input=user_message.content, chat_history=chat_history)

    # Sanitize the response
    sanitized_response = sanitize_output(llm_response)

    # Add the LLM response to the state's message history
    add_messages(
        state,
        [AIMessage(content=sanitized_response)],
    )

    # Return the updated state
    return state
