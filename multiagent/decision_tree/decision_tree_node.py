from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from .decision_tree_agent import LLM_init, generate_chat_history, sanitize_output, reason_question_type, send_chat_via_post

def decision_tree_node(state, config):
    """Node function for the Decision Tree Agent."""
    # Initialize the LLM Chain for the Decision Tree Agent
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

    # Check for specific keywords (e.g., "ready") to trigger additional actions
    if "ready" in user_message.content.lower():
        parts = user_message.content.lower().split(",")
        if len(parts) == 2 and parts[0].strip() == "ready":
            visa_type = parts[1].strip()
            send_chat_via_post(
                chat_history=chat_history,
                visa_type=visa_type,
                llm=llm_chain,  # Pass the LLM instance if required
                question_type=reason_question_type(sanitized_response),
            )

    # Return the updated state
    return state
