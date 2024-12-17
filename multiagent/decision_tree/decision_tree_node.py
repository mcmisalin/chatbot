from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage

# Import your agent initialization logic
from .decision_tree_agent import LLM_init, generate_chat_history, sanitize_output, reason_question_type, send_chat_via_post

# Define the agent node function
def decision_tree_node(state, config):
    """Node function for the Decision Tree Agent."""
    # Initialize the LLM Chain for the Decision Tree Agent
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
            name="Decision Tree Agent",
        )
    )

    # Handle any post-response logic (e.g., forwarding data to APIs)
    if "ready" in last_message.lower():
        parts = last_message.lower().split(",")
        if len(parts) == 2 and parts[0].strip() == "ready":
            visa_type = parts[1].strip()
            send_chat_via_post(
                chat_history=chat_history,
                visa_type=visa_type,
                llm=llm_chain,  # Assuming the chain has the required info
                question_type=reason_question_type(sanitized_response),
            )

    return state
