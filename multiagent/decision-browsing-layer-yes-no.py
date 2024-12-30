#########################################
# streamlit_langchain_vertexai.py
#########################################
import streamlit as st
from typing import Any, List, Tuple, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import format_document
from langchain.schema import Document
from langchain_google_community import VertexAISearchRetriever

# --- Constants & Setup ---
PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "global"
DATA_STORE_ID = "q-a-decision_1726662176097"
MODEL_NAME = "gemini-2.0-flash-exp"
TEMPERATURE = 0.1

# We store chat history in st.session_state so it persists across user messages
if "CHAT_HISTORY" not in st.session_state:
    st.session_state.CHAT_HISTORY = []

# --- Functions ---

def search_immigration_database(query: str) -> Union[str, Tuple[str, List[Any]]]:
    """Search for visa information using VertexAI Search Retriever."""
    retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        data_store_id=DATA_STORE_ID,
        location_id=LOCATION,
        engine_data_type=0
    )
    return retriever.invoke(query)

def generate_response(query: str, chat_history: List[Union[AIMessage, HumanMessage]]) -> str:
    """
    Generates a response to a user query using the Gemini model,
    incorporating data store retrieval.
    """
    model = ChatVertexAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    
    # Prompt setup: incorporate the datastore context
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are an expert immigration legal assistant. "
             "Use the context below to answer the user's question. "
             "If you cannot answer the question using only the given information, respond with "
             "'I cannot answer the question.'. If the question is not relevant to immigration, "
             "respond with 'I cannot answer the question.'. Do not invent new answers. "
             "Do not hallucinate answers. Keep the answers very short and concise. "
             "If not related to the question, respond with 'I cannot answer the question.'"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Question: {question}\n\nContext: {context}")
        ]
    )

    # Chain creation for data retrieval, document formatting, and LLM response
    chain = (
        RunnablePassthrough.assign(context=lambda x: search_immigration_database(x["question"]))
        | prompt
        | model
    )

    response = chain.invoke({
        "question": query, 
        "chat_history": chat_history
    })

    return response.content

# --- Streamlit Chatbot App ---

st.set_page_config(page_title="Gemini Immigration Chatbot")

st.title("Gemini Immigration Chatbot")

# Display chat history
for msg in st.session_state.CHAT_HISTORY:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# The user input at the bottom of the chat
if user_input := st.chat_input("Type your question here..."):
    # 1) Display user message
    st.session_state.CHAT_HISTORY.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # 2) Generate response
    with st.spinner("Processing..."):
        response = generate_response(user_input, st.session_state.CHAT_HISTORY)
    
    # 3) Display response
    st.session_state.CHAT_HISTORY.append(AIMessage(content=response))
    st.chat_message("assistant").write(response)
