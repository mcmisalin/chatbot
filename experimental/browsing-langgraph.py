from typing import Any, List, Tuple, Union
import streamlit as st
import requests
import json
import re
import uuid

from langchain_core.prompts import PromptTemplate
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai.chat_models import ChatVertexAI, AIMessage, HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold

PROJECT_ID = "vaulted-zodiac-253111"
DATA_STORE_ID = "attorney-search-datastore_1725005156525"
LOCATION_ID = "global"

thread_id = uuid.uuid4()

eligible_visas = []

button_html = """
    <a href="https://www.usvisatest.com/intakeForm" target="_self">
        <button style="background-color:purple; color:white; border:none; padding:10px 20px; cursor:pointer;">
            Start
        </button>
    </a>
"""

def search_immigration_database(query: str) -> Union[str, Tuple[str, List[Any]]]:
    """Search for visa information using VertexAI Search Retriever."""
    retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        data_store_id=DATA_STORE_ID,
        location_id=LOCATION_ID,
        engine_data_type=0
    )
    return retriever.invoke(query)

def sanitize_output(msg):
    unwanted_phrases = ["search_immigration_database"]
    for phrase in unwanted_phrases:
        msg = msg.replace(phrase, "")
    return msg.strip()

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

generation_config = {
    "temperature": 0.2,
    "max_output_tokens": 2048,
    "top_p": 0.9,
    "max_retries": 3,
    "stop": None,
    "request_timeout": 13
}

memory = MemorySaver()

template = """
You are a highly knowledgeable US immigration and tax expert, similar in style to the Gemini model. 
When the user asks a question, respond by:
- Addressing their question directly and honestly. If the answer is "no," state that clearly.
- If there are no direct legal ways to achieve the user's exact goal, still suggest legitimate steps, alternatives, or strategies.
- Be concise, factual, and organized.
- Highlight key terms or strategies using **bold text**.
- Include a short list of possible actions, credits, or considerations that could help the user.
- Keep the tone neutral, professional, and helpful.

Do not mention internal processes or that you are using a database.
{chat_history}
Human: {human_input}
Assistant:
"""

promptllm = PromptTemplate(template=template, input_variables=["chat_history","human_input"])

llm = ChatVertexAI(
    model="gemini-2.0-flash-exp",
    generation_config=generation_config,
    safety_settings=safety_settings
)

workflow = StateGraph(state_schema=MessagesState)

def model_node(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    
    # Normalize the response into role/content dictionaries
    response_dicts = []
    if not isinstance(response, list):
        response = [response]

    for msg in response:
        if isinstance(msg, AIMessage):
            response_dicts.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            response_dicts.append({"role": "user", "content": msg.content})
        elif isinstance(msg, dict) and "role" in msg and "content" in msg:
            response_dicts.append(msg)

    new_messages = state["messages"] + response_dicts
    return {"messages": new_messages}

workflow.add_node("model_node", model_node)
workflow.add_edge(START, "model_node")

app = workflow.compile(checkpointer=memory)

st.set_page_config(page_title="Chatbot")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {
            visibility: hidden;
        }
    .stChatMessage > div:first-child {
        display: none;
    }
    .block-container{
        padding: 50px 18px 18px;
    }
    header{
         visibility: hidden;
         display: none;
    }
    h1{
        text-align: left;
        margin: 0px 0px 0px 18px;
    }
    .css-15zrgzn {display: none}
    .css-eczf16 {display: none}
    .css-jn99sy {display: none}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.title("ImmPath Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello, my name is ImmPath Chatbot and I am an expert for visas and immigration to the USA."},
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display current chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Enter your question here...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("Processing..."):
        response_stream = app.stream(
            {"messages": st.session_state["messages"]},
            config={"configurable": {"thread_id": str(thread_id)}},
            stream_mode="messages"
        )

        # We'll accumulate all chunks/final message and only print once after receiving the final dictionary
        assistant_content = ""
        final_dict_received = False

        for result in response_stream:
            if isinstance(result, tuple):
                result = result[0]

            if isinstance(result, AIMessageChunk):
                # Accumulate partial chunks
                assistant_content += result.content
            elif isinstance(result, AIMessage):
                # Final AIMessage: append its content too
                assistant_content += result.content
            elif isinstance(result, dict) and "messages" in result:
                # The final state dictionary with all messages
                final_dict_received = True
                updated_messages = result["messages"]
                st.session_state["messages"] = updated_messages

        # Now print the final assistant message once
        final_assistant_message = assistant_content.strip()
        if not final_assistant_message:
            for m in reversed(st.session_state["messages"]):
                if m["role"] == "assistant":
                    final_assistant_message = m["content"]
                    break

        if "Hello, my name is ImmPath Chatbot" in final_assistant_message:
            intro_index = final_assistant_message.rfind("Hello, my name is ImmPath Chatbot")
            if intro_index > 0:
                # Keep only the text before intro re-appeared
                final_assistant_message = final_assistant_message[:intro_index].strip()

        st.chat_message("assistant").write(final_assistant_message)
