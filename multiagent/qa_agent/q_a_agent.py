from langchain.chains import LLMChain
from typing import Any, List, Tuple, Union
import streamlit as st
import requests
import json
import re

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai import ChatVertexAI

from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold

from extract_agent import extract_visa_types_and_answers
from reason_question_type_agent import reason_question_type

PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "us-central1"
STAGING_BUCKET = "gs:/immigration_pathways_agent_buckets"
DATA_STORE_ID = "attorney-search-datastore_1725005156525"
LOCATION_ID = "global"


eligible_visas = []

# HTML code for a button that acts as a link
button_html = """
    <a href="https://www.usvisatest.com/intakeForm" target="_self">
        <button style="background-color:purple; color:white; border-radius:5px; border:none; padding:10px 20px; cursor:pointer;">
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
    
    
def generate_chat_history(messages):
    chat_history = ""
    for message in messages:
        if message["role"] == "user":
            chat_history += f"Human: {message['content']}\n"
        elif message["role"] == "assistant":
            chat_history += f"Assistant: {message['content']}\n"
    return chat_history

# send chat history and user input via HTTP POST
def send_chat_via_post(chat_history, visa_type, llm, question_type):
    url_private= "http://127.0.0.1:5000/api/chat" # For local testing
    url_public =  "https://10.40.0.107:443/api/chatbothistory" # Replace with actual API endpoint
    headers = {'Content-Type': 'application/json'}
   
    try:
        data = {
        "userId": st.query_params.get("userId"),
        "chat_history": chat_history,
        "choosen_visa" : visa_type,
        "eligible_visas": llm ["visa_types"],
        "answers": llm ["intake_form_questions"],
        "question_type" : question_type
        }
        response = requests.post(url_public, headers=headers, data=json.dumps(data), verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
def sanitize_output(msg):
    # List of unwanted phrases
    unwanted_phrases = ["search_immigration_database"]
    for phrase in unwanted_phrases:
        msg = msg.replace(phrase, "")
    return msg.strip()  
            
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@st.cache_resource(show_spinner=False)
def LLM_init():
    template = """ 
    Task Overview:
    You are Immigration Pathways Chatbot, an expert for visas and immigration to the United States of America. 
    Based on the user's query, analyze the specific needs and provide tailored information.

    Instructions:
        1. Carefully analyze the user's query to identify the core question or request.
        2. Leverage the knowledge base of 5000+ scraped websites and PDFs to provide accurate and up-to-date information.
        3. If the query is specific to visa eligibility, follow the established guidelines:
            * Identify potential visa types.
            * Outline eligibility criteria.
            * Provide clear next steps.
        4. For more open-ended queries, provide informative and helpful responses, drawing on the knowledge base as needed.
        5. Maintain a professional and helpful tone throughout the conversation.
        6. Collect all the user answers and assess his eligibility for visa type he is inquiring for in the background and present him that ONLY when you are sure he is eligible for that visa type.

    Output:
        - **Tailored Responses:** Provide responses that directly address the user's query, avoiding irrelevant information.
        - **Clear and Concise Explanations:** Use plain language to explain complex concepts.
        - **Actionable Advice:** Offer practical advice and guidance.


    GUARDRAILS: 
    - Before you reply, attend, think and remember all the instructions set here.
    - You are truthful and never lie. 
    - Ensure your answers are complete, unless the user requests a more concise approach.
    - Never make up facts and if you are not 100 percent sure, reply with why you cannot answer in a truthful way.
    - When presented with inquiries seeking information, provide answers that reflect a deep understanding of the field, guaranteeing their correctness.
    - Always keep the conversation focused visa types in the USA, immigration to USA and on helping the user determine their visa options, avoiding unrelated topics.
    - Politely but firmly guide the user back to the visa inquiry if they attempt to go off-topic.
    - Providing helpful explanations, pros and cons, and relevant questions ensures users can make informed decisions.

    - NEVER show the summary of user answers/converstion.
    - NEVER show the source (excels,pdfs) of information to user.
    - NEVER repeat any questions that have already been asked. 

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    promptllm = PromptTemplate(template=template, input_variables=["chat_history","human_input"])
    
    safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 2048,
        "temperature": 1.0,
        "top_p": 0.2,
        "max_retries": 5,
        "stop":None,
        "request_timeout": 13
    }
    
    llm_chain = LLMChain(
        prompt=promptllm, 
        llm=ChatVertexAI(
            model="gemini-1.5-flash-002",
            generation_config=generation_config,
            safety_settings=safety_settings), 
        memory=memory, 
        verbose=True
    )
    
    return llm_chain

st.set_page_config(page_title="Chatbot")

# Hiding 'Deploy' and ... menu,  icons,  footer, add message padding
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
# Insert new font
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.title("ImmPath Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello, my name is ImmPath Chatbot and I am an expert for visas and immigration to the USA."},
        {"role": "assistant", "content": "How can I help you?"}
    ]

if "show_button" not in st.session_state:
    st.session_state["show_button"] = False  # Initialize the flag

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner('Processing...'):
        llm_chain = LLM_init()
        chat_history = generate_chat_history(st.session_state["messages"])

        msg2 = llm_chain.predict(human_input=prompt, chat_history=chat_history)
        msg1 = sanitize_output(msg2)

        # Append the assistant's response to the messages list
        st.session_state["messages"].append({"role": "assistant", "content": msg1})

        # Render the assistant's response in the chat
        st.chat_message("assistant").write(msg1)

        memory.save_context({"input": prompt}, {"output": msg1})

