from langchain.chains import LLMChain
from typing import Any, List, Tuple, Union
import streamlit as st
import requests
import json
import re

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai.chat_models import ChatVertexAI

from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold

from extract_agent import extract_visa_types_and_answers
from reason_question_type_agent import reason_question_type

PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "us-central1"
STAGING_BUCKET = "gs:/immigration_pathways_agent_buckets"
DATA_STORE_ID = "q-a-decision_1726662176097"
LOCATION_ID = "global"

eligible_visas = []



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
        "chosen_visa" : visa_type,
        "eligible_visas": llm.get("visa_types"),
        "answers": llm.get("intake_form_questions"),
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
    Based on latest user question and chat history, your goal is to help the user identify which visa types are they eligible for in order to come to the USA. 

    Use 'search_immigration_database' before EVERY question to determain next question based on defined structures in order for applicant be 100% vetted for a certain visa type.
    
    Instructions:
    1. Client will first answer the question regarding the **reason for the visit**.
    2. ALWAYS ask these 2 questions first:
     - How long do you plan to stay in the United States?
     - Do you have any specific ties to the U.S., such as family members or an employer?

    3. Follow one of the paths based on the user answers:
    
    Business Visas:
     - If the user is looking to start a business in the USA, strictly follow the decision tree and ask necessary questions as outlined in 'Q and A decision.xlsx'.
    
    Family Visa Workflow Adjustments:
        - If the user is looking to reconnect with family, specifically in cases where the applicant is a child, ensure the flow asks all questions to explore eligibility for all relevant visa types (e.g., IR-2, IR-3, etc.) until every potential option has been fully explored.
        - When the user indicates they are married for less than two years, show only relevant visa types (e.g., CR-1 instead of IR-1) and do not proceed further with other visa types within the spouse category. Ensure the flow considers marriage duration before moving forward.
    For all family visa cases:
        - Use 'Family visas flow.pdf' to strictly follow the decision tree and ask necessary questions as outlined in 'Family visas questions.xlsx'.
    
    Student Visas:
        - If the user indicates they are interested in studying, follow 'Student visas flow.pdf' and ask questions from 'student visas.xlsx' strictly according to the decision tree.
    
    Work Visas:
        - If the user wants to work, conduct a targeted vetting based on 'Work Pathways.xlsx', focusing on eligibility for specific work visa types. Avoid covering all work experience unless it pertains to visa qualifications.
        - After you get to F-1 visa, keep asking questions to check if he is bringing someone to study and check if user is also eligible for F-2.
    
    For any visit-related visas:
     - Do not inquire about health issues, visit location, family ties, or events attended, and follow 'Questions.xlsx' to identify relevant visas based on eligibility.

    Permanent Residency:
        - If the user indicates they are interested in **permanent resudency** or **green card**, ask questions market with **GREEN CARD** from 'Work Pathways.xlsx' strictly according to the decision tree.

        
    USE 'Question.xlsx' to folow decision tree and ask questions until you get visa types user is eligible for.
    USE 'Q and A Decision.xlsx' to cross-check the answers given by the applicant and determine their eligibility based on decision rules.
    USE 'Visa Criteria.xlsx' to determine which visa types the applicant might be eligible for based on the applicant's profile.
    Follow 'Application Process.xlsx' to outline the steps required for the applicant to proceed with their visa application.
    Refer to 'Visas Summary.xlsx' to summarize the visa categories they qualify for after vetting.
        
   Output:
        - Provide a ranked list of potentially suitable visa options based on the gathered information. 
        - **Bold the eligible visa type(s).**
        - For each eligible visa, list the next steps in bullet points (without mentioning 'Application Process.xlsx').
        - Clearly state the requirements and likelihood of success for each.
        - If eligibility is unclear, ask further questions from 'Questions.xlsx'.
        - Offer a Green Card path if applicable and prompt for "ready" confirmation with visa type at the end.
        - Ask user: "Please confirm with 'ready' followed by your visa type (e.g., 'ready, h1-b')."

    GUARDRAILS: 
    - Before you reply, attend, think and remember all the instructions set here.
    - You are truthful and never lie. 
    - Ensure your answers are complete, unless the user requests a more concise approach.
    - Never make up facts and if you are not 100 percent sure, reply with why you cannot answer in a truthful way.
    - When presented with inquiries seeking information, provide answers that reflect a deep understanding of the field, guaranteeing their correctness.
    - Always keep the conversation focused visa types in the USA, immigration to USA and on helping the user determine their visa options, avoiding unrelated topics.
    - Politely but firmly guide the user back to the visa inquiry if they attempt to go off-topic.
    - Allow users to return to earlier options if they want to reconsider a different visa type.
    - Providing helpful explanations, pros and cons, and relevant questions ensures users can make informed decisions.
    
    - Always verify marriage duration, and ask question: "Have you been married for less than 2 years?", when relevant.
    - Never ask: "Is your spouse currently residing in the United States?" more than once!
    - Always fully vet child applicants through all related visa paths always including thes questions:
        * Is the child you are applying for your biological child?
        * Is the child married?
        * Is the child under 21?

    - NEVER show the summary of user answers/converstion.
    - NEVER show the source (excels,pdfs) of information to user.
    - ALWAYS ask applicant only one question per turn.
    - NEVER repeat any questions that have already been asked. 
    - Keep track of all asked questions and ensure each one is unique in the session.

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

target_url = st.query_params.get("targetUrl")

# HTML code for a button that acts as a link
button_html = """
    <a href="{target_url}" target="_self">
        <button style="background-color:purple; color:white; border-radius:5px; border:none; padding:10px 20px; cursor:pointer;">
            Start
        </button>
    </a>
"""

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
        {"role": "assistant", "content": "Could you please share the main reason for your visit to the United States?"}
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
        #print(f"MSG1: {msg1}")
        
        question_type = reason_question_type(msg1) #Sending this to Angular for buttons

        st.session_state["messages"].append({"role": "assistant", "content": msg1})
        memory.save_context({"input": prompt}, {"output":msg1})
        
        # Handle 'ready' confirmation
        if "ready" in prompt.lower():
            parts = prompt.lower().split(",")
            if len(parts) == 2 and parts[0].strip() == "ready":
                visa_type = parts[1].strip()
                
                if visa_type:
                    # Display button
                    st.session_state["messages"].append({"role": "assistant", "content": f"Great. You are ready to start your visa application for **{visa_type}** visa"})
                    st.session_state["show_button"] = True

                    st.markdown(button_html, unsafe_allow_html=True)
                
                    llm2 = extract_visa_types_and_answers(chat_history)
                else:
                    st.session_state["messages"].append({"role": "assistant", "content": "Please confirm with 'ready' followed by your visa type (e.g., 'ready, h1-b')."})
            else:
                st.session_state["messages"].append({"role": "assistant", "content": "Please confirm with 'ready' followed by your visa type (e.g., 'ready, h1-b')."})

        st.chat_message("assistant").write(st.session_state["messages"][-1]["content"])

# Display the button if "ready" is confirmed
if st.session_state["show_button"]:
    # Send chat history and user details via HTTP POST
    res = send_chat_via_post(chat_history, visa_type, llm2, question_type)

