#Do this first!
#pip install streamlit google-genai

from typing import Any, List, Tuple, Union
import streamlit as st
import requests
import json
import re
import uuid

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_community import VertexAISearchRetriever
from google import genai
from google.genai import types

from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold

from extract_agent import extract_visa_types_and_answers
from reason_question_type_agent import reason_question_type

PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "us-central1"
STAGING_BUCKET = "gs:/immigration_pathways_agent_buckets"
DATA_STORE_ID = "q-a-decision_1726662176097"
LOCATION_ID = "global"

eligible_visas = []

# Initialize GenAI client
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello, my name is ImmPath Chatbot and I am an expert for visas and immigration to the USA."},
        {"role": "assistant", "content": "Could you please share the main reason for your visit to the United States?"}
    ]

if "show_button" not in st.session_state:
    st.session_state["show_button"] = False

# For info and questions from JSON blocks
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "yes_no_questions" not in st.session_state:
    st.session_state.yes_no_questions = {}

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

def send_chat_via_post(chat_history, visa_type, llm, question_type):
    url_private= "http://127.0.0.1:5000/api/chat"
    url_public =  "https://10.40.0.107:443/api/chatbothistory"
    headers = {'Content-Type': 'application/json'}

    try:
        data = {
            "userId": st.query_params.get("userId"),
            "chat_history": chat_history,
            "chosen_visa": visa_type,
            "eligible_visas": llm.get("visa_types") if llm else [],
            "answers": llm.get("intake_form_questions") if llm else [],
            "question_type": question_type
        }
        response = requests.post(url_public, headers=headers, data=json.dumps(data), verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def sanitize_output(msg):
    unwanted_phrases = ["search_immigration_database"]
    for phrase in unwanted_phrases:
        msg = msg.replace(phrase, "")
    return msg.strip()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def extract_json_blocks(response):
    # Find all JSON blocks between ```json and ```
    json_blocks = re.findall(r'```json\n(.*?)\n```', response, re.DOTALL)
    parsed_blocks = []
    for block in json_blocks:
        try:
            parsed_json = json.loads(block)
            parsed_blocks.append(parsed_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON block: {e}")
    return parsed_blocks

st.set_page_config(page_title="Chatbot")
target_url = st.query_params.get("targetUrl")

button_html = f"""
    <a href="{target_url}" target="_self">
        <button style="background-color:purple; color:white; border-radius:5px; border:none; padding:10px 20px; cursor:pointer;">
            Start
        </button>
    </a>
"""

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
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

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input()
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Prepare the full prompt
    prompt = """
Task Overview:
You are Immigration Pathways Chatbot, an expert for visas and immigration to the United States of America. 
Based on the latest user question and the chat history, your goal is to help the user identify which visa types they may be eligible for to come to the USA.

Use 'search_immigration_database' before EVERY question to determine the next question based on defined structures, ensuring the applicant is fully vetted for a certain visa type.

Instructions:
1. The client will first answer a question regarding their **reason for the visit**.
2. ALWAYS start by asking these 2 questions (if not already asked):
   - How long do you plan to stay in the United States?
   - Do you have any specific ties to the U.S., such as family members or an employer?

3. Follow the appropriate paths based on the user's answers:

   Business Visas:
   - If the user is looking to start a business in the USA, strictly follow the decision tree and ask necessary questions as outlined in 'Q and A decision.xlsx'.

   Family Visa Workflow Adjustments:
   - If the user is looking to reconnect with family, and especially if the applicant is a child, ensure the flow explores eligibility for all relevant child-related visa types (e.g., IR-2, IR-3, etc.) until every potential option is considered.
   - If the user has been married for less than two years, show only relevant visa types (e.g., CR-1 instead of IR-1) and do not proceed further with other spouse-category visas. Consider marriage duration before moving forward.
   - For all family visa cases, use 'Family visas flow.pdf' and 'Family visas questions.xlsx' to strictly follow the decision trees and ask necessary questions.

   Student Visas:
   - If the user is interested in studying, follow 'Student visas flow.pdf' and strictly use 'student visas.xlsx' questions according to the decision tree.

   Work Visas:
   - If the user wants to work, conduct a targeted vetting based on 'Work Pathways.xlsx', focusing on eligibility for specific work visa types. Avoid covering all work experience unless it pertains to visa qualifications.
   - After identifying the F-1 visa as an option, keep asking if they are bringing someone to study and whether they are eligible for F-2.

   Visit-Related Visas:
   - Do not inquire about health issues, exact visit locations, family ties, or events attended. Follow 'Questions.xlsx' to identify relevant visas based on eligibility.

   Permanent Residency (Green Card):
   - If the user indicates interest in **permanent residency** or **green card**, ask questions marked with **GREEN CARD** in 'Work Pathways.xlsx' strictly according to the decision tree.

Use 'Question.xlsx' to follow the decision tree and ask questions until you determine which visa types the user is eligible for.
Use 'Q and A Decision.xlsx' to cross-check answers and determine eligibility based on decision rules.
Use 'Visa Criteria.xlsx' to determine which visa types the applicant might be eligible for.
Follow 'Application Process.xlsx' to outline steps required for the applicant to proceed (but do not mention the file names to the user).
Refer to 'Visas Summary.xlsx' to summarize the visa categories they qualify for after vetting.

Output:
- Provide a ranked list of potentially suitable visa options based on gathered information.
- **Bold the eligible visa type(s)**.
- For each eligible visa, list the next steps in bullet points (without mentioning 'Application Process.xlsx').
- Clearly state the requirements and the likelihood of success for each.
- If eligibility is unclear, ask further questions from 'Questions.xlsx'.
- Offer a Green Card path if applicable and prompt the user with: "Please confirm with 'ready' followed by your visa type (e.g., 'ready, h1-b')."

GUARDRAILS:
- Before you reply, attend to and remember all instructions here.
- You are truthful and never lie.
- Ensure answers are complete, unless the user requests more conciseness.
- Never make up facts; if unsure, explain why you cannot answer fully.
- Provide deep understanding and correctness.
- Keep the conversation focused on visa types and US immigration, avoiding unrelated topics.
- Politely redirect the user back to visa inquiries if they go off-topic.
- Allow the user to return to earlier options if they want to reconsider a visa type.
- Provide helpful explanations, pros and cons, and relevant questions for informed decisions.
- Always verify marriage duration when relevant, and ask "Have you been married for less than 2 years?" if applicable.
- Never ask "Is your spouse currently residing in the United States?" more than once.
- Always fully vet applicants WITH CHILD, always including:
  * Is the child your biological child?
  * Is the child married?
  * Is the child under 21?
- NEVER show a summary of user answers/conversation to the user.
- NEVER show the source (excels, PDFs) of information to the user.
- ALWAYS ask only one question per turn.
- NEVER repeat any previously asked questions.
- Keep track of asked questions to ensure uniqueness.
    
{chat_history}
Human: {human_input}
Chatbot:"""

    promptllm = PromptTemplate(template=prompt, input_variables=["chat_history","human_input"])
    chat_history = generate_chat_history(st.session_state["messages"])
    textsi_1 = promptllm.format(chat_history=chat_history, human_input=user_input)

    # Build contents for streaming
    contents = []
    # Add user input first
    contents.append(types.Content(role="user", parts=[types.Part.from_text(user_input)]))
    for message in st.session_state["messages"]:
        is_user = (message["role"] == "user")
        role = "user" if is_user else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(message["content"])]))

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        # Add tools or safety_settings if needed
        system_instruction=[types.Part.from_text(textsi_1)]
    )

    with st.spinner('Processing...'):
        response = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates and chunk.candidates[0].content.parts:
                response += chunk.text

        msg1 = sanitize_output(response.strip())
        question_type = reason_question_type(msg1)
        st.session_state["messages"].append({"role": "assistant", "content": msg1})
        st.chat_message("assistant").write(msg1)

        # Extract JSON blocks for sidebars
        parsed_json_blocks = extract_json_blocks(msg1)
        for block in parsed_json_blocks:
            if "info_to_confirm" in block:
                st.session_state.user_info = block["info_to_confirm"]
            if "yes_no_questions" in block:
                st.session_state.yes_no_questions = block["yes_no_questions"]

        # Handle 'ready' confirmation
        if "ready" in user_input.lower():
            parts = user_input.lower().split(",")
            if len(parts) == 2 and parts[0].strip() == "ready":
                visa_type = parts[1].strip()
                if visa_type:
                    st.session_state["messages"].append({"role": "assistant", "content": f"Great. You are ready to start your visa application for **{visa_type}** visa"})
                    st.session_state["show_button"] = True
                    st.markdown(button_html, unsafe_allow_html=True)
                    llm2 = extract_visa_types_and_answers(chat_history)
                else:
                    st.session_state["messages"].append({"role": "assistant", "content": "Please confirm with 'ready' followed by your visa type (e.g., 'ready, h1-b')."})
            else:
                st.session_state["messages"].append({"role": "assistant", "content": "Please confirm with 'ready' followed by your visa type (e.g., 'ready, h1-b')."})

        # Re-display last assistant message if we added more
        if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "assistant":
            st.chat_message("assistant").write(st.session_state["messages"][-1]["content"])

# Display the button if "ready" is confirmed
if st.session_state["show_button"]:
    if 'visa_type' in locals() and 'llm2' in locals():
        res = send_chat_via_post(chat_history, visa_type, llm2, question_type)

# Now display sidebars for user info and yes/no questions, same as first code snippet:
# Sidebar: Confirm Your Information
if st.session_state.user_info:
    st.sidebar.title("Confirm Your Information")
    st.sidebar.markdown("---")
    for key, value in st.session_state.user_info.items():
        include = st.sidebar.radio(
            f"{key.replace('_', ' ').capitalize()}: {value}",
            ["Include", "Omit"],
            horizontal=True,
            key=f"info_{key}"
        )
        if include == "Omit":
            st.session_state.user_info[key] = None  # Remove if omitted

# Sidebar: Yes/No Questions
if st.session_state.yes_no_questions:
    st.sidebar.title("Quick Options")
    st.sidebar.markdown("---")
    for question_key, question_text in st.session_state.yes_no_questions.items():
        answer = st.sidebar.radio(
            f"{question_text}", ["Yes", "No"],
            horizontal=True,
            key=f"yesno_{question_key}"
        )
        st.session_state.user_info[question_key] = answer
