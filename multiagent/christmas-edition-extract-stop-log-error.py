#########################################
# streamlit_chatbot_with_stop_and_extract.py
#########################################

import streamlit as st
from typing import Any, List, Tuple, Union
import json
import re

from google import genai
from google.genai import types

# 1) Import or define your "extract_visa_types_and_answers" function:
# from your_extract_agent_module import extract_visa_types_and_answers

# For demonstration, we'll inline your snippet:
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

generation_model = GenerativeModel("gemini-1.5-flash-002")
generation_config = GenerationConfig(
    temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8
)


def extract_visa_types_and_answers(chat_history: str) -> dict:
    """
    Provided snippet that calls the gemini model and parses JSON.
    """
    prompt = """
    Task Overview:
    You are an agent that extracts visa types and answers from a chat history. 
    Given a chat history and a list of intake form questions, your job is to:

    1. Identify all visa types client is eligible for, through the chat history and return them as a JSON array.
    2. Analyze the chat history and map user responses to the array called 'intake form questions'. 
    Create a JSON object where keys are intake form questions and values are the corresponding user answers from the chat. 
    If a question's answer is not found, the value should be "Not Found."
    The questions MAY NOT be exactly the same, but try to reason and map if there is a certain match.

    Important:
    - Ensure that each intake form question has a corresponding answer, even if it is "Not Found."
    - ONLY return the JSON object with no extra explanations or comments.

    Input:
    - Chat History: {chat_history}
    - Available visa types: ["B-1", "B-2", "H-1B", "H-2A", ... etc...]

    Output:
    Return ONLY the JSON object:
        - visa_types: []
        - intake_form_questions

    GUARDRAILS:
     - ALWAYS remove word json from the beginning of response.
    """

    # We combine the prompt + the chat_history + the visa_types (omitted for brevity).
    # For simplicity, we'll just embed the chat_history in the prompt. Real code: pass them as separate contents.
    final_prompt = prompt.replace("{chat_history}", chat_history)

    # Call the gemini model
    response = generation_model.generate_content(
        contents=[final_prompt],
        generation_config=generation_config
    ).text

    # Minimal error handling for JSON parse
    # If response has triple backticks or starts with "```json", remove them
    response_clean = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE).strip()

    # Attempt to parse as JSON
    try:
        data = json.loads(response_clean)
    except json.JSONDecodeError:
        data = {
            "visa_types": [],
            "intake_form_questions": {}
        }

    return data


# 2) Basic "Gemini" chat with streaming.
#    We'll store partial logs in st.session_state for logging.

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # store (role, content) for the conversation
if "logs" not in st.session_state:
    st.session_state["logs"] = []  # store text from search results, links, etc.

# Function to generate responses from the Gemini model
def generate_response(user_input):
    client = genai.Client(
        vertexai=True,
        project="vaulted-zodiac-253111",  # Replace with your project ID
        location="us-central1"  # Replace with your preferred location
    )

    system_instruction = """You are a US-based immigration expert AI Agent. Your goal is to find the right pathway for applicants to get visas in the US.
    Use Google Search to establish the right set of questions to a user. Limit to 2 questions per turn of the chat.
    Consider:
    1) Time to get a visa
    2) Path to residency and green card
    3) Country of origin
    4) cover big areas that are fast like Family, study and investment
    5) Always ask if they live in the US currently and their status and if they have family in US that are permenent residents

    Have a chat with the user, guiding them through questions to understand which visa they can apply for. Use Google Search throughout.
    Summarize applicable visas. This will then get packaged and sent to a lawyer.
    Even when you have a number of applicable visas, continue to look at 2-3 other options that may be applicable.
    Give the user information in bite-sized pieces. Assume the applicant is not familiar with immigration processes. Don't overwhelm them with text or information.
    Take the questions from the PDF inot consideration as you decide what visa that they are applicable for.
    Summarize the viable options regularly throughout the chat.
    IF an option is not clearly applicable discard it. 
    Continue to explore viable options until the user has 5 viable visas.
    After 6 questions and when you have a few viable visas, summarize again and guide the person to our full service offering at ImmigrationPathways.com
    If the case seems to be getting complicated or the person is getting frustrated guide the to ImmigrationPathways.com.
    """
    file_uri = "gs://immigration-pathways2/immigration-pathways/ELIGIBILITY_CRITERIA.pdf"
    file_part = types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf")
    contents = [types.Content(role="user", parts=[types.Part.from_text(user_input), types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf")])]
    for message in st.session_state.messages:
        role = "user" if message["is_user"] else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(message["text"])]))

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=2048,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=[types.Part.from_text(system_instruction)],
    )

    # We'll accumulate the final response
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash-exp",
        contents=contents,
        config=generate_content_config,
    ):
        # partial streaming chunk
        if chunk.candidates and chunk.candidates[0].content.parts:
            partial_text = chunk.text
            response_text += partial_text
            # 3) Logging
            # Store search results or relevant data:
            st.session_state["logs"].append({
                "event": "search",
                "extracted_data": chunk.text
            })

    return response_text


# 3) Streamlit App

st.set_page_config(page_title="Gemini Immigration Chatbot with STOP + Extract")

st.title("Gemini Immigration Chatbot with STOP & Extraction")

# Insert new font
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# Show conversation
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])


# Chat input
user_input = st.chat_input("Ask your question or type 'stop' to end.")
if user_input:
    # 1) Check if user wants to stop
    if user_input.lower() in ["stop", "done", "end"]:
        # 2) We do the extraction logic
        full_chat = ""
        for m in st.session_state["messages"]:
            # accumulate the entire chat
            full_chat += f"{m['role'].upper()}: {m['content']}\n"

        # Call extract agent
        st.write("**Ending chat. Extracting your answers...**")
        try:
            extracted_data = extract_visa_types_and_answers(full_chat)
            st.write("**Extraction Results:**")
            st.json(extracted_data)
        except Exception as e:
            st.error(f"Error extracting data: {e}")
            extracted_data = {}


        # 4) Provide next steps or redirect
        #   Option 1: Just display a link to the intake form:
        intake_url = "https://www.immigrationpathways.com/intakeForm"
        st.markdown(f"**You can now fill out our intake form: [Start your application]({intake_url})**")

        # Option 2: If you wanted to auto-redirect, 
        # you might do something with st.experimental_rerun or st.stop
        # but this is the simplest approach:
        st.stop()

    # If user didn't type "stop", we handle the normal flow
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate LLM response
    with st.spinner("Thinking..."):
        assistant_reply = generate_response(user_input)

    st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.write(assistant_reply)

    # Optionally store or parse any search result text if your code returns it.
    # e.g. if partial search results are in the chunk, do st.session_state["logs"].append(...)


# If user has typed nothing, do nothing
st.write("Type your question or 'stop' to finish.")
