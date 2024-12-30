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
    # The 'prompt' snippet from your code. Keep a single large prompt referencing chat_history:
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

# Function to call Gemini 2.0 "flash-exp" with partial streaming
def generate_response(user_input: str) -> str:
    # Create the client
    client = genai.Client(
        vertexai=True,
        project="vaulted-zodiac-253111",  # Replace with your project ID
        location="us-central1"           # Replace with your location
    )

    # This system instruction is an example
    system_instruction = """You are a US-based immigration expert AI Agent.
    If user says 'stop' or 'done', do not continue. 
    Provide next steps or keep conversation short. 
    Searching the web for info as needed. 
    """

    contents = []
    # Add the conversation so far
    for msg in st.session_state["messages"]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(msg["content"])]))

    # Add the new user input
    contents.append(types.Content(role="user", parts=[types.Part.from_text(user_input)]))

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
        # Tools are optional if you want google_search=...
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

    return response_text


# 3) Streamlit App

st.set_page_config(page_title="Gemini Immigration Chatbot with STOP + Extract")

st.title("Gemini Immigration Chatbot with STOP & Extraction")

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

        # 3) Logging
        # e.g. store some logs in st.session_state["logs"]. 
        # This is where you'd store search results or relevant data:
        logs = ""
        st.session_state["logs"].append({
            "event": "extraction_done",
            "extracted_data": extracted_data
        })
        logs = st.session_state["logs"]
        st.json(logs)

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
