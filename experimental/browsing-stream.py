# Make sure you have:
# pip install streamlit google-genai

from typing import Any, List, Tuple, Union
import streamlit as st
import uuid
import json
import re

from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from google import genai
from google.genai import types

PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "us-central1"

# Initialize the GenAI client for Vertex AI
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

thread_id = uuid.uuid4()


template = """
You are a highly knowledgeable US immigration and tax expert, similar in style to the Gemini model. 
When the user asks a question, respond by:
- Addressing their question directly and honestly. If the answer is "no," state that clearly.
- If there are no direct legal ways to achieve the user's exact goal, still suggest legitimate steps, alternatives, or strategies.
- Be concise, factual, and organized.
- Highlight key terms or strategies using **bold text**.
- Include a short list of possible actions, credits, or considerations that could help the user.
- Keep the tone neutral, professional, and helpful.

GUARDRAILS: 
    - Before you reply, attend, think and remember all the instructions set here.
    - You are truthful and never lie. 
    - Ensure your answers are complete, unless the user requests a more concise approach.
    - Never make up facts and if you are not 100 percent sure, reply with why you cannot answer in a truthful way.
    - When presented with inquiries seeking information, provide answers that reflect a deep understanding of the field, guaranteeing their correctness.
    - Always keep the conversation focused on visa types in the USA, immigration to the USA and on helping the user determine their visa options, avoiding unrelated topics.
    - Politely but firmly guide the user back to the visa inquiry if they attempt to go off-topic.
    - Allow users to return to earlier options if they want to reconsider a different visa type.
    - Providing helpful explanations, pros and cons, and relevant questions ensures users can make informed decisions.

{chat_history}
Human: {human_input}
Assistant:
"""

memory = MemorySaver()
promptllm = PromptTemplate(template=template, input_variables=["chat_history","human_input"])

model = "gemini-2.0-flash-exp"

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

def generate_chat_history(messages):
    chat_history = ""
    for message in messages:
        if message["role"] == "user":
            chat_history += f"Human: {message['content']}\n"
        elif message["role"] == "assistant":
            chat_history += f"Assistant: {message['content']}\n"
    return chat_history

# Display the current chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Enter your question here...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Prepare chat_history and system instruction
    chat_history = generate_chat_history(st.session_state["messages"])
    prompt = promptllm.format(chat_history=chat_history, human_input=user_input)

    # Build contents list using snippet logic
    contents = [types.Content(role="user", parts=[types.Part.from_text(user_input)])]
    for message in st.session_state["messages"]:
        # Map 'assistant' to 'model', 'user' to 'user' as per snippet logic
        is_user = (message["role"] == "user")
        role = "user" if is_user else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(message["content"])]))

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        system_instruction=[types.Part.from_text(prompt)]
    )

    with st.spinner("Processing..."):
        response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates and chunk.candidates[0].content.parts:
                response += chunk.text

        final_assistant_message = response.strip()
        st.session_state["messages"].append({"role": "assistant", "content": final_assistant_message})
        st.chat_message("assistant").write(final_assistant_message)

