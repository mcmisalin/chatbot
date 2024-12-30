import streamlit as st
from google import genai
from google.genai import types
import json
import re
import base64

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {}



# Function to generate responses from the Gemini model
def generate_response(user_input):
    client = genai.Client(
        vertexai=True,
        project="vaulted-zodiac-253111",  # Replace with your project ID
        location="us-central1"  # Replace with your preferred location
    )

    textsi_1 = """You are a US-based immigration expert AI Agent. Your goal is to find the right pathway for applicants to get visas in the US.
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
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=[types.Part.from_text(textsi_1)],
    )

    response = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash-exp",
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.candidates and chunk.candidates[0].content.parts:
            response += chunk.text
    return response





# Streamlit UI
st.title("Immigration Pathways Assistent")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message("user" if message["is_user"] else "assistant"):
        st.markdown(message["text"])

# User input
if prompt := st.chat_input("""Hi, I'm here to guide you on your immigration journey. I'll find the Immigration Pathways that are best for you and we will navigate the path together. 
Who do I have the pleasure of chatting with and what is your reason for looking for a visa to US?"""):
    st.session_state.messages.append({"text": prompt, "is_user": True})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your information..."):
            response = generate_response(prompt)
            st.markdown(response)



    # Add assistant response to chat
    st.session_state.messages.append({"text": response, "is_user": False})