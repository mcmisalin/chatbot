from langchain.chains import LLMChain
from typing import Any, List, Tuple, Union
import requests
import json

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai import ChatVertexAI

from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold

from .extract_agent import extract_visa_types_and_answers
from .reason_question_type_agent import reason_question_type

PROJECT_ID = "vaulted-zodiac-253111"
LOCATION_ID = "global"
DATA_STORE_ID = "q-a-decision_1726662176097"

# Search immigration database
def search_immigration_database(query: str) -> Union[str, Tuple[str, List[Any]]]:
    """Search for visa information using VertexAI Search Retriever."""
    retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        data_store_id=DATA_STORE_ID,
        location_id=LOCATION_ID,
        engine_data_type=0
    )
    return retriever.invoke(query)

# Generate chat history
def generate_chat_history(messages: List[dict]) -> str:
    chat_history = ""
    for message in messages:
        if message["role"] == "user":
            chat_history += f"Human: {message['content']}\n"
        elif message["role"] == "assistant":
            chat_history += f"Assistant: {message['content']}\n"
    return chat_history

# Send chat history and user input via HTTP POST
def send_chat_via_post(chat_history, visa_type, llm, question_type):
    url_public = "https://10.40.0.107:443/api/chatbothistory"  # Replace with actual API endpoint
    headers = {'Content-Type': 'application/json'}

    try:
        data = {
            "chat_history": chat_history,
            "chosen_visa": visa_type,
            "eligible_visas": llm.get("visa_types"),
            "answers": llm.get("intake_form_questions"),
            "question_type": question_type
        }
        response = requests.post(url_public, headers=headers, data=json.dumps(data), verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Sanitize output
def sanitize_output(msg: str) -> str:
    unwanted_phrases = ["search_immigration_database"]
    for phrase in unwanted_phrases:
        msg = msg.replace(phrase, "")
    return msg.strip()

# Initialize the LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def LLM_init() -> LLMChain:
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
        6. Collect all the user answers and assess their eligibility for visa type inquiring in the background and present that ONLY when you are sure they are eligible.

    Output:
        - **Tailored Responses:** Provide responses that directly address the user's query, avoiding irrelevant information.
        - **Clear and Concise Explanations:** Use plain language to explain complex concepts.
        - **Actionable Advice:** Offer practical advice and guidance.

    GUARDRAILS:
    - Always ensure your answers are complete and truthful.
    - Keep the conversation focused on visas and immigration to the USA.

    {chat_history}
    Human: {human_input}
    Assistant:
    """

    prompt = PromptTemplate(template=template, input_variables=["chat_history", "human_input"])

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 2048,
        "top_p": 0.2,
        "max_retries": 5,
        "stop": None,
        "request_timeout": 13
    }

    llm_chain = LLMChain(
        prompt=prompt,
        llm=ChatVertexAI(
            model="gemini-1.5-flash-002",
            generation_config=generation_config,
            safety_settings=safety_settings),
        memory=memory,
        verbose=True
    )

    return llm_chain

# Decision Agent Function
def decision_agent(messages: List[dict], user_input: str) -> str:
    llm_chain = initialize_llm()
    chat_history = generate_chat_history(messages)

    response = llm_chain.predict(human_input=user_input, chat_history=chat_history)
    sanitized_response = sanitize_output(response)

    return sanitized_response
