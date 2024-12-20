from langchain.chains import LLMChain
from typing import Any, List
import json

from langchain_community.llms import VertexAI

import vertexai


PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "us-central1" 
STAGING_BUCKET = "gs:/immigration_pathways_agent_buckets" 
DATA_STORE_ID = "q-a-decision_1726662176097"
LOCATION_ID = "global" 

vertexai.init(project=PROJECT_ID, location=LOCATION)
from vertexai.generative_models import GenerationConfig, GenerativeModel
generation_model = GenerativeModel("gemini-2.0-flash-exp")

generation_config = GenerationConfig(
    temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8
)

visa_types = [
    "B-1", "B-2",  # Business and Tourism
    "H-1B", "H-2A", "H-2B", "H-3",  # Employment visas
    "L-1A", "L-1B",  # Intra-company transfers
    "O-1", "P-1", "P-2", "P-3",  # Individuals with extraordinary ability, athletes, entertainers
    "R-1",  # Religious workers
    "TN",  # NAFTA professionals
    "E-1", "E-2",  # Treaty traders and investors
    "F-1", "M-1",  # Students (academic and vocational)
    "J-1", "J-2",  # Exchange visitors
    "K-1", "K-2",  # Fiancé(e) visas
    "I",  # Representatives of foreign media
    "V",  # Family members of legal permanent residents
    "EB-1", "EB-2", "EB-3", "EB-4", "EB-5",  # Employment-based immigrant visas
    "IR-1", "IR-2", "CR-1", "CR-2",  # Immediate relatives
    "F-2A", "F-2B", "F-3", "F-4",  # Family preference immigrant visas
    "C",  # Transit visa
    "A",  # Diplomatic visas
    "G",  # Employees of international organizations
    "NATO",  # NATO officials and employees
    "R-1",  # Temporary religious worker visa
    "U", "T",  # Visas for victims of crimes or human trafficking
    "S",  # Witnesses or informants
    "C-1", "C-1/D",  # Crewmember visa
    "R",  # Religious Worker
    "NATO" # International Organization NATO
    ]


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
    - Available visa types: {visa_types}
    - Intake Form Questions:
    [
    "What is your country of citizenship?",
    "What is your country of residence?",
    "What visa type do you currently have?",
    "What are the dates of your current visa? (from - to)",
    "Have you ever visited the US before?",
    "Do you have a spouse?",
    "What is your spouse's country of citizenship?",
    "Have you ever applied for a US visa before and been denied?",
    "If you were denied a US visa, what was the reason?",
    "What are the dates of your denied visa application? (from - to)",
    "Do you have children?",
    "What is the country of citizenship of your child(ren)?",
    "Do you have any criminal history?",
    "If you have a criminal history, what was the crime and the results of the ruling?",
    "Have you ever entered the US illegally?",
    "If you entered the US illegally, when did you enter and when did you exit? (dates)",
    "Have you ever overstayed your visa's allotted time in the US?",
    "If you overstayed, what were the dates of the overstay? (from - to)",
    "Have you ever been deported?",
    "How many times have you been deported?",
    "What are the dates of your deportation(s)?"
    ]

    Output:
    Return ONLY the JSON object:
        - visa_types: []
        - intake_form_questions
        
    GUARDRAILS:
     - AlWAYS remove word json from the beginning of response.
"""


#Agent 2:  Extract Visa Types and Answers
def extract_visa_types_and_answers(chat_history):
    

    response = generation_model.generate_content(
    contents=[prompt,chat_history,visa_types], generation_config=generation_config
    ).text
    
    if response.startswith("```json"):
        response = response[7:].strip()
    if response.endswith("```"):
        response = response[:-3].strip() 


    response_json = json.loads(response)
    return response_json