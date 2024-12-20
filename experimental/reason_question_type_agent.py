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
    temperature=0.2, max_output_tokens=256, top_k=40, top_p=0.8
)

prompt = """
    Task Overview:
    You are an agent that needs to determain question type from a LLM response. 
    Based on the question from chatbot send to you, your job is to:
     - Specify the question type with one of: "yes/no", "multiple_choice", "text_input", "number_input", in separate parameter, separated by new line.
        - Use **EXACTLY** this format for every question and question type:
            - Question Type: [type]
    
    Input:
        - Question: {question}
    
    Desired output:
        - Question Type: yes/no
        
    Examples:
        Input:  What is your name?
        Output: Question Type: text_input

        Input: Do you have a passport?
        Output: Question Type: yes/no

        Input: What is your date of birth?
        Output: Question Type: date_input

        Input: What is your preferred method of contact (email, phone, mail)?
        Output: Question Type: multiple_choice

        Input: How many years have you lived in this country?
        Output: Question Type: number_input

        Input: What is your country of citizenship?
        Output: Question Type: text_input


    Important:
    - Ensure that you output EXACTLY "Question Type: [type]" 
    - If you are unsure, default to text_input.

    Important:
    - Ensure that each question has a corresponding type

"""


#Agent 2:  Extract Visa Types and Answers
def reason_question_type(question):
    
    response = generation_model.generate_content(
    contents=[prompt,question], generation_config=generation_config
    ).text
    return response