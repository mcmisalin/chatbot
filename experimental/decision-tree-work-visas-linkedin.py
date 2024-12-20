import streamlit as st
from google import genai
from google.genai import types
import json
import re
import requests

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "yes_no_questions" not in st.session_state:
    st.session_state.yes_no_questions = {}
if "info_to_confirm" not in st.session_state:
    st.session_state.info_to_confirm = {}

# Function to extract JSON blocks from response
def extract_json_blocks(response):
    json_blocks = re.findall(r'```json\n(.*?)\n```', response, re.DOTALL)
    parsed_blocks = []
    for block in json_blocks:
        try:
            parsed_json = json.loads(block)
            parsed_blocks.append(parsed_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON block: {e}")
    return parsed_blocks

# Function to extract LinkedIn data
def extract_from_linkedin(url):
    api_key = 'yiYjekZ4h5uyX0Rhl2K-qQ'  # Replace with your actual API key

    headers = {'Authorization': 'Bearer ' + api_key}
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    params = {
        'linkedin_profile_url': url,
        'extra': 'include',
        'inferred_salary': 'include',
        'skills': 'include',
        'use_cache': 'if-present',
        'fallback_to_cache': 'on-error',
    }
    response = requests.get(api_endpoint, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # Extracting the required fields
        city = data.get('city', 'Unknown')
        country_full_name = data.get('country_full_name', 'Unknown')
        field_of_study = None
        school = None
        for edu_entry in data.get('education', []):
            if edu_entry.get('degree_name') == 'BSc':
                field_of_study = edu_entry.get('field_of_study', 'Unknown')
                school = edu_entry.get('school', 'Unknown')
                break
        latest_experience = data.get('experiences', [{}])[0]
        latest_company = latest_experience.get('company', 'Unknown')
        latest_title = latest_experience.get('title', 'Unknown')
        first_name = data.get('first_name', 'Unknown')
        last_name = data.get('last_name', 'Unknown')

        return {
            "city": city,
            "country_full_name": country_full_name,
            "field_of_study": field_of_study,
            "school": school,
            "latest_company": latest_company,
            "latest_title": latest_title,
            "first_name": first_name,
            "last_name": last_name,
        }
    else:
        st.error("Failed to fetch data from LinkedIn.")
        return {}

# Mock search_immigration_database function
def search_immigration_database(query):
    print(f"Searching immigration database with query: {query}")
    return {
        "yes_no_questions": {
            "Do you have a job offer in the USA?": "This is to determine if you are eligible for H1-B or similar visas.",
            "Do you have a degree in a specialized field?": "Specialized degrees may be required for certain work visas.",
        },
        "info_to_confirm": {
            "Degree Field": "Computer Science",
            "Years of Work Experience": "5",
        }
    }

# Function to generate work visa questions
def generate_work_visa_questions(extracted_data):
    profile_query = f"Work visa eligibility for profile: {json.dumps(extracted_data)}"
    return search_immigration_database(profile_query)

# Streamlit UI
st.title("ImmPath Chatbot for Work Visa Funnel")

# LinkedIn URL Input
linkedin_url = st.text_input("Enter your LinkedIn profile URL:")

if linkedin_url:
    with st.spinner("Extracting LinkedIn data..."):
        extracted_data = extract_from_linkedin(linkedin_url)
        if extracted_data:
            # Display extracted LinkedIn data in the sidebar
            st.sidebar.title("LinkedIn Profile Information")
            st.sidebar.markdown("---")
            for key, value in extracted_data.items():
                st.sidebar.text(f"{key.replace('_', ' ').capitalize()}: {value}")

            # Generate questions based on the work visa funnel
            with st.spinner("Generating work visa questions..."):
                visa_questions = generate_work_visa_questions(extracted_data)
                st.session_state.user_info = visa_questions.get("info_to_confirm", {})
                st.session_state.yes_no_questions = visa_questions.get("yes_no_questions", {})

                # Sidebar: Confirm Your Information
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
                    st.sidebar.title("Quick Yes/No Questions")
                    st.sidebar.markdown("---")
                    for question_key, question_text in st.session_state.yes_no_questions.items():
                        answer = st.sidebar.radio(
                            f"{question_key}", ["Yes", "No"],
                            horizontal=True,
                            key=f"yesno_{question_key}"
                        )
                        st.session_state.user_info[question_key] = answer

                st.success("Questions and information have been updated.")
