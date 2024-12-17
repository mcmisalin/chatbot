FROM python:3.9-slim

# Install dependencies from your requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8084

# Run the Streamlit app
CMD ["streamlit", "run", "decision_tree_agent.py", "--server.port=8084", "--server.enableCORS=false"]
