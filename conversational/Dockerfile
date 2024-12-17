FROM python:3.9-slim

# Install dependencies from your requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8083

# Run the Streamlit app
CMD ["streamlit", "run", "browsingagent.py", "--server.port=8083", "--server.enableCORS=false"]
