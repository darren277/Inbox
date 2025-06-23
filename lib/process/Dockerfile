FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy model during build
RUN python -m spacy download en_core_web_sm

COPY process.py src/
COPY settings.py .

CMD ["python", "src/process.py"]
