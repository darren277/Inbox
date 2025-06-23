FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install 'flask[async]'

RUN pip install gunicorn
# Install Uvicorn for async support
RUN pip install uvicorn
RUN pip install asgiref

COPY dashboard.py ./
COPY settings.py ./

COPY topic_keywords.json ./

COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:5000", "dashboard:wsgi"]
