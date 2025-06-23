FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install 'flask[async]'
RUN pip install hypercorn

COPY webhook.py src/
COPY settings.py src/

CMD ["python", "src/webhook.py"]
