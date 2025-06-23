FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apicronjob.py src/
COPY settings.py src/

CMD ["python", "src/apicronjob.py"]
