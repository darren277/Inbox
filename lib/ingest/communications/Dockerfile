FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for pydub/ffmpeg if handling audio
# For Debian/Ubuntu based images:
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY communications.py src/
COPY settings.py src/
#COPY text_data/ text_data/
#COPY audio_data/ audio_data/

CMD ["python", "src/communications.py"]
