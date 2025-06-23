""""""
import json
import time
import os
import random
from datetime import datetime
from kafka import KafkaProducer
from pydub import AudioSegment
from speech_recognition import Recognizer, AudioFile, UnknownValueError, RequestError

from settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, SIMULATED_DATA_INTERVAL_SECONDS

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


# --- Speech-to-Text (STT) Function ---
def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file to text using Google Web Speech API (online).
    For a production environment, consider more robust/offline solutions like Whisper or Vosk.
    """
    r = Recognizer()
    try:
        audio = AudioSegment.from_file(audio_file_path)

        temp_wav_path = "temp_audio.wav"
        audio.export(temp_wav_path, format="wav")

        with AudioFile(temp_wav_path) as source:
            audio_data = r.record(source)

        # os.remove(temp_wav_path) # Clean up temporary file

        # Recognize speech using Google Web Speech API
        text = r.recognize_google(audio_data)
        return text
    except UnknownValueError:
        print(f"Google Speech Recognition could not understand audio in {audio_file_path}")
        return ""
    except RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {e}")
        return ""
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def simulate_communications_stream(text_data_dir="text_data", audio_data_dir="audio_data"):
    text_files = [os.path.join(text_data_dir, f) for f in os.listdir(text_data_dir) if f.endswith(".txt")]
    audio_files = [os.path.join(audio_data_dir, f) for f in os.listdir(audio_data_dir) if f.endswith((".wav", ".mp3"))]

    print(f"Starting simulated data stream to Kafka topic: {KAFKA_TOPIC}")
    print(f"Found {len(text_files)} text files and {len(audio_files)} audio files.")

    while True:
        communication_type = random.choice(["text", "audio"])

        message_data = {
            "timestamp": datetime.now().isoformat(),
            "source_id": f"sim_user_{random.randint(1, 100)}",
            "channel": f"{communication_type}_chat",
            "content": "",
            "type": communication_type,
            "sentiment": "neutral" # Placeholder for later analysis
        }

        if communication_type == "text" and text_files:
            file_path = random.choice(text_files)
            with open(file_path, "r", encoding="utf-8") as f:
                message_data["content"] = f.read().strip()
            print(f"Producing text message from {os.path.basename(file_path)}")
        elif communication_type == "audio" and audio_files:
            file_path = random.choice(audio_files)
            print(f"Transcribing and producing audio message from {os.path.basename(file_path)}")
            transcript = transcribe_audio(file_path)
            message_data["content"] = transcript
            message_data["original_audio_path"] = file_path
        else:
            print("No data files available for simulation.")
            time.sleep(SIMULATED_DATA_INTERVAL_SECONDS)
            continue

        if message_data["content"]:
            producer.send(KAFKA_TOPIC, message_data)
            print(f"Produced message: {message_data['type']} communication at {message_data['timestamp']}")
        else:
            print(f"Skipped producing empty message for {communication_type} communication.")

        time.sleep(SIMULATED_DATA_INTERVAL_SECONDS)


if __name__ == "__main__":
    os.makedirs("text_data", exist_ok=True)
    os.makedirs("audio_data", exist_ok=True)

    with open("text_data/chat_1.txt", "w") as f:
        f.write("Hey team, I think we should reconsider that trade on company X. The market looks volatile. Also, remember the compliance guidelines.")
    with open("text_data/chat_2.txt", "w") as f:
        f.write("Just confirming the meeting for tomorrow. Please bring all relevant documents. No sensitive information in public channels!")
    with open("text_data/chat_3.txt", "w") as f:
        f.write("The new regulations might impact our strategy. Need to analyze the impact quickly. Let's touch base ASAP.")
    with open("text_data/chat_4_risky.txt", "w") as f:
        f.write("Pssst. Heard a rumor about acquisition of ZYX. Buy before the announcement. Keep it quiet. This is insider info.")
    with open("text_data/chat_5_negative.txt", "w") as f:
        f.write("I am extremely frustrated with the progress. This project is a disaster. We need a new approach, urgently.")

    # You would need actual .wav files in 'audio_data/' for audio transcription to work.
    # For demonstration, you can create dummy empty files, but transcription will yield empty strings.
    # To test STT, put some actual short .wav or .mp3 files here.
    with open("audio_data/call_1.wav", "w") as f: f.write("")
    with open("audio_data/call_2.wav", "w") as f: f.write("")

    # To run this, you'll need a Kafka instance running (e.g., using Docker).
    # docker run -p 9092:9092 -e KAFKA_ADVERTISED_LISTENERS='PLAINTEXT://localhost:9092' -e KAFKA_LISTENERS='PLAINTEXT://0.0.0.0:9092' -e KAFKA_ZOOKEEPER_CONNECT='localhost:2181' -t confluentinc/cp-kafka:latest
    # Or for a quick local setup without Zookeeper:
    # docker run -p 9092:9092 -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092 -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 -e KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0 -t confluentinc/cp-kafka:latest

    # And you might need to install ffmpeg for pydub to handle .mp3 etc.
    # On macOS: brew install ffmpeg
    # On Ubuntu: sudo apt-get install ffmpeg
    # On Windows: Download from official site and add to PATH

    simulate_communications_stream()