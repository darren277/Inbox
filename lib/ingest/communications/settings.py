""""""
import os

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "inbox_topic")

# NLP Configuration
SPACY_MODEL = "en_core_web_sm" # Or "en_core_web_md" for better embeddings
STOP_WORDS = ["the", "a", "an", "and", "or", "but"] # Extend with financial stopwords

# Topic Modeling
NUM_TOPICS = 5 # Number of topics to discover

# Simulated Data
SIMULATED_DATA_INTERVAL_SECONDS = 5
