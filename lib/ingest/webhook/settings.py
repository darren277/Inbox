""""""
import os

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "inbox_topic")

# Webhooks
WEBHOOK_FLASK_PORT = int(os.getenv("WEBHOOK_FLASK_PORT", 5001)) # Different port than the dashboard app
SLACK_VERIFICATION_TOKEN = os.getenv("SLACK_VERIFICATION_TOKEN", None) # Optional, but highly recommended for security

# Topic Modeling
NUM_TOPICS = 5 # Number of topics to discover

# Simulated Data
SIMULATED_DATA_INTERVAL_SECONDS = 5
