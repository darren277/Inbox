""""""
import os

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "inbox_topic")

# APIs
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "https://jsonplaceholder.typicode.com/posts") # Example: JSONPlaceholder posts
EXTERNAL_API_POLLING_CRON_SCHEDULE = os.getenv("EXTERNAL_API_POLLING_CRON_SCHEDULE", "*/10 * * * *") # Every 10 minutes
EXTERNAL_API_TIMEOUT_SECONDS = int(os.getenv("EXTERNAL_API_TIMEOUT_SECONDS", 15))
# Optional: API Key for real APIs (e.g., if you use a news API)
# EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", None)

# Topic Modeling
NUM_TOPICS = 5 # Number of topics to discover

# Simulated Data
SIMULATED_DATA_INTERVAL_SECONDS = 5
