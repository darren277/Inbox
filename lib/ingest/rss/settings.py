""""""
import os

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "inbox_topic")

# RSS Feeds
HACKERNEWS_RSS_URL = os.getenv("HACKERNEWS_RSS_URL", "https://hnrss.org/newest")
RSS_POLLING_INTERVAL_SECONDS = int(os.getenv("RSS_POLLING_INTERVAL_SECONDS", 300))

# Topic Modeling
NUM_TOPICS = 5 # Number of topics to discover

# Simulated Data
SIMULATED_DATA_INTERVAL_SECONDS = 5
