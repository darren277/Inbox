""""""
import os

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "inbox_topic")

# SurrealDB Configuration
SURREALDB_HOST = os.getenv("SURREALDB_HOST", "localhost")
SURREALDB_PORT = int(os.getenv("SURREALDB_PORT", 8000))
SURREALDB_DB = os.getenv("SURREALDB_DB", "inbox")
SURREALDB_NAMESPACE = os.getenv("SURREALDB_NAMESPACE", "inbox")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASSWORD", "Your$tr0ngP@ss")

# Topic Modeling
NUM_TOPICS = 5 # Number of topics to discover

SIMULATED_DATA_INTERVAL_SECONDS = 5
