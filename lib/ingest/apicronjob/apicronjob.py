""""""
import json
import requests
from datetime import datetime, timezone
from kafka import KafkaProducer
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    EXTERNAL_API_URL,
    EXTERNAL_API_TIMEOUT_SECONDS
    # EXTERNAL_API_KEY # Uncomment if your API needs a key
)


def get_kafka_producer():
    """Initializes and returns a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            request_timeout_ms=30000,
            api_version=(0, 10, 1)
        )
        logger.info(f"Kafka producer initialized for topic: {KAFKA_TOPIC}")
        return producer
    except Exception as e:
        logger.error(f"Error initializing Kafka producer: {e}")
        return None


def fetch_and_process_api_data(producer, api_url):
    """Fetches data from the API, processes items, and sends them to Kafka."""
    logger.info(f"Fetching data from API: {api_url}")

    headers = {
        "Accept": "application/json",
        # "Authorization": f"Bearer {EXTERNAL_API_KEY}" # Uncomment if using an API key
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=EXTERNAL_API_TIMEOUT_SECONDS)
        # Raise HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()
        try:
            data = response.json()

            if not isinstance(data, list):
                logger.warning(f"API response is not a list. Expected a list of items. Got: {type(data)}")
                data = [data]

            processed_count = 0
            for item in data:
                # --- TODO: Adapt this section to specific API response structure or convert to swappable function ---
                # For JSONPlaceholder /posts, typical item: {"userId": 1, "id": 1, "title": "...", "body": "..."}
                item_id = item.get("id")
                title = item.get("title", "No Title")
                body = item.get("body", "No Content")

                # Combine title and body for comprehensive content for NLP
                full_content = f"Title: {title}. Body: {body}"

                # Use current UTC time if API doesn't provide a specific timestamp
                # If your API has a field like 'createdAt' or 'published_at', use that!
                # Example: item_timestamp_str = item.get("createdAt")
                # If using item_timestamp_str, try: datetime.fromisoformat(item_timestamp_str).astimezone(timezone.utc).isoformat()
                timestamp = datetime.now(timezone.utc).isoformat()

                if not item_id:  # Generate a unique ID if API doesn't provide one
                    item_id = f"api_item_{hash(full_content)}_{timestamp}"
                    logger.warning(f"API item has no 'id'. Generated fallback ID: {item_id}")

                communication_data = {
                    "source_id": f"api_item_{item_id}",
                    "type": "external_api_event",
                    "timestamp": timestamp,
                    "content": full_content,
                    "metadata": {
                        "api_source": api_url,
                        "original_item_id": item_id,
                        "item_title": title,
                        # Add any other relevant fields from the API item here
                        # e.g., "author": item.get("author"), "category": item.get("category")
                    }
                }
                # --- End of API specific adaptation ---

                if producer:
                    producer.send(KAFKA_TOPIC, value=communication_data)
                    processed_count += 1
                    logger.info(f"Produced API message to Kafka: {title[:70]}...")
                else:
                    logger.error("Kafka producer not initialized. Cannot send message to Kafka.")

            if producer:
                producer.flush(timeout=30)
            logger.info(f"Finished processing {processed_count} API entries from {api_url}.")

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse API response as JSON: {json_err} - Response text: {response.text[:200]}...")

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred: {conn_err} - Check API URL or network.")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Request timed out: {timeout_err} - API might be slow or unresponsive.")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An unexpected request error occurred: {req_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during API processing: {e}", exc_info=True)


if __name__ == "__main__":
    producer = get_kafka_producer()
    if producer:
        fetch_and_process_api_data(producer, EXTERNAL_API_URL)
        producer.close()
    else:
        logger.critical("Kafka producer not initialized. Exiting API producer.")
        sys.exit(1)
