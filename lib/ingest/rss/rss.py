""""""
import time
import json
import feedparser
from datetime import datetime, timezone
from kafka import KafkaProducer
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    HACKERNEWS_RSS_URL,
    RSS_POLLING_INTERVAL_SECONDS
)


def get_kafka_producer():
    """Initializes and returns a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Kafka producer initialized for topic: {KAFKA_TOPIC}")
        return producer
    except Exception as e:
        logger.error(f"Error initializing Kafka producer: {e}")
        return None


def clean_html(html_content):
    """Removes HTML tags from a string using BeautifulSoup."""
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    return ""


def fetch_and_process_rss(producer, rss_url):
    """Fetches the RSS feed, processes items, and sends them to Kafka."""
    logger.info(f"Fetching RSS feed from: {rss_url}")
    try:
        feed = feedparser.parse(rss_url)

        if feed.bozo:
            logger.warning(f"Error parsing RSS feed: {feed.bozo_exception}")
            # Continue processing if there are entries despite errors
            if not feed.entries:
                return

        for entry in feed.entries:
            # Extract and format data
            title = entry.title if hasattr(entry, 'title') else "No Title"

            description_html = entry.description if hasattr(entry, 'description') else ""
            content = clean_html(description_html)

            full_content = f"Title: {title}. Content: {content}"

            link = entry.link if hasattr(entry, 'link') else "No Link"
            source_id = entry.guid if hasattr(entry, 'guid') else link

            # Parse publication date to ISO format
            pub_date_str = entry.published if hasattr(entry, 'published') else datetime.now(timezone.utc).isoformat()
            try:
                # feedparser.parse() automatically converts dates to datetime objects in .published_parsed
                # We want ISO format string for consistency in our system
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    # Convert struct_time to datetime object, then to ISO format with UTC timezone
                    dt_object = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    timestamp = dt_object.isoformat()
                else:
                    # Fallback to current UTC time
                    timestamp = datetime.now(timezone.utc).isoformat()
            except Exception as date_e:
                logger.warning(
                    f"Could not parse date '{pub_date_str}' for entry '{title}': {date_e}. Using current time.")
                timestamp = datetime.now(timezone.utc).isoformat()

            communication_data = {
                "source_id": source_id,
                "type": "rss_post",
                "timestamp": timestamp,
                "content": full_content,
                "metadata": {
                    "title": title,
                    "link": link,
                    "creator": entry.author if hasattr(entry, 'author') else "Unknown",
                    "comments_url": entry.comments if hasattr(entry, 'comments') else None
                }
            }

            producer.send(KAFKA_TOPIC, value=communication_data)
            logger.info(f"Produced RSS message to Kafka: {title[:70]}...")

        producer.flush()
        logger.info("Finished processing RSS feed entries.")

    except Exception as e:
        logger.error(f"Error fetching or processing RSS feed: {e}")


if __name__ == "__main__":
    producer = get_kafka_producer()
    if producer:
        while True:
            fetch_and_process_rss(producer, HACKERNEWS_RSS_URL)
            logger.info(f"Waiting for {RSS_POLLING_INTERVAL_SECONDS} seconds before next RSS fetch...")
            time.sleep(RSS_POLLING_INTERVAL_SECONDS)
    else:
        logger.critical("Kafka producer not initialized. Exiting RSS producer.")
