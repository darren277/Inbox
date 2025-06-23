""""""
from flask import Flask, request, jsonify
from kafka import KafkaProducer
import json
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    WEBHOOK_FLASK_PORT,
    SLACK_VERIFICATION_TOKEN
)

app = Flask(__name__)

producer = None
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    logger.info("Kafka producer initialized for Slack webhook server.")
except Exception as e:
    logger.error(f"Error initializing Kafka producer for webhook server: {e}")
    # Handle this more gracefully in production (e.g., retry logic, health checks)


@app.route('/slack/events', methods=['POST'])
def slack_events():
    """
    Receives Slack webhook events.
    Handles URL verification and message parsing.
    """
    logger.info(f"Received Slack webhook request. Headers: {request.headers}")

    # Check for Slack's URL verification challenge
    if request.json and "challenge" in request.json and "type" in request.json and request.json[
        "type"] == "url_verification":
        challenge = request.json.get("challenge")
        logger.info(f"Responding to Slack URL verification challenge: {challenge}")
        return jsonify({"challenge": challenge})

    if request.json and "event" in request.json:
        payload = request.json
        logger.debug(f"Slack Event Payload: {json.dumps(payload, indent=2)}")

        # Basic token verification (optional but recommended)
        # For production, implement full request signature verification as per Slack API docs
        if SLACK_VERIFICATION_TOKEN and payload.get("token") != SLACK_VERIFICATION_TOKEN:
            logger.warning("Invalid Slack verification token.")
            return jsonify({"status": "error", "message": "Invalid verification token"}), 403

        event_type = payload["event"]["type"]

        # Process different event types
        if event_type == "message":
            # Ignore bot messages to prevent infinite loops or processing of internal messages
            if "subtype" in payload["event"] and payload["event"]["subtype"] in ["bot_message", "message_changed", "message_deleted"]:
                logger.info(f"Ignoring Slack bot message or message modification: {payload['event'].get('subtype')}")
                return jsonify({"status": "ignored", "message": "Bot message or message modification."}), 200

            # Extract message content
            message_text = payload["event"].get("text")
            channel_id = payload["event"].get("channel")
            user_id = payload["event"].get("user")
            event_ts = payload["event"].get("event_ts")

            if message_text:
                # Convert Slack's event_ts (Unix timestamp string) to ISO format
                try:
                    timestamp = datetime.fromtimestamp(float(event_ts), tz=timezone.utc).isoformat()
                except (TypeError, ValueError):
                    timestamp = datetime.now(timezone.utc).isoformat()
                    logger.warning(f"Could not parse Slack event_ts '{event_ts}'. Using current UTC time.")

                communication_data = {
                    "source_id": f"slack_message_{channel_id}_{event_ts}", # Unique ID for the message
                    "type": "slack_message",
                    "timestamp": timestamp,
                    "content": message_text,
                    "metadata": {
                        "channel_id": channel_id,
                        "user_id": user_id,
                        "source_event_type": event_type,
                        "team_id": payload.get("team_id")
                    }
                }

                if producer:
                    try:
                        producer.send(KAFKA_TOPIC, value=communication_data)
                        producer.flush()
                        logger.info(
                            f"Sent Slack message to Kafka: Channel={channel_id}, User={user_id}, Text='{message_text[:50]}...'")
                    except Exception as kafka_e:
                        logger.error(f"Error sending Slack message to Kafka: {kafka_e}")
                        # Slack webhooks expect a quick 200 response to avoid retries
                        # TODO: A separate retry mechanism for Kafka failures might be better
                else:
                    logger.error("Kafka producer not initialized. Cannot send message to Kafka.")
            else:
                logger.info(f"Received Slack message event with no 'text' content. Event: {payload['event']}")
        else:
            logger.info(f"Received unhandled Slack event type: {event_type}")

        return jsonify({"status": "ok", "message": "Event received and processed."}), 200

    logger.warning(f"Received unhandled webhook request: {request.json}")
    return jsonify({"status": "error", "message": "Unhandled request type"}), 400


@app.route('/healthz', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if producer and producer.bootstrap_connected():
        return "OK", 200
    return "Kafka Producer Not Ready", 503


if __name__ == '__main__':
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"0.0.0.0:{WEBHOOK_FLASK_PORT}"]

    import asyncio

    asyncio.run(serve(app, config))
