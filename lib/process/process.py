""""""
from collections import defaultdict
from datetime import datetime
import json
import re
import logging
from gensim import corpora, models
from kafka import KafkaConsumer
from nltk.corpus import stopwords
import spacy
from surrealdb import Surreal
from transformers import pipeline

from settings import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, SPACY_MODEL, STOP_WORDS, NUM_TOPICS,
    SURREALDB_HOST, SURREALDB_PORT, SURREALDB_NAMESPACE, SURREALDB_DB, SURREALDB_USER, SURREALDB_PASS,
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load NLP Model ---
try:
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"SpaCy model '{SPACY_MODEL}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading SpaCy model: {e}. Please run `python -m spacy download {SPACY_MODEL}`")
    exit()

# Extend NLTK stopwords
nltk_stopwords = set(stopwords.words('english'))
all_stopwords = nltk_stopwords.union(set(STOP_WORDS))

# --- Language model for Sentiment (Optional - requires `transformers` and `torch` or `tensorflow`) ---
# You can choose a smaller, more efficient model for production.
# For demo, 'distilbert-base-uncased-finetuned-sst-2-english' is good.
# Or 'cardiffnlp/twitter-roberta-base-sentiment-latest' for more general sentiment.
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logger.info("Sentiment analyzer (Hugging Face pipeline) loaded.")
except Exception as e:
    sentiment_analyzer = None
    logger.warning(f"Could not load sentiment analyzer pipeline: {e}. Sentiment analysis will be basic.")


# --- NLP Preprocessing Functions ---
def preprocess_text(text):
    """
    Cleans and tokenizes text for NLP.
    """
    if not text:
        return []

    # Remove special characters, numbers (keep some punctuation for context if needed, but remove most for topic modeling)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 2
    ]
    return tokens


# --- Topic Modeling ---
# Initialize LDA model (will be updated over time or retrained)
lda_model = None
dictionary = None
topic_keywords = defaultdict(list)
topic_over_time = defaultdict(lambda: defaultdict(int))  # {topic_id: {date: count}}


def update_topic_model(corpus_data):
    """
    Updates or retrains the LDA model with new data.
    In a real system, you'd likely retrain periodically or use online LDA.
    """
    global lda_model, dictionary, topic_keywords

    if not corpus_data:
        return

    # Create a dictionary from the new corpus
    new_dictionary = corpora.Dictionary(corpus_data)

    # Combine with existing dictionary if it exists
    if dictionary:
        dictionary.merge_with(new_dictionary)
    else:
        dictionary = new_dictionary

    # Create a BoW corpus
    corpus = [dictionary.doc2bow(doc) for doc in corpus_data]

    if lda_model:
        # If LDA model exists, we can train it further (online learning)
        # Note: Gensim's LdaModel has `update` method for online learning.
        # For simplicity here, we'll re-train, but `update` is better for real-time.
        lda_model.update(corpus)
    else:
        # Train a new LDA model if none exists
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=NUM_TOPICS,
            passes=10,
            random_state=42,
            alpha='auto',
            eta='auto'
        )

    logger.info(f"LDA model updated/trained with {len(corpus_data)} documents.")
    # Extract top keywords for each topic
    topic_keywords.clear()
    for i, topic in lda_model.print_topics(num_words=5):
        words = [word.split('*')[1].strip().replace('"', '') for word in topic.split('+')]
        topic_keywords[i] = words
        logger.info(f"Topic {i}: {', '.join(words)}")


def get_document_topics(processed_tokens):
    """
    Infers topics for a single document.
    """
    if not lda_model or not dictionary or not processed_tokens:
        return []

    bow_vector = dictionary.doc2bow(processed_tokens)
    topics = lda_model.get_document_topics(bow_vector, minimum_probability=0.1)
    return sorted(topics, key=lambda x: x[1], reverse=True)  # Sort by probability


# --- Sentiment Analysis ---
def analyze_sentiment(text):
    if sentiment_analyzer:
        result = sentiment_analyzer(text)[0]
        label = result['label']
        score = result['score']
        # Normalize labels to 'positive', 'negative', 'neutral' if needed
        if label == 'POSITIVE':
            return 'positive', score
        elif label == 'NEGATIVE':
            return 'negative', score
        else:  # Covers 'NEITHER', 'NEU', etc. for some models
            return 'neutral', score
    else:
        # Fallback to a very basic rule-based sentiment if LLM not loaded
        text_lower = text.lower()
        if any(word in text_lower for word in ['risk', 'breach', 'fraud', 'bad', 'problem', 'failure', 'violation', 'disaster', 'crisis', 'insider']):
            return 'negative', 0.8
        elif any(word in text_lower for word in ['good', 'success', 'opportunity', 'compliant', 'approved', 'positive']):
            return 'positive', 0.8
        else:
            return 'neutral', 0.5


async def get_surrealdb_client():
    """Connects to SurrealDB and returns a client object."""
    from surrealdb import Surreal
    db = Surreal(f"ws://{SURREALDB_HOST}:{SURREALDB_PORT}/rpc")
    await db.connect()
    await db.signin({"user": SURREALDB_USER, "pass": SURREALDB_PASS})
    await db.use(SURREALDB_NAMESPACE, SURREALDB_DB)
    return db


# --- Kafka Consumer & Processing Loop ---

async def process_communications_stream():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='nlp_processing_group'
    )
    logger.info(f"Listening for messages on Kafka topic: {KAFKA_TOPIC}")

    surreal_db_client = None
    try:
        surreal_db_client = await get_surrealdb_client()
        logger.info(f"Connected to SurrealDB: {SURREALDB_NAMESPACE}:{SURREALDB_DB}")
    except Exception as e:
        logger.error(f"Failed to connect to SurrealDB: {e}")
        return  # Exit if DB connection fails

    text_buffer = []
    buffer_size_for_update = 50

    for message in consumer:
        communication = message.value
        content = communication.get("content")
        timestamp_str = communication.get("timestamp")  # Use this for actual timestamp if available

        if not content:
            logger.warning(f"Skipping empty communication: {communication}")
            continue

        logger.info(f"Received message: Type='{communication['type']}', Source='{communication['source_id']}', Content='{content[:50]}...'")

        # 1. NLP Preprocessing
        processed_tokens = preprocess_text(content)

        # 2. Sentiment Analysis
        sentiment_label, sentiment_score = analyze_sentiment(content)
        communication["sentiment"] = sentiment_label
        communication["sentiment_score"] = sentiment_score
        logger.info(f"  Sentiment: {sentiment_label} ({sentiment_score:.2f})")

        # 3. Topic Modeling
        text_buffer.append(processed_tokens)
        if len(text_buffer) >= buffer_size_for_update:
            logger.info(f"Updating topic model with {len(text_buffer)} new documents.")
            update_topic_model(text_buffer)
            text_buffer.clear()

        message_topics = get_document_topics(processed_tokens)
        communication["topics"] = [{"id": int(t[0]), "probability": float(t[1])} for t in message_topics]  # Ensure float for JSON

        # Update topic evolution data for visualization (this part for internal state, not DB)
        if timestamp_str and message_topics:
            message_date = datetime.fromisoformat(timestamp_str).strftime('%Y-%m-%d')
            for topic_info in message_topics:
                topic_id = int(topic_info[0])
                topic_over_time[topic_id][message_date] += 1

        # 4. Store in SurrealDB
        try:
            # Prepare data for SurrealDB insertion
            communication_for_db = communication.copy()
            # Convert timestamp string to a proper datetime object for SurrealDB
            communication_for_db['timestamp'] = datetime.fromisoformat(communication_for_db['timestamp'])

            # Using INSERT statement to create new records
            result = await surreal_db_client.query("INSERT INTO communications $data", {"data": communication_for_db})
            logger.info(f"  Stored processed message in SurrealDB. Topics: {communication['topics']}")
        except Exception as e:
            logger.error(f"Error inserting into SurrealDB: {e}")

        # ... (high-risk alert logic remains the same) ...

    if surreal_db_client:
        await surreal_db_client.close()




if __name__ == "__main__":
    import asyncio
    # Ensure SpaCy model is downloaded
    # Run: python -m spacy download en_core_web_sm

    # To run this, you'll need Kafka running (see data_ingestion.py comments)
    # And SurrealDB running: docker run -p 8000:8000 -p 8001:8001 surrealdb/surrealdb

    asyncio.run(process_communications_stream())
