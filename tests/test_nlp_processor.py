""""""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
import logging
import re

from pytest_mock import mocker

import src.nlp_processing

@pytest.fixture(autouse=True)
def caplog_fixture(caplog):
    caplog.set_level(logging.INFO)


# Fixture to reset global variables in the nlp_processor module between tests
# because script uses global state for LDA model, dictionary, etc.
@pytest.fixture(autouse=True)
def reset_module_globals():
    src.nlp_processing.lda_model = None
    src.nlp_processing.dictionary = None
    src.nlp_processing.topic_keywords.clear()
    src.nlp_processing.topic_over_time.clear()

    src.nlp_processing.nlp = None
    src.nlp_processing.sentiment_analyzer = None

    yield


@pytest.fixture
def mock_nlp_model(mocker):
    mock_spacy_load = mocker.patch('spacy.load')
    mock_nlp_instance = MagicMock()

    # Define the behavior of calling the nlp instance (e.g., nlp(text))
    def mock_nlp_call(text):
        mock_doc = MagicMock()
        mock_tokens = []
        # Simulate basic tokenization and attribute assignment for testing `preprocess_text`
        # This mock needs to be somewhat intelligent to correctly test filtering logic.
        words = re.findall(r'[a-zA-Z]+', text.lower())
        for word in words:
            mock_token = MagicMock()
            mock_token.lemma_ = word # Simplistic lemmatization for mock
            mock_token.is_stop = word in src.nlp_processing.all_stopwords # Use actual stop words
            mock_token.is_punct = not word.isalnum()
            mock_token.is_space = False
            mock_tokens.append(mock_token)

        # Mock the __iter__ method to allow iteration over tokens
        mock_doc.__iter__.return_value = iter(mock_tokens)
        return mock_doc

    mock_nlp_instance.side_effect = mock_nlp_call
    mock_spacy_load.return_value = mock_nlp_instance

    # Explicitly set the global nlp variable in the module to our mock
    src.nlp_processing.nlp = mock_nlp_instance
    yield mock_nlp_instance


# Fixture for a mock sentiment analyzer (Hugging Face pipeline)
@pytest.fixture
def mock_sentiment_analyzer(mocker):
    mock_pipeline = mocker.patch('transformers.pipeline')
    mock_analyzer_instance = MagicMock()

    mock_analyzer_instance.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
    mock_pipeline.return_value = mock_analyzer_instance

    # Explicitly set the global sentiment_analyzer variable in the module to our mock
    src.nlp_processing.sentiment_analyzer = mock_analyzer_instance
    yield mock_analyzer_instance


# Fixture for a mock KafkaConsumer
@pytest.fixture
def mock_kafka_consumer(mocker):
    # Patch the KafkaConsumer class itself
    mock_consumer_class = mocker.patch('kafka.KafkaConsumer')
    # Create a mock instance for when KafkaConsumer(...) is called
    mock_consumer_instance = MagicMock()

    # Configure the mock instance to be iterable and yield mock messages
    # This will be overridden in specific tests that need to yield messages.
    mock_consumer_instance.__iter__.return_value = iter([]) # Default to no messages
    mock_consumer_class.return_value = mock_consumer_instance
    yield mock_consumer_class # Yield the patched class, so tests can configure its return_value


# Fixture for a mock SurrealDB client
@pytest.fixture
def mock_surrealdb_client(mocker):
    # Patch the Surreal class from surrealdb
    mock_surreal_class = mocker.patch('surrealdb.Surreal')
    mock_db_instance = AsyncMock()

    mock_db_instance.connect.return_value = None
    mock_db_instance.signin.return_value = None
    mock_db_instance.use.return_value = None
    mock_db_instance.query.return_value = [{"status": "OK", "result": {"id": "comm:1", "content": "mock"}}]
    mock_db_instance.close.return_value = None

    mock_surreal_class.return_value = mock_db_instance
    yield mock_db_instance


# Fixture to mock Gensim components
@pytest.fixture
def mock_gensim(mocker):
    mocker.patch('gensim.corpora.Dictionary')
    mocker.patch('gensim.models.LdaModel')

    # Configure mocked Dictionary to return a mock doc2bow
    # A simple mock: for each token, return (index, count)
    src.nlp_processing.corpora.Dictionary.return_value.doc2bow.side_effect = lambda tokens: [(i, 1) for i, _ in enumerate(tokens)]
    src.nlp_processing.corpora.Dictionary.return_value.merge_with.return_value = None

    mock_lda_model = MagicMock()
    # Mock print_topics to return predefined topic keywords
    mock_lda_model.print_topics.return_value = [
        (0, '0.6*"product" + 0.4*"new"'),
        (1, '0.7*"security" + 0.3*"risk"')
    ]
    # Mock get_document_topics to return predefined topics and probabilities
    mock_lda_model.get_document_topics.return_value = [(0, 0.7), (1, 0.3)]
    # Mock the update method for online learning simulation
    mock_lda_model.update.return_value = None

    src.nlp_processing.models.LdaModel.return_value = mock_lda_model

    src.nlp_processing.lda_model = mock_lda_model
    src.nlp_processing.dictionary = src.nlp_processing.corpora.Dictionary.return_value

    yield mock_lda_model


# --- Unit Tests ---

def test_preprocess_text_standard(mock_nlp_model):
    """
    Test preprocessing with a standard sentence, ensuring cleaning and tokenization.
    """
    text = "This is a Test sentence, with numbers 123 and symbols!."
    processed = src.nlp_processing.preprocess_text(text)
    assert 'test' in processed
    assert 'sentence' in processed
    assert 'number' not in processed # Numbers should be removed
    assert 'symbol' not in processed # Symbols should be removed
    assert 'this' not in processed # 'this' should be removed as it's a stop word
    assert 'is' not in processed # 'is' should be removed as it's a stop word
    assert 'a' not in processed # 'a' should be removed as it's a stop word
    assert len(processed) > 0 # Should contain valid tokens


def test_preprocess_text_empty(mock_nlp_model):
    """
    Test preprocessing with an empty string.
    """
    text = ""
    processed = src.nlp_processing.preprocess_text(text)
    assert processed == []


def test_preprocess_text_only_stopwords(mock_nlp_model):
    """
    Test preprocessing with only stop words.
    """
    text = "The is a an"
    processed = src.nlp_processing.preprocess_text(text)
    assert processed == []


def test_preprocess_text_with_custom_stopwords(mock_nlp_model):
    """
    Test preprocessing with custom stop words defined in settings.
    """
    # Temporarily add a custom stop word to the actual set used by the processor
    original_stopwords_len = len(src.nlp_processing.all_stopwords)
    src.nlp_processing.all_stopwords.add('custom_stopword') # This is from settings.py example
    text = "This is a custom_stopword about a very important test."
    processed = src.nlp_processing.preprocess_text(text)
    assert 'custom_stopword' not in processed
    assert 'important' in processed
    assert len(processed) > 0
    # Clean up the added stop word for other tests
    src.nlp_processing.all_stopwords.remove('custom_stopword')
    assert len(src.nlp_processing.all_stopwords) == original_stopwords_len


def test_analyze_sentiment_llm_positive(mock_sentiment_analyzer):
    """
    Test sentiment analysis when LLM is active and predicts positive.
    """
    mock_sentiment_analyzer.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
    label, score = src.nlp_processing.analyze_sentiment("This is a fantastic product!")
    assert label == 'positive'
    assert score == 0.95


def test_analyze_sentiment_llm_negative(mock_sentiment_analyzer):
    """
    Test sentiment analysis when LLM is active and predicts negative.
    """
    mock_sentiment_analyzer.return_value = [{'label': 'NEGATIVE', 'score': 0.85}]
    label, score = src.nlp_processing.analyze_sentiment("The system failed due to an error.")
    assert label == 'negative'
    assert score == 0.85


def test_analyze_sentiment_llm_neutral(mock_sentiment_analyzer):
    """
    Test sentiment analysis when LLM is active and predicts neutral.
    """
    mock_sentiment_analyzer.return_value = [{'label': 'NEUTRAL', 'score': 0.6}]
    label, score = src.nlp_processing.analyze_sentiment("This is a factual report.")
    assert label == 'neutral'
    assert score == 0.6


def test_analyze_sentiment_fallback_negative(mocker):
    """
    Test sentiment analysis using the rule-based fallback for negative keywords.
    """
    # Temporarily set sentiment_analyzer to None to force fallback
    mocker.patch.object(src.nlp_processing, 'sentiment_analyzer', None)
    label, score = src.nlp_processing.analyze_sentiment("There was a security breach discovered.")
    assert label == 'negative'
    assert score == 0.8


def test_analyze_sentiment_fallback_positive(mocker):
    """
    Test sentiment analysis using the rule-based fallback for positive keywords.
    """
    mocker.patch.object(src.nlp_processing, 'sentiment_analyzer', None)
    label, score = src.nlp_processing.analyze_sentiment("The project was a huge success.")
    assert label == 'positive'
    assert score == 0.8


def test_analyze_sentiment_fallback_neutral(mocker):
    """
    Test sentiment analysis using the rule-based fallback for neutral text.
    """
    mocker.patch.object(src.nlp_processing, 'sentiment_analyzer', None)
    label, score = src.nlp_processing.analyze_sentiment("The quick brown fox jumps over the lazy dog.")
    assert label == 'neutral'
    assert score == 0.5


async def test_get_surrealdb_client_success(mock_surrealdb_client):
    """
    Test successful connection to SurrealDB.
    """
    client = await src.nlp_processing.get_surrealdb_client()
    mock_surrealdb_client.connect.assert_called_once()
    mock_surrealdb_client.signin.assert_called_once_with({"user": "mock_user", "pass": "mock_pass"})
    mock_surrealdb_client.use.assert_called_once_with("mock_namespace", "mock_db_name")
    assert client == mock_surrealdb_client # Ensure the returned client is our mock


async def test_get_surrealdb_client_failure(mock_surrealdb_client):
    """
    Test failure to connect to SurrealDB.
    """
    mock_surrealdb_client.connect.side_effect = Exception("DB connection failed for test")
    with pytest.raises(Exception, match="DB connection failed for test"):
        await src.nlp_processing.get_surrealdb_client()


def test_update_topic_model_initial(mock_gensim):
    """
    Test initial training of the LDA model.
    """
    corpus_data = [["word1", "word2"], ["word3", "word4", "word5"]]
    src.nlp_processing.update_topic_model(corpus_data)

    src.nlp_processing.corpora.Dictionary.assert_called_once_with(corpus_data)
    # Check that LdaModel constructor was called once for initial training
    src.nlp_processing.models.LdaModel.assert_called_once_with(
        corpus=mocker.ANY, id2word=src.nlp_processing.dictionary, num_topics=mocker.ANY,
        passes=mocker.ANY, random_state=mocker.ANY, alpha=mocker.ANY, eta=mocker.ANY
    )
    assert src.nlp_processing.lda_model is not None
    assert src.nlp_processing.dictionary is not None
    assert src.nlp_processing.topic_keywords # Should be populated by mock_gensim fixture


def test_update_topic_model_update(mock_gensim):
    """
    Test updating an existing LDA model with new data.
    """
    # Simulate initial training by calling it once
    src.nlp_processing.update_topic_model([["initial_word1", "initial_word2"]])
    initial_dict_call_count = src.nlp_processing.corpora.Dictionary.call_count
    initial_lda_call_count = src.nlp_processing.models.LdaModel.call_count

    # Now update with new data
    new_corpus_data = [["new_word1", "new_word2"]]
    src.nlp_processing.update_topic_model(new_corpus_data)

    # Dictionary should have been called again (for new_dictionary), and then merged
    assert src.nlp_processing.corpora.Dictionary.call_count == initial_dict_call_count + 1
    src.nlp_processing.corpora.Dictionary.return_value.merge_with.assert_called_once()
    # The existing LDA model's update method should have been called
    src.nlp_processing.lda_model.update.assert_called_once()
    # The LdaModel constructor should not be called again
    assert src.nlp_processing.models.LdaModel.call_count == initial_lda_call_count
    assert src.nlp_processing.topic_keywords # Should be updated by mock


def test_update_topic_model_empty_corpus(mock_gensim):
    """
    Test that the topic model is not updated with empty corpus data.
    """
    src.nlp_processing.update_topic_model([])
    src.nlp_processing.corpora.Dictionary.assert_not_called()
    src.nlp_processing.models.LdaModel.assert_not_called()
    assert src.nlp_processing.lda_model is None
    assert src.nlp_processing.dictionary is None


def test_get_document_topics_with_model(mock_gensim):
    """
    Test inferring topics for a document when a model exists.
    """
    # The mock_gensim fixture ensures lda_model and dictionary are set
    processed_tokens = ["test", "document", "word1"]
    topics = src.nlp_processing.get_document_topics(processed_tokens)
    src.nlp_processing.dictionary.doc2bow.assert_called_once_with(processed_tokens)
    src.nlp_processing.lda_model.get_document_topics.assert_called_once()
    # Based on mock_gensim return value for get_document_topics
    assert topics == [(0, 0.7), (1, 0.3)]


def test_get_document_topics_without_model():
    """
    Test inferring topics when no LDA model is loaded.
    """
    # Ensure global state is clean for this specific test
    src.nlp_processing.lda_model = None
    src.nlp_processing.dictionary = None
    processed_tokens = ["test", "document"]
    topics = src.nlp_processing.get_document_topics(processed_tokens)
    assert topics == []


def test_get_document_topics_empty_tokens(mock_gensim):
    """
    Test inferring topics with empty processed tokens.
    """
    processed_tokens = []
    topics = src.nlp_processing.get_document_topics(processed_tokens)
    assert topics == []


async def test_process_communications_stream_success(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, mock_sentiment_analyzer, mock_gensim, caplog
):
    """
    Test the full message processing stream with multiple valid messages.
    Covers NLP, sentiment, topic modeling updates, and DB storage.
    """
    # Create mock messages for Kafka consumer
    mock_message_1 = MagicMock()
    mock_message_1.value = {
        "type": "email",
        "source_id": "email_123",
        "content": "This is a positive communication about a new product. It is great!",
        "timestamp": "2023-01-01T10:00:00Z"
    }
    mock_message_2 = MagicMock()
    mock_message_2.value = {
        "type": "chat",
        "source_id": "chat_456",
        "content": "There was a small risk involved but we mitigated it successfully.",
        "timestamp": "2023-01-01T11:00:00Z"
    }
    mock_message_3 = MagicMock()
    mock_message_3.value = {
        "type": "doc",
        "source_id": "doc_789",
        "content": "Another normal document, providing information.",
        "timestamp": "2023-01-02T12:00:00Z"
    }

    # Simulate the Kafka consumer yielding messages using an async generator
    async def message_generator():
        yield mock_message_1
        yield mock_message_2
        yield mock_message_3
        # The loop in process_communications_stream will exit after these messages

    # Set the return value for the KafkaConsumer constructor to our async generator
    mock_kafka_consumer.return_value = message_generator()

    with caplog.at_level(logging.INFO):
        await src.nlp_processing.process_communications_stream()

    # Assertions for Kafka consumer initialization
    mock_kafka_consumer.assert_called_once_with(
        src.nlp_processing.KAFKA_TOPIC,
        bootstrap_servers=src.nlp_processing.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=mocker.ANY, # Deserializer function is hard to compare directly
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='nlp_processing_group'
    )

    # Assertions for SurrealDB client operations
    mock_surrealdb_client.connect.assert_called_once()
    mock_surrealdb_client.signin.assert_called_once()
    mock_surrealdb_client.use.assert_called_once()
    mock_surrealdb_client.close.assert_called_once()

    # Verify that NLP preprocessing, sentiment analysis, and topic inference were called for each message
    assert mock_nlp_model.call_count == 3
    assert mock_sentiment_analyzer.call_count == 3
    assert src.nlp_processing.dictionary.doc2bow.call_count == 3
    assert src.nlp_processing.lda_model.get_document_topics.call_count == 3

    # Verify topic model updates (buffer_size_for_update is 2 in modified script)
    # First update after 2 messages, then no more updates because only 3 messages total
    assert src.nlp_processing.corpora.Dictionary.call_count == 2 # One for initial, one for merge
    assert src.nlp_processing.models.LdaModel.call_count == 1 # Initial training call
    assert src.nlp_processing.lda_model.update.call_count == 1 # One update after first 2 messages

    # Verify SurrealDB insertions (called once for each message)
    assert mock_surrealdb_client.query.call_count == 3
    # Check content of inserted data for the first message (positive from LLM mock)
    inserted_data_1 = mock_surrealdb_client.query.call_args_list[0].kwargs['data']
    assert inserted_data_1['sentiment'] == 'positive'
    assert isinstance(inserted_data_1['timestamp'], datetime)
    assert inserted_data_1['topics'] == [{"id": 0, "probability": 0.7}, {"id": 1, "probability": 0.3}]

    # Check content of inserted data for the second message (negative from LLM mock)
    # Note: If sentiment_analyzer mock returns POSITIVE by default, it will be positive.
    # To test fallback, we'd need to control sentiment_analyzer.return_value per message or patch it for that part.
    # Here, mock_sentiment_analyzer.return_value is positive by default for all calls.
    inserted_data_2 = mock_surrealdb_client.query.call_args_list[1].kwargs['data']
    assert inserted_data_2['sentiment'] == 'positive' # Based on mock_sentiment_analyzer's default
    assert isinstance(inserted_data_2['timestamp'], datetime)

    # Check content of inserted data for the third message
    inserted_data_3 = mock_surrealdb_client.query.call_args_list[2].kwargs['data']
    assert inserted_data_3['sentiment'] == 'positive' # Based on mock_sentiment_analyzer's default
    assert isinstance(inserted_data_3['timestamp'], datetime)


    # Verify logging messages
    assert "Listening for messages on Kafka topic: test_topic" in caplog.text
    assert "Connected to SurrealDB: test:test_db_name" in caplog.text # From settings mock
    assert "Received message: Type='email'" in caplog.text
    assert "Sentiment: positive (0.95)" in caplog.text # From LLM mock
    assert "Updating topic model with 2 new documents." in caplog.text
    assert "Stored processed message in SurrealDB." in caplog.text


async def test_process_communications_stream_empty_content(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, caplog
):
    """
    Test handling of messages with empty content.
    Should skip processing and logging a warning.
    """
    mock_message = MagicMock()
    mock_message.value = {
        "type": "email",
        "source_id": "email_empty",
        "content": "",
        "timestamp": "2023-01-01T10:00:00Z"
    }

    async def message_generator():
        yield mock_message
    mock_kafka_consumer.return_value = message_generator()

    with caplog.at_level(logging.WARNING):
        await src.nlp_processing.process_communications_stream()

    assert "Skipping empty communication" in caplog.text
    # Ensure no NLP, sentiment, topic modeling, or DB ops for empty content
    mock_nlp_model.assert_not_called()
    src.nlp_processing.analyze_sentiment.assert_not_called()
    src.nlp_processing.update_topic_model.assert_not_called()
    src.nlp_processing.get_document_topics.assert_not_called()
    mock_surrealdb_client.query.assert_not_called()
    mock_surrealdb_client.close.assert_called_once() # Client should still be closed


async def test_process_communications_stream_db_connection_failure(
    mock_kafka_consumer, mock_surrealdb_client, caplog
):
    """
    Test behavior when SurrealDB connection fails at startup.
    The stream processing loop should not start.
    """
    mock_surrealdb_client.connect.side_effect = Exception("Failed to connect for test")
    # Even if consumer is mocked, its iterator shouldn't be called if DB fails before loop
    mock_kafka_consumer.return_value = MagicMock()
    mock_kafka_consumer.return_value.__iter__.return_value = iter([]) # Ensure no messages

    with caplog.at_level(logging.ERROR):
        await src.nlp_processing.process_communications_stream()

    assert "Failed to connect to SurrealDB: Failed to connect for test" in caplog.text
    # Ensure the consumer loop is not even attempted if DB connection fails
    mock_kafka_consumer.return_value.__iter__.assert_not_called()
    mock_surrealdb_client.close.assert_not_called() # No close if connection never established


async def test_process_communications_stream_db_insertion_failure(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, mock_sentiment_analyzer, mock_gensim, caplog
):
    """
    Test error handling when SurrealDB insertion fails for a message.
    Processing should continue to the next message.
    """
    mock_message_1 = MagicMock()
    mock_message_1.value = {
        "type": "email",
        "source_id": "email_1",
        "content": "This is a communication that will fail DB insertion.",
        "timestamp": "2023-01-01T10:00:00Z"
    }
    mock_message_2 = MagicMock()
    mock_message_2.value = {
        "type": "email",
        "source_id": "email_2",
        "content": "This is a communication that should succeed DB insertion.",
        "timestamp": "2023-01-01T10:01:00Z"
    }

    async def message_generator():
        yield mock_message_1
        yield mock_message_2
    mock_kafka_consumer.return_value = message_generator()

    # Make the first query call fail, and subsequent ones succeed
    mock_surrealdb_client.query.side_effect = [
        Exception("DB insertion failed for test message 1"),
        {"status": "OK", "result": {"id": "comm:2", "content": "mock"}}
    ]

    with caplog.at_level(logging.ERROR):
        await src.nlp_processing.process_communications_stream()

    assert "Error inserting into SurrealDB: DB insertion failed for test message 1" in caplog.text
    assert "Stored processed message in SurrealDB." in caplog.text # The second message should succeed

    # All steps for both messages should have been attempted
    assert mock_nlp_model.call_count == 2
    assert mock_sentiment_analyzer.call_count == 2
    assert mock_surrealdb_client.query.call_count == 2
    mock_surrealdb_client.close.assert_called_once() # Client should still be closed at the end


async def test_process_communications_stream_no_timestamp(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, mock_sentiment_analyzer, mock_gensim, caplog
):
    """
    Test handling of messages without a timestamp.
    Topic evolution data should not be updated for such messages.
    """
    mock_message = MagicMock()
    mock_message.value = {
        "type": "chat",
        "source_id": "chat_no_ts",
        "content": "Content without a timestamp.",
        "topics": [{"id": 0, "probability": 0.9}], # Will be overwritten by processing
    }

    async def message_generator():
        yield mock_message
    mock_kafka_consumer.return_value = message_generator()

    # Clear topic_over_time to ensure it's not populated
    src.nlp_processing.topic_over_time.clear()

    with caplog.at_level(logging.INFO):
        await src.nlp_processing.process_communications_stream()

    # Ensure processing steps occurred
    mock_nlp_model.assert_called_once()
    mock_sentiment_analyzer.assert_called_once()
    mock_surrealdb_client.query.assert_called_once()

    # Crucially, topic_over_time should remain empty if timestamp is missing
    assert not src.nlp_processing.topic_over_time
    mock_surrealdb_client.close.assert_called_once()

