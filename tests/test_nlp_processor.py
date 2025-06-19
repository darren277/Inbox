import pytest
from unittest.mock import AsyncMock, MagicMock, ANY
from datetime import datetime
import logging
import re

import src.nlp_processing as nlp_processor

# Ensure logging is captured by pytest
@pytest.fixture(autouse=True)
def caplog_fixture(caplog):
    caplog.set_level(logging.INFO) # Adjust as needed for specific tests


# Fixture to reset global variables in the nlp_processor module between tests
# because script uses global state for LDA model, dictionary, etc.
@pytest.fixture(autouse=True)
def reset_module_globals():
    nlp_processor.lda_model = None
    nlp_processor.dictionary = None
    nlp_processor.topic_keywords.clear()
    nlp_processor.topic_over_time.clear()

    nlp_processor.nlp = None
    nlp_processor.sentiment_analyzer = None

    yield


@pytest.fixture
def mock_nlp_model(mocker):
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
            mock_token.is_stop = word in nlp_processor.all_stopwords # Use actual stop words
            mock_token.is_punct = not word.isalnum()
            mock_token.is_space = False
            mock_tokens.append(mock_token)

        # Mock the __iter__ method to allow iteration over tokens
        mock_doc.__iter__.return_value = iter(mock_tokens)
        return mock_doc

    mock_nlp_instance.side_effect = mock_nlp_call

    mocker.patch.object(nlp_processor, 'nlp', new=mock_nlp_instance)
    yield mock_nlp_instance


# Fixture for a mock sentiment analyzer (Hugging Face pipeline)
@pytest.fixture
def mock_sentiment_analyzer(mocker):
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance.return_value = [{'label': 'POSITIVE', 'score': 0.95}]

    # ### CHANGED: Patch the 'sentiment_analyzer' global variable directly within nlp_processor. ###
    mocker.patch.object(nlp_processor, 'sentiment_analyzer', new=mock_analyzer_instance)

    # We want the original analyze_sentiment function to run and use the mocked sentiment_analyzer.
    yield mock_analyzer_instance


# Fixture for a mock KafkaConsumer
@pytest.fixture
def mock_kafka_consumer(mocker):
    class MockMessage:
        def __init__(self, value):
            self.value = value

    mock_consumer_instance = MagicMock()

    # Configure the mock instance to be iterable and yield MockMessage objects.
    # This will be overridden in specific tests that need to yield messages.
    mock_consumer_instance.__iter__.return_value = iter([]) # Default to no messages

    # Patch the KafkaConsumer class itself directly in the nlp_processor module's namespace
    mocker.patch('nlp_processor.KafkaConsumer', return_value=mock_consumer_instance)
    mock_consumer_instance.MockMessage = MockMessage
    yield mock_consumer_instance # Yield the instance directly


# Fixture for a mock SurrealDB client
@pytest.fixture
def mock_surrealdb_client(mocker):
    # Patch the Surreal class from surrealdb
    mock_surreal_class = mocker.patch('surrealdb.Surreal')
    mock_db_instance = AsyncMock()

    mock_db_instance.connect.return_value = None
    mock_db_instance.signin.return_value = None
    # Patch the constants used in get_surrealdb_client to match the mock
    mocker.patch('nlp_processor.SURREALDB_USER', 'mock_user')
    mocker.patch('nlp_processor.SURREALDB_PASS', 'mock_pass')
    mocker.patch('nlp_processor.SURREALDB_NAMESPACE', 'mock_namespace')
    mocker.patch('nlp_processor.SURREALDB_DB', 'mock_db_name')

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
    nlp_processor.corpora.Dictionary.return_value.doc2bow.side_effect = lambda tokens: [(i, 1) for i, _ in enumerate(tokens)]
    nlp_processor.corpora.Dictionary.return_value.merge_with.return_value = None

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

    nlp_processor.models.LdaModel.return_value = mock_lda_model

    nlp_processor.lda_model = mock_lda_model
    nlp_processor.dictionary = nlp_processor.corpora.Dictionary.return_value

    yield mock_lda_model


# --- Unit Tests ---

def test_preprocess_text_standard(mock_nlp_model):
    """
    Test preprocessing with a standard sentence, ensuring cleaning and tokenization.
    """
    text = "This is a Test sentence, with numbers 123 and symbols!."
    processed = nlp_processor.preprocess_text(text)
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
    processed = nlp_processor.preprocess_text(text)
    assert processed == []


def test_preprocess_text_only_stopwords(mock_nlp_model):
    """
    Test preprocessing with only stop words.
    """
    text = "The is a an"
    processed = nlp_processor.preprocess_text(text)
    assert processed == []


def test_preprocess_text_with_custom_stopwords(mock_nlp_model):
    """
    Test preprocessing with custom stop words defined in settings.
    """
    # Temporarily add a custom stop word to the actual set used by the processor
    original_stopwords_len = len(nlp_processor.all_stopwords)
    nlp_processor.all_stopwords.add('custom_stopword') # This is from settings.py example
    text = "This is a custom_stopword about a very important test."
    processed = nlp_processor.preprocess_text(text)
    assert 'custom_stopword' not in processed
    assert 'important' in processed
    assert len(processed) > 0
    # Clean up the added stop word for other tests
    nlp_processor.all_stopwords.remove('custom_stopword')
    assert len(nlp_processor.all_stopwords) == original_stopwords_len


def test_analyze_sentiment_llm_positive(mock_sentiment_analyzer):
    """
    Test sentiment analysis when LLM is active and predicts positive.
    """
    nlp_processor.sentiment_analyzer.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
    label, score = nlp_processor.analyze_sentiment("This is a fantastic product!")
    assert label == 'positive'
    assert score == 0.95


def test_analyze_sentiment_llm_negative(mock_sentiment_analyzer):
    """
    Test sentiment analysis when LLM is active and predicts negative.
    """
    nlp_processor.sentiment_analyzer.return_value = [{'label': 'NEGATIVE', 'score': 0.85}]
    label, score = nlp_processor.analyze_sentiment("The system failed due to an error.")
    assert label == 'negative'
    assert score == 0.85


def test_analyze_sentiment_llm_neutral(mock_sentiment_analyzer):
    """
    Test sentiment analysis when LLM is active and predicts neutral.
    """
    nlp_processor.sentiment_analyzer.return_value = [{'label': 'NEUTRAL', 'score': 0.6}]
    label, score = nlp_processor.analyze_sentiment("This is a factual report.")
    assert label == 'neutral'
    assert score == 0.6


def test_analyze_sentiment_fallback_negative(mocker):
    """
    Test sentiment analysis using the rule-based fallback for negative keywords.
    """
    # Temporarily set sentiment_analyzer to None to force fallback
    mocker.patch.object(nlp_processor, 'sentiment_analyzer', None)
    label, score = nlp_processor.analyze_sentiment("There was a security breach discovered.")
    assert label == 'negative'
    assert score == 0.8


def test_analyze_sentiment_fallback_positive(mocker):
    """
    Test sentiment analysis using the rule-based fallback for positive keywords.
    """
    mocker.patch.object(nlp_processor, 'sentiment_analyzer', None)
    label, score = nlp_processor.analyze_sentiment("The project was a huge success.")
    assert label == 'positive'
    assert score == 0.8


def test_analyze_sentiment_fallback_neutral(mocker):
    """
    Test sentiment analysis using the rule-based fallback for neutral text.
    """
    mocker.patch.object(nlp_processor, 'sentiment_analyzer', None)
    label, score = nlp_processor.analyze_sentiment("The quick brown fox jumps over the lazy dog.")
    assert label == 'neutral'
    assert score == 0.5


@pytest.mark.asyncio
async def test_get_surrealdb_client_success(mock_surrealdb_client):
    """
    Test successful connection to SurrealDB.
    """
    client = await nlp_processor.get_surrealdb_client()
    mock_surrealdb_client.connect.assert_called_once()
    mock_surrealdb_client.signin.assert_called_once_with({"user": "mock_user", "pass": "mock_pass"})
    mock_surrealdb_client.use.assert_called_once_with("mock_namespace", "mock_db_name")
    assert client == mock_surrealdb_client # Ensure the returned client is our mock


@pytest.mark.asyncio
async def test_get_surrealdb_client_failure(mock_surrealdb_client):
    """
    Test failure to connect to SurrealDB.
    """
    mock_surrealdb_client.connect.side_effect = Exception("DB connection failed for test")
    with pytest.raises(Exception, match="DB connection failed for test"):
        await nlp_processor.get_surrealdb_client()


def test_update_topic_model_initial(mock_gensim):
    """
    Test initial training of the LDA model.
    """
    corpus_data = [["word1", "word2"], ["word3", "word4", "word5"]]
    nlp_processor.update_topic_model(corpus_data)

    nlp_processor.corpora.Dictionary.assert_called_once_with(corpus_data)
    # Check that LdaModel constructor was called once for initial training
    nlp_processor.models.LdaModel.assert_called_once_with(
        corpus=ANY, id2word=nlp_processor.dictionary, num_topics=ANY,
        passes=ANY, random_state=ANY, alpha=ANY, eta=ANY
    )
    assert nlp_processor.lda_model is not None
    assert nlp_processor.dictionary is not None
    assert nlp_processor.topic_keywords # Should be populated by mock_gensim fixture


def test_update_topic_model_update(mock_gensim):
    """
    Test updating an existing LDA model with new data.
    """
    # Simulate initial training by calling it once
    nlp_processor.update_topic_model([["initial_word1", "initial_word2"]])
    initial_dict_call_count = nlp_processor.corpora.Dictionary.call_count
    initial_lda_call_count = nlp_processor.models.LdaModel.call_count

    # Now update with new data
    new_corpus_data = [["new_word1", "new_word2"]]
    nlp_processor.update_topic_model(new_corpus_data)

    # Dictionary should have been called again (for new_dictionary), and then merged
    assert nlp_processor.corpora.Dictionary.call_count == initial_dict_call_count + 1
    nlp_processor.corpora.Dictionary.return_value.merge_with.assert_called_once()
    # The existing LDA model's update method should have been called
    nlp_processor.lda_model.update.assert_called_once()
    # The LdaModel constructor should not be called again
    assert nlp_processor.models.LdaModel.call_count == initial_lda_call_count
    assert nlp_processor.topic_keywords # Should be updated by mock


def test_update_topic_model_empty_corpus(mock_gensim):
    """
    Test that the topic model is not updated with empty corpus data.
    """
    nlp_processor.update_topic_model([])
    nlp_processor.corpora.Dictionary.assert_not_called()
    nlp_processor.models.LdaModel.assert_not_called()
    assert nlp_processor.lda_model is None
    assert nlp_processor.dictionary is None


def test_get_document_topics_with_model(mock_gensim):
    """
    Test inferring topics for a document when a model exists.
    """
    # The mock_gensim fixture ensures lda_model and dictionary are set
    processed_tokens = ["test", "document", "word1"]
    topics = nlp_processor.get_document_topics(processed_tokens)
    nlp_processor.dictionary.doc2bow.assert_called_once_with(processed_tokens)
    nlp_processor.lda_model.get_document_topics.assert_called_once()
    # Based on mock_gensim return value for get_document_topics
    assert topics == [(0, 0.7), (1, 0.3)]


def test_get_document_topics_without_model():
    """
    Test inferring topics when no LDA model is loaded.
    """
    # Ensure global state is clean for this specific test
    nlp_processor.lda_model = None
    nlp_processor.dictionary = None
    processed_tokens = ["test", "document"]
    topics = nlp_processor.get_document_topics(processed_tokens)
    assert topics == []


def test_get_document_topics_empty_tokens(mock_gensim):
    """
    Test inferring topics with empty processed tokens.
    """
    processed_tokens = []
    topics = nlp_processor.get_document_topics(processed_tokens)
    assert topics == []


@pytest.mark.asyncio
async def test_process_communications_stream_success(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, mock_sentiment_analyzer, mock_gensim, caplog
):
    """
    Test the full message processing stream with multiple valid messages.
    Covers NLP, sentiment, topic modeling updates, and DB storage.
    """
    # Create mock messages data for Kafka consumer
    messages_data = [
        {
            "type": "email",
            "source_id": "email_123",
            "content": "This is a positive communication about a new product. It is great!",
            "timestamp": "2023-01-01T10:00:00Z"
        },
        {
            "type": "chat",
            "source_id": "chat_456",
            "content": "There was a small risk involved but we mitigated it successfully.",
            "timestamp": "2023-01-01T11:00:00Z"
        },
        {
            "type": "doc",
            "source_id": "doc_789",
            "content": "Another normal document, providing information.",
            "timestamp": "2023-01-02T12:00:00Z"
        }
    ]

    # This ensures that `message.value` in the loop is a plain dictionary.
    mock_kafka_consumer.__iter__.return_value = [mock_kafka_consumer.MockMessage(data) for data in messages_data]

    with caplog.at_level(logging.INFO):
        await nlp_processor.process_communications_stream()

    # Assertions for Kafka consumer initialization (it's now patched at the module level)
    nlp_processor.KafkaConsumer.assert_called_once_with(
        nlp_processor.KAFKA_TOPIC,
        bootstrap_servers=nlp_processor.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=ANY, # Deserializer function is hard to compare directly
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
    # The analyze_sentiment function itself is no longer patched in mock_sentiment_analyzer fixture.
    assert nlp_processor.analyze_sentiment.call_count == 3
    assert nlp_processor.dictionary.doc2bow.call_count == 3
    assert nlp_processor.lda_model.get_document_topics.call_count == 3

    # Verify topic model updates (buffer_size_for_update is 50 in original script)
    # With 3 messages, no update should occur yet as buffer_size_for_update is 50.
    # The initial `LdaModel` call will still happen if dictionary is created.
    assert nlp_processor.corpora.Dictionary.call_count == 1 # Only initial dictionary creation
    assert nlp_processor.models.LdaModel.call_count == 1 # Only initial LDA model training (no update)
    assert nlp_processor.lda_model.update.call_count == 0 # No update because buffer not full

    # Verify SurrealDB insertions (called once for each message)
    assert mock_surrealdb_client.query.call_count == 3
    # Check content of inserted data for the first message (positive from LLM mock)
    inserted_data_1 = mock_surrealdb_client.query.call_args_list[0].kwargs['data']
    assert inserted_data_1['sentiment'] == 'positive'
    assert isinstance(inserted_data_1['timestamp'], datetime)
    assert inserted_data_1['topics'] == [{"id": 0, "probability": 0.7}, {"id": 1, "probability": 0.3}]

    # Check content of inserted data for the second message (positive from LLM mock)
    inserted_data_2 = mock_surrealdb_client.query.call_args_list[1].kwargs['data']
    assert inserted_data_2['sentiment'] == 'positive'
    assert isinstance(inserted_data_2['timestamp'], datetime)

    # Check content of inserted data for the third message
    inserted_data_3 = mock_surrealdb_client.query.call_args_list[2].kwargs['data']
    assert inserted_data_3['sentiment'] == 'positive'
    assert isinstance(inserted_data_3['timestamp'], datetime)


    # Verify logging messages
    log_messages = [record.message for record in caplog.records]
    assert "Listening for messages on Kafka topic: test_topic" in log_messages
    assert "Connected to SurrealDB: mock_namespace:mock_db_name" in log_messages
    assert "Received message: Type='email'" in log_messages
    # The sentiment score from the mock sentiment_analyzer pipeline is 0.95
    assert any(f"Sentiment: positive (0.95)" in msg for msg in log_messages)
    # Should NOT see "Updating topic model" if buffer not full
    assert not any("Updating topic model" in msg for msg in log_messages)
    assert any("Stored processed message in SurrealDB." in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_process_communications_stream_empty_content(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, caplog
):
    """
    Test handling of messages with empty content.
    Should skip processing and logging a warning.
    """
    mock_message_data = {
        "type": "email",
        "source_id": "email_empty",
        "content": "",
        "timestamp": "2023-01-01T10:00:00Z"
    }

    mock_kafka_consumer.__iter__.return_value = [mock_kafka_consumer.MockMessage(mock_message_data)]

    with caplog.at_level(logging.WARNING):
        await nlp_processor.process_communications_stream()

    assert "Skipping empty communication" in caplog.text
    # Ensure no NLP, sentiment, topic modeling, or DB ops for empty content
    mock_nlp_model.assert_not_called()
    assert nlp_processor.analyze_sentiment.call_count == 0
    nlp_processor.update_topic_model.assert_not_called()
    nlp_processor.get_document_topics.assert_not_called()
    mock_surrealdb_client.query.assert_not_called()
    mock_surrealdb_client.close.assert_called_once() # Client should still be closed


@pytest.mark.asyncio
async def test_process_communications_stream_db_connection_failure(
    mock_kafka_consumer, mock_surrealdb_client, caplog
):
    """
    Test behavior when SurrealDB connection fails at startup.
    The stream processing loop should not start.
    """
    mock_surrealdb_client.connect.side_effect = Exception("Failed to connect for test")
    mock_kafka_consumer.__iter__.return_value = [] # No messages will be yielded

    with caplog.at_level(logging.ERROR):
        await nlp_processor.process_communications_stream()

    assert "Failed to connect to SurrealDB: Failed to connect for test" in caplog.text
    # Ensure the consumer loop is not even attempted if DB connection fails
    # The __iter__ method of the consumer instance should be called once to get the empty iterator.
    mock_kafka_consumer.__iter__.assert_called_once()
    mock_surrealdb_client.close.assert_not_called() # No close if connection never established


@pytest.mark.asyncio
async def test_process_communications_stream_db_insertion_failure(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, mock_sentiment_analyzer, mock_gensim, caplog
):
    """
    Test error handling when SurrealDB insertion fails for a message.
    Processing should continue to the next message.
    """
    messages_data = [
        {
            "type": "email",
            "source_id": "email_1",
            "content": "This is a communication that will fail DB insertion.",
            "timestamp": "2023-01-01T10:00:00Z"
        },
        {
            "type": "email",
            "source_id": "email_2",
            "content": "This is a communication that should succeed DB insertion.",
            "timestamp": "2023-01-01T10:01:00Z"
        }
    ]

    mock_kafka_consumer.__iter__.return_value = [mock_kafka_consumer.MockMessage(data) for data in messages_data]

    # Make the first query call fail, and subsequent ones succeed
    mock_surrealdb_client.query.side_effect = [
        Exception("DB insertion failed for test message 1"),
        {"status": "OK", "result": {"id": "comm:2", "content": "mock"}}
    ]

    with caplog.at_level(logging.ERROR):
        await nlp_processor.process_communications_stream()

    log_messages = [record.message for record in caplog.records]
    assert any("Error inserting into SurrealDB: DB insertion failed for test message 1" in msg for msg in log_messages)
    assert any("Stored processed message in SurrealDB." in msg for msg in log_messages)

    # All steps for both messages should have been attempted
    assert mock_nlp_model.call_count == 2
    assert nlp_processor.analyze_sentiment.call_count == 2
    assert mock_surrealdb_client.query.call_count == 2
    mock_surrealdb_client.close.assert_called_once() # Client should still be closed at the end


@pytest.mark.asyncio
async def test_process_communications_stream_no_timestamp(
    mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, mock_sentiment_analyzer, mock_gensim, caplog
):
    """
    Test handling of messages without a timestamp (or with a None timestamp).
    Verifies topic evolution data is NOT updated and that DB insertion is SKIPPED
    due to TypeError from datetime.fromisoformat(None) in the original script.
    """
    mock_message_data = {
        "type": "chat",
        "source_id": "chat_no_ts",
        "content": "Content without a timestamp.",
        "timestamp": None, # ### CHANGED: Explicitly set timestamp to None for testing behavior. ###
        "topics": [{"id": 0, "probability": 0.9}], # Will be overwritten by processing
    }

    mock_kafka_consumer.__iter__.return_value = [mock_kafka_consumer.MockMessage(mock_message_data)]

    # Clear topic_over_time to ensure it's not populated
    nlp_processor.topic_over_time.clear()

    with caplog.at_level(logging.ERROR): # ### CHANGED: Set level to ERROR to capture the expected DB error. ###
        await nlp_processor.process_communications_stream()

    # Ensure processing steps occurred before the DB insertion attempt
    mock_nlp_model.assert_called_once()
    assert nlp_processor.analyze_sentiment.call_count == 1
    mock_surrealdb_client.query.assert_not_called()

    # Crucially, topic_over_time should remain empty if timestamp is missing
    assert not nlp_processor.topic_over_time
    mock_surrealdb_client.close.assert_called_once()

    found_error_log = False
    expected_error_substring = "Error inserting into SurrealDB: TypeError: fromisoformat: argument must be str, bytes, or os.PathLike object, not NoneType"
    for record in caplog.records:
        if record.levelname == 'ERROR' and expected_error_substring in record.message:
            found_error_log = True
            break
    assert found_error_log, f"Expected error log '{expected_error_substring}' not found in logs."

