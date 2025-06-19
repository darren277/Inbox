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
    # This mocks the Hugging Face pipeline object that the analyze_sentiment function uses.
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = [{'label': 'POSITIVE', 'score': 0.95}]

    # Patch the 'sentiment_analyzer' global variable in nlp_processor to be our mock pipeline.
    mocker.patch.object(nlp_processor, 'sentiment_analyzer', new=mock_pipeline_instance)

    # Patch the analyze_sentiment *function* itself.
    # This is crucial to be able to assert .call_count on it.
    # We configure its side_effect to call the *original* function's implementation
    # using our mocked `sentiment_analyzer` (pipeline). This allows the actual logic
    # within `analyze_sentiment` to run, while still making the function a mock.
    original_analyze_sentiment = nlp_processor.analyze_sentiment
    mock_analyze_sentiment_func = mocker.patch('src.nlp_processing.analyze_sentiment')

    # When the mocked analyze_sentiment_func is called, it will call the original function,
    # which in turn will use the mocked nlp_processor.sentiment_analyzer pipeline.
    # This setup ensures we can assert on `mock_analyze_sentiment_func.call_count`.
    mock_analyze_sentiment_func.side_effect = original_analyze_sentiment

    yield mock_pipeline_instance


# Fixture for a mock KafkaConsumer
@pytest.fixture
def mock_kafka_consumer(mocker):
    # Define a simple mock message class to ensure message.value is a dictionary.
    class MockMessage:
        def __init__(self, value):
            self.value = value

    mock_consumer_instance = MagicMock()

    # Configure the mock instance to be iterable and yield MockMessage objects.
    mock_consumer_instance.__iter__.return_value = iter([]) # Default to no messages

    # Patch the KafkaConsumer class itself directly in the nlp_processor module's namespace
    mocker.patch('src.nlp_processing.KafkaConsumer', return_value=mock_consumer_instance)
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
    mocker.patch('src.nlp_processing.SURREALDB_USER', 'mock_user')
    mocker.patch('src.nlp_processing.SURREALDB_PASS', 'mock_pass')
    mocker.patch('src.nlp_processing.SURREALDB_NAMESPACE', 'mock_namespace')
    mocker.patch('src.nlp_processing.SURREALDB_DB', 'mock_db_name')

    mock_db_instance.use.return_value = None
    mock_db_instance.query.return_value = [{"status": "OK", "result": {"id": "comm:1", "content": "mock"}}]
    mock_db_instance.close.return_value = None

    mock_surreal_class.return_value = mock_db_instance
    yield mock_db_instance


# Fixture to mock Gensim components
@pytest.fixture
def mock_gensim(mocker):
    # Patch gensim.corpora.Dictionary and gensim.models.LdaModel
    # as they are referenced *within* nlp_processor.py via nlp_processor.corpora and nlp_processor.models.
    mock_dictionary_class = mocker.patch('src.nlp_processing.corpora.Dictionary')
    mock_lda_model_class = mocker.patch('src.nlp_processing.models.LdaModel')

    # Configure mocked Dictionary to return a NEW mock instance each time it's called.
    # This is crucial for distinguishing between `dictionary` and `new_dictionary` within `update_topic_model`.
    def create_mock_dictionary(*args, **kwargs):
        mock_dict_instance = MagicMock()
        mock_dict_instance.doc2bow.side_effect = lambda tokens: [(i, 1) for i, _ in enumerate(tokens)]
        mock_dict_instance.merge_with.return_value = None
        return mock_dict_instance

    mock_dictionary_class.side_effect = create_mock_dictionary

    # Configure mocked LdaModel instance behavior
    mock_lda_model_instance = MagicMock()
    mock_lda_model_instance.print_topics.return_value = [
        (0, '0.6*"product" + 0.4*"new"'),
        (1, '0.7*"security" + 0.3*"risk"')
    ]
    mock_lda_model_instance.get_document_topics.return_value = [(0, 0.7), (1, 0.3)]
    mock_lda_model_instance.update.return_value = None

    # Configure the LdaModel class to return our specific mock instance when called.
    mock_lda_model_class.return_value = mock_lda_model_instance

    # No longer directly setting nlp_processor.lda_model and nlp_processor.dictionary here.
    # These globals are managed by the nlp_processor.py logic itself, which is what we want to test.
    yield


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
    # mock_sentiment_analyzer is the mock for the pipeline instance.
    # Here we are testing the behavior of the `analyze_sentiment` function itself
    # when its internal `sentiment_analyzer` (the pipeline) is mocked.
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
    # Temporarily set sentiment_analyzer to None to force fallback.
    # Note: For this test, we are bypassing the mock_sentiment_analyzer fixture's default setup
    # to specifically test the fallback path of the *original* analyze_sentiment function.
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
    assert client == mock_surrealdb_client  # Ensure the returned client is our mock


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
    # Ensure LDA model is None before calling update_topic_model for initial training scenario
    nlp_processor.lda_model = None
    nlp_processor.dictionary = None

    corpus_data = [["word1", "word2"], ["word3", "word4", "word5"]]
    nlp_processor.update_topic_model(corpus_data)

    # Assert that Dictionary was called (to create the first dictionary)
    nlp_processor.corpora.Dictionary.assert_called_once_with(corpus_data)
    # Assert that LdaModel class was instantiated for initial training
    nlp_processor.models.LdaModel.assert_called_once_with(
        corpus=ANY, id2word=ANY, num_topics=ANY,
        passes=ANY, random_state=ANY, alpha=ANY, eta=ANY
    )
    assert nlp_processor.lda_model is not None
    assert nlp_processor.dictionary is not None
    assert nlp_processor.topic_keywords  # Should be populated by mock_gensim fixture's return for print_topics


def test_update_topic_model_update(mock_gensim):
    """
    Test updating an existing LDA model with new data.
    """
    # Manually set up initial state as if LDA model was already trained.
    # This uses actual mocks for the instances from the fixture, ensuring they are distinct.
    initial_dict_mock = MagicMock()
    initial_dict_mock.doc2bow.side_effect = lambda tokens: [(i, 1) for i, _ in enumerate(tokens)]
    initial_dict_mock.merge_with.return_value = None # This will be called on the *initial* dictionary mock

    mock_lda_model_instance = MagicMock()
    mock_lda_model_instance.update.return_value = None
    mock_lda_model_instance.print_topics.return_value = [
        (0, '0.6*"product" + 0.4*"new"'), (1, '0.7*"security" + 0.3*"risk"')
    ]
    mock_lda_model_instance.get_document_topics.return_value = [(0, 0.7), (1, 0.3)]

    nlp_processor.lda_model = mock_lda_model_instance
    nlp_processor.dictionary = initial_dict_mock

    # Clear calls before the actual test logic begins to avoid double counting from setup
    nlp_processor.corpora.Dictionary.reset_mock()
    nlp_processor.models.LdaModel.reset_mock()
    initial_dict_mock.merge_with.reset_mock()
    mock_lda_model_instance.update.reset_mock()

    new_corpus_data = [["new_word1", "new_word2"]]
    nlp_processor.update_topic_model(new_corpus_data)

    # A new dictionary should be created from `new_corpus_data`.
    nlp_processor.corpora.Dictionary.assert_called_once_with(new_corpus_data)
    # The initial dictionary's `merge_with` method should have been called once with the new dictionary.
    initial_dict_mock.merge_with.assert_called_once_with(nlp_processor.corpora.Dictionary.return_value)
    # The existing LDA model's update method should have been called once.
    mock_lda_model_instance.update.assert_called_once()
    # The LdaModel constructor should *not* be called again as we are updating an existing model.
    nlp_processor.models.LdaModel.assert_not_called()
    assert nlp_processor.topic_keywords  # Should be updated by mock


def test_update_topic_model_empty_corpus(mock_gensim):
    """
    Test that the topic model is not updated with empty corpus data.
    """
    # Ensure initial state is clean (lda_model and dictionary are None)
    nlp_processor.lda_model = None
    nlp_processor.dictionary = None

    nlp_processor.update_topic_model([])

    # Assert that neither Dictionary nor LdaModel were interacted with for an empty corpus.
    nlp_processor.corpora.Dictionary.assert_not_called()
    nlp_processor.models.LdaModel.assert_not_called()
    assert nlp_processor.lda_model is None
    assert nlp_processor.dictionary is None


def test_get_document_topics_with_model(mock_gensim):
    """
    Infers topics for a document when a model exists.
    """
    # Manually set up a mock LDA model and dictionary for this test.
    # The `mock_gensim` fixture ensures the patched classes exist, but we need instances here.
    nlp_processor.dictionary = MagicMock()
    nlp_processor.dictionary.doc2bow.return_value = [(0, 1)] # Sample BoW vector

    nlp_processor.lda_model = MagicMock()
    nlp_processor.lda_model.get_document_topics.return_value = [(0, 0.7), (1, 0.3)]

    processed_tokens = ["test", "document", "word1"]
    topics = nlp_processor.get_document_topics(processed_tokens)

    nlp_processor.dictionary.doc2bow.assert_called_once_with(processed_tokens)
    nlp_processor.lda_model.get_document_topics.assert_called_once()
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
    # Manually set up a mock LDA model and dictionary for this test.
    nlp_processor.dictionary = MagicMock()
    nlp_processor.dictionary.doc2bow.return_value = []

    nlp_processor.lda_model = MagicMock()
    nlp_processor.lda_model.get_document_topics.return_value = []

    processed_tokens = []
    topics = nlp_processor.get_document_topics(processed_tokens)

    # Assertions based on the `if not ... or not processed_tokens:` check
    nlp_processor.dictionary.doc2bow.assert_not_called()
    nlp_processor.lda_model.get_document_topics.assert_not_called()
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
    # Assert on the patched analyze_sentiment function.
    nlp_processor.analyze_sentiment.assert_called_once_with("This is a positive communication about a new product. It is great!")
    nlp_processor.analyze_sentiment.assert_any_call("There was a small risk involved but we mitigated it successfully.")
    nlp_processor.analyze_sentiment.assert_any_call("Another normal document, providing information.")
    assert nlp_processor.analyze_sentiment.call_count == 3

    # Need to setup mocks for Dictionary and LdaModel here as they are reset between tests
    mock_dict = MagicMock()
    mock_dict.doc2bow.return_value = [(0, 1)]
    mock_lda = MagicMock()
    mock_lda.get_document_topics.return_value = [(0, 0.7), (1, 0.3)]

    # Assert that Dictionary was called for each document's BoW
    assert nlp_processor.corpora.Dictionary.call_count == 1 # Only one initial dictionary creation
    assert nlp_processor.corpora.Dictionary().doc2bow.call_count == 3 # `doc2bow` should be called for each document.

    # Assert that LdaModel was called for initial training
    assert nlp_processor.models.LdaModel.call_count == 1

    # `get_document_topics` on the LDA model should be called for each message
    assert nlp_processor.models.LdaModel().get_document_topics.call_count == 3

    # Verify topic model updates (buffer_size_for_update is 50 in original script)
    # With 3 messages, no update should occur yet as buffer_size_for_update is 50.
    # The initial `LdaModel` call will still happen if dictionary is created.
    # This is checked by `nlp_processor.models.LdaModel.call_count == 1` above.
    nlp_processor.models.LdaModel().update.assert_not_called() # No update because buffer not full

    # Verify SurrealDB insertions (called once for each message)
    assert mock_surrealdb_client.query.call_count == 3
    # Check content of inserted data for the first message (positive from LLM mock)
    inserted_data_1 = mock_surrealdb_client.query.call_args_list[0].kwargs['data']
    assert inserted_data_1['sentiment'] == 'positive'
    assert isinstance(inserted_data_1['timestamp'], datetime)
    # The topics are determined by mock_lda_model_instance.get_document_topics.return_value
    assert inserted_data_1['topics'] == [{"id": 0, "probability": 0.7}, {"id": 1, "probability": 0.3}]

    # Verify logging messages
    log_messages = [record.message for record in caplog.records]
    assert "Listening for messages on Kafka topic: test_topic" in log_messages
    assert "Connected to SurrealDB: mock_namespace:mock_db_name" in log_messages
    assert any("Received message: Type='email'" in msg for msg in log_messages)
    assert any("Received message: Type='chat'" in msg for msg in log_messages)
    assert any("Received message: Type='doc'" in msg for msg in log_messages)
    # The sentiment score from the mock analyze_sentiment function (via side_effect) is 0.95
    assert any(f"Sentiment: positive (0.95)" in msg for msg in log_messages)
    # Should NOT see "Updating topic model" if buffer not full
    assert not any("Updating topic model" in msg for msg in log_messages)
    assert any("Stored processed message in SurrealDB." in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_process_communications_stream_empty_content(
        mock_kafka_consumer, mock_surrealdb_client, mock_nlp_model, mock_sentiment_analyzer, caplog
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
    nlp_processor.analyze_sentiment.assert_not_called()  # This now works because analyze_sentiment is patched.
    nlp_processor.update_topic_model.assert_not_called()
    nlp_processor.get_document_topics.assert_not_called()
    mock_surrealdb_client.query.assert_not_called()
    mock_surrealdb_client.close.assert_called_once()  # Client should still be closed


@pytest.mark.asyncio
async def test_process_communications_stream_db_connection_failure(
        mock_kafka_consumer, mock_surrealdb_client, caplog
):
    """
    Test behavior when SurrealDB connection fails at startup.
    The stream processing loop should not start.
    """
    mock_surrealdb_client.connect.side_effect = Exception("Failed to connect for test")
    mock_kafka_consumer.__iter__.return_value = []  # No messages will be yielded

    with caplog.at_level(logging.ERROR):
        await nlp_processor.process_communications_stream()

    assert "Failed to connect to SurrealDB: Failed to connect for test" in caplog.text
    # Ensure the consumer loop is not even attempted if DB connection fails
    # The __iter__ method of the consumer instance should be called once to get the empty iterator.
    nlp_processor.KafkaConsumer.assert_called_once()  # Consumer class is instantiated
    mock_kafka_consumer.__iter__.assert_not_called()  # But its iterator method is NOT called
    mock_surrealdb_client.close.assert_not_called()  # No close if connection never established


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
    assert nlp_processor.analyze_sentiment.call_count == 2  # This now works.
    assert mock_surrealdb_client.query.call_count == 2
    mock_surrealdb_client.close.assert_called_once()  # Client should still be closed at the end


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
        "timestamp": None,
        "topics": [{"id": 0, "probability": 0.9}],  # Will be overwritten by processing
    }

    mock_kafka_consumer.__iter__.return_value = [mock_kafka_consumer.MockMessage(mock_message_data)]

    # Clear topic_over_time to ensure it's not populated
    nlp_processor.topic_over_time.clear()

    with caplog.at_level(logging.ERROR):
        await nlp_processor.process_communications_stream()

    # Ensure processing steps occurred before the DB insertion attempt
    mock_nlp_model.assert_called_once()
    assert nlp_processor.analyze_sentiment.call_count == 1  # This now works.
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

