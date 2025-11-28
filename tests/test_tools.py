from unittest.mock import patch, MagicMock
from src.tools import fetch_weather_api, query_qdrant_db

# --- WEATHER TOOL TESTS ---

def test_fetch_weather_success(mock_weather_response):
    """Test standard successful API call."""
    with patch("src.tools.requests.get") as mock_get:
        # Configure the mock to return 200 and our JSON data
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_weather_response

        result = fetch_weather_api("London")
        
        assert "25.5°C" in result
        assert "clear sky" in result
        assert "London" in result  # Our function usually includes the city name

def test_fetch_weather_city_not_found():
    """Test 404 handling."""
    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value.status_code = 404
        
        result = fetch_weather_api("Atlantis")
        assert "not found" in result.lower()

def test_fetch_weather_no_input():
    """Test empty input handling."""
    result = fetch_weather_api("")
    assert "Error" in result

# --- RAG TOOL TESTS ---

def test_query_qdrant_success(mock_qdrant_docs):
    """Test vector store retrieval."""
    # We must mock QdrantVectorStore AND Embeddings to prevent real AWS calls
    with patch("src.tools.BedrockEmbeddings"), \
         patch("src.tools.QdrantClient"), \
         patch("src.tools.QdrantVectorStore") as MockStore:
        
        # Setup the mock store to return our fake docs
        instance = MockStore.return_value
        instance.similarity_search.return_value = mock_qdrant_docs
        
        result = query_qdrant_db("what to wear")
        
        assert "Wear linen" in result
        assert "Cotton absorbs" in result

def test_query_qdrant_empty():
    """Test when no documents are found."""
    with patch("src.tools.BedrockEmbeddings"), \
         patch("src.tools.QdrantClient"), \
         patch("src.tools.QdrantVectorStore") as MockStore:
        
        instance = MockStore.return_value
        instance.similarity_search.return_value = []
        
        result = query_qdrant_db("weird query")
        assert "No relevant documents" in result