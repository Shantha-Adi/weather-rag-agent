import pytest
from unittest.mock import MagicMock
from src.state import RouterOutput

@pytest.fixture
def mock_weather_response():
    """Mock successful OpenWeatherMap API response"""
    return {
        "main": {"temp": 25.5, "humidity": 60},
        "weather": [{"description": "clear sky"}]
    }

@pytest.fixture
def mock_qdrant_docs():
    """Mock LangChain Documents returned from vector store"""
    doc1 = MagicMock()
    doc1.page_content = "Wear linen in high heat."
    doc2 = MagicMock()
    doc2.page_content = "Cotton absorbs moisture."
    return [doc1, doc2]

@pytest.fixture
def mock_router_weather():
    """Mock LLM output for Weather intent"""
    return RouterOutput(intent="weather", city="London", rag_query=None)

@pytest.fixture
def mock_router_rag():
    """Mock LLM output for RAG intent"""
    return RouterOutput(intent="rag", city=None, rag_query="summer fabrics")