import pytest
from unittest.mock import patch, MagicMock
from src.graph import intent_classifier, decide_next_node, weather_node, rag_node

# --- ROUTER LOGIC TESTS ---

def test_intent_classifier_weather(mock_router_weather):
    """Test that the classifier correctly extracts weather intent."""
    
    # Mock the LLM inside the classifier
    with patch("src.graph.ChatBedrock") as MockLLM:
        # The LLM's .with_structured_output().invoke() chain
        mock_chain = MockLLM.return_value.with_structured_output.return_value
        mock_chain.invoke.return_value = mock_router_weather
        
        fake_state = {"messages": [MagicMock(content="Weather in London")]}
        result = intent_classifier(fake_state)
        
        assert result["intent"] == "weather"
        assert result["extracted_city"] == "London"

def test_decide_next_node():
    """Test the conditional edge logic."""
    assert decide_next_node({"intent": "weather"}) == "weather_node"
    assert decide_next_node({"intent": "rag"}) == "rag_node"

# --- NODE INTEGRATION TESTS ---

def test_weather_node_integration():
    """Test that the weather node calls the tool correctly."""
    with patch("src.graph.fetch_weather_api") as mock_tool:
        mock_tool.return_value = "Sunny 25C"
        
        state = {"extracted_city": "Paris"}
        result = weather_node(state)
        
        mock_tool.assert_called_once_with("Paris")
        assert result["context"] == "Sunny 25C"

def test_rag_node_integration():
    """Test that the RAG node calls the tool correctly."""
    with patch("src.graph.query_qdrant_db") as mock_tool:
        mock_tool.return_value = "Fabric advice..."
        
        state = {"extracted_query": "summer fabric"}
        result = rag_node(state)
        
        mock_tool.assert_called_once_with("summer fabric")
        assert result["context"] == "Fabric advice..."
# ```

# ---

# ### **3. How to Run the Tests**

# 1.  **Install Testing Libraries:**
#     Add these to your `requirements.txt`:
#     ```text
#     pytest
#     pytest-mock
#     requests
#     langchain
#     langgraph
#     boto3
#     # ... your other dependencies
#     ```
#     Then run: `pip install -r requirements.txt`

# 2.  **Run Tests:**
#     Open your terminal in the root folder (`weather-rag-agent/`) and type:
#     ```bash
#     pytest -v
#     ```

# **Expected Output:**
# ```text
# tests/test_tools.py::test_fetch_weather_success PASSED [ 20%]
# tests/test_tools.py::test_fetch_weather_city_not_found PASSED [ 40%]
# tests/test_tools.py::test_fetch_weather_no_input PASSED [ 60%]
# tests/test_graph.py::test_intent_classifier_weather PASSED [ 80%]
# tests/test_graph.py::test_decide_next_node PASSED [100%]
# ```

# This setup ensures that even if you don't have AWS credentials or API keys on your local machine, the tests will pass because everything external is mocked. This is the professional standard for CI/CD pipelines on GitHub.