import requests
import json
from langchain_aws import BedrockEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from src.config import *
import os

def fetch_weather_api(city_name: str) -> str:
    """Fetches current weather data for a given city."""
    if not city_name:
        return "Error: No city name provided."

    full_url = f"{OPENWEATHER_BASE_URL}?q={city_name}&appid={os.getenv("OPENWEATHER_API_KEY")}&units=metric"

    try:
        response = requests.get(full_url)
        if response.status_code == 200:
            data = response.json()
            # Return a clean summary string instead of raw JSON for the LLM
            main = data.get('main', {})
            weather_desc = data.get('weather', [{}])[0].get('description', 'unknown')
            return f"Weather in {city_name}: {main.get('temp')}°C, {weather_desc}. Humidity: {main.get('humidity')}%"
        elif response.status_code == 404:
            return f"Error: City '{city_name}' not found."
        else:
            return f"Error: API returned status {response.status_code}"
    except Exception as e:
        return f"Error connecting to weather API: {str(e)}"

def query_qdrant_db(query: str) -> str:
    """Queries the Qdrant vector store."""
    if not query:
        return "Error: No search query provided."

    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, region_name=REGION_NAME)
    
    client = QdrantClient(path=QDRANT_PATH)
    
    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )

    docs = store.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant documents found."
        
    return "\n\n".join([d.page_content for d in docs])