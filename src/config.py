import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Qdrant Config
# QDRANT_PATH = "clothing-v10"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.path.join(SRC_DIR, "clothing-v10")

COLLECTION_NAME = "clothing-v10"

# AWS Bedrock Config
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0" 
VECTOR_SIZE = 1024  

# LLM Model IDs
ROUTER_MODEL_ID = "us.amazon.nova-lite-v1:0"
ANSWER_MODEL_ID = "us.amazon.nova-pro-v1:0"
REGION_NAME = "us-east-1"


# eval configs
DEFAULT_EVAL_DATASET_PATH = "data/evaluation_dataset.json"
DATASET_NAME = "Weather_Clothing_Router_Test_v3"
JUDGE_MODEL_ID = "us.amazon.nova-pro-v1:0"


# INDEX 
DEFAULT_RAG_DATASET_PATH = "../data/clothing_documents"


