🌤️ Weather & Style Agent

An intelligent LangGraph agent that routes user queries between real-time weather data and expert fashion advice using RAG (Retrieval Augmented Generation).

📖 Overview

This project implements a conditional routing agent. Instead of using a generic LLM for everything, it intelligently decides whether a user needs:

Real-Time Data: Fetches live weather from the OpenWeatherMap API.

Expert Knowledge: Retrieves fashion/fabric advice from a local Qdrant vector store (RAG).

It uses LangGraph to manage the state and workflow, AWS Bedrock for LLM/Embeddings, and Streamlit for the chat interface.

🏗️ Architecture

The agent follows a Router-Tool pattern:

Classifier Node: Analyzes user intent and extracts entities (City vs. Search Query).

Conditional Edge: Routes the flow to either weather_node or rag_node.

Tool Nodes: Execute the specific API call or Vector Search.

Answer Node: Synthesizes the retrieved context into a final natural language response.

📂 Project Structure

weather-rag-agent/
├── .env                    # API Keys (Not included in repo)
├── app.py                  # Streamlit User Interface
├── evaluate.py             # LangSmith Evaluation Script
├── main.py                 # CLI Entry Point
├── pytest.ini              # Testing Configuration
├── requirements.txt        # Dependencies
├── data/                   # Assets (PDFs and Test JSONs)
│   ├── clothing_documents/ # Place PDFs here
│   └── evaluation_dataset.json
├── src/
│   ├── config.py           # Configuration & Env Loading
│   ├── graph.py            # LangGraph Workflow Definition
│   ├── ingest.py           # PDF Ingestion Script
│   ├── state.py            # Pydantic Models & State TypedDict
│   └── tools.py            # External Tool Logic (API & DB)
└── tests/
    ├── conftest.py         # Pytest Fixtures
    ├── test_graph.py       # Graph Logic Tests
    └── test_tools.py       # Tool Unit Tests


🚀 Installation

Clone the repository:

git clone [https://github.com/your-username/weather-rag-agent.git](https://github.com/your-username/weather-rag-agent.git)
cd weather-rag-agent


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Configure Environment Variables:
Create a .env file in the root directory. You can copy the structure below:

# .env file

# 1. Weather API (Required)
OPENWEATHER_API_KEY="your_openweather_key"
OPENWEATHER_BASE_URL="[https://api.openweathermap.org/data/2.5/weather](https://api.openweathermap.org/data/2.5/weather)"

# 2. AWS Bedrock Credentials (Required)
AWS_ACCESS_KEY_ID="your_aws_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret"
REGION_NAME="us-east-1"

# 3. Model IDs (Optional - defaults provided in src/config.py)
ROUTER_MODEL_ID="us.amazon.nova-lite-v1:0"
ANSWER_MODEL_ID="us.amazon.nova-pro-v1:0"
EMBEDDING_MODEL_ID="amazon.titan-embed-text-v2:0"

# 4. LangSmith Evaluation (Optional)
LANGCHAIN_API_KEY="your_langsmith_key"
LANGCHAIN_TRACING_V2=true


💻 Usage

1. Ingest Knowledge (RAG)

Before asking about fashion, you need to populate the vector database. Place your PDF guides inside data/clothing_documents/.

python src/ingest.py


2. Run the Web App

Launch the Streamlit interface to chat with the agent visually.

streamlit run app.py


3. Run via CLI

For quick testing in the terminal without a UI:

python main.py


🧪 Testing & Evaluation

This project uses a hybrid testing strategy to ensure reliability.

Unit Tests (Logic Stability)

We use pytest with mocks to test the logic without hitting real APIs. This is fast and free.

python -m pytest -v


Evaluation (LLM Intelligence)

We use LangSmith to evaluate if the Router is making smart decisions and if the answers are faithful to the context. This uses the dataset in data/evaluation_dataset.json.

python evaluate.py


🛠️ Tech Stack

Orchestration: LangGraph

LLMs: AWS Bedrock (Nova Lite for Routing, Nova Pro for Answering)

Embeddings: Amazon Titan V2

Vector Database: Qdrant (Local)

External API: OpenWeatherMap

Testing: Pytest, Unittest.mock

Evaluation: LangSmith SDK

📝 License

This project is licensed under the MIT License.