from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_aws import ChatBedrock
from src.config import *
from src.state import AgentState, RouterOutput
from src.tools import fetch_weather_api, query_qdrant_db


def intent_classifier(state: AgentState):
    llm = ChatBedrock(
        model_id=ROUTER_MODEL_ID,
        region_name=REGION_NAME
    ).with_structured_output(RouterOutput)
    
    system_prompt = (
        "You are a smart router. "
        "1. If user asks about weather, intent='weather', extract city. "
        "2. If user asks for advice/clothing rules, intent='rag', extract query."
    )
    
    msg = state["messages"][-1].content
    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg}
    ])
    
    return {
        "intent": result.intent,
        "extracted_city": result.city,
        "extracted_query": result.rag_query
    }

def weather_node(state: AgentState):
    city = state.get("extracted_city")
    print(f"--- Weather Tool: {city} ---")
    return {"context": fetch_weather_api(city)}

def rag_node(state: AgentState):
    query = state.get("extracted_query")
    print(f"--- RAG Tool: {query} ---")
    return {"context": query_qdrant_db(query)}

def answer_node(state: AgentState):
    llm = ChatBedrock(model_id=ANSWER_MODEL_ID, region_name=REGION_NAME)
    
    prompt = f"""
    Answer the question based on the context.
    Question: {state["messages"][-1].content}
    Context: {state["context"]}
    """
    response = llm.invoke(prompt)
    return {"messages": [response]}

def decide_next_node(state: AgentState) -> Literal["weather_node", "rag_node"]:
    if state["intent"] == "weather":
        return "weather_node"
    return "rag_node"

# --- GRAPH BUILDER ---

def build_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("classifier", intent_classifier)
    builder.add_node("weather_node", weather_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("answer_node", answer_node)
    
    builder.add_edge(START, "classifier")
    
    builder.add_conditional_edges(
        "classifier",
        decide_next_node
    )
    
    builder.add_edge("weather_node", "answer_node")
    builder.add_edge("rag_node", "answer_node")
    builder.add_edge("answer_node", END)
    
    return builder.compile()