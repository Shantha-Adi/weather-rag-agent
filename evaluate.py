# # NEW: Import from the src folder
# from src.graph import build_graph
# import os
# import pandas as pd
# from langsmith import Client, evaluate
# from langsmith.schemas import Run, Example
# from langchain_aws import ChatBedrock

# # --- IMPORT YOUR GRAPH ---
# # Assuming your main file is named 'my_agent.py' and has a function 'build_graph'
# # from my_agent import build_graph

# # Initialize Clients
# client = Client()
# judge_llm = ChatBedrock(model_id="us.amazon.nova-pro-v1:0", region_name="us-east-1")

# # ============================================================
# # 1. DEFINE THE DATASET
# # ============================================================
# # We create a mix of Weather and RAG questions to test the Router.

# dataset_name = "Weather_Clothing_Router_Test_v2"

# examples = [
#     # --- WEATHER INTENTS (API) ---
#     {
#         "inputs": {"question": "What is the current temperature in New York?"},
#         "outputs": {"expected_intent": "weather", "expected_city": "New York"}
#     },
#     {
#         "inputs": {"question": "Is it going to rain in London today?"},
#         "outputs": {"expected_intent": "weather", "expected_city": "London"}
#     },
#     {
#         "inputs": {"question": "Check weather forecast for Mumbai."},
#         "outputs": {"expected_intent": "weather", "expected_city": "Mumbai"}
#     },
#     {
#         "inputs": {"question": "How cold is it in Toronto right now?"},
#         "outputs": {"expected_intent": "weather", "expected_city": "Toronto"}
#     },
#     {
#         "inputs": {"question": "Dubai weather report please."},
#         "outputs": {"expected_intent": "weather", "expected_city": "Dubai"}
#     },

#     # --- RAG INTENTS (Expert Knowledge) ---
#     {
#         "inputs": {"question": "What kind of fabric is best for high humidity?"},
#         "outputs": {"expected_intent": "rag", "expected_query": "fabric for high humidity"}
#     },
#     {
#         "inputs": {"question": "How should I layer my clothes for freezing temperatures?"},
#         "outputs": {"expected_intent": "rag", "expected_query": "layering clothes freezing cold"}
#     },
#     {
#         "inputs": {"question": "Is it okay to wear linen to a formal winter wedding?"},
#         "outputs": {"expected_intent": "rag", "expected_query": "linen fabric winter formal wear"}
#     },
#     {
#         "inputs": {"question": "Suggest a color palette for autumn skin tones."},
#         "outputs": {"expected_intent": "rag", "expected_query": "color palette autumn skin tone"}
#     },
#     {
#         "inputs": {"question": "What are the rules for wearing white after Labor Day?"},
#         "outputs": {"expected_intent": "rag", "expected_query": "wearing white after labor day rules"}
#     }
# ]

# # Check if dataset exists, if not, create it
# if not client.has_dataset(dataset_name=dataset_name):
#     dataset = client.create_dataset(dataset_name=dataset_name)
#     client.create_examples(
#         inputs=[e["inputs"] for e in examples],
#         outputs=[e["outputs"] for e in examples],
#         dataset_id=dataset.id
#     )
#     print(f"Created dataset '{dataset_name}' with {len(examples)} examples.")
# else:
#     print(f"Using existing dataset '{dataset_name}'.")

# # ============================================================
# # 2. DEFINE THE TARGET FUNCTION
# # ============================================================
# # This function wraps your graph. It takes the dataset input and 
# # returns the graph's final state for evaluation.

# def target_graph(inputs: dict) -> dict:
#     app = build_graph()
#     # Map dataset "question" -> Graph "messages"
#     result = app.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
#     return result

# # ============================================================
# # 3. DEFINE CUSTOM EVALUATORS
# # ============================================================

# def eval_router_intent(run: Run, example: Example) -> dict:
#     """
#     Checks if the classifier picked the correct intent (weather vs rag).
#     """
#     # 1. Get actual output from the graph state
#     actual_intent = run.outputs.get("intent")
    
#     # 2. Get expected output from the dataset
#     expected_intent = example.outputs.get("expected_intent")
    
#     # 3. Score
#     score = 1 if actual_intent == expected_intent else 0
    
#     return {
#         "key": "router_intent_accuracy",
#         "score": score,
#         "comment": f"Predicted: {actual_intent}, Expected: {expected_intent}"
#     }

# def eval_entity_extraction(run: Run, example: Example) -> dict:
#     """
#     Checks if the correct city or query was extracted.
#     """
#     intent = example.outputs.get("expected_intent")
#     score = 0
#     comment = ""
    
#     if intent == "weather":
#         actual = run.outputs.get("extracted_city")
#         expected = example.outputs.get("expected_city")
#         # Simple fuzzy match or exact match
#         if expected and actual and expected.lower() in actual.lower():
#             score = 1
#         comment = f"City Check - Actual: {actual}, Expected: {expected}"
        
#     elif intent == "rag":
#         actual = run.outputs.get("extracted_query")
#         # We just check if a query was generated at all for RAG
#         if actual and len(actual) > 2:
#             score = 1
#         comment = f"RAG Query Check - Generated: {actual}"

#     return {
#         "key": "entity_extraction_accuracy",
#         "score": score,
#         "comment": comment
#     }

# def eval_answer_faithfulness(run: Run, example: Example) -> dict:
#     """
#     Uses an LLM (Bedrock) to check if the answer is based on the context.
#     Only runs if context exists.
#     """
#     answer = run.outputs.get("answer")
#     context = run.outputs.get("context")
    
#     if not context or "Error" in context:
#         return {"key": "faithfulness", "score": 0, "comment": "No context retrieved"}

#     # LLM-as-a-Judge Prompt
#     prompt = f"""
#     You are a grader. Rate the following Answer on whether it is faithful to the Context.
#     Context: {context}
#     Answer: {answer}
    
#     Respond with ONLY a number: 1 (faithful) or 0 (hallucinated/unsupported).
#     """
    
#     response = judge_llm.invoke(prompt)
    
#     try:
#         score = int(response.content.strip())
#     except:
#         score = 0
        
#     return {
#         "key": "faithfulness",
#         "score": score
#     }

# # ============================================================
# # 4. RUN EVALUATION
# # ============================================================

# experiment_results = evaluate(
#     target_graph,
#     data=dataset_name,
#     evaluators=[
#         eval_router_intent,
#         eval_entity_extraction,
#         eval_answer_faithfulness
#     ],
#     experiment_prefix="Weather-Clothing-Bot",
#     metadata={"version": "1.0", "description": "Testing Router Logic"}
# )

# print(f"View results at: {experiment_results.url}")


import os
import json
import argparse
import re
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_aws import ChatBedrock
from src.config import *
# Import from the src folder
from src.graph import build_graph



# Initialize Clients
client = Client()
judge_llm = ChatBedrock(model_id=JUDGE_MODEL_ID, region_name=REGION_NAME)

# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def load_dataset_from_file(filepath: str):
    """Loads the evaluation examples from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON from '{filepath}'")
        return None

def ensure_dataset_exists(examples):
    """Creates the dataset in LangSmith if it doesn't exist."""
    if not examples:
        return False

    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_id=dataset.id
        )
        print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples.")
    else:
        print(f"Using existing dataset '{DATASET_NAME}' in LangSmith.")
    return True

def target_graph(inputs: dict) -> dict:
    """Wraps the LangGraph agent for evaluation."""
    app = build_graph()
    # Map dataset "question" -> Graph "messages"
    result = app.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
    
    # FIX: Extract the actual string content from the last AI Message
    # LangGraph stores the conversation in 'messages', so we need to grab the last one.
    if "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        result["answer"] = last_msg.content  # Explicitly add 'answer' key for evaluators
    else:
        result["answer"] = "No response generated."
        
    return result

# ============================================================
# 2. CUSTOM EVALUATORS
# ============================================================

def eval_router_intent(run: Run, example: Example) -> dict:
    """Checks if the router selected the correct intent (weather vs rag)."""
    actual_intent = run.outputs.get("intent")
    expected_intent = example.outputs.get("expected_intent")
    score = 1 if actual_intent == expected_intent else 0
    return {
        "key": "router_intent_accuracy",
        "score": score,
        "comment": f"Predicted: {actual_intent}, Expected: {expected_intent}"
    }

def eval_entity_extraction(run: Run, example: Example) -> dict:
    """Checks if the city or query was extracted correctly."""
    intent = example.outputs.get("expected_intent")
    score = 0
    comment = ""
    
    if intent == "weather":
        actual = run.outputs.get("extracted_city")
        expected = example.outputs.get("expected_city")
        if expected and actual and expected.lower() in actual.lower():
            score = 1
        comment = f"City Check - Actual: {actual}, Expected: {expected}"
        
    elif intent == "rag":
        actual = run.outputs.get("extracted_query")
        # Check if a query was generated and isn't empty
        if actual and len(actual) > 2:
            score = 1
        comment = f"RAG Query Check - Generated: {actual}"

    return {
        "key": "entity_extraction_accuracy",
        "score": score,
        "comment": comment
    }

def eval_answer_faithfulness(run: Run, example: Example) -> dict:
    """Uses an LLM Judge to check if the answer is supported by the context."""
    answer = run.outputs.get("answer")
    context = run.outputs.get("context")
    
    # 1. Guard against empty context (automatic fail)
    if not context or "Error" in str(context):
        return {"key": "faithfulness", "score": 0, "comment": "No context retrieved"}

    # 2. Strict Prompt
    prompt = f"""
    You are a strict grader.
    Task: Rate if the Answer is faithful to the Context.
    
    Context: {context}
    Answer: {answer}
    
    CRITERIA:
    - 1: The answer is derived solely from the context.
    - 0: The answer contains information not present in the context or ignores the context.
    
    Output format: Return ONLY the integer 0 or 1. No text. No punctuation.
    """
    
    try:
        response = judge_llm.invoke(prompt)
        raw_output = response.content.strip()
        
        # 3. Robust Regex Parsing
        # Looks for the numbers 0 or 1, even if buried in text like "Score: 1."
        match = re.search(r'\b[01]\b', raw_output)
        
        if match:
            score = int(match.group())
            comment = "Parsed successfully"
        else:
            score = 0
            comment = f"Parse Error: LLM returned '{raw_output}'"
            print(f"⚠️ Judge LLM Parse Error: {raw_output}")

    except Exception as e:
        score = 0
        comment = f"Exception: {str(e)}"
        print(f"❌ Judge LLM Exception: {e}")
        
    return {
        "key": "faithfulness",
        "score": score,
        "comment": comment
    }

# ============================================================
# 3. MAIN EXECUTION
# ============================================================

def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Run LangSmith Evaluation for Weather Agent")
    parser.add_argument(
        "dataset_path", 
        nargs="?", 
        default=DEFAULT_EVAL_DATASET_PATH, 
        help="Path to the JSON dataset file (default: data/evaluation_dataset.json)"
    )
    args = parser.parse_args()

    print(f"--- Starting Evaluation Process ---")
    print(f"📂 Loading dataset from: {args.dataset_path}")
    
    # 1. Load Data
    examples = load_dataset_from_file(args.dataset_path)
    
    if examples:
        # 2. Sync with LangSmith
        if ensure_dataset_exists(examples):
            
            # 3. Run Evaluation
            print("🚀 Running experiments...")
            experiment_results = evaluate(
                target_graph,
                data=DATASET_NAME,
                evaluators=[
                    eval_router_intent,
                    eval_entity_extraction,
                    eval_answer_faithfulness
                ],
                experiment_prefix="Weather-Clothing-Bot",
                metadata={"version": "2.3", "dataset_source": args.dataset_path}
            )

            print(f"\n✅ Evaluation Complete! View results here:\n{experiment_results.url}")
    else:
        print("❌ Evaluation aborted due to dataset errors.")

if __name__ == "__main__":
    main()