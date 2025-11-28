from typing import TypedDict, Annotated, Literal, List, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# --- Graph State ---
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    context: str
    intent: str
    extracted_city: Optional[str]
    extracted_query: Optional[str]

# --- Router Output Schema ---
class RouterOutput(BaseModel):
    intent: Literal["weather", "rag"] = Field(
        ..., description="The user's intent: 'weather' for live data, 'rag' for static knowledge/advice."
    )
    city: Optional[str] = Field(
        None, description="If intent is weather, extract the specific city name."
    )
    rag_query: Optional[str] = Field(
        None, description="If intent is rag, extract the search query."
    )