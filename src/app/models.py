from pydantic import BaseModel
from typing import List, Any

class QuestionRequest(BaseModel):
    question: str

class QueryRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    answer: Any

class VectorRAGResponse(BaseModel):
    docs: List[str]

class KGGraphResponse(BaseModel):
    result: Any
