from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QuestionRequest, QueryRequest, AgentResponse, VectorRAGResponse, KGGraphResponse
from services import agent_service, vector_rag_service, kg_graph_service
import logging

app = FastAPI(
    title="RAG API",
    description="API for Hybrid Retrieval, Vector Search, and Knowledge Graph",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/agent", response_model=AgentResponse)
def run_agent(request: QuestionRequest):
    try:
        answer = agent_service(request.question)
        return AgentResponse(answer=answer)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-rag", response_model=VectorRAGResponse)
def run_vector_rag(request: QueryRequest):
    try:
        docs = vector_rag_service(request.query)
        return VectorRAGResponse(docs=docs)
    except Exception as e:
        logger.error(f"Vector RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/kg-graph", response_model=KGGraphResponse)
def run_kg_graph(request: QueryRequest):
    try:
        result = kg_graph_service(request.query)
        return KGGraphResponse(result=result)
    except Exception as e:
        logger.error(f"KG Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  
    )
