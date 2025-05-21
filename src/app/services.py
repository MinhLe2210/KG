from RAG.agent import create_agent
from RAG.kg_rag import kg_graph as kg_graph_func, vector_rag as vector_rag_func

def agent_service(question: str):
    result = create_agent(question)
    return result

def vector_rag_service(query: str):
    result = vector_rag_func(query)
    return result

def kg_graph_service(query: str):
    result = kg_graph_func(query)
    return result
