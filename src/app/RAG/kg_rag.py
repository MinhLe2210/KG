import os 
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from RAG.prompt import text2cypher, rewrite_to_czech
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from langchain_openai import AzureOpenAIEmbeddings

def clean_embedding(text):
    for record in text:
        for k, v in record.items():
            if isinstance(v, (tuple, list)):
                for d in v:
                    if isinstance(d, dict) and "embedding" in d:
                        d.pop("embedding")
            elif isinstance(v, dict) and "embedding" in v:
                v.pop("embedding")
    return text
def kg_graph(query):
    llm = AzureChatOpenAI(
                azure_deployment="gpt-4o-mini",  
                api_version="2024-12-01-preview",
                temperature=0,
                max_tokens=4096,
                timeout=30,
                max_retries=2,
            )
    graph = Neo4jGraph()
    text2cypher_formated = text2cypher.replace("<user_question_replace>",query)
    res = llm.invoke(text2cypher_formated).content
    response = graph.query(str(res))
    cleaned_response = clean_embedding(response)
    return cleaned_response

def vector_rag(query):
    llm = AzureChatOpenAI(
                azure_deployment="gpt-4o-mini",  
                api_version="2024-12-01-preview",
                temperature=0,
                max_tokens=4096,
                timeout=30,
                max_retries=2,
            )
    rewrite_query = llm.invoke(rewrite_to_czech.replace("<input_replace>", query)).content
    embeddings = AzureOpenAIEmbeddings(model="ace-text-embedding-3-large")

    QDRANT_URL = "https://20840cd3-a3bf-4a62-af36-72b49fe3bed0.us-east-1-0.aws.cloud.qdrant.io"
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Ensure your API key is set in env
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    vectorstore = QdrantVectorStore(
                                    client=client,
                                    collection_name="law",
                                    embedding=embeddings
                                )


    found_docs = vectorstore.similarity_search(rewrite_query, k=5)
    docs = [doc.page_content for doc in found_docs]
    return docs

if '__main__' == __name__:
    query = "How is the taxpayer's tax calculated?"
    # print(kg_graph(llm, query, text2cypher))
    search_result = vector_rag(query)
    print(search_result[0])
