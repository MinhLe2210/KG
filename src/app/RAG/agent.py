import os

# === LangSmith tracing setup ===
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "tax-qa-rag"
import json
from pprint import pprint
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from RAG.kg_rag import kg_graph, vector_rag
from RAG.prompt import CoT_reasoning_critique, router_prompt
from json_repair import repair_json

# Load environment variables
load_dotenv()

# Initialize the LLM globally
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  
    api_version="2024-12-01-preview",
    temperature=0,
    max_tokens=4096,
    timeout=30,
    max_retries=2,
)

@dataclass(kw_only=True)
class GraphState:
    question: str = field(default=None)      # User input question
    router: Optional[str] = field(default=None)
    documents: List = field(default_factory=list)
    answer: Optional[str] = field(default=None)

def router_node(state: GraphState):
    print("=============== GENERATE SMART QUESTION ========================")
    question = state.question
    response = llm.invoke([HumanMessage(content=router_prompt.replace("<input_replace>", question))])
    print(response.content)
    return {"router": response.content}

def choose_tool_to_use(state: GraphState):
    print("=============== CHOOSE TOOL FOR SMART QUESTION ========================")
    return 'none' if state.router == 'none' else 'rag'

def rag_node(state: GraphState):
    print("=============== Knowledge Graph and Vector Retrieval NODE ========================")
    question = state.question
    documents = [kg_graph(question), vector_rag(question)]
    return {"documents": documents}

def normal_node(state: GraphState):
    print("=============== NORMAL NODE ========================")
    return {"answer": "Your question is not relevant to law. Please provide a question relevant to law."}

def final_node(state: GraphState):
    print("=============== FINAL ANSWER NODE ========================")
    question = state.question
    docs = state.documents
    # Đảm bảo prompt có đúng context/question
    cot_final_prompt = CoT_reasoning_critique.replace("<context_replace>", question).replace("<context_replace>", str(docs))
    final_result = llm.invoke([HumanMessage(content=cot_final_prompt)])
    ans_json = json.loads(repair_json(final_result.content))

    if isinstance(ans_json, list):
        ans_json = ans_json[0]
    pprint(ans_json, indent=2, width=170)

    answer = ans_json.get('chosen_answer')
    if ans_json.get('deeper_wider_than_chosen_answer') and ans_json['deeper_wider_than_chosen_answer'] != 'None':
        answer = ans_json['deeper_wider_than_chosen_answer']
    return {"answer": answer}

def create_agent(question):
    workflow = StateGraph(GraphState)
    workflow.add_node("router_node", router_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("normal_answer_node", normal_node)
    workflow.add_node("final_node", final_node)

    workflow.add_edge(START, "router_node")
    workflow.add_conditional_edges(
        "router_node",
        choose_tool_to_use,
        {
            "rag": "rag_node",
            "no": "normal_answer_node",
        },
    )
    workflow.add_edge("rag_node", "final_node")
    workflow.add_edge("normal_answer_node", END)
    workflow.add_edge("final_node", END)

    app = workflow.compile()
    inputs = {"question": question}
    value = None
    for output in app.stream(inputs, config={"configurable": {"thread_id": 42}}):
        for key, value in output.items():
            pprint(f"Finished running: {key}:", indent=2, width=80, depth=None)
    return value['answer']

# if __name__ == "__main__":
#     ans = create_agent("How is the taxpayer's tax calculated?")
#     pprint(ans)
