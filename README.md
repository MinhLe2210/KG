# Project Setup and Usage

This document provides clear instructions to set up and run the project for processing documents, generating a knowledge graph, and semantic search using Neo4j and Qdrant, integrated with LangSmith for LLM tracing.

---

## Environment Setup

Follow these steps to configure the Python virtual environment and install dependencies:

```bash
pip install uv
uv .venv
.venv/Scripts/activate
uv pip install -r requirements.txt
```

## Configure `.env` File

Create a `.env` file in the root directory of the project and add the following:

* **LangSmith API**: Obtain your API key for LLM tracing from [LangSmith](https://smith.langchain.com/).
* **Neo4j Aura**: Sign up at [Neo4j Aura](https://neo4j.com/product/auradb/), create an instance, and get your `URI` and API key.
* **Qdrant API**: Obtain your API key from [Qdrant](https://qdrant.tech/).

Example `.env` file structure:

```dotenv
LANGSMITH_API_KEY=your_langsmith_api_key
NEO4J_URI=your_neo4j_aura_uri
NEO4J_API_KEY=your_neo4j_api_key
QDRANT_API_KEY=your_qdrant_api_key
```

---

## Processing PDF Files

Run the following script to clean noise from PDF files and chunk them into `all_chunk.csv`:

```bash
python process_pdf.py
```

---

## Knowledge Graph Creation

Generate Subject-Predicate-Object (SPO) triplets for the knowledge graph:

```bash
python create_triplets.py
```

This will create a CSV file named `extract_KG.csv` containing the triplets.

Upload generated triplets to your Neo4j Aura instance:

```bash
python kg_to_neo4j.py
```

---

## Semantic Search with Qdrant

Upload chunks from `all_chunk.csv` into Qdrant for semantic search:

```bash
python qdrant_rag.py
```

---

## Running the API

Navigate to the application directory and run the API:

```bash
cd src/app
python main.py
```

Access Swagger UI at [http://localhost:8000](http://localhost:8000) to interact with and test the API endpoints.

---
