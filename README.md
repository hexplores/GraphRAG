# GraphRAG for Smart Retrieval

A lightweight GraphRAG-style retrieval pipeline that builds a document graph and combines vector similarity with graph expansion for smarter context retrieval.

## Features
- Ingests plain text and Markdown files
- Chunks text with overlap
- Builds TF-IDF embeddings
- Constructs a graph linking documents, chunks, and keywords
- Retrieves top chunks with optional graph expansion
- Simple CLI for indexing and querying

## Quick Start
1. Create and activate a Python environment.
2. Install dependencies from requirements.txt.
3. Build an index.
4. Run a query.

## CLI Usage
Build an index:
- Command: graphrag-smart-retrieval build --input data --output index

Query:
- Command: graphrag-smart-retrieval query --index index --query "your question" --top-k 5

## Configuration
A sample configuration file is available at configs/config.example.json. You can pass it with:
- Command: graphrag-smart-retrieval build --input data --output index --config configs/config.example.json

## Output Artifacts
The index folder contains:
- chunks.json
- vectorizer.pkl
- matrix.npz
- graph.json
- metadata.json

<<<<<<< HEAD
=======
## Public Deployment
The app now includes a public HTTP API for deployment as a web service.

Start locally:
- Command: uvicorn graphrag_smart_retrieval.api:app --host 0.0.0.0 --port 8000

Health check:
- URL: http://localhost:8000/health

Query example:
- URL: http://localhost:8000/query
- Body: {"index_dir": "./index", "query": "your question", "top_k": 5, "use_graph": true}

The project also includes a Dockerfile and a Heroku-style Procfile for deployment to cloud platforms.

## Production notes
- Set a public port via the PORT environment variable.
- Point the app to a persistent index directory, or rebuild the index in a deployment step.
- For public traffic, prefer a managed platform such as Render, Railway, Fly.io, or Azure App Service.

>>>>>>> d6105a8 (Add public API deployment support)
