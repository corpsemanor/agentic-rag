# Agentic RAG System

Set up complete Docker infrastructure with Backend API that proxies requests to Ollama container, preparing for RAG functionality.

## 🚀 1. Start Foundation

1. **Clone and setup:**
   ```bash
   git clone https://github.com/corpsemanor/agentic-rag
   cd agentic-rag
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Start all services:**
   ```bash
   # First run or rebuild
   sudo docker compose up --build -d
   
   # Regular start (without changes)
   docker compose up -d
   ```

4. **Verify services are running:**
   ```bash
   docker compose ps
   ```

## 🌐 Services

### Running Services:
- **OpenWebUI**: http://localhost:3000 (Frontend chat interface)
- **Backend API**: http://localhost:8000 (FastAPI backend)
- **Phoenix (Observability)**: http://localhost:6006 (Monitoring & tracing)
- **PostgreSQL**: http://localhost:5432 (Database with pgvector)
- **Ollama**: http://localhost:11434 (LLM model hosting)

## 🚀 2. Start Agentic RAG with CrewAI Agents and Ollama LLM

The system uses two CrewAI agents for enhanced answer quality:

1. **Researcher Agent** - Validates facts from retrieved documents
2. **Finisher Agent** - Formats response with proper citations

### Setup Models:

```bash
# Pull embedding model (required for document indexing)
docker exec ollama ollama pull nomic-embed-text

# Pull generation model (required for chat responses)
docker exec ollama ollama pull qwen3:0.6b
```

### Upload and Index Documents:

```bash
# Upload a document - automatically chunks AND indexes it into vector database
# chunk_size is optional (default: 1000 characters = ~170 words, ~250 tokens)
# Range: 100-10000 characters

# Upload with default chunk size (1000 characters)
curl -X POST \
  -F "file=@test2.docx" \
  http://localhost:8000/api/upload-and-chunk

# Upload with custom chunk size
curl -X POST \
  -F "file=@test2.docx" \
  -F "chunk_size=500" \
  http://localhost:8000/api/upload-and-chunk

# You can upload multiple documents sequentially - each will be automatically indexed
curl -X POST -F "file=@document1.pdf" http://localhost:8000/api/upload-and-chunk
curl -X POST -F "file=@document2.docx" -F "chunk_size=800" http://localhost:8000/api/upload-and-chunk
curl -X POST -F "file=@document3.md" http://localhost:8000/api/upload-and-chunk

# Supported file formats: .pdf, .docx, .doc, .md
# Process flow:
# 1. Original file → saved to data/raw_docs/
# 2. Document is chunked → saved to data/clear_docs/filename_chunks.json
# 3. Chunks are automatically indexed → stored in PostgreSQL vector database
# Document is immediately available for RAG queries after upload!
# All uploaded documents are stored in the same vector index and can be searched together
```

### Useful Docker Commands:

```bash
# View logs for specific service
docker compose logs -f backend

# Restart backend (after code changes)
docker compose restart backend
```

## 🔍 3. Testing

### Test RAG Retrieval with Metadata:

```bash
# Test retrieval API - should return metadata
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Outside missions mean?"}'

# Expected response includes:
# - document_id: source document ID
# - source: source file name (from config)
# - metadata: full metadata object
# - score: relevance score
```

### Test Agentic Chat:

```bash
# Test agentic chat endpoint (uses agents for better quality)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Valkyria Chronicles?", "model": "qwen3:0.6b"}'

# Expected: Response with citations [1][2][3] and sources array
```

### Verify System Components:

```bash
# Quick check: OpenWebUI → Backend → Agents → RAG
docker compose logs --tail=100 backend | grep -E "\[AGENT\]|Retrieved.*documents|POST.*chat"

# Check RAG retrieval works
docker compose logs -f backend | grep "Retrieved.*documents"

# Check both agents are working
docker compose logs -f backend | grep "\[AGENT\]"

# Verify metadata in database
docker exec postgres_db psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM data_rag_vectors;"
```

### Manage Indexed Documents:

If you need to clear and re-index all documents:

```bash
# Drop existing index (removes all indexed documents)
docker exec postgres_db psql -U rag_user -d rag_db -c "DROP TABLE IF EXISTS data_rag_vectors CASCADE;"

# Re-index by re-uploading documents through /api/upload-and-chunk
# Documents will be automatically indexed on upload
```

## 📊 4. Observability with Phoenix

### Overview

Phoenix automatically collects traces from your RAG pipeline. Open http://localhost:6006 to view traces.

### What's Being Traced

The following operations are instrumented:

1. **`document.upload`** - Document upload and indexing
   - Attributes: filename, chunk_size, file_size, total_chunks, indexed_count

2. **`rag.retrieve`** - Vector similarity search
   - Attributes: query, limit, documents_found

3. **`agents.process`** - CrewAI agent processing
   - Attributes: model, query_length, docs_count

4. **`rag.chat`** - Complete RAG pipeline (parent span)
   - Attributes: model
   - Child spans: rag.retrieve + agents.process

5. **LLM calls** - Automatically instrumented via `auto_instrument=True`
   - Ollama API calls, embeddings, generations

### Viewing Traces

1. Open Phoenix UI: http://localhost:6006
2. Upload a document and make a RAG query
3. View the complete trace showing all pipeline stages with timing and metadata

### Code quality
```sh
# Check code for linting issues without making changes
uv run ruff check

# Fix linting issues using ruff
uv run ruff check --fix
```