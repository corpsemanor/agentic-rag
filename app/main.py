"""
FastAPI Backend for Agentic RAG System
Foundation Stage - Ollama Proxy API for OpenWebUI integration
"""

import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chunking import DocumentChunkingService
from app.vectorstorage import VectorStorageService
from app.cache import add_message_to_history, get_conversation_history
from app.agent import run_rag_crew
from app.tracing import trace
from app.config import (
    OLLAMA_HOST, RAW_DOCS_DIR, CLEAR_DOCS_DIR,
    OLLAMA_CLIENT_TIMEOUT, DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE, ALLOWED_UPLOAD_EXTENSIONS, DEFAULT_SIMILARITY_TOP_K,
    BACKEND_HOST, BACKEND_PORT, DEFAULT_USER_ID
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Agentic RAG Backend",
    description="Backend API for RAG system with OpenWebUI integration",
    version="0.1.0"
)

# Initialize services
chunking_service = DocumentChunkingService()
vector_repository = VectorStorageService()

# Configure CORS for OpenWebUI frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    service: str

class UploadChunkResponse(BaseModel):
    filename: str
    status: str
    total_chunks: int
    chunk_size: int
    message: str


# HTTP client for Ollama communication
async def _get_ollama_client():
    """Returns an async HTTP client for Ollama."""
    return httpx.AsyncClient(timeout=OLLAMA_CLIENT_TIMEOUT)

@app.get("/health")
async def get_health():
    """Health check endpoint for service monitoring"""
    return HealthResponse(
        status="healthy",
        service="agentic-rag-backend"
    )

# --- Ollama Proxy Endpoints ---

@app.get("/api/tags")
async def proxy_get_models():
    """Get available models from Ollama"""
    async with await _get_ollama_client() as client:
        try:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/pull")
async def proxy_pull_model(request: Request):
    """Pull a model from Ollama"""
    body = await request.json()
    async with await _get_ollama_client() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/pull", json=body)
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/generate")
async def proxy_generate_response(request: Request):
    """Generate response from Ollama"""
    body = await request.json()
    async with await _get_ollama_client() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=body)
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/chat")
async def handle_chat_completion(request: Request):
    """Chat completion endpoint with RAG agents for OpenWebUI"""
    body = await request.json()
    load_dotenv()
    print("Request body:", body)

    # Force non-streaming mode for agent processing
    body["stream"] = False

    with trace("rag.chat", {"model": body.get("model", "unknown")}):
        try:
            # Detect request format
            if "messages" in body:
                # OpenWebUI format
                messages = body.get("messages", [])
                user_query = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_query = msg.get("content")
                        break
                if not user_query:
                    raise HTTPException(status_code=400, detail="No user message found")
            else:
                # Direct API format
                user_query = body.get("prompt", body.get("query", ""))
                if not user_query:
                    raise HTTPException(status_code=400, detail="No query provided")

            # 1. Retrieve documents
            with trace("rag.retrieve", {"query": user_query[:100], "limit": DEFAULT_SIMILARITY_TOP_K}):
                response_nodes = vector_repository.search_similar_documents(
                    user_query, DEFAULT_SIMILARITY_TOP_K
                )

            # 2. Format documents
            retrieved_docs = [
                {
                    "text": node.text,
                    "score": node.score if hasattr(node, 'score') else 0.0,
                    "metadata": node.metadata if hasattr(node, 'metadata') and node.metadata else {},
                    "source": node.metadata.get('source_file', 'unknown') if hasattr(node, 'metadata') and node.metadata else 'unknown',
                    "document_id": node.metadata.get('id', node.metadata.get('document_id', None)) if hasattr(node, 'metadata') and node.metadata else None
                }
                for node in response_nodes
            ]

            # 3. Process with agents
            with trace("agents.process", {
                "model": body["model"],
                "query_length": len(user_query),
                "docs_count": len(retrieved_docs)
            }):
                result = await run_rag_crew(body["model"], user_query, retrieved_docs)

            # 4. Update memory
            await add_message_to_history(DEFAULT_USER_ID, "assistant", result.get('response', ''))
            history = await get_conversation_history(DEFAULT_USER_ID)
            logging.info(f"Conversation memory: {history}")

            # 5. Format response
            if "messages" in body:
                # Ollama-compatible format for OpenWebUI
                return {
                    "model": body["model"],
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "message": {
                        "role": "assistant",
                        "content": result.get('response', '')
                    },
                    "done": True
                }
            else:
                # Custom format for direct API calls
                return result

        except Exception as e:
            logger.error(f"Chat completion failed: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"Chat failed: {e!s}")

@app.get("/api/version")
async def proxy_get_ollama_version():
    """Get Ollama version"""
    async with await _get_ollama_client() as client:
        try:
            response = await client.get(f"{OLLAMA_HOST}/api/version")
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/rag")
async def perform_rag_query(request: Request):
    """Perform RAG query using Ollama embeddings"""
    body = await request.json()
    load_dotenv()
    query_string = body.get("query", "")
    if not query_string:
        raise HTTPException(status_code=400, detail="No query provided")

    try:
        response_nodes = vector_repository.search_similar_documents(
            query_string, DEFAULT_SIMILARITY_TOP_K
        )
        
        response = [
            {
                "text": node.text,
                "score": node.score if hasattr(node, 'score') else 0.0,
                "metadata": node.metadata if hasattr(node, 'metadata') and node.metadata else {},
                "source": node.metadata.get('source_file', 'unknown') if hasattr(node, 'metadata') and node.metadata else 'unknown',
                "document_id": node.metadata.get('id', node.metadata.get('document_id', None)) if hasattr(node, 'metadata') and node.metadata else None
            }
            for node in response_nodes
        ]
        return {
            "response": response,
            "query": query_string,
            "count": len(response)
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"RAG query failed: {e!s}")

@app.get("/")
async def get_root():
    """Root endpoint with basic API information"""
    return {
        "message": "Agentic RAG Backend API",
        "version": "0.1.0",
        "stage": "Foundation",
        "ollama_host": OLLAMA_HOST,
        "endpoints": {
            "health": "/health",
            "ollama_proxy": "/api/*",
            "docs": "/docs"
        }
    }

@app.post("/api/upload-and-chunk", response_model=UploadChunkResponse)
async def upload_and_process_document(
    file: UploadFile = File(...), 
    chunk_size: int = Form(
        DEFAULT_CHUNK_SIZE, 
        ge=MIN_CHUNK_SIZE, 
        le=MAX_CHUNK_SIZE, 
        description=f"Maximum chunk size (default: {DEFAULT_CHUNK_SIZE}, range: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE})"
    )
):
    """
    Unified endpoint for document upload, chunking, and indexing.
    """
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed: {ALLOWED_UPLOAD_EXTENSIONS}"
        )

    # Ensure RAW_DOCS_DIR exists
    Path(RAW_DOCS_DIR).mkdir(parents=True, exist_ok=True)
    file_path = Path(RAW_DOCS_DIR) / file.filename

    try:
        with trace("document.upload", {
            "filename": file.filename,
            "chunk_size": chunk_size
        }) as span:
            
            # 1. Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_size = file_path.stat().st_size
            if span: span.set_attribute("file_size", file_size)

            # 2. Process and chunk
            chunks = chunking_service.chunk_document_from_file(file.filename, chunk_size)
            if span: span.set_attribute("total_chunks", len(chunks))

            # 3. Automatically index chunks
            chunk_filename = f"{Path(file.filename).stem}_chunks.json"
            chunk_file_path = Path(CLEAR_DOCS_DIR) / chunk_filename
            
            indexed_count = 0
            indexing_error = None
            try:
                indexed_count = vector_repository.load_and_insert_from_json(
                    json_file_path=str(chunk_file_path),
                    source_name=Path(file.filename).stem
                )
                logger.info(f"Indexed {indexed_count} chunks from {chunk_filename}")
                if span: span.set_attribute("indexed_count", indexed_count)
            except Exception as index_error:
                indexing_error = str(index_error)
                logger.error(f"Indexing failed (chunks saved but not indexed): {indexing_error}")
                if span: span.set_attribute("indexing_error", indexing_error)

        message = f"Successfully processed {len(chunks)} chunks"
        if indexed_count > 0:
            message += f", indexed {indexed_count}"
        if indexing_error:
            message += f". Warning: indexing failed: {indexing_error}"

        return UploadChunkResponse(
            filename=file.filename,
            status="completed",
            total_chunks=len(chunks),
            chunk_size=chunk_size,
            message=message
        )

    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {e!s}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
        log_level="info"
    )
