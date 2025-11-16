# Personal AI Twin - Technical Design Document

## 1. Executive Summary

**Project**: Personal AI Biography & Knowledge Base  
**Purpose**: Digital twin that serves as queryable repository of professional life and knowledge  
**Architecture**: FastAPI backend, HTMX frontend, RAG-based AI with local embeddings, LLM-agnostic design

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Browser                        │
│              (HTMX + TailwindCSS)                       │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Application                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           API Layer (Routes)                     │   │
│  └──────────────────┬──────────────────────────────┘   │
│  ┌──────────────────▼──────────────────────────────┐   │
│  │          Service Layer                           │   │
│  │  - RAG Service                                   │   │
│  │  - Document Processing                           │   │
│  │  - LLM Abstraction                              │   │
│  └──────┬────────────────────┬─────────────────────┘   │
└─────────┼────────────────────┼──────────────────────────┘
          │                    │
┌─────────▼─────────┐  ┌──────▼──────────────┐
│  Vector Store     │  │  Relational DB      │
│  (pgvector/       │  │  (PostgreSQL)       │
│   ChromaDB)       │  │                     │
└───────────────────┘  └─────────────────────┘
          │
┌─────────▼──────────────────┐
│    Local LLM (Ollama)      │
│    - llama3.2              │
│    - mistral               │
│    - qwen                  │
└────────────────────────────┘
```

### 2.2 Technology Stack Summary

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend | HTMX + TailwindCSS | Minimal JS, server-driven interactions |
| Backend | FastAPI | Async, type-safe, auto-docs |
| Database | PostgreSQL + pgvector | ACID compliance, vector extension for scale |
| Vector Store (Dev) | ChromaDB | Easy local development |
| Embeddings | sentence-transformers | Local, free, multilingual |
| LLM Runtime | Ollama | Easy model switching, local privacy |
| Testing | pytest + httpx | Standard Python testing |
| ORM | SQLAlchemy 2.0 | Type-safe, async support |
| Migrations | Alembic | Database version control |

---

## 3. Database Design

### 3.1 Scaling Strategy

**Phase 1 (MVP)**: SQLite + ChromaDB  
**Phase 2 (Production)**: PostgreSQL + pgvector  
**Phase 3 (Scale)**: PostgreSQL + Dedicated vector DB (Qdrant/Weaviate)

### 3.2 Schema Design

#### Relational Tables (PostgreSQL)

```sql
-- Document metadata and versioning
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    document_type VARCHAR(50) NOT NULL, -- 'journal', 'project', 'publication', 'note'
    source_path TEXT,
    content_hash VARCHAR(64) UNIQUE, -- SHA-256 for deduplication
    file_size_bytes INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    indexed_at TIMESTAMP,
    metadata JSONB, -- Flexible: {tags, author, project_name, etc}
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);

-- Document chunks for RAG
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    embedding vector(384), -- all-MiniLM-L6-v2 dimension
    metadata JSONB, -- {heading, section, page_number, etc}
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_embedding ON document_chunks USING ivfflat(embedding vector_cosine_ops);

-- Conversation history
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB -- {tags, context, summary}
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    llm_model VARCHAR(100), -- Track which model generated response
    retrieved_chunks UUID[], -- Array of chunk IDs used
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);

-- LLM configurations and usage tracking
CREATE TABLE llm_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL, -- 'ollama:llama3.2', 'openai:gpt-4'
    provider VARCHAR(50) NOT NULL, -- 'ollama', 'openai', 'anthropic'
    model_identifier VARCHAR(200) NOT NULL,
    parameters JSONB, -- {temperature, top_p, max_tokens, etc}
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    llm_config_id UUID REFERENCES llm_configs(id),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_logs_created ON usage_logs(created_at DESC);
```

### 3.3 Vector Store Strategy

**ChromaDB (Development)**:
```python
# Simple, embedded, no config
collection = chroma_client.create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)
```

**pgvector (Production)**:
```python
# Integrated with PostgreSQL, ACID compliance
# Use document_chunks.embedding column with ivfflat index
# Supports hybrid search (vector + full-text)
```

**Migration Path**: Export ChromaDB → Import to pgvector using bulk insert with progress tracking.

---

## 4. RAG Implementation

### 4.1 Document Processing Pipeline

```
Raw Document → Text Extraction → Chunking → Embedding → Storage
     │              │                │           │          │
     │         (unstructured/    (semantic)  (sentence-  (DB +
     │          pypdf/docx)      splitter)  transformers) Vector)
     │
     └─→ Metadata Extraction (title, dates, tags)
```

### 4.2 Chunking Strategy

**Approach**: Semantic chunking with overlap

```python
class ChunkingStrategy:
    """
    Hybrid approach:
    1. Respect document structure (paragraphs, sections)
    2. Target chunk size: 500-1000 tokens
    3. Overlap: 100 tokens (20%)
    4. Metadata preservation
    """
    
    CHUNK_SIZE = 800  # tokens
    OVERLAP = 150     # tokens
    MIN_CHUNK_SIZE = 200
```

**Why this matters**:
- Small chunks: Better precision, more API calls
- Large chunks: Better context, worse precision
- Overlap: Prevents context loss at boundaries

**Document Type Handlers**:
```python
handlers = {
    '.txt': TextHandler,
    '.md': MarkdownHandler,  # Preserve headers
    '.pdf': PDFHandler,      # Extract page numbers
    '.docx': DocxHandler,
    '.html': HTMLHandler,    # Strip boilerplate
    '.eml': EmailHandler,    # Parse headers
}
```

### 4.3 Embedding Model

**Primary**: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Speed: ~1000 sentences/sec on CPU
- Quality: Good for English, decent multilingual
- Size: 80MB

**Upgrade Path**: `all-mpnet-base-v2` (768d, slower but better quality)

```python
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True
        ).tolist()
```

### 4.4 Retrieval Strategy

**Hybrid Search** (Phase 2):
1. **Vector similarity** (semantic): Get top 20 candidates
2. **BM25 full-text** (keyword): Get top 20 candidates
3. **Reciprocal Rank Fusion**: Merge results
4. **Reranking** (optional): Cross-encoder for final top-k

**Simple Version (Phase 1)**:
- Pure cosine similarity
- Top-k = 5-10 chunks
- Distance threshold: 0.3-0.5

```python
async def retrieve_context(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.4
) -> list[DocumentChunk]:
    query_embedding = embed_text(query)
    
    # Vector search
    results = await db.execute(
        select(DocumentChunk)
        .order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        )
        .limit(top_k * 2)  # Get extras for filtering
    )
    
    # Filter by threshold and return top_k
    filtered = [r for r in results if r.similarity > similarity_threshold]
    return filtered[:top_k]
```

---

## 5. LLM Abstraction Layer

### 5.1 Provider Interface

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class LLMProvider(ABC):
    """Abstract base for all LLM providers"""
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        pass
```

### 5.2 Ollama Implementation

```python
import httpx
from typing import AsyncGenerator

class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": stream
        }
        
        if stream:
            return self._stream_response(payload)
        else:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    async def _stream_response(self, payload) -> AsyncGenerator[str, None]:
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        yield chunk["message"]["content"]
    
    async def count_tokens(self, text: str) -> int:
        # Approximation: 1 token ≈ 4 chars
        return len(text) // 4
    
    def get_model_info(self) -> dict:
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url
        }
```

### 5.3 LLM Factory

```python
class LLMFactory:
    """Factory for creating LLM providers"""
    
    _providers: dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider,
        # Future: "openai": OpenAIProvider,
        # "anthropic": AnthropicProvider,
    }
    
    @classmethod
    def create(cls, provider: str, **kwargs) -> LLMProvider:
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
        return cls._providers[provider](**kwargs)
    
    @classmethod
    def register(cls, name: str, provider_class: type[LLMProvider]):
        cls._providers[name] = provider_class
```

### 5.4 Configuration Management

```yaml
# config/llm_configs.yaml
default: ollama_llama32

models:
  ollama_llama32:
    provider: ollama
    model: llama3.2
    temperature: 0.7
    max_tokens: 2000
    system_prompt: "You are an AI assistant representing Alan's professional knowledge..."
  
  ollama_mistral:
    provider: ollama
    model: mistral
    temperature: 0.8
    max_tokens: 2500
  
  ollama_qwen:
    provider: ollama
    model: qwen2.5
    temperature: 0.7
    max_tokens: 2000
```

---

## 6. API Design

### 6.1 Core Endpoints

```python
# app/api/routes.py

from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()

# === Document Management ===

@router.post("/api/documents/upload")
async def upload_document(
    file: UploadFile,
    document_type: str,
    tags: list[str] | None = None
) -> DocumentResponse:
    """Upload and index a new document"""
    pass

@router.get("/api/documents")
async def list_documents(
    document_type: str | None = None,
    limit: int = 50,
    offset: int = 0
) -> list[DocumentResponse]:
    """List all documents with pagination"""
    pass

@router.get("/api/documents/{document_id}")
async def get_document(document_id: str) -> DocumentDetailResponse:
    """Get document details including chunks"""
    pass

@router.delete("/api/documents/{document_id}")
async def delete_document(document_id: str) -> dict:
    """Soft delete document and its chunks"""
    pass

@router.post("/api/documents/{document_id}/reindex")
async def reindex_document(document_id: str) -> dict:
    """Re-chunk and re-embed document"""
    pass

# === Chat / Query ===

@router.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main RAG endpoint
    - Retrieves relevant context
    - Generates response with LLM
    - Returns sources
    """
    pass

@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming version of chat endpoint"""
    pass

@router.get("/api/conversations")
async def list_conversations(
    limit: int = 50,
    offset: int = 0
) -> list[ConversationResponse]:
    """List conversation history"""
    pass

@router.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str
) -> ConversationDetailResponse:
    """Get full conversation with messages"""
    pass

# === Search ===

@router.get("/api/search")
async def search(
    query: str,
    document_type: str | None = None,
    limit: int = 10
) -> SearchResponse:
    """Semantic search across all documents"""
    pass

# === Admin / Config ===

@router.get("/api/llm/models")
async def list_llm_models() -> list[LLMConfigResponse]:
    """List available LLM configurations"""
    pass

@router.post("/api/llm/models/switch")
async def switch_llm_model(model_name: str) -> dict:
    """Switch active LLM model"""
    pass

@router.get("/api/stats")
async def get_stats() -> StatsResponse:
    """System statistics: document count, token usage, etc."""
    pass
```

### 6.2 Request/Response Models

```python
from pydantic import BaseModel, Field
from datetime import datetime

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str | None = None
    model: str | None = None  # Override default LLM
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    include_sources: bool = True

class SourceChunk(BaseModel):
    document_id: str
    document_title: str
    content: str
    similarity_score: float
    metadata: dict

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    model_used: str
    sources: list[SourceChunk] | None = None
    token_count: int
    latency_ms: int

class DocumentResponse(BaseModel):
    id: str
    title: str
    document_type: str
    created_at: datetime
    chunk_count: int
    tags: list[str]
```

---

## 7. Frontend Design (HTMX + TailwindCSS)

### 7.1 Architecture

```
/templates
  /pages
    - index.html          # Chat interface
    - documents.html      # Document management
    - search.html         # Advanced search
  /components
    - chat_message.html   # Single message component
    - document_card.html  # Document list item
    - source_citation.html # Retrieved source display
  /layouts
    - base.html           # Main layout with nav
```

### 7.2 Main Chat Interface

```html
<!-- templates/pages/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Twin - Chat</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto max-w-4xl p-4">
        <!-- Header -->
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800">Your AI Twin</h1>
            <p class="text-gray-600">Professional knowledge assistant</p>
        </header>
        
        <!-- Chat Container -->
        <div class="bg-white rounded-lg shadow-lg">
            <!-- Messages -->
            <div id="messages" 
                 class="h-96 overflow-y-auto p-4 space-y-4">
                <!-- Messages loaded via HTMX -->
            </div>
            
            <!-- Input Form -->
            <form hx-post="/api/chat" 
                  hx-target="#messages" 
                  hx-swap="beforeend"
                  hx-indicator="#loading"
                  class="border-t p-4">
                
                <div class="flex gap-2">
                    <input type="text" 
                           name="message" 
                           placeholder="Ask me anything..."
                           class="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                           required>
                    
                    <button type="submit" 
                            class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                        Send
                    </button>
                </div>
                
                <!-- Loading Indicator -->
                <div id="loading" class="htmx-indicator mt-2">
                    <span class="text-sm text-gray-500">Thinking...</span>
                </div>
            </form>
        </div>
        
        <!-- Sources Section -->
        <div id="sources" 
             class="mt-4 bg-white rounded-lg shadow p-4 hidden">
            <h3 class="font-semibold text-gray-700 mb-2">Sources</h3>
            <div id="source-list"></div>
        </div>
    </div>
</body>
</html>
```

### 7.3 HTMX Patterns

**Streaming Response**:
```html
<div hx-post="/api/chat/stream"
     hx-ext="sse"
     sse-connect="/api/chat/stream"
     sse-swap="message">
    <!-- Content streams here -->
</div>
```

**Infinite Scroll for Documents**:
```html
<div hx-get="/api/documents?offset=20"
     hx-trigger="revealed"
     hx-swap="afterend">
    <!-- Load more trigger -->
</div>
```

**File Upload with Progress**:
```html
<form hx-post="/api/documents/upload"
      hx-encoding="multipart/form-data"
      hx-indicator="#upload-progress">
    <input type="file" name="file">
    <progress id="upload-progress" class="htmx-indicator"></progress>
</form>
```

---

## 8. Testing Strategy

### 8.1 Test Structure

```
/tests
  /unit
    - test_chunking.py
    - test_embeddings.py
    - test_llm_providers.py
    - test_rag_service.py
  /integration
    - test_api_endpoints.py
    - test_document_pipeline.py
    - test_conversation_flow.py
  /e2e
    - test_chat_workflow.py
  /fixtures
    - sample_documents.py
    - mock_llm_responses.py
  conftest.py
```

### 8.2 Unit Test Examples

```python
# tests/unit/test_chunking.py

import pytest
from app.services.chunking import ChunkingService

@pytest.fixture
def chunking_service():
    return ChunkingService(chunk_size=500, overlap=100)

def test_chunk_text_basic(chunking_service):
    text = "A" * 1000  # Simple long text
    chunks = chunking_service.chunk_text(text)
    
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)  # With tolerance
    
def test_chunk_overlap(chunking_service):
    text = "The quick brown fox jumps over the lazy dog. " * 100
    chunks = chunking_service.chunk_text(text)
    
    # Check overlap exists
    for i in range(len(chunks) - 1):
        assert chunks[i][-50:] in chunks[i+1][:150]

def test_preserve_paragraph_boundaries(chunking_service):
    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = chunking_service.chunk_text(text, respect_paragraphs=True)
    
    # Should not break mid-paragraph
    for chunk in chunks:
        assert not chunk.strip().startswith("aragraph")
```

```python
# tests/unit/test_embeddings.py

import pytest
import numpy as np
from app.services.embeddings import EmbeddingService

@pytest.fixture
def embedding_service():
    return EmbeddingService(model_name="all-MiniLM-L6-v2")

def test_embed_single_text(embedding_service):
    text = "This is a test sentence."
    embedding = embedding_service.embed_text(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Model dimension
    assert all(isinstance(x, float) for x in embedding)

def test_embed_batch(embedding_service):
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = embedding_service.embed_batch(texts)
    
    assert len(embeddings) == 3
    assert all(len(e) == 384 for e in embeddings)

def test_similarity_score(embedding_service):
    text1 = "The cat sat on the mat."
    text2 = "A feline rested on the rug."
    text3 = "Python programming language."
    
    emb1 = embedding_service.embed_text(text1)
    emb2 = embedding_service.embed_text(text2)
    emb3 = embedding_service.embed_text(text3)
    
    # Cosine similarity
    sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    assert sim_12 > sim_13  # Similar sentences should be closer
```

### 8.3 Integration Test Examples

```python
# tests/integration/test_api_endpoints.py

import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_upload_document():
    async with AsyncClient(app=app, base_url="http://test") as client:
        files = {"file": ("test.txt", b"Test content", "text/plain")}
        data = {"document_type": "note", "tags": "test,sample"}
        
        response = await client.post(
            "/api/documents/upload",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "id" in result
        assert result["title"] == "test.txt"

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        request = {
            "message": "What do you know about my projects?",
            "temperature": 0.7,
            "include_sources": True
        }
        
        response = await client.post("/api/chat", json=request)
        
        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        assert "sources" in result
        assert len(result["sources"]) > 0

@pytest.mark.asyncio
async def test_search_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/search",
            params={"query": "healthcare informatics", "limit": 5}
        )
        
        assert response.status_code == 200
        results = response.json()
        assert len(results["results"]) <= 5
```

### 8.4 Mock LLM for Testing

```python
# tests/fixtures/mock_llm_responses.py

class MockLLMProvider(LLMProvider):
    """Mock LLM for testing without actual model calls"""
    
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        # Return predictable responses for testing
        last_message = messages[-1]["content"]
        return f"Mock response to: {last_message[:50]}..."
    
    async def count_tokens(self, text: str) -> int:
        return len(text.split())
    
    def get_model_info(self) -> dict:
        return {"provider": "mock", "model": "test"}

# Register in tests
@pytest.fixture(autouse=True)
def use_mock_llm(monkeypatch):
    monkeypatch.setattr(
        "app.services.llm.LLMFactory._providers",
        {"mock": MockLLMProvider}
    )
```

### 8.5 Test Configuration

```python
# conftest.py

import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """Create test database"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "title": "Research Paper 1",
            "content": "This is about AI in healthcare...",
            "document_type": "publication"
        },
        # ... more samples
    ]
```

### 8.6 Coverage Goals

- **Unit tests**: 80%+ coverage
- **Integration tests**: Cover all API endpoints
- **E2E tests**: Main user workflows (upload → chat → search)

```bash
# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term
```

---

## 9. Project Structure

```
personal-ai-twin/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry
│   ├── config.py               # Configuration management
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # API endpoints
│   │   ├── dependencies.py     # DI for DB, services
│   │   └── middleware.py       # Logging, CORS, etc.
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy models
│   │   └── schemas.py          # Pydantic request/response models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_service.py      # Main RAG orchestration
│   │   ├── document_service.py # Document management
│   │   ├── embeddings.py       # Embedding generation
│   │   ├── chunking.py         # Text chunking logic
│   │   ├── vector_store.py     # Vector DB abstraction
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── base.py         # Abstract LLM interface
│   │       ├── ollama.py       # Ollama implementation
│   │       └── factory.py      # LLM factory
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_extraction.py  # PDF, DOCX, etc.
│   │   ├── token_counter.py
│   │   └── logger.py
│   │
│   └── templates/              # HTMX templates
│       ├── layouts/
│       ├── pages/
│       └── components/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── fixtures/
│   └── conftest.py
│
├── migrations/                 # Alembic migrations
│   ├── versions/
│   └── env.py
│
├── data/                       # Local data storage
│   ├── uploads/
│   ├── processed/
│   └── embeddings/
│
├── scripts/                    # Utility scripts
│   ├── migrate_chromadb_to_pg.py
│   ├── bulk_import.py
│   └── reindex_all.py
│
├── config/
│   ├── llm_configs.yaml
│   ├── logging.yaml
│   └── app_settings.yaml
│
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml             # Poetry alternative
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 10. Deployment & Scaling

### 10.1 Development Environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_USER: aitwin
      POSTGRES_PASSWORD: dev_password
      POSTGRES_DB: knowledge_base
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    # Pull models on startup
    command: serve
  
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://aitwin:dev_password@postgres/knowledge_base
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - postgres
      - ollama
    volumes:
      - ./data:/app/data
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
  ollama_models:
```

### 10.2 Production Considerations

**Infrastructure**:
- **App**: Uvicorn + Gunicorn (multiple workers)
- **DB**: PostgreSQL with pgvector extension
- **LLM**: Ollama on GPU instance OR cloud API
- **Reverse Proxy**: Nginx for static files + rate limiting
- **Monitoring**: Prometheus + Grafana

**Scaling Path**:

| Metric | Small (MVP) | Medium | Large |
|--------|-------------|--------|-------|
| Documents | <1,000 | <100,000 | 1M+ |
| Vector DB | ChromaDB | pgvector | Qdrant cluster |
| LLM | Local Ollama | Ollama + API fallback | Distributed inference |
| Compute | 4 CPU, 8GB RAM | 8 CPU, 16GB, GPU | Multi-node |

**Optimization Strategies**:
1. **Caching**: Redis for frequent queries
2. **Async Processing**: Celery for document ingestion
3. **CDN**: Static assets (Tailwind, HTMX)
4. **Connection Pooling**: pgbouncer for PostgreSQL
5. **Vector Index**: Tune ivfflat lists (sqrt(rows))

### 10.3 Environment Variables

```bash
# .env.example

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/aitwin
VECTOR_DIMENSIONS=384

# LLM
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=llama3.2
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32

# RAG
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.4

# App
APP_NAME="AI Twin"
APP_VERSION=1.0.0
DEBUG=False
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=http://localhost:8000

# Storage
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=50
```

---

## 11. Implementation Roadmap

### Phase 1: MVP (Weeks 1-2)
- [ ] Setup FastAPI + basic HTMX UI
- [ ] Implement document upload (PDF, TXT)
- [ ] Text extraction & chunking service
- [ ] Local embeddings with sentence-transformers
- [ ] ChromaDB integration
- [ ] Basic Ollama LLM integration
- [ ] Simple RAG query endpoint
- [ ] Unit tests for core services

### Phase 2: Core Features (Weeks 3-4)
- [ ] PostgreSQL + pgvector migration
- [ ] Full CRUD for documents
- [ ] Conversation history
- [ ] Source citations in responses
- [ ] Advanced chunking (respect structure)
- [ ] LLM provider abstraction
- [ ] Multiple model support (llama, mistral, qwen)
- [ ] Integration tests

### Phase 3: Enhanced UX (Weeks 5-6)
- [ ] Streaming responses
- [ ] Advanced search interface
- [ ] Document type filtering
- [ ] Tag-based organization
- [ ] Usage statistics dashboard
- [ ] Model comparison interface
- [ ] Batch document import
- [ ] E2E tests

### Phase 4: Optimization (Weeks 7-8)
- [ ] Hybrid search (vector + BM25)
- [ ] Query caching with Redis
- [ ] Async document processing
- [ ] Re-ranking for better retrieval
- [ ] Token usage optimization
- [ ] Performance profiling
- [ ] Load testing

### Phase 5: Production Ready (Weeks 9-10)
- [ ] Docker deployment
- [ ] CI/CD pipeline
- [ ] Monitoring & logging
- [ ] Backup strategy
- [ ] Security hardening
- [ ] Documentation
- [ ] User guide

---

## 12. Security Considerations

### 12.1 Data Privacy

Given your medical background, consider:
- **No PHI/PII in knowledge base** (unless properly secured)
- **Local-first approach**: All data stays on your infrastructure
- **Encryption at rest**: PostgreSQL encryption
- **Audit logging**: Track all document access

### 12.2 Authentication (Future)

```python
# For multi-user version
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(credentials: HTTPBearer = Depends(security)):
    # JWT validation
    # Return user object
    pass
```

### 12.3 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request, ...):
    pass
```

---

## 13. Monitoring & Observability

```python
# app/utils/logger.py

import structlog
from prometheus_client import Counter, Histogram

# Metrics
chat_requests = Counter('chat_requests_total', 'Total chat requests')
chat_latency = Histogram('chat_latency_seconds', 'Chat response time')
document_uploads = Counter('document_uploads_total', 'Documents uploaded')

# Structured logging
logger = structlog.get_logger()

async def log_request(request_id: str, endpoint: str):
    logger.info(
        "api_request",
        request_id=request_id,
        endpoint=endpoint,
        timestamp=datetime.utcnow()
    )
```

---

## 14. Future Enhancements

### Short-term
- [ ] Multi-modal support (images, audio)
- [ ] Fine-tuning embeddings on your data
- [ ] Scheduled re-indexing
- [ ] Export conversations as markdown
- [ ] Browser extension for quick queries

### Long-term
- [ ] Multi-user support with access control
- [ ] Knowledge graph integration (Neo4j)
- [ ] Federated search across multiple knowledge bases
- [ ] Active learning: feedback loop to improve retrieval
- [ ] Voice interface
- [ ] Mobile app (React Native + FastAPI)

---

## 15. Appendix

### A. Sample Requirements Files

```txt
# requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
asyncpg==0.29.0
alembic==1.13.1
pydantic==2.5.3
pydantic-settings==2.1.0
sentence-transformers==2.3.1
chromadb==0.4.22
httpx==0.26.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
jinja2==3.1.3
aiofiles==23.2.1
PyPDF2==3.0.1
python-docx==1.1.0
tiktoken==0.5.2
```

```txt
# requirements-dev.txt
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
httpx==0.26.0
black==24.1.1
ruff==0.1.14
mypy==1.8.0
```

### B. Ollama Model Setup

```bash
# Pull recommended models
ollama pull llama3.2        # 2B params, fast, good quality
ollama pull mistral         # 7B params, excellent reasoning
ollama pull qwen2.5:7b      # Alternative, multilingual

# Check running models
ollama list

# Test a model
ollama run llama3.2 "Explain RAG in one sentence"
```

### C. Database Migration Example

```python
# migrations/versions/001_initial_schema.py

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

def upgrade():
    op.create_table(
        'documents',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('content_hash', sa.String(64), unique=True),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        # ... more columns
    )
    
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('document_id', sa.UUID(), sa.ForeignKey('documents.id')),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(384)),
        # ... more columns
    )
    
    # Create vector index
    op.execute(
        'CREATE INDEX idx_chunks_embedding ON document_chunks '
        'USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)'
    )

def downgrade():
    op.drop_table('document_chunks')
    op.drop_table('documents')
```

---

## Summary

This architecture provides:

✅ **Minimal dependencies** - Core stack only (FastAPI, PostgreSQL, Ollama)  
✅ **Scalability path** - Start with ChromaDB, grow to pgvector, then dedicated vector DB  
✅ **LLM flexibility** - Abstract interface, easy to swap models  
✅ **Local-first** - Privacy-preserving, no cloud dependencies (optional)  
✅ **Testable** - Clear separation of concerns, mockable components  
✅ **Maintainable** - Type hints, async/await, modern Python patterns  

**Next Steps**:
1. Set up development environment (Docker Compose)
2. Implement Phase 1 MVP
3. Test with sample documents from your work
4. Iterate based on retrieval quality

---

**Document Version**: 1.0  
**Last Updated**: November 15, 2025  
**Author**: Technical Planning for Personal AI Twin Project
