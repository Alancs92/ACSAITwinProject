# Personal AI Twin - Coding Patterns & Best Practices Cheatsheet

## Table of Contents
1. [Python Type Hints](#python-type-hints)
2. [Async/Await Patterns](#asyncawait-patterns)
3. [FastAPI Patterns](#fastapi-patterns)
4. [SQLAlchemy 2.0 Async](#sqlalchemy-20-async)
5. [Dependency Injection](#dependency-injection)
6. [Error Handling](#error-handling)
7. [Testing Patterns](#testing-patterns)
8. [Logging](#logging)
9. [Configuration Management](#configuration-management)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Python Type Hints

### Basic Types
```python
from typing import List, Dict, Set, Tuple, Optional, Union, Any

# Simple types
name: str = "John"
age: int = 30
price: float = 19.99
is_active: bool = True

# Collections
names: List[str] = ["Alice", "Bob"]
scores: Dict[str, int] = {"Alice": 95, "Bob": 87}
unique_ids: Set[int] = {1, 2, 3}
coordinates: Tuple[float, float] = (10.5, 20.3)

# Optional (can be None)
middle_name: Optional[str] = None  # Same as: str | None

# Union (multiple types)
id_or_name: Union[int, str] = 123  # Same as: int | str

# Any (avoid when possible)
data: Any = {"anything": "goes"}
```

### Function Annotations
```python
# Basic function
def greet(name: str, age: int) -> str:
    return f"Hello {name}, you are {age} years old"

# Function with optional parameters
def create_user(
    name: str,
    email: str,
    age: Optional[int] = None
) -> Dict[str, Any]:
    return {"name": name, "email": email, "age": age}

# Async function
async def fetch_data(url: str) -> Dict[str, Any]:
    # Implementation
    pass

# Function returning None
def log_message(message: str) -> None:
    print(message)

# Generator
def generate_numbers(n: int) -> Generator[int, None, None]:
    for i in range(n):
        yield i
```

### Class Annotations
```python
from typing import List, Optional
from dataclasses import dataclass

class User:
    """User class with type hints."""
    
    def __init__(
        self, 
        name: str, 
        email: str,
        age: Optional[int] = None
    ):
        self.name: str = name
        self.email: str = email
        self.age: Optional[int] = age
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "age": self.age
        }

# Using dataclass (recommended)
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
```

### Advanced Types
```python
from typing import Callable, TypeVar, Generic, Protocol

# Callable (function as parameter)
def execute_callback(
    callback: Callable[[int, str], bool],
    num: int,
    text: str
) -> bool:
    return callback(num, text)

# TypeVar for generic functions
T = TypeVar('T')

def first_element(items: List[T]) -> Optional[T]:
    return items[0] if items else None

# Generic class
class Container(Generic[T]):
    def __init__(self, value: T):
        self.value: T = value
    
    def get(self) -> T:
        return self.value

# Protocol (structural subtyping)
class Drawable(Protocol):
    def draw(self) -> None:
        ...

def render(obj: Drawable) -> None:
    obj.draw()
```

---

## Async/Await Patterns

### Basic Async
```python
import asyncio
from typing import List

# Simple async function
async def fetch_url(url: str) -> str:
    await asyncio.sleep(1)  # Simulating I/O
    return f"Content from {url}"

# Calling async functions
async def main():
    # Single call
    result = await fetch_url("https://example.com")
    
    # Multiple concurrent calls
    urls = ["https://a.com", "https://b.com", "https://c.com"]
    results = await asyncio.gather(*[fetch_url(url) for url in urls])
    
    # With error handling
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(result)

# Run async function
if __name__ == "__main__":
    asyncio.run(main())
```

### Async Context Managers
```python
from typing import AsyncGenerator

class AsyncDatabaseConnection:
    """Async context manager for database connections."""
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        print("Connecting...")
    
    async def disconnect(self):
        print("Disconnecting...")

# Usage
async def use_database():
    async with AsyncDatabaseConnection() as conn:
        # Use connection
        pass
```

### Async Generators
```python
from typing import AsyncGenerator

async def fetch_pages(url: str) -> AsyncGenerator[str, None]:
    """Async generator for paginated results."""
    page = 1
    while True:
        await asyncio.sleep(0.1)  # Simulating API call
        content = f"Page {page} from {url}"
        yield content
        
        page += 1
        if page > 5:  # Stop condition
            break

# Usage
async def process_pages():
    async for page in fetch_pages("https://api.example.com"):
        print(page)
```

### Async with Timeouts
```python
import asyncio

async def slow_operation() -> str:
    await asyncio.sleep(10)
    return "Done"

async def with_timeout():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
    return result
```

---

## FastAPI Patterns

### Basic Application Setup
```python
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup/shutdown."""
    # Startup
    print("Starting up...")
    # Initialize database, load models, etc.
    yield
    # Shutdown
    print("Shutting down...")
    # Close connections, cleanup

app = FastAPI(
    title="Personal AI Twin",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Request/Response Models (Pydantic)
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class DocumentBase(BaseModel):
    """Base document schema."""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    document_type: str
    tags: List[str] = []

class DocumentCreate(DocumentBase):
    """Schema for creating documents."""
    pass

class DocumentResponse(DocumentBase):
    """Schema for document responses."""
    id: str
    created_at: datetime
    chunk_count: int
    
    class Config:
        from_attributes = True  # SQLAlchemy 2.0 compatibility

class ChatRequest(BaseModel):
    """Chat request schema."""
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v

class ChatResponse(BaseModel):
    """Chat response schema."""
    message: str
    conversation_id: str
    sources: Optional[List[Dict[str, Any]]] = None
    token_count: int
```

### Route Patterns
```python
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api", tags=["documents"])

# Simple GET
@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> DocumentResponse:
    """Get document by ID."""
    document = await db.get(Document, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    return document

# POST with body
@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def create_document(
    document: DocumentCreate,
    db: AsyncSession = Depends(get_db)
) -> DocumentResponse:
    """Create new document."""
    db_document = Document(**document.dict())
    db.add(db_document)
    await db.commit()
    await db.refresh(db_document)
    return db_document

# File upload
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = "note",
    db: AsyncSession = Depends(get_db)
) -> DocumentResponse:
    """Upload and process document."""
    if not file.filename.endswith(('.pdf', '.txt', '.docx')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type"
        )
    
    content = await file.read()
    # Process file...
    return document

# Query parameters
@router.get("/search")
async def search_documents(
    query: str,
    limit: int = 10,
    offset: int = 0,
    document_type: Optional[str] = None
) -> List[DocumentResponse]:
    """Search documents."""
    # Implementation
    pass
```

### Dependency Injection Examples
```python
from typing import AsyncGenerator
from fastapi import Depends, HTTPException, Header

# Database dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Service dependency
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

# Composed dependency
async def get_document_service(
    db: AsyncSession = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> DocumentService:
    return DocumentService(db, embedding_service)

# Authentication dependency
async def get_current_user(
    authorization: str = Header(...)
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token = authorization.replace("Bearer ", "")
    # Validate token...
    return user

# Using dependencies
@router.post("/documents")
async def create_document(
    document: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
    user: User = Depends(get_current_user)
) -> DocumentResponse:
    return await service.create(document, user.id)
```

---

## SQLAlchemy 2.0 Async

### Model Definition
```python
from sqlalchemy import String, Integer, TIMESTAMP, Text, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    """Base class for all models."""
    pass

class Document(Base):
    """Document model."""
    __tablename__ = "documents"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Required fields
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Optional fields
    source_path: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    
    # JSON field
    metadata_: Mapped[dict] = mapped_column(JSONB, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Boolean
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationship
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title}')>"

class DocumentChunk(Base):
    """Document chunk model."""
    __tablename__ = "document_chunks"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Relationship
    document: Mapped["Document"] = relationship(back_populates="chunks")
```

### Database Setup
```python
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession,
    async_sessionmaker
)

# Create async engine
async_engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost/dbname",
    echo=True,  # Log SQL
    pool_pre_ping=True,  # Verify connections
    pool_size=5,
    max_overflow=10
)

# Create session factory
async_session = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create tables
async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

### CRUD Operations
```python
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

# CREATE
async def create_document(
    db: AsyncSession,
    title: str,
    content: str
) -> Document:
    document = Document(title=title, content=content)
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document

# READ - Single
async def get_document(
    db: AsyncSession,
    document_id: uuid.UUID
) -> Document | None:
    result = await db.get(Document, document_id)
    return result

# READ - Multiple with filter
async def get_documents_by_type(
    db: AsyncSession,
    document_type: str,
    limit: int = 10
) -> List[Document]:
    stmt = (
        select(Document)
        .where(Document.document_type == document_type)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

# READ - With relationship loading
async def get_document_with_chunks(
    db: AsyncSession,
    document_id: uuid.UUID
) -> Document | None:
    stmt = (
        select(Document)
        .where(Document.id == document_id)
        .options(selectinload(Document.chunks))
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

# UPDATE
async def update_document_title(
    db: AsyncSession,
    document_id: uuid.UUID,
    new_title: str
) -> None:
    stmt = (
        update(Document)
        .where(Document.id == document_id)
        .values(title=new_title)
    )
    await db.execute(stmt)
    await db.commit()

# DELETE
async def delete_document(
    db: AsyncSession,
    document_id: uuid.UUID
) -> None:
    stmt = delete(Document).where(Document.id == document_id)
    await db.execute(stmt)
    await db.commit()
```

### Complex Queries
```python
from sqlalchemy import func, and_, or_

# Count
async def count_documents_by_type(
    db: AsyncSession,
    document_type: str
) -> int:
    stmt = (
        select(func.count(Document.id))
        .where(Document.document_type == document_type)
    )
    result = await db.execute(stmt)
    return result.scalar()

# Join
async def get_documents_with_chunk_count(
    db: AsyncSession
) -> List[tuple[Document, int]]:
    stmt = (
        select(Document, func.count(DocumentChunk.id))
        .join(DocumentChunk, Document.id == DocumentChunk.document_id)
        .group_by(Document.id)
    )
    result = await db.execute(stmt)
    return result.all()

# Complex WHERE
async def search_documents(
    db: AsyncSession,
    query: str,
    document_type: str | None = None
) -> List[Document]:
    conditions = [Document.title.ilike(f"%{query}%")]
    
    if document_type:
        conditions.append(Document.document_type == document_type)
    
    stmt = select(Document).where(and_(*conditions))
    result = await db.execute(stmt)
    return result.scalars().all()
```

---

## Dependency Injection

### Service Layer Pattern
```python
from typing import Protocol

# 1. Define interface (Protocol)
class VectorStoreProtocol(Protocol):
    """Interface for vector store implementations."""
    
    async def add_embeddings(
        self, 
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[dict]
    ) -> None:
        ...
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[dict]:
        ...

# 2. Implement service
class ChromaVectorStore:
    """ChromaDB implementation of vector store."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        # Initialize ChromaDB client
    
    async def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[dict]
    ) -> None:
        # Implementation
        pass
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[dict]:
        # Implementation
        pass

# 3. Create dependency
def get_vector_store() -> VectorStoreProtocol:
    return ChromaVectorStore(collection_name="knowledge_base")

# 4. Use in service
class DocumentService:
    def __init__(
        self,
        db: AsyncSession,
        vector_store: VectorStoreProtocol
    ):
        self.db = db
        self.vector_store = vector_store
    
    async def add_document(self, document: Document) -> None:
        # Use both dependencies
        self.db.add(document)
        await self.db.commit()
        
        # Add to vector store
        await self.vector_store.add_embeddings(...)

# 5. Inject in route
@router.post("/documents")
async def create_document(
    document: DocumentCreate,
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStoreProtocol = Depends(get_vector_store)
) -> DocumentResponse:
    service = DocumentService(db, vector_store)
    return await service.add_document(document)
```

### Singleton Pattern with lru_cache
```python
from functools import lru_cache

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Singleton embedding service."""
    return EmbeddingService(model_name="all-MiniLM-L6-v2")

@lru_cache()
def get_settings() -> Settings:
    """Singleton settings."""
    return Settings()
```

---

## Error Handling

### Custom Exceptions
```python
class AITwinException(Exception):
    """Base exception for AI Twin."""
    pass

class DocumentNotFoundError(AITwinException):
    """Document not found."""
    pass

class EmbeddingError(AITwinException):
    """Error generating embeddings."""
    pass

class LLMError(AITwinException):
    """LLM generation error."""
    pass
```

### Exception Handlers in FastAPI
```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(
    request: Request, 
    exc: DocumentNotFoundError
):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)}
    )

@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    return JSONResponse(
        status_code=500,
        content={"detail": "LLM generation failed", "error": str(exc)}
    )
```

### Try-Except Patterns
```python
from typing import Optional

# Simple try-except
async def safe_operation() -> Optional[str]:
    try:
        result = await risky_operation()
        return result
    except SpecificError as e:
        logger.error(f"Operation failed: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error")
        raise

# With finally
async def with_cleanup():
    resource = None
    try:
        resource = await acquire_resource()
        result = await use_resource(resource)
        return result
    except ResourceError as e:
        logger.error(f"Resource error: {e}")
        raise
    finally:
        if resource:
            await release_resource(resource)

# Multiple exceptions
async def handle_multiple():
    try:
        result = await operation()
    except (ValueError, TypeError) as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
```

---

## Testing Patterns

### Pytest Fixtures
```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    class MockEmbeddingService:
        def embed_text(self, text: str) -> List[float]:
            return [0.1] * 384
        
        def embed_batch(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * 384 for _ in texts]
    
    return MockEmbeddingService()
```

### Unit Tests
```python
# Test service
@pytest.mark.asyncio
async def test_create_document(test_db, mock_embedding_service):
    service = DocumentService(test_db, mock_embedding_service)
    
    document = await service.create_document(
        title="Test",
        content="Content",
        document_type="note"
    )
    
    assert document.id is not None
    assert document.title == "Test"
    assert document.content == "Content"

# Test with exception
@pytest.mark.asyncio
async def test_create_document_invalid_type(test_db):
    service = DocumentService(test_db, mock_embedding_service)
    
    with pytest.raises(ValueError, match="Invalid document type"):
        await service.create_document(
            title="Test",
            content="Content",
            document_type="invalid"
        )

# Parametrized test
@pytest.mark.parametrize("text,expected_length", [
    ("Hello", 384),
    ("This is a longer sentence", 384),
    ("", 384),
])
def test_embed_text_dimension(embedding_service, text, expected_length):
    embedding = embedding_service.embed_text(text)
    assert len(embedding) == expected_length
```

### Integration Tests
```python
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_document_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/documents",
            json={
                "title": "Test Document",
                "content": "This is test content",
                "document_type": "note"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Document"
        assert "id" in data

@pytest.mark.asyncio
async def test_upload_file():
    async with AsyncClient(app=app, base_url="http://test") as client:
        files = {
            "file": ("test.txt", b"Test content", "text/plain")
        }
        
        response = await client.post(
            "/api/documents/upload",
            files=files
        )
        
        assert response.status_code == 201
```

### Mocking
```python
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    # Mock async function
    mock_llm = AsyncMock(return_value="Generated response")
    
    service = RAGService(db, vector_store, mock_llm)
    response = await service.generate_response("query")
    
    assert response == "Generated response"
    mock_llm.assert_called_once()

# Patch external dependency
@patch('app.services.embeddings.SentenceTransformer')
def test_embedding_service_init(mock_transformer):
    service = EmbeddingService()
    mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")
```

---

## Logging

### Setup Logging
```python
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO"):
    """Configure application logging."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler
            logging.FileHandler(log_dir / "app.log"),
        ]
    )
    
    # Set specific loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

# Usage
logger = logging.getLogger(__name__)
```

### Logging Best Practices
```python
import logging

logger = logging.getLogger(__name__)

class DocumentService:
    async def create_document(self, document: DocumentCreate) -> Document:
        logger.info(f"Creating document: {document.title}")
        
        try:
            # Process document
            db_document = Document(**document.dict())
            
            logger.debug(f"Document object created: {db_document.id}")
            
            # Save to database
            self.db.add(db_document)
            await self.db.commit()
            
            logger.info(f"Document created successfully: {db_document.id}")
            return db_document
            
        except Exception as e:
            logger.error(
                f"Failed to create document '{document.title}': {e}",
                exc_info=True  # Include traceback
            )
            raise

# Different log levels
logger.debug("Detailed information for debugging")
logger.info("General informational messages")
logger.warning("Warning messages for potentially harmful situations")
logger.error("Error messages for serious problems")
logger.critical("Critical messages for very serious errors")

# Structured logging
logger.info(
    "Document processed",
    extra={
        "document_id": doc.id,
        "user_id": user.id,
        "processing_time": elapsed
    }
)
```

---

## Configuration Management

### Using pydantic-settings
```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List

class Settings(BaseSettings):
    """Application settings."""
    
    # App
    app_name: str = "Personal AI Twin"
    debug: bool = False
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # LLM
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL"
    )
    default_llm_model: str = Field(
        default="llama3.2",
        env="DEFAULT_LLM_MODEL"
    )
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    similarity_threshold: float = Field(default=0.4, ge=0, le=1)
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    allowed_origins: List[str] = ["http://localhost:8000"]
    
    @validator("database_url")
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql", "sqlite")):
            raise ValueError("Invalid database URL")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage
settings = Settings()

# Access settings
print(settings.database_url)
print(settings.chunk_size)
```

---

## Common Pitfalls & Solutions

### ❌ Pitfall 1: Forgetting await
```python
# ❌ Wrong - Missing await
async def get_document(db: AsyncSession, id: str):
    document = db.get(Document, id)  # Returns coroutine, not result!
    return document

# ✅ Correct
async def get_document(db: AsyncSession, id: str):
    document = await db.get(Document, id)
    return document
```

### ❌ Pitfall 2: Not using async context managers
```python
# ❌ Wrong - Session not properly closed
async def wrong_way():
    session = async_session()
    result = await session.execute(stmt)
    return result  # Session leaked!

# ✅ Correct
async def correct_way():
    async with async_session() as session:
        result = await session.execute(stmt)
        return result.scalars().all()
```

### ❌ Pitfall 3: Blocking operations in async code
```python
import time

# ❌ Wrong - Blocks event loop
async def slow_function():
    time.sleep(5)  # Blocks entire application!
    return "Done"

# ✅ Correct - Use asyncio.sleep
async def fast_function():
    await asyncio.sleep(5)  # Doesn't block
    return "Done"

# ✅ For CPU-bound work, use executor
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def cpu_intensive():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            expensive_computation,
            data
        )
    return result
```

### ❌ Pitfall 4: Mutable default arguments
```python
# ❌ Wrong - Mutable default is shared!
def add_item(item: str, items: List[str] = []):
    items.append(item)
    return items

# ✅ Correct
def add_item(item: str, items: List[str] | None = None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### ❌ Pitfall 5: Not handling exceptions properly
```python
# ❌ Wrong - Bare except catches everything
try:
    result = await operation()
except:  # Catches KeyboardInterrupt, SystemExit, etc.
    logger.error("Error")

# ✅ Correct - Catch specific exceptions
try:
    result = await operation()
except ValueError as e:
    logger.error(f"Validation error: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error")
    raise
```

### ❌ Pitfall 6: SQLAlchemy 1.x patterns in 2.0
```python
# ❌ Wrong - Old query syntax
query = session.query(Document).filter_by(title="Test")
documents = query.all()

# ✅ Correct - 2.0 syntax
stmt = select(Document).where(Document.title == "Test")
result = await session.execute(stmt)
documents = result.scalars().all()
```

### ❌ Pitfall 7: Not using type hints
```python
# ❌ Wrong - No type information
def process_data(data):
    return [item.upper() for item in data]

# ✅ Correct - Clear types
def process_data(data: List[str]) -> List[str]:
    return [item.upper() for item in data]
```

### ❌ Pitfall 8: Forgetting to commit database changes
```python
# ❌ Wrong - Changes not persisted
async def create_document(db: AsyncSession, document: Document):
    db.add(document)
    return document  # Not committed!

# ✅ Correct
async def create_document(db: AsyncSession, document: Document):
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document
```

---

## Quick Reference Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_embeddings.py

# Run specific test
pytest tests/unit/test_embeddings.py::test_embed_text

# Run with verbose output
pytest -v

# Run with print statements
pytest -s

# Run async tests
pytest -v --asyncio-mode=auto
```

### Code Quality
```bash
# Format code
black app/ tests/

# Check formatting (don't modify)
black --check app/ tests/

# Lint with ruff
ruff check app/ tests/

# Fix auto-fixable issues
ruff check --fix app/ tests/

# Type check
mypy app/

# Run all quality checks
black app/ tests/ && ruff check app/ tests/ && mypy app/
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history
```

---

## Summary Checklist

When writing code, ensure:

- [ ] **Type hints** on all function signatures
- [ ] **Docstrings** for classes and public methods
- [ ] **Async/await** for all I/O operations
- [ ] **Error handling** with specific exceptions
- [ ] **Logging** at appropriate levels
- [ ] **Tests** for new functionality
- [ ] **DI pattern** for services
- [ ] **SQLAlchemy 2.0** syntax
- [ ] **Proper cleanup** (context managers, finally blocks)
- [ ] **Code formatting** (black, ruff)

---

**Remember**: Write code that is:
- **Readable** - Clear variable names, logical structure
- **Maintainable** - Modular, well-documented
- **Testable** - Dependency injection, mockable
- **Type-safe** - Full type hints
- **Async** - Non-blocking I/O
