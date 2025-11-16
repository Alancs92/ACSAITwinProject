# Personal AI Twin - Knowledge Base

> Your professional digital twin powered by RAG (Retrieval-Augmented Generation)

## Overview

A personal AI assistant that serves as a queryable repository of your professional knowledge, work history, and achievements. Built with FastAPI, HTMX, and local LLMs for privacy-first operation.

## Key Features

- 🧠 **RAG-based Knowledge Retrieval** - Semantic search across all your documents
- 🔒 **Privacy-First** - All data and processing stays local
- 🔄 **LLM Agnostic** - Easy switching between Ollama models (llama3.2, mistral, qwen)
- 📝 **Document Management** - Support for PDF, TXT, DOCX, MD, and more
- 💬 **Conversational Interface** - Natural chat with source citations
- 📊 **Usage Analytics** - Track token usage and model performance
- 🎯 **Minimal Dependencies** - Lean tech stack for maintainability

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI | Async API with auto-docs |
| Frontend | HTMX + TailwindCSS | Server-driven UI, minimal JS |
| Database | PostgreSQL + pgvector | ACID + vector search |
| Embeddings | sentence-transformers | Local, free embeddings |
| LLM | Ollama | Local model inference |
| Testing | pytest | Comprehensive test coverage |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 8GB RAM minimum (16GB recommended)
- 20GB disk space for models

### Installation

1. **Clone and setup**:
```bash
git clone <your-repo>
cd personal-ai-twin
cp .env.example .env
```

2. **Start services**:
```bash
docker-compose up -d
```

3. **Pull Ollama models**:
```bash
docker exec -it personal-ai-twin-ollama-1 ollama pull llama3.2
docker exec -it personal-ai-twin-ollama-1 ollama pull mistral
```

4. **Install Python dependencies**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

5. **Initialize database**:
```bash
alembic upgrade head
```

6. **Run the application**:
```bash
uvicorn app.main:app --reload
```

7. **Access the app**:
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

## Project Structure

```
personal-ai-twin/
├── app/                    # Application code
│   ├── api/               # API endpoints
│   ├── services/          # Business logic
│   ├── models/            # Data models
│   └── templates/         # HTMX templates
├── tests/                 # Test suite
├── migrations/            # Database migrations
├── config/                # Configuration files
├── data/                  # Document storage
└── scripts/               # Utility scripts
```

## Usage

### Upload Documents

```bash
# Via API
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "document_type=publication" \
  -F "tags=research,ai"

# Via Web UI
# Navigate to http://localhost:8000/documents
```

### Chat with Your Knowledge Base

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/chat",
        json={
            "message": "What are my key achievements in health informatics?",
            "include_sources": True
        }
    )
    print(response.json())
```

### Search Documents

```bash
curl "http://localhost:8000/api/search?query=machine%20learning&limit=5"
```

## Configuration

Key environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://aitwin:password@localhost/knowledge_base

# LLM
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=llama3.2

# RAG Settings
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.4

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

## Development Workflow

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature
```

2. **Make changes and test**:
```bash
pytest
black app/ tests/
ruff check app/ tests/
mypy app/
```

3. **Run the app locally**:
```bash
uvicorn app.main:app --reload
```

4. **Commit and push**:
```bash
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature
```

## Roadmap

- [x] Phase 1: MVP with basic RAG
- [ ] Phase 2: PostgreSQL + pgvector migration
- [ ] Phase 3: Enhanced UX (streaming, advanced search)
- [ ] Phase 4: Optimization (hybrid search, caching)
- [ ] Phase 5: Production deployment

See [TECHNICAL_DESIGN.md](./TECHNICAL_DESIGN.md) for detailed implementation plan.

## Architecture Highlights

### RAG Pipeline

```
User Query → Embedding → Vector Search → Top-K Chunks → LLM Context → Response
                                                              ↓
                                               Source Citations Attached
```

### Scaling Strategy

| Stage | Documents | Vector DB | Compute |
|-------|-----------|-----------|---------|
| MVP | <1K | ChromaDB | 4 CPU, 8GB |
| Production | <100K | pgvector | 8 CPU, 16GB, GPU |
| Scale | 1M+ | Qdrant | Multi-node |

## Security

- 🔐 Local-first: No external API calls required
- 🛡️ Data encryption at rest
- 📝 Audit logging for document access
- 🚫 No PHI/PII unless properly secured
- 🔒 Optional authentication for multi-user setup

## Troubleshooting

**Ollama not responding**:
```bash
docker logs personal-ai-twin-ollama-1
docker restart personal-ai-twin-ollama-1
```

**Database connection error**:
```bash
docker ps  # Check if postgres is running
docker-compose restart postgres
```

**Out of memory during embedding**:
- Reduce `EMBEDDING_BATCH_SIZE` in .env
- Process documents in smaller batches

**Slow retrieval**:
- Check vector index: `EXPLAIN ANALYZE SELECT ... ORDER BY embedding <=> ...`
- Tune ivfflat lists parameter
- Consider upgrading to dedicated vector DB

## Performance Tips

1. **Chunking**: Adjust `CHUNK_SIZE` based on your document types
2. **Top-K**: Higher values = better context but slower/more expensive
3. **Threshold**: Lower = more results but potentially less relevant
4. **Models**: llama3.2 (fast) vs mistral (better quality)
5. **Batch Processing**: Use bulk import scripts for large document sets

## Contributing

1. Read [TECHNICAL_DESIGN.md](./TECHNICAL_DESIGN.md)
2. Follow the coding standards (black, ruff, mypy)
3. Write tests for new features
4. Update documentation
5. Submit PR with clear description

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [HTMX Documentation](https://htmx.org/)
- [Ollama Models](https://ollama.ai/library)
- [sentence-transformers](https://www.sbert.net/)
- [pgvector](https://github.com/pgvector/pgvector)

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the [TECHNICAL_DESIGN.md](./TECHNICAL_DESIGN.md) for detailed documentation
- Review existing issues for similar problems

---

**Built with ❤️ for personal knowledge management and professional growth**
