# ⚡ Adaptive Event-Driven RAG System with Hybrid Retrieval running on Local LLM Models

**An Adaptive Event-Driven RAG System with Hybrid Retrieval (FAISS + MMR + BM25 + BGE Reranker + HyDE) using Event-Driven Architecture (Microservices with Kafka Message Broker) running on Local LLM Models (OLLAMA)**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-purple)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-MMR-purple)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/BM25-Sparse-yellow)](#)
[![BGE](https://img.shields.io/badge/BGE-Reranker-blue)](https://github.com/FlagOpen/FlagEmbedding)
[![HyDE](https://img.shields.io/badge/HyDE-Query%20Transform-pink)](#)
[![Kafka](https://img.shields.io/badge/Message%20Broker-Kafka-red)](https://kafka.apache.org/)
[![Microservices](https://img.shields.io/badge/Architecture-Microservices-purple)](#)
[![OLLAMA](https://img.shields.io/badge/Local%20Models-OLLAMA-orange)](https://ollama.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-24.0-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Introduction

One of the most valuable capabilities language models have brought to organizations is reading documents through AI and using it to automate responses. **RAG (Retrieval-Augmented Generation)** is the most practical technique for this, allowing us to feed the latest documents to the system and retrieve them based on user queries using dense vector search, sparse vector search, reranking, and other methods before passing them to the LLM.

While many resources explain RAG well, most use APIs and are not local, which is a concern for organizations with confidential documents. Another issue is that their code is usually limited to a Python notebook, lacking a real chatbot product perspective.

**This project solves both problems.**

This is a **production-ready, end-to-end RAG chatbot** that:
- ✅ Runs **100% locally** with OLLAMA (Gemma 3 4B/12B)
- ✅ No external API calls - complete **data privacy**
- ✅ Full **chatbot experience** with React frontend and WebSocket streaming
- ✅ **Document upload** support (PDF, DOCX, TXT, Markdown)
- ✅ **Hybrid retrieval** with FAISS + BM25 + BGE Reranker
- ✅ **Adaptive pipeline** using LangGraph orchestration
- ✅ **Event-driven architecture** with Kafka microservices
- ✅ **Real-time streaming** with character-by-character responses
- ✅ **Source citations** with filename and page numbers

Local models (8B-12B) prove highly effective for RAG, delivering strong retrieval and generation quality while maintaining full data privacy and cost efficiency. The system retrieves the most relevant chunks and generates accurate responses with references to source files and page numbers.

---

## 📖 Overview

This project presents an **adaptive Retrieval-Augmented Generation (RAG)** framework designed for scalable, privacy-preserving conversational AI.

Unlike traditional RAG pipelines that rely on a fixed retrieval strategy, this system dynamically adapts its retrieval workflow based on query complexity and retrieval quality. By combining **dense semantic search**, **sparse lexical search**, **adaptive query transformation**, and **cross-encoder reranking**, the system delivers highly relevant context for local Large Language Models (LLMs).

The architecture follows an **event-driven microservices design**, where independent services communicate asynchronously through **Apache Kafka**, enabling scalable document ingestion, distributed processing, and real-time response generation.

Since all AI models run locally through **OLLAMA** with **Gemma 3** models (4B or 12B), the entire pipeline operates without external API calls, ensuring complete data privacy and eliminating cloud inference costs.

---

## ✨ Key Features

### 📚 Document Processing
- Upload PDF, DOCX, TXT and Markdown documents
- Automatic text extraction using PyMuPDF
- Intelligent document chunking with overlap
- Metadata extraction and tracking
- Incremental document indexing
- Document status tracking (pending → indexing → completed → failed)

### 🔍 Adaptive Hybrid Retrieval
- **Dense Retrieval** - Semantic search using FAISS with Jina Embeddings v3
- **MMR (Maximum Marginal Relevance)** - Diversity-enhanced retrieval
- **Sparse Retrieval** - Lexical keyword matching using BM25
- **Hybrid Retrieval** - Combined Dense + Sparse with configurable ratio
- **Adaptive Query Rewriting** - Short query expansion and long query decomposition
- **Quality Evaluation** - Automatic assessment of retrieval quality
- **HyDE (Hypothetical Document Embeddings)** - Query transformation for better retrieval
- **Cross-Encoder Reranking** - BGE-Reranker v2 for precision scoring
- **Coreference Resolution** - Pronoun resolution using conversation history

### 🤖 Local AI Models (OLLAMA)
- **Gemma 3 4B** - Lightweight, fast inference (default)
- **Gemma 3 12B** - More capable, higher quality responses
- **Local Embedding Models** - Jina Embeddings v3
- **Local Reranker** - BGE Reranker v2-m3
- **Fully Offline** - No external API calls

### ⚡ Event-Driven Architecture
- **Apache Kafka** - Reliable message broker
- **Independent Microservices** - Decoupled, scalable services
- **Asynchronous Processing** - Non-blocking event streams
- **Fault Tolerance** - Service isolation
- **Horizontal Scalability** - Scale services independently

### 💬 Modern Chat Experience
- **React + TypeScript UI** - Modern, responsive interface
- **Real-time WebSocket Streaming** - Live token-by-token responses
- **Character-by-Character Streaming** - True typing effect with configurable delays
- **Multi-document Conversations** - Select multiple documents per conversation
- **Persistent Chat History** - SQLite database storage
- **Source Attribution** - Citations with filename and page numbers

### 🔒 Privacy First
- ✅ **100% Local** - No external API calls
- ✅ **Zero Cloud Costs** - No per-token or per-request fees
- ✅ **Air-Gap Ready** - Works in isolated environments
- ✅ **Data Sovereignty** - Complete control over your data
- ✅ **No Data Leakage** - Documents never leave your infrastructure

---

## 🏗️ System Architecture

The architecture consists of two independent workflows:

### 📚 Knowledge Base Construction
```
Document Upload → Text Extraction → Chunking → Embedding → FAISS + BM25 Index
```

### 💬 Adaptive Conversational Retrieval
```
Query → Coreference Resolution → Query Analysis → Query Rewriting → 
Dense Retrieval (FAISS + MMR) → Quality Evaluation → 
[Hybrid Retrieval | HyDE] → Reranker → OLLAMA (Gemma 3) → Answer + Citations
```

---

## 🔧 Adaptive Retrieval Pipeline

### Pipeline Nodes (LangGraph)

| # | Node | Description |
|---|------|-------------|
| 1 | **Coreference Resolution** | Resolves pronouns using conversation history |
| 2 | **Query Analysis** | Analyzes query length and complexity |
| 3 | **Query Rewriting** | Expands short or decomposes long queries |
| 4 | **HyDE Generation** | Generates hypothetical document (fallback) |
| 5 | **Dense Retrieval** | FAISS similarity search with MMR |
| 6 | **Quality Evaluation** | Checks if documents meet threshold |
| 7 | **Sparse Attachment** | Attaches BM25 results (if quality passes) |
| 8 | **Reranking** | BGE cross-encoder reranking |
| 9 | **Generation** | OLLAMA (Gemma 3) answer generation with citations |

### Conditional Routing

```
Quality Evaluation
        │
        ├─── QUALITY PASSED ───> Rerank → Generation
        │
        └─── QUALITY FAILED ───> HyDE (if enabled) → Retry Retrieval
                                  │
                                  └─── HyDE Already Used → Generation (fallback)
```

---

## 📂 Project Structure

```
adaptive-RAG/
│
├── backend/
│   ├── services/
│   │   ├── chat-service/
│   │   │   ├── Dockerfile
│   │   │   ├── Dockerfile-worker
│   │   │   └── src/
│   │   │       ├── api/
│   │   │       │   ├── routes/
│   │   │       │   │   ├── conversation_routes.py
│   │   │       │   │   ├── document_routes.py
│   │   │       │   │   └── websocket_routes.py
│   │   │       │   ├── schemas/
│   │   │       │   │   ├── conversation_schemas.py
│   │   │       │   │   └── document_schemas.py
│   │   │       │   └── websocket/
│   │   │       │       └── manager.py
│   │   │       ├── consumers/
│   │   │       │   └── chat_consumer.py
│   │   │       ├── database/
│   │   │       │   ├── models.py
│   │   │       │   └── sqlite_session.py
│   │   │       ├── handlers/
│   │   │       │   ├── document_embedding_done_handler.py
│   │   │       │   ├── prompt_answer_chunk_streamed_handler.py
│   │   │       │   └── prompt_answer_completed_handler.py
│   │   │       ├── services/
│   │   │       │   ├── conversation_service.py
│   │   │       │   └── document_service.py
│   │   │       ├── workers/
│   │   │       │   └── worker.py
│   │   │       ├── dependencies.py
│   │   │       ├── main.py
│   │   │       └── registry.py
│   │   │
│   │   └── llm-service/
│   │       ├── Dockerfile
│   │       └── src/
│   │           ├── graph/
│   │           │   ├── orchestrator_graph.py
│   │           │   ├── orchestrator_nodes.py
│   │           │   └── orchestrator_state.py
│   │           ├── handlers/
│   │           │   ├── document_uploaded_handler.py
│   │           │   └── prompt_requested_graph_handler.py
│   │           ├── services/
│   │           │   ├── core/
│   │           │   │   ├── document_loader.py
│   │           │   │   ├── embedding_model.py
│   │           │   │   ├── jina_wrapper.py
│   │           │   │   ├── text_chunker.py
│   │           │   │   └── text_preprocessor.py
│   │           │   ├── coreference_resolver.py
│   │           │   ├── generation_service.py
│   │           │   ├── hyde_service.py
│   │           │   ├── ingestion_service.py
│   │           │   ├── llm_service.py
│   │           │   ├── query_rewriting_service.py
│   │           │   ├── query_state.py
│   │           │   ├── reranker.py
│   │           │   └── retrieval_service.py
│   │           ├── workers/
│   │           │   └── worker.py
│   │           ├── dependencies.py
│   │           ├── main.py
│   │           └── registry.py
│   │
│   └── shared/
│       ├── common/
│       │   ├── events/
│       │   │   ├── base_event.py
│       │   │   ├── document_embedding_done.py
│       │   │   ├── document_uploaded.py
│       │   │   ├── prompt_answer_chunk_streamed.py
│       │   │   ├── prompt_answer_completed.py
│       │   │   └── prompt_answer_requested.py
│       │   ├── kafka/
│       │   │   ├── consumer.py
│       │   │   └── producer.py
│       │   └── message_bus/
│       │       ├── bus.py
│       │       └── interfaces.py
│       ├── requirements.txt
│       ├── setup.py
│       └── README.md
│
├── frontend/
│   └── rag-react-app/
│       ├── src/
│       ├── public/
│       └── package.json
│
├── models/
│   ├── snapshot/
│   │   └── jina-embeddings-v3/
│   ├── BAAI/
│   │   └── models--BAAI--bge-reranker-v2-m3/
│   ├── faiss_index/
│   ├── bm25_index/
│   └── hf_cache/
│
├── data/
│   ├── uploads/
│   ├── faiss_index/
│   └── bm25_index/
│
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- [OLLAMA](https://ollama.com/) installed locally
- Make sure ports 8001, 3000, 9092, 9000 are available

### 1. Install OLLAMA

```bash
# Download from https://ollama.com/
# Or install via command line
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download Gemma 3 Model

```bash
# Download Gemma 3 4B (lightweight, fast)
ollama pull gemma3:4b

# OR download Gemma 3 12B (more capable)
ollama pull gemma3:12b
```

### 3. Start OLLAMA Service

```bash
ollama serve
```

### 4. Configure Model in Environment

Set the model in `docker-compose.yml` or `.env`:

```yaml
# For Gemma 3 4B (default)
OLLAMA_MODEL=gemma3:4b

# OR for Gemma 3 12B
OLLAMA_MODEL=gemma3:12b
```

### 5. Start Services

```bash
docker-compose up -d --build
```

### 6. Access the Application

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **Chat Service API** | http://localhost:8001 |
| **API Documentation** | http://localhost:8001/docs |
| **Kafka UI (Kafdrop)** | http://localhost:9000 |

---

## ⚙️ Configuration

### Gemma 3 Model Selection

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **Gemma 3 4B** | 4B | Fast | Good | Quick responses, limited hardware |
| **Gemma 3 12B** | 12B | Slower | Excellent | High quality responses, better hardware |

### Key Environment Variables

```yaml
# Model Selection
OLLAMA_MODEL=gemma3:12b          # or gemma3:4b
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_TEMPERATURE=0.3

# MMR Settings
MMR_FETCH_K=200
MMR_LAMBDA_MULT=0.8

# Retrieval Settings
SPARSE_RETRIEVAL_RATIO=0.2
SIMILARITY_THRESHOLD=0.5
MIN_DOCS_REQUIRED=3

# HyDE
USE_HYDE=True

# Streaming
STREAM_CHAR_DELAY=0.02
STREAM_CHUNK_SIZE=3
STREAM_SOURCE_DELAY=0.3
```

### MMR Lambda Values

| Lambda | Meaning | Use Case |
|--------|---------|----------|
| **0.3** | High diversity | Broad topics |
| **0.5** | Balanced | Default |
| **0.7** | High relevance | Specific queries |
| **0.8** | Very high relevance | Focused queries |

---

## 🌐 API Endpoints

### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/conversation/new` | Create a new conversation |
| GET | `/conversation/user/{user_id}` | Get user conversations |
| GET | `/conversation/{conversation_id}` | Get conversation details |
| DELETE | `/conversation/{conversation_id}` | Delete a conversation |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload a document |
| GET | `/documents/` | Get all documents |
| DELETE | `/documents/{document_id}` | Delete a document |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/{user_id}` | WebSocket connection for real-time chat |

---

## 🔌 WebSocket Streaming

### Sending a Chat Message

```json
{
    "type": "chat",
    "conversation_id": "abc-123",
    "prompt": "What is RAG?",
    "file_ids": ["doc-1", "doc-2"],
    "retrieval_k": 20,
    "similarity_threshold": 0.5,
    "top_k": 5,
    "use_hyde": true,
    "use_mmr": true,
    "mmr_lambda_mult": 0.8
}
```

### Receiving Chunks

```json
{
    "type": "answer_chunk",
    "chunk": "RAG stands for ",
    "chunk_index": 0,
    "is_last": false
}
```

---

## 🔄 Event Flow

### Document Ingestion

```
Upload → Gateway → Kafka → Ingestion Service → Chunking → 
Embedding → FAISS + BM25 → Kafka → Chat Service → Database
```

### User Request

```
User → Gateway → Kafka → LLM Service → LangGraph Pipeline → 
Retrieval → Reranker → OLLAMA (Gemma 3) → Kafka → Chat Service → WebSocket → User
```

---

## 🔒 Privacy & Security

- ✅ **100% Local Inference** - All models run locally
- ✅ **Zero External API Calls** - No data sent to external services
- ✅ **Air-Gap Ready** - Works in isolated environments
- ✅ **Data Sovereignty** - Complete control over all data
- ✅ **No Cloud Costs** - No per-token fees

---

## 🐛 Troubleshooting

### OLLAMA Connection Error
```bash
# Check OLLAMA is running
ollama ps
# Restart OLLAMA
ollama serve
# Test connection
curl http://localhost:11434/api/tags
```

### Model Not Found
```bash
# Pull the model
ollama pull gemma3:4b
# or
ollama pull gemma3:12b
```

### Kafka Not Starting
```bash
docker-compose logs kafka
docker-compose restart kafka
```

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [OLLAMA](https://ollama.com/) - Local LLM inference (Gemma 3)
- [Google DeepMind](https://deepmind.google/) - Gemma 3 models
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Pipeline orchestration
- [Jina AI](https://jina.ai/) - Embedding models
- [BAAI](https://www.baai.ac.cn/) - BGE Reranker
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Apache Kafka](https://kafka.apache.org/) - Event streaming
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

---

