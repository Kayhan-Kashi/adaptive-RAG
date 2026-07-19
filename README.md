# ⚡ Adaptive RAG System - Event-Driven Architecture with Hybrid Retrieval

**An intelligent, event-driven Adaptive RAG (Retrieval-Augmented Generation) system with hybrid retrieval, LangGraph orchestration, and real-time streaming, running entirely on local LLM models.**

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

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [LangGraph Pipeline Flow](#langgraph-pipeline-flow)
- [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [RAG Configuration](#rag-configuration)
- [WebSocket Streaming](#websocket-streaming)
- [API Endpoints](#api-endpoints)
- [Privacy First](#privacy-first)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

An event-driven **Adaptive RAG (Retrieval-Augmented Generation)** system that combines:

- 🔄 **Event-Driven Microservices** - Decoupled, scalable services with Kafka
- 🔀 **LangGraph Orchestration** - Intelligent pipeline routing and state management
- 🎯 **Hybrid Retrieval** - Dense (FAISS + MMR) + Sparse (BM25) search with BGE reranking
- 🧠 **HyDE** - Hypothetical Document Embeddings for query transformation
- 🎛️ **Adaptive Retrieval** - Dynamically adjusts strategy based on query quality
- 🤖 **Local Models** - 100% privacy-first with OLLAMA, zero cloud costs
- 📡 **Real-time Streaming** - Token-by-token responses via WebSocket

---

## 🎯 Key Features

### 🏗️ Event-Driven Microservices
- **Decoupled Services** - Independent, scalable microservices
- **Async Communication** - Non-blocking Kafka events
- **Fault Tolerance** - Service isolation
- **Horizontal Scaling** - Scale services independently

### 🔀 LangGraph Orchestration
- **State Management** - Centralized pipeline state
- **Conditional Routing** - Intelligent node selection based on quality
- **Modular Nodes** - Each pipeline step is a separate node
- **Smart Fallbacks** - Automatic HyDE fallback when quality is poor

### 🎯 Hybrid Retrieval Pipeline
- **FAISS + MMR** - Dense semantic retrieval with diversity
- **BM25** - Sparse lexical keyword matching
- **BGE Reranker** - Cross-encoder precision scoring
- **Intelligent Routing** - Quality-based pipeline decisions

### 🧠 Adaptive Retrieval Strategy
- **Quality Evaluation** - Automatic assessment of retrieval quality
- **Conditional Sparse Attachment** - Adds BM25 results only when quality passes
- **HyDE Fallback** - Triggers query transformation when quality is poor
- **Configurable Thresholds** - Adjustable quality thresholds

### 📡 Real-time Streaming
- **WebSocket Support** - Real-time bidirectional communication
- **Token-by-Token** - Stream responses as they're generated
- **Character-by-Character** - True typing effect with configurable delays
- **Live Updates** - Real-time progress tracking

### 🔒 Privacy First
- **100% Local** - No external API calls
- **Zero Cloud Costs** - No per-token or per-request fees
- **Air-Gap Ready** - Works in isolated environments
- **Data Sovereignty** - Complete control over your data
- **No Data Leakage** - Your documents never leave your infrastructure

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              React Frontend                                    │
│                          (WebSocket + REST API)                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway (Chat Service)                        │
│                      REST API + Kafka Producer + WebSocket                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Apache Kafka Message Broker                           │
│                          ⚡ Event-Driven Communication                         │
│                                                                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────────────┐ │
│  │  prompt-requested       │  │  prompt-answer-chunk-streamed               │ │
│  │  prompt-completed       │  │  document-uploaded                          │ │
│  │  document-embedding-done│  │  document-indexed                           │ │
│  └─────────────────────────┘  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────────┐
│   LLM Service     │    │   Chat Worker     │    │   Ingestion Service   │
│   (Consumer)      │    │   (Consumer)      │    │    (Consumer)         │
│                   │    │                   │    │                       │
│  🔀 Adaptive      │    │  💬 Chat Logic    │    │  📄 Document          │
│  RAG Pipeline     │    │  Database Update  │    │     Processing        │
│  (LangGraph)      │    │                   │    │                       │
└───────────────────┘    └───────────────────┘    └───────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    🔀 Adaptive LangGraph RAG Pipeline                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         PIPELINE FLOW                                   │   │
│  │                                                                          │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │   │
│  │  │ Coreference  │ → │   Query      │ → │    Query                  │  │   │
│  │  │ Resolution   │    │   Analysis   │    │    Rewriting             │  │   │
│  │  └──────────────┘    └──────────────┘    └──────────────────────────┘  │   │
│  │                                                       │                 │   │
│  │                                                       ▼                 │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │          Dense Retrieval (FAISS + MMR)                          │  │   │
│  │  │  • 20 chunks retrieved with MMR diversity                      │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │                                                       │                 │   │
│  │                                                       ▼                 │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │          Quality Evaluation                                     │  │   │
│  │  │  ┌────────────────────┐  ┌────────────────────────────────┐    │  │   │
│  │  │  │ ✅ QUALITY PASSED  │  │ ❌ QUALITY FAILED              │    │  │   │
│  │  │  │  → Attach Sparse   │  │  → Trigger HyDE Fallback      │    │  │   │
│  │  │  │  (BM25)            │  │  → Retry Retrieval            │    │  │   │
│  │  │  └────────────────────┘  └────────────────────────────────┘    │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │                    │                      │                           │   │
│  │                    ▼                      ▼                           │   │
│  │  ┌────────────────────────────┐  ┌────────────────────────────────┐  │   │
│  │  │  BGE Reranker +            │  │  HyDE Generation +             │  │   │
│  │  │  Generation                │  │  Retry Retrieval              │  │   │
│  │  └────────────────────────────┘  └────────────────────────────────┘  │   │
│  │                                                       │                 │   │
│  │                                                       ▼                 │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │          OLLAMA Generation (Streaming)                          │  │   │
│  │  │  • Token-by-token streaming                                     │  │   │
│  │  │  • Citation extraction                                          │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔀 LangGraph Pipeline Flow

### Pipeline Nodes

| # | Node | Description | Input | Output |
|---|------|-------------|-------|--------|
| 1 | **Coreference Resolution** | Resolves pronouns using conversation history | Query + History | Resolved Query |
| 2 | **Query Analysis** | Analyzes query length and complexity | Resolved Query | Query State |
| 3 | **Query Rewriting** | Expands short or decomposes long queries | Query + State | Rewritten Queries |
| 4 | **HyDE Generation** | Generates hypothetical document (fallback) | Query | HyDE Document |
| 5 | **Dense Retrieval** | FAISS similarity search with MMR | Query | Dense Chunks |
| 6 | **Quality Evaluation** | Checks if documents meet threshold | Dense Chunks | Quality Pass/Fail |
| 7 | **Sparse Attachment** | Attaches BM25 results (if quality passes) | Query | Combined Chunks |
| 8 | **Reranking** | BGE cross-encoder reranking | Combined Chunks | Ranked Chunks |
| 9 | **Generation** | OLLAMA answer generation with citations | Ranked Chunks | Answer |

### Conditional Routing Logic

```
Quality Evaluation
        │
        ├─── QUALITY PASSED ───> Rerank → Generation
        │
        └─── QUALITY FAILED ───> HyDE (if enabled) → Retry Retrieval
                                  │
                                  └─── HyDE Already Used → Generation (fallback)
```

### Adaptive Decisions

| Condition | Action |
|-----------|--------|
| **Quality Passed** | Attach sparse results → Rerank → Generate |
| **Quality Failed + HyDE Enabled** | Generate HyDE → Retry Retrieval |
| **Quality Failed + HyDE Disabled** | Skip HyDE → Generate with best available |
| **Quality Failed + HyDE Already Used** | Proceed to generation with fallback |

---

## 🔧 Hybrid Retrieval Pipeline

### Stage 1: Coreference Resolution
- Resolves pronouns (it, this, that, these, those) using conversation history
- Replaces pronouns with correct entities
- Improves query clarity for better retrieval

### Stage 2: Query Analysis & Rewriting
- **Short Queries** (< 30 chars) → Expanded with relevant keywords
- **Long Queries** (> 100 chars) → Decomposed into sub-queries
- **Well-formed** → Passed through unchanged

### Stage 3: HyDE (Query Transformation) - Optional
- Generates a hypothetical document from the user query
- Transforms the query into a document-like format for better retrieval
- **Smart Activation** - Only triggered when quality fails

### Stage 4: FAISS + MMR (Dense Retrieval with Diversity)
- FAISS for fast similarity search on dense embeddings
- MMR (Maximum Marginal Relevance) for diverse, non-redundant results
- **Configurable Lambda** - Balance between relevance (0.8) and diversity (0.2)

### Stage 5: Quality Evaluation
- Checks if documents meet the similarity threshold
- **Pass** → Proceed to sparse attachment + reranking
- **Fail** → Trigger HyDE fallback (if enabled)

### Stage 6: BM25 (Sparse Lexical Retrieval) - Conditional
- BM25 for keyword-based lexical matching
- **Only executed when quality passes**
- Adds 20% sparse results to the candidate pool

### Stage 7: BGE Reranker (Cross-Encoder Precision)
- BGE Reranker v2 for final precision scoring
- Cross-encoder for deep relevance assessment
- Output - Highly relevant, precision-ranked final results

### Stage 8: Generation (OLLAMA)
- OLLAMA local LLM for answer generation
- Streaming token-by-token responses
- Citation extraction from sources
- Character-by-character streaming with configurable delays

---

## 🛠️ Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Orchestration** | LangGraph | Pipeline orchestration & state management |
| **Message Broker** | Apache Kafka | Async event communication |
| **LLM** | OLLAMA (Local) | Text generation + HyDE |
| **Query Transform** | HyDE | Hypothetical Document Embeddings |
| **Dense Search** | FAISS + MMR | Semantic retrieval with diversity |
| **Sparse Search** | BM25 | Lexical keyword matching |
| **Reranker** | BGE Reranker v2 | Cross-encoder precision |
| **Embeddings** | Jina AI v3 | Dense vector encoding |
| **Coreference** | Custom LLM | Pronoun resolution |
| **Query Rewriting** | LLM-based | Query expansion/decomposition |
| **Backend** | Python 3.11 + FastAPI | Kafka producers/consumers |
| **Frontend** | React + TypeScript | Modern UI |
| **WebSocket** | FastAPI WebSockets | Real-time streaming |
| **Database** | SQLite | Conversation storage |
| **Orchestration** | Docker Compose | Container management |
| **Monitoring** | Kafdrop | Kafka UI |

---

## 📁 Project Structure

```
advanced-RAG/
├── backend/
│   ├── services/
│   │   ├── chat-service/
│   │   │   ├── Dockerfile
│   │   │   ├── requirements.txt
│   │   │   └── src/
│   │   │       ├── api/
│   │   │       ├── handlers/
│   │   │       ├── services/
│   │   │       ├── workers/
│   │   │       └── main.py
│   │   └── llm-service/
│   │       ├── Dockerfile
│   │       ├── requirements.txt
│   │       └── src/
│   │           ├── orchestrator/
│   │           │   ├── orchestrator_graph.py   # LangGraph pipeline
│   │           │   ├── orchestrator_nodes.py   # Pipeline nodes
│   │           │   ├── orchestrator_state.py   # State definitions
│   │           │   ├── generation_service.py   # LLM generation
│   │           │   ├── retrieval_service.py    # Hybrid retrieval
│   │           │   ├── coreference_resolver.py # Pronoun resolution
│   │           │   ├── hyde_service.py         # HyDE generation
│   │           │   ├── query_rewriting_service.py
│   │           │   └── reranker.py             # BGE reranker
│   │           ├── handlers/
│   │           ├── workers/
│   │           └── main.py
│   └── shared/
│       ├── common/
│       │   ├── events/
│       │   └── kafka/
│       └── utils/
├── frontend/
│   └── rag-react-app/
│       ├── src/
│       ├── public/
│       └── package.json
├── data/
│   ├── uploads/
│   ├── faiss_index/
│   └── bm25_index/
├── models/
│   ├── .cache/
│   ├── BAAI/
│   │   └── models--BAAI--bge-reranker-v2-m3/
│   ├── bm25_index/
│   ├── faiss_index/
│   ├── hf_cache/
│   └── snapshot/
│       └── jina-embeddings-v3/
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- [OLLAMA](https://ollama.com/) installed locally
- Make sure ports 8001, 3000, 9092, 9000 are available

### 1. Install OLLAMA

Download and install OLLAMA from [https://ollama.com/](https://ollama.com/)

### 2. Download a Model

```bash
ollama pull gemma3:4b
# or
ollama pull gemma3:12b
# or
ollama pull llama3.2:3b
# or
ollama pull mistral
```

### 3. Start OLLAMA Service

```bash
ollama serve
```

### 4. Clone the Repository

```bash
git clone https://github.com/Kayhan-Kashi/advanced-RAG.git
cd advanced-RAG
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 6. Build and Start All Services

```bash
docker-compose up -d --build
```

### 7. Access the Application

- **Frontend**: http://localhost:3000
- **Chat Service API**: http://localhost:8001
- **Kafka UI (Kafdrop)**: http://localhost:9000
- **API Documentation**: http://localhost:8001/docs

### 8. Verify Services

```bash
# Check all containers are running
docker-compose ps

# Check Kafka health
docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Check service logs
docker-compose logs -f rag-llm-service
```

---

## 📊 Environment Variables

### LLM Service Configuration

```yaml
rag-llm-service:
  environment:
    # Kafka
    - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
    - CONSUMER_GROUP=llm-worker-group
    
    # OLLAMA - Local LLM
    - OLLAMA_MODEL=gemma3:4b
    - OLLAMA_BASE_URL=http://host.docker.internal:11434
    - OLLAMA_TEMPERATURE=0.3
    
    # Embedding Model
    - MODEL_PATH=/app/models/snapshot/jina-embeddings-v3
    
    # HuggingFace Cache
    - HF_HUB_OFFLINE=0
    - HF_HUB_ENABLE_OFFLINE=0
    - TRANSFORMERS_OFFLINE=0
    - HF_HUB_DISABLE_SYMLINKS_WARNING=1
    - HF_HOME=/app/models/hf_cache
    
    # Reranker (BGE)
    - RERANKER_MODEL_PATH=/app/models/BAAI/models--BAAI--bge-reranker-v2-m3
    - RERANKER_REPO_ID=BAAI/bge-reranker-v2-m3
    - RERANKER_USE_FP16=true
    - RERANKER_BATCH_SIZE=32
    - RERANKER_MAX_LENGTH=512
    - RERANKER_LIMIT=30
    
    # MMR Settings
    - MMR_FETCH_K=200
    - MMR_LAMBDA_MULT=0.8
    
    # Retrieval Settings
    - FAISS_WEIGHT=0.6
    - BM25_WEIGHT=0.4
    - SPARSE_RETRIEVAL_RATIO=0.2
    - MIN_SPARSE_RESULTS=1
    
    # Quality Settings
    - SIMILARITY_THRESHOLD=0.5
    - MIN_DOCS_REQUIRED=3
    
    # HyDE
    - USE_HYDE=True
    
    # Streaming
    - STREAM_CHAR_DELAY=0.02
    - STREAM_CHUNK_SIZE=3
    - STREAM_SOURCE_DELAY=0.3
```

---

## 🎛️ RAG Configuration

### HyDE Configuration

HyDE (Hypothetical Document Embeddings) improves retrieval quality by generating a hypothetical document from the query.

```yaml
- USE_HYDE=True  # Enable HyDE fallback
```

| Scenario | Recommendation |
|----------|----------------|
| Short queries (1-3 words) | Enable HyDE |
| Ambiguous queries | Enable HyDE |
| Technical/domain-specific questions | Enable HyDE |
| Long, well-formed questions | Disable HyDE |

### MMR Configuration

MMR (Maximum Marginal Relevance) balances relevance and diversity:

```
MMR Score = λ × Relevance - (1-λ) × Diversity
```

| Lambda | Meaning | Use Case |
|--------|---------|----------|
| **0.3** | High diversity | Broad topics, different perspectives |
| **0.5** | Balanced | Default for most cases |
| **0.7** | High relevance | Specific fact-finding |
| **0.8** | Very high relevance | Focused, specific queries |

```yaml
- MMR_LAMBDA_MULT=0.8  # High relevance focus
- MMR_FETCH_K=200      # Number of candidates
```

### Quality Thresholds

```yaml
- SIMILARITY_THRESHOLD=0.5  # Minimum score to pass quality check
- MIN_DOCS_REQUIRED=3       # Minimum documents above threshold
```

### Sparse Retrieval Ratio

```yaml
- SPARSE_RETRIEVAL_RATIO=0.2  # 20% of results from BM25
- MIN_SPARSE_RESULTS=1        # Minimum sparse results
```

---

## 📡 WebSocket Streaming

### Connection

```
ws://localhost:8001/ws/{user_id}
```

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

### Receiving Completion

```json
{
    "type": "answer",
    "conversation_id": "abc-123",
    "answer": "RAG stands for Retrieval Augmented Generation..."
}
```

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
| GET | `/documents/{document_id}/status` | Get document status |
| DELETE | `/documents/{document_id}` | Delete a document |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |

---

## 🔒 Privacy First

- ✅ **100% Local** - No external API calls
- ✅ **Zero Cloud Costs** - No per-token or per-request fees
- ✅ **Air-Gap Ready** - Works in isolated environments
- ✅ **Data Sovereignty** - Complete control over your data
- ✅ **No Data Leakage** - Your documents never leave your infrastructure

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OLLAMA](https://ollama.com/) - Local LLM inference
- [Jina AI](https://jina.ai/) - Embeddings
- [BAAI](https://www.baai.ac.cn/) - BGE reranker
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Pipeline orchestration
- [Apache Kafka](https://kafka.apache.org/) - Event streaming
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

---

## 📊 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **Dense Retrieval** | Chunks per query | 20 |
| **Sparse Retrieval** | Chunks per query | 4 |
| **Reranking** | Chunks processed | 24 |
| **Generation** | Tokens per second | ~10-15 |
| **Pipeline Latency** | End-to-end | ~5-10s |
| **Chunk Streaming** | Characters per chunk | 3 |
| **Chunk Delay** | Between chunks | 20ms |

---

## 🐛 Troubleshooting

### Common Issues

**OLLAMA not responding:**
```bash
# Check OLLAMA is running
ollama ps
# Restart OLLAMA
ollama serve
```

**Kafka not starting:**
```bash
# Check Kafka logs
docker-compose logs kafka
# Restart Kafka
docker-compose restart kafka
```

**Models not downloading:**
```bash
# Check network connectivity
docker-compose exec rag-llm-service ping google.com
# Check HuggingFace access
docker-compose exec rag-llm-service curl -I https://huggingface.co
```

---

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OLLAMA Documentation](https://github.com/ollama/ollama)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)

---
