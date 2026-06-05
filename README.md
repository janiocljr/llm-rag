# rag-chat-llm

**Sistema RAG (Retrieval-Augmented Generation) offline** para consulta inteligente de documentos PDF via linguagem natural, com extração robusta de tabelas, embeddings locais e resposta contextual.

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![Tests](https://img.shields.io/badge/Tests-109%2F109%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/Coverage-100%25%20core%20modules-brightgreen)]()

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Características](#características)
3. [Requisitos](#requisitos)
4. [Instalação](#instalação)
5. [Quick Start](#quick-start)
6. [Estrutura do Projeto](#estrutura-do-projeto)
7. [Arquitetura](#arquitetura)
8. [Uso](#uso)
9. [Configuração](#configuração)
10. [Testes](#testes)
11. [Documentação](#documentação)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

Um sistema RAG completo e offline que permite:

- **Ingestão de PDFs**: Extração inteligente de texto e tabelas usando `pdfplumber` + `Camelot`
- **Processamento de texto**: Chunking token-aware com sobreposição semântica
- **Embeddings locais**: Modelo multilíngue `BAAI/bge-m3` (1024-dim)
- **Busca semântica**: Índice FAISS com `IndexFlatIP` (busca exata)
- **Re-ranking inteligente**: MMR (Maximal Marginal Relevance) para diversidade
- **Resposta contextual**: LLM local `Mistral 7B` com citações de fonte
- **100% Offline**: Zero dependência de APIs externas

---

## ✨ Características

### Extração Inteligente de Tabelas
- ✅ **Camelot** com fallback automático (lattice → stream → pdfplumber)
- ✅ **Persistência em CSV**: Tabelas salvas em `backend/data/index/tables/`
- ✅ **Detecção de headers/footers**: Remoção automática de conteúdo desnecessário
- ✅ **Metadata enriched**: Chunks com `is_table=true` e caminho do CSV

### Chunking Token-Aware
- ✅ **Recursive character splitter**: Respeita limites de tokens
- ✅ **Separadores semânticos**: Parágrafos → sentenças → palavras
- ✅ **Overlapping**: 64 tokens de sobreposição para continuidade
- ✅ **Page boundaries**: Chunks respeitam limites de página para citações precisas

### Busca e Ranking
- ✅ **Top-K retrieval**: Recupera 5 chunks similares por padrão
- ✅ **Similarity threshold**: Filtra por relevância (default 0.35)
- ✅ **MMR re-ranking**: λ=0.6 para balancear relevância vs. diversidade
- ✅ **Final K selection**: Injeta apenas 3 chunks no prompt do LLM

### Código Profissional
- ✅ **Type hints completos**: Todos os módulos com annotations
- ✅ **Docstrings detalhadas**: Documentação nas funções e classes
- ✅ **109 testes unitários**: 100% de cobertura em módulos core
- ✅ **Zero comentários inline**: Código auto-documentado

---

## 💻 Requisitos

### Hardware

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8+ cores |
| **Disco** | 8 GB livres | 15 GB |

### Software

- **macOS 11+** ou **Linux** (Ubuntu 20.04+)
- **Python 3.10 — 3.12** (3.13+ não testado)
- **Ghostscript** (para Camelot; instalado automaticamente via Conda/Homebrew)
- **Conda/Miniforge** (recomendado) ou **venv** local

---

## 🚀 Instalação

### Opção 1: Conda (Recomendado - macOS/Linux)

```bash
# Clone o repositório
git clone <REPO_URL>
cd llm-rag

# Instale Miniforge e crie o ambiente (macOS)
bash scripts/setup_backend_mac.sh

# Ative o conda
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"  # zsh
# ou para bash:
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

conda activate rag

# Instale o modelo LLM (~4.1 GB)
mkdir -p backend/models
# Baixe de: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# E coloque em backend/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Inicie a aplicação
python3 start.py --no-ingest
```

### Opção 2: venv Local

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 start.py --no-ingest
```

### Opção 3: Docker

```bash
cd backend
docker compose up --build
```

---

## ⚡ Quick Start

**Após instalação, execute em ~5 minutos:**

```bash
# 1. Adicione seus PDFs
mkdir -p backend/data/pdfs
cp /caminho/para/seus/documentos/*.pdf backend/data/pdfs/

# 2. Inicie o sistema
python3 start.py

# 3. Abra no navegador
# Chat: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

**URLs importantes:**
- 🎨 **Frontend (Streamlit)**: `http://localhost:8501`
- 🔌 **API (FastAPI)**: `http://localhost:8000`
- 📚 **Swagger Docs**: `http://localhost:8000/docs`
- ⚙️ **ReDoc**: `http://localhost:8000/redoc`

---

## 📁 Estrutura do Projeto

```
llm-rag/
├── README.md                     # Este arquivo
├── pyproject.toml               # Configuração build + dependências
├── start.py                     # Launcher (Conda/venv auto-detection)
├── environment.yml              # Conda environment (Python 3.11)
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app entrypoint
│   │   ├── api/
│   │   │   ├── routes.py       # /query, /ingest, /stats, /health
│   │   │   └── memory_routes.py # Memory management (optional)
│   │   ├── core/
│   │   │   ├── config.py       # Settings (Pydantic)
│   │   │   ├── ingestion.py    # PDFIngester + RecursiveCharSplitter
│   │   │   ├── embedder.py     # BAAI/bge-m3 wrapper
│   │   │   ├── vector_store.py # FAISS IndexFlatIP
│   │   │   ├── llm.py          # Mistral 7B via llama-cpp-python
│   │   │   ├── pipeline.py     # RAGPipeline orchestrator
│   │   │   └── text_utils.py   # Utilities (clean_text, estimate_tokens)
│   │   ├── models/
│   │   │   └── schemas.py      # Pydantic models (Request/Response)
│   │   └── utils/
│   │       ├── logging.py      # Log configuration
│   │       └── text.py         # Text utilities
│   ├── tests/
│   │   ├── conftest.py         # Pytest fixtures
│   │   ├── test_config.py      # 32 tests, 100% coverage
│   │   ├── test_embedder.py    # 13 tests, 100% coverage
│   │   ├── test_schemas.py     # 20 tests, 100% coverage
│   │   ├── test_text_utils.py  # 24 tests, 100% coverage
│   │   ├── test_ingestion.py   # 20 tests
│   │   └── test_rag_system.py  # Integration tests
│   ├── requirements.txt         # Dependencies
│   ├── Dockerfile             # Docker image
│   └── docker-compose.yml      # Docker services
│
├── frontend/
│   ├── rag_chat.py            # Streamlit main app
│   ├── components/            # UI components
│   │   ├── chat.py
│   │   ├── sidebar.py
│   │   ├── diagram.py
│   │   └── ...
│   └── utils/
│       ├── api_client.py      # Backend connector
│       ├── formatting.py      # Text formatting
│       └── demo_data.py       # Sample data
│
├── integrations/              # Optional integrations
│   ├── chroma_store.py       # ChromaDB (optional)
│   ├── mongo_store.py        # MongoDB (optional)
│   ├── memory.py             # Memory management
│   └── ...
│
├── docs/                      # Professional documentation
│   ├── README.md             # Documentation hub
│   ├── ARCHITECTURE.md       # System design (10+ Mermaid diagrams)
│   ├── COMPONENTS.md         # Component details
│   └── DATA_FLOWS.md         # End-to-end flows
│
└── scripts/
    └── setup_backend_mac.sh   # Conda setup for macOS
```

---

## 🏗️ Arquitetura

### Visão de Alto Nível

```
User Browser (Streamlit)
        ↓
   API Client (HTTP/SSE)
        ↓
 FastAPI Backend
        ↓
 RAG Pipeline
  ├─ PDFIngester (pdfplumber + Camelot)
  ├─ Embedder (BAAI/bge-m3, 1024-dim)
  ├─ VectorStore (FAISS IndexFlatIP)
  ├─ MMR Re-ranker
  └─ LLM (Mistral 7B GGUF)
        ↓
  Persistent Storage
  ├─ FAISS index
  ├─ Metadata JSON
  ├─ CSV tables
  └─ Models directory
```

### Pipeline de Query

```
User Question
    ↓
Embed Query (bge-m3)
    ↓
FAISS Search (top-5)
    ↓
Filter by Threshold (0.35)
    ↓
MMR Re-rank (λ=0.6)
    ↓
Build Prompt (system + context + question)
    ↓
Generate with LLM (Mistral)
    ↓
Stream Response to User
```

### Pipeline de Ingestão

```
PDF Files
    ↓
Load & Extract (pdfplumber)
    ↓
Extract Tables (Camelot lattice → stream → pdfplumber)
    ↓
Clean Text (Unicode normalization, remove headers/footers)
    ↓
Chunk (RecursiveCharSplitter, token-aware)
    ↓
Embed Chunks (bge-m3, batch processing)
    ↓
FAISS Index (IndexFlatIP)
    ↓
Persist (index + metadata + tables)
```

**Para diagrama completo, veja [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

---

## 🎮 Uso

### Via Frontend (Streamlit)

```bash
# Acesse http://localhost:8501
# 1. Adicione PDFs via sidebar
# 2. Digite sua pergunta
# 3. Veja respostas com citações
```

### Via API (cURL)

```bash
# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é o tema principal?"}'

# Ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": false}'

# Stats
curl http://localhost:8000/api/v1/stats

# Health check
curl http://localhost:8000/health
```

### Via Python SDK

```python
from app.core.ingestion import PDFIngester
from app.core.pipeline import RAGPipeline
from app.core.config import settings
from pathlib import Path

pipeline = RAGPipeline(settings)

pipeline.ingest(force_reindex=False)

response = pipeline.query(question="What is the main topic?")
print(response.answer)
print(response.found_in_documents)
for chunk in response.retrieved_chunks:
    print(f"[{chunk.citation}] {chunk.text}")
```

---

## ⚙️ Configuração

Defina em `backend/.env` ou variáveis de ambiente:

```bash
# Paths
DATA_DIR=backend/data/pdfs          # Pasta com PDFs
INDEX_DIR=backend/data/index        # Índice persistido
MODEL_DIR=backend/models            # Modelos GGUF

# Chunking
CHUNK_SIZE=512                      # Tokens por chunk
CHUNK_OVERLAP=64                    # Tokens overlap

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3         # Model ID
EMBEDDING_DIM=1024                  # Dimensionalidade
EMBEDDING_BATCH_SIZE=128            # Batch size

# Retrieval
RETRIEVAL_TOP_K=5                   # Chunks recuperados
RETRIEVAL_FINAL_K=3                 # Chunks no prompt
SIMILARITY_THRESHOLD=0.35           # Score mínimo
MMR_LAMBDA=0.6                      # Balance relevância/diversidade

# LLM
LLM_MODEL_PATH=models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
LLM_CONTEXT_LENGTH=4096             # Context window
LLM_MAX_NEW_TOKENS=512              # Max output
LLM_TEMPERATURE=0.1                 # Deterministic
LLM_N_GPU_LAYERS=0                  # GPU layers (0=CPU)
LLM_N_THREADS=8                     # CPU threads

# Optional
USE_CHROMA=false                    # ChromaDB (optional)
MONGO_URI=mongodb://...             # MongoDB (optional)
```

---

## 🧪 Testes

### Executar Testes

```bash
# Todos os testes
PYTHONPATH=./backend python -m pytest backend/tests/ -v

# Apenas testes unitários
PYTHONPATH=./backend python -m pytest backend/tests/ -v -m unit

# Com cobertura
PYTHONPATH=./backend python -m pytest backend/tests/ \
  --cov=app \
  --cov-report=html \
  --cov-report=term-missing

# Testes específicos
PYTHONPATH=./backend python -m pytest backend/tests/test_config.py -v
```

### Resultado Esperado

```
✅ 109 testes passando
✅ 100% de cobertura em módulos core:
   - app/core/config.py: 100%
   - app/core/embedder.py: 100%
   - app/core/text_utils.py: 100%
   - app/models/schemas.py: 100%
   - app/core/ingestion.py: 90%+ (testes completos)
```

---

## 📚 Documentação

### Documentação Técnica

**Todos os diagramas em Mermaid, renderizáveis no GitHub/GitLab:**

- 📐 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
  - Visão de 3 camadas (Client/Frontend/Backend)
  - Arquitetura de componentes
  - Pipeline de query (sequence diagram)
  - Pipeline de ingestão (flowchart)
  - Estrutura do backend (class diagrams)
  - 10+ diagramas Mermaid

- 🔧 **[docs/COMPONENTS.md](docs/COMPONENTS.md)**
  - Detalhes de cada componente
  - Schemas Pydantic
  - Endpoints da API
  - Interações entre componentes
  - 5+ diagramas

- 🔄 **[docs/DATA_FLOWS.md](docs/DATA_FLOWS.md)**
  - Query flow detalhado (sequence diagram)
  - Ingest flow completo (flowchart)
  - Data structures
  - Error handling
  - Performance timelines
  - 8+ diagramas

### Guias de Configuração

- [README-SETUP.md](README-SETUP.md) — Setup detalhado para macOS
- [environment.yml](environment.yml) — Conda environment
- [pyproject.toml](pyproject.toml) — Build config + dependencies

---

## 📊 Status do Projeto

| Componente | Status | Cobertura |
|-----------|--------|-----------|
| **Backend** | ✅ Production-Ready | 100% core |
| **Frontend** | ✅ Functional | N/A |
| **Testes** | ✅ Comprehensive | 109/109 passing |
| **Documentação** | ✅ Complete | 23+ diagramas |
| **Camelot** | ✅ Robust | Fallback automático |
| **Type Safety** | ✅ Full | Type hints everywhere |

---

## 🐛 Troubleshooting

### Porta Ocupada (8000/8501)

```bash
lsof -ti:8000 | xargs kill -9  # Kill backend
lsof -ti:8501 | xargs kill -9  # Kill frontend
```

### Conda Não Encontrado

```bash
# Ative o hook do conda
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"

# Ou use conda run
$HOME/miniforge3/bin/conda run -n rag python3 start.py
```

### NumPy Falhando em macOS

```bash
# Use Conda em vez de venv
bash scripts/setup_backend_mac.sh
```

### Modelo LLM Não Encontrado

```bash
# Verifique o caminho configurado
echo $LLM_MODEL_PATH

# Baixe de: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
mkdir -p backend/models
# Coloque o arquivo .gguf em backend/models/
```

### FAISS Index Corrompido

```bash
# Delete e reindexe
rm -rf backend/data/index
python3 start.py  # Reconstrói o índice
```

---

## 🚀 Próximos Passos

1. **Personalize o system prompt** em `backend/app/core/pipeline.py`
2. **Adicione seus PDFs** em `backend/data/pdfs/`
3. **Explore os documentos técnicos** em `docs/` para entender a arquitetura
4. **Customize retrieval parameters** em `backend/.env` conforme necessário

---

## 📖 Referências Adicionais

- [Arquitetura do Sistema](docs/ARCHITECTURE.md) — 10+ diagramas Mermaid, arquitetura de 3 camadas
- [Componentes](docs/COMPONENTS.md) — Detalhes de PDFIngester, Embedder, VectorStore, LLM, RAGPipeline
- [Fluxos de Dados](docs/DATA_FLOWS.md) — Flows end-to-end de query e ingestão, timelines de performance

---

## 📦 Dependências Principais

| Componente | Pacote | Versão | Propósito |
|-----------|--------|--------|----------|
| **Text Extraction** | pdfplumber | 0.11+ | Extração de texto de PDFs |
| **Table Extraction** | camelot-py[cv] | 0.12+ | Extração inteligente de tabelas |
| **Embeddings** | sentence-transformers | 3.0+ | BAAI/bge-m3 (1024-dim) |
| **Vector DB** | faiss-cpu | 1.8+ | Indexação e busca semântica |
| **LLM** | llama-cpp-python | 0.3+ | Mistral 7B GGUF inference |
| **Backend** | FastAPI | 0.110+ | API HTTP |
| **Frontend** | Streamlit | 1.40+ | Interface web |
| **Utilities** | pydantic | 2.0+ | Validação de schemas |

---

## 📄 Licença

[Adicione sua licença aqui]

---

## ✉️ Contato

- **Autor**: Janio Lima
- **Email**: pereira.zanellagra@gmail.com
- **Repositório**: [GitHub URL]

---

**Status**: Production-Ready ✅  
**Última atualização**: Junho 2026  
**Versão**: 1.0.0