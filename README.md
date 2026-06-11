# rag-chat-llm

**Sistema RAG (Retrieval-Augmented Generation) 100% offline** para consulta inteligente de documentos PDF via linguagem natural, com extração robusta de tabelas, embeddings locais e respostas contextuais com citação de fonte.

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()
[![Tests](https://img.shields.io/badge/Tests-109%2F109%20passing-brightgreen)]()

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
11. [Documentação Técnica](#documentação-técnica)
12. [Troubleshooting](#troubleshooting)
13. [Dependências Principais](#dependências-principais)
14. [Contato](#contato)

---

## 🎯 Visão Geral

Sistema RAG completo e offline que permite consultar documentos PDF através de linguagem natural. Toda inferência é local — sem envio de dados para APIs externas.

**Componentes principais:**

| Camada | Tecnologia | Detalhe |
|--------|-----------|---------|
| Extração de texto | `pdfplumber` + `Camelot` | Texto e tabelas de PDFs |
| Embeddings | `intfloat/multilingual-e5-small` | 384-dim, multilíngue |
| Índice vetorial | FAISS `IndexFlatIP` | Busca exata por similaridade |
| Índice lexical | BM25 (Okapi) | Busca por termos exatos, puro Python |
| Fusão de rankings | Reciprocal Rank Fusion (RRF) | Combina vetorial + lexical |
| Re-ranking | MMR (λ = 0.6) | Diversidade nos resultados |
| LLM | `Qwen2.5-7B-Instruct` Q4\_K\_M | Inferência local via llama-cpp |
| Frontend | Streamlit | Interface web interativa |
| Backend | FastAPI + SSE | API REST com streaming |

---

## ✨ Características

### Extração Inteligente de PDFs
- **Camelot** com fallback automático: lattice → stream → pdfplumber
- **Separação de notas de rodapé**: cada nota gera um chunk dedicado para preservar fatos numéricos
- **Detecção de headers/footers**: remoção automática de conteúdo repetitivo
- **Exportação de tabelas em CSV**: salvas em `backend/data/index/tables/`
- **Limpeza de tabelas pdfplumber**: remove linhas e colunas inteiramente vazias antes de gerar o chunk, evitando "| | |" que polui embeddings e BM25

### Chunking Token-Aware
- **Recursive character splitter**: respeita limites semânticos (parágrafos → sentenças → palavras)
- **Semantic paragraph chunker**: agrupa parágrafos por importância estrutural
- **Overlap de 50 tokens**: evita perda de informação nas bordas dos chunks
- **Page-aware**: chunks nunca cruzam limites de página — citações sempre precisas

### Busca Híbrida (Vetorial + Lexical)
- **FAISS IndexFlatIP**: recupera os 10 chunks semanticamente mais similares
- **BM25 Okapi** (`lexical.py`): busca por termos exatos — números, datas, siglas ("IPCA", "5,35%", "julho 2025") que embeddings pequenos não distinguem bem
- **Reciprocal Rank Fusion (RRF, k=60)**: funde os dois rankings de forma robusta a escalas distintas de score
- **Threshold de similaridade**: filtra candidatos com score vetorial < 0.45
- **MMR re-ranking**: seleciona os 5 melhores chunks balanceando relevância e diversidade
- **SSE streaming**: resposta transmitida token a token para o frontend

### Qualidade da Resposta
- **`classify_answer()`**: detecta e corrige o padrão de falso negativo de modelos quantizados — quando o LLM inicia com "Não encontrei..." mas em seguida apresenta o dado correto com citação, o prefixo espúrio é removido automaticamente
- **`stream_with_false_negative_guard()`**: versão streaming do mesmo mecanismo, bufferiza apenas o início sem bloquear o fluxo
- **`context_char_budget()`**: calcula o orçamento de caracteres para o bloco de contexto reservando espaço para a resposta e para o system prompt, evitando `ValueError` por estouro da janela de contexto do modelo

### Aceleração por Hardware
- **Apple Silicon (MPS)**: `LLM_N_GPU_LAYERS=-1` ativa Metal GPU para inferência (~30× vs. CPU)
- **Embeddings em MPS**: `EMBEDDING_DEVICE=mps` processa vetores na GPU unificada
- **CPU fallback**: configurável via `LLM_N_GPU_LAYERS=0` para x86/Linux

### Qualidade de Código
- **Type hints** completos em todos os módulos
- **109 testes unitários** com ≥ 90% de cobertura nos módulos core
- **Código limpo**: zero comentários inline, zero docstrings — auto-documentado por nomes

---

## 💻 Requisitos

### Hardware

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8+ cores |
| **Disco** | 10 GB livres | 15 GB |
| **GPU** | Opcional | Apple M1/M2/M3 (MPS) |

> **Nota Apple Silicon**: com `LLM_N_GPU_LAYERS=-1` e `EMBEDDING_DEVICE=mps` a inferência utiliza a GPU unificada do chip, reduzindo o tempo de resposta em ~30× em relação à execução exclusiva em CPU.

### Software

- **macOS 12+** ou **Linux** (Ubuntu 20.04+)
- **Python 3.10 — 3.12** (3.13+ não testado)
- **Ghostscript** (para Camelot; instalado via Conda/Homebrew)
- **Conda/Miniforge** (recomendado) ou **venv** local

---

## 🚀 Instalação

### Opção 1: Conda (Recomendado — macOS/Linux)

```bash
# Clone o repositório
git clone <REPO_URL>
cd llm-rag

# Instale Miniforge e crie o ambiente
bash scripts/setup_backend_mac.sh

# Ative o ambiente
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"  # zsh
# ou:
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"  # bash
conda activate rag

# Baixe o modelo LLM (~4.7 GB)
cd backend
python scripts/download_model.py

# Baixe o modelo de embeddings (~118 MB)
python scripts/download_embeddings.py

# Inicie a aplicação
cd ..
python3 start.py
```

### Opção 2: venv Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
python3 start.py
```

### Opção 3: Docker

```bash
cd backend
docker compose up --build
```

---

## ⚡ Quick Start

```bash
# 1. Adicione seus PDFs
mkdir -p backend/data/pdfs
cp /caminho/para/seus/documentos/*.pdf backend/data/pdfs/

# 2. Inicie o sistema (indexação automática na primeira execução)
python3 start.py

# 3. Acesse no navegador
#    Chat:        http://localhost:8501
#    API Docs:    http://localhost:8000/docs
```

**URLs disponíveis:**

| Serviço | URL |
|---------|-----|
| Frontend (Streamlit) | `http://localhost:8501` |
| API (FastAPI) | `http://localhost:8000` |
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |

---

## 📁 Estrutura do Projeto

```
llm-rag/
├── README.md
├── pyproject.toml               # Build config + dependências
├── start.py                     # Launcher (auto-detecta Conda/venv)
├── environment.yml              # Conda environment (Python 3.11)
│
├── backend/
│   ├── .env                     # Configuração de runtime
│   ├── app/
│   │   ├── main.py              # FastAPI app + lifespan
│   │   ├── api/
│   │   │   ├── routes.py        # /query, /query/stream, /ingest, /stats
│   │   │   └── memory_routes.py # Rotas de memória persistente (opcional)
│   │   ├── core/
│   │   │   ├── config.py        # Settings via Pydantic (lê .env)
│   │   │   ├── ingestion.py     # PDFIngester + RecursiveCharSplitter
│   │   │   ├── advanced_ingestion.py  # Chunking semântico, detecção H/F
│   │   │   ├── embedder.py      # multilingual-e5-small wrapper
│   │   │   ├── vector_store.py  # FAISS IndexFlatIP + MMR
│   │   │   ├── lexical.py       # BM25Index (Okapi) — busca lexical
│   │   │   ├── llm.py           # Qwen2.5-7B + classify_answer + context_char_budget
│   │   │   ├── pipeline.py      # RAGPipeline — busca híbrida + RRF + assinatura
│   │   │   ├── text_utils.py    # clean_text, estimate_tokens
│   │   │   ├── chroma_store.py  # ChromaDB (opcional)
│   │   │   ├── memory.py        # MemoryOrchestrator (opcional)
│   │   │   └── mongo_store.py   # MongoDB (opcional)
│   │   ├── models/
│   │   │   └── schemas.py       # Pydantic models (Request/Response)
│   │   └── utils/
│   │       ├── logging.py       # Configuração de logging
│   │       └── text.py          # Utilitários de texto
│   ├── scripts/
│   │   ├── download_model.py        # Baixa modelos LLM GGUF
│   │   ├── download_embeddings.py   # Baixa modelos de embeddings
│   │   ├── ensure_models.py         # Verifica cache antes do startup
│   │   ├── reindex_pdfs.py          # Reindexação forçada
│   │   ├── analyze_embeddings.py    # Análise comparativa de modelos
│   │   └── plot_embedding_analysis.py
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_config.py
│   │   ├── test_embedder.py
│   │   ├── test_ingestion.py
│   │   ├── test_schemas.py
│   │   ├── test_table_extraction.py
│   │   ├── test_text_utils.py
│   │   ├── test_utils.py
│   │   └── test_rag_system.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── frontend/
│   ├── rag_chat.py              # Streamlit main app
│   ├── components/
│   │   ├── chat.py              # Componente de chat
│   │   ├── sidebar.py           # Painel de configuração
│   │   ├── diagram.py           # Visualização da arquitetura
│   │   ├── prompt_viewer.py     # Visualizador do prompt enviado ao LLM
│   │   └── thinking.py          # Indicador de carregamento
│   └── utils/
│       ├── api_client.py        # Conector HTTP/SSE com o backend
│       ├── formatting.py        # Formatação de texto
│       └── demo_data.py         # Dados de exemplo
│
├── integrations/                # Integrações opcionais
│   ├── chroma_store.py          # ChromaDB
│   ├── mongo_store.py           # MongoDB
│   ├── memory.py
│   ├── memory_routes.py
│   ├── pipeline.py
│   ├── config.py
│   └── main.py
│
├── docs/
│   ├── ARCHITECTURE.md          # Diagramas Mermaid da arquitetura
│   ├── COMPONENTS.md            # Detalhes de cada componente
│   └── DATA_FLOWS.md            # Fluxos de dados end-to-end
│
└── scripts/
    └── setup_backend_mac.sh     # Setup Conda para macOS
```

---

## 🏗️ Arquitetura

### Visão de Alto Nível

```
Usuário (Navegador)
        │
   Streamlit Frontend
        │  HTTP / SSE
   FastAPI Backend
        │
   RAGPipeline
   ├── PDFIngester       ← pdfplumber + Camelot
   ├── Embedder          ← intfloat/multilingual-e5-small (384-dim)
   ├── VectorStore       ← FAISS IndexFlatIP
   ├── BM25Index         ← busca lexical (Okapi, puro Python)
   ├── RRF Fusion        ← Reciprocal Rank Fusion (k=60)
   ├── MMR Re-ranker     ← λ = 0.6
   └── LLM               ← Qwen2.5-7B-Instruct Q4_K_M (llama-cpp)
        │
   Armazenamento Local
   ├── faiss.index
   ├── metadata.json
   ├── index_signature.json   ← valida compatibilidade do modelo de embedding
   ├── tables/ (CSV)
   └── models/ (GGUF)
```

### Pipeline de Consulta (Query)

```
Pergunta do Usuário
        │
        ├─── Embed Query (multilingual-e5-small)
        │           │
        │    FAISS Search — top-10 candidatos (threshold ≥ 0.45)
        │
        └─── BM25 Search — top-10 por termos exatos
                    │
        Reciprocal Rank Fusion (RRF, k=60)
        — funde ambos os rankings em lista unificada
                    │
        MMR Re-rank (λ = 0.6) → top-5 chunks finais
                    │
        Build Prompt (context_char_budget — evita estouro)
        system prompt + contexto + pergunta + instruções
                    │
        Qwen2.5-7B generate (stream token-a-token)
                    │
        classify_answer() — corrige falso negativo
                    │
        SSE → Frontend → Usuário
```

### Pipeline de Ingestão

```
Arquivos PDF (backend/data/pdfs/)
        │
pdfplumber — extração página a página
        │
Camelot — detecção de tabelas (lattice → stream → fallback)
_format_pdfplumber_table() — remove linhas/colunas vazias
        │
Remoção de headers/footers + extração de notas de rodapé
        │
clean_text — normalização Unicode, colapso de espaços
        │
SemanticParagraphChunker / RecursiveCharSplitter
        │
Embed chunks (multilingual-e5-small, batch=32, MPS)
        │
FAISS IndexFlatIP — adição de vetores
BM25Index.build() — índice lexical em memória
        │
Persistência
├── faiss.index + metadata.json + tables/*.csv
└── index_signature.json  ← modelo/dim/prefixos atuais
```

> Para diagramas completos com sequências e class diagrams, veja [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🎮 Uso

### Via Frontend (Streamlit)

1. Acesse `http://localhost:8501`
2. Use o painel lateral para ajustar `top_k`, `threshold` e `final_k`
3. Digite sua pergunta e visualize a resposta com chunks recuperados e prompt completo

### Via API REST (cURL)

```bash
# Consulta simples (resposta completa)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é o tema principal do documento?"}'

# Consulta com streaming SSE
curl -N http://localhost:8000/api/v1/query/stream \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual foi a inflação em 2024?"}'

# Reindexar PDFs
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'

# Status do índice
curl http://localhost:8000/api/v1/stats

# Health check
curl http://localhost:8000/health
```

### Via Python

```python
from app.core.pipeline import RAGPipeline
from app.core.config import settings
from app.models.schemas import QueryRequest

pipeline = RAGPipeline(settings)

# Indexar PDFs (pula se o índice já existe)
pipeline.ingest(force_reindex=False)

# Consultar
request = QueryRequest(question="Qual é o tema principal?")
response = pipeline.query(request)

print(response.answer)
print(f"Encontrado nos documentos: {response.found_in_documents}")
print(f"Latência: {response.latency_ms:.0f} ms")

for chunk in response.retrieved_chunks:
    print(f"[{chunk.citation}] score={chunk.score:.3f}")
    print(chunk.text[:200])
```

---

## ⚙️ Configuração

Todas as variáveis são definidas em `backend/.env`. Valores abaixo refletem a configuração de referência:

```bash
# Diretórios
DATA_DIR=data/pdfs          # PDFs de entrada
INDEX_DIR=data/index        # Índice FAISS persistido
MODEL_DIR=models            # Modelos GGUF

# Chunking
CHUNK_SIZE=300              # Tokens máximos por chunk
CHUNK_OVERLAP=50            # Tokens de sobreposição

# Embeddings
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DIM=384
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=mps        # mps (Apple Silicon) | cuda | cpu

# Recuperação
RETRIEVAL_TOP_K=10          # Candidatos recuperados do FAISS
RETRIEVAL_FINAL_K=5         # Chunks injetados no prompt após MMR
SIMILARITY_THRESHOLD=0.45   # Score mínimo de similaridade (0–1)
MMR_LAMBDA=0.6              # 1.0 = só relevância, 0.0 = só diversidade

# LLM
LLM_MODEL_PATH=models/qwen2.5-7b-instruct-q4_k_m.gguf
LLM_CONTEXT_LENGTH=4096
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.1
LLM_N_GPU_LAYERS=-1         # -1 = todas as camadas na GPU (MPS/CUDA) | 0 = CPU
LLM_N_THREADS=12            # Threads de CPU (usado quando GPU parcial ou ausente)

# Integrações opcionais
USE_CHROMA=false
MONGO_URI=mongodb://ragadmin:ragpassword@localhost:27017/rag_knowledge?authSource=admin
```

> **Ajuste de threshold**: valores entre `0.40` e `0.55` são recomendados para documentos de domínio específico. Thresholds muito baixos aumentam recall mas introduzem ruído; muito altos podem retornar "Não encontrei" para perguntas legítimas.

---

## 🧪 Testes

```bash
# Todos os testes
PYTHONPATH=./backend python -m pytest backend/tests/ -v

# Com cobertura
PYTHONPATH=./backend python -m pytest backend/tests/ \
  --cov=app \
  --cov-report=html \
  --cov-report=term-missing

# Módulo específico
PYTHONPATH=./backend python -m pytest backend/tests/test_config.py -v
```

### Resultado esperado

```
✅ 109 testes passando
✅ Cobertura ≥ 90% nos módulos core:
   - app/core/config.py       100%
   - app/core/text_utils.py   100%
   - app/models/schemas.py    100%
   - app/core/embedder.py     100%
   - app/core/ingestion.py     90%+
```

---

## 📚 Documentação Técnica

| Documento | Conteúdo |
|-----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitetura em 3 camadas, diagramas Mermaid de componentes e sequências |
| [docs/COMPONENTS.md](docs/COMPONENTS.md) | Detalhes de PDFIngester, Embedder, VectorStore, LLM, RAGPipeline |
| [docs/DATA_FLOWS.md](docs/DATA_FLOWS.md) | Fluxos end-to-end de query e ingestão, timelines de performance |

---

## 🐛 Troubleshooting

### Porta já em uso (8000 / 8501)

```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

### Conda não encontrado

```bash
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"
conda activate rag
```

### Modelo LLM não encontrado

```bash
# Baixe o Qwen2.5-7B (~4.7 GB)
cd backend
python scripts/download_model.py --model qwen2.5-7b

# Verifique o caminho configurado em .env
grep LLM_MODEL_PATH .env
```

### Modelo de embeddings não encontrado

```bash
cd backend
python scripts/download_embeddings.py --model multilingual-e5-small
```

### Índice FAISS corrompido ou desatualizado

```bash
rm -rf backend/data/index
python3 start.py   # Reindexação automática
```

### Aviso de incompatibilidade de embedding no log

Se os logs mostrarem `INCOMPATIBILIDADE de embedding: índice construído com {...}`, o índice foi gerado com um modelo de embedding diferente do configurado atualmente. Scores sem sentido fazem a busca retornar resultados aleatórios ou vazios. Solução:

```bash
# Reconstrói o índice com o modelo atual
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'
```

> O arquivo `index_signature.json` é salvo junto ao índice e registra o modelo, dimensão e prefixos usados na geração dos vetores.

### MPS não ativado (macOS)

```bash
# Verifique suporte MPS
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Se True, configure em backend/.env:
# LLM_N_GPU_LAYERS=-1
# EMBEDDING_DEVICE=mps
```

### NumPy / FAISS falhando em macOS

```bash
# Use Conda em vez de pip/venv puro
bash scripts/setup_backend_mac.sh
```

### Respostas "Não encontrei essa informação" em excesso

Ajuste os parâmetros de recuperação em `backend/.env`:

```bash
SIMILARITY_THRESHOLD=0.40   # Reduza o threshold (mais recall)
RETRIEVAL_TOP_K=15          # Aumente os candidatos
RETRIEVAL_FINAL_K=7         # Mais chunks no contexto do LLM
```

---

## 📦 Dependências Principais

| Categoria | Pacote | Versão | Função |
|-----------|--------|--------|--------|
| Extração de texto | `pdfplumber` | 0.11+ | Extração de texto de PDFs |
| Extração de tabelas | `camelot-py[cv]` | 0.10+ | Detecção e extração de tabelas |
| Embeddings | `sentence-transformers` | 3.0+ | `multilingual-e5-small` (384-dim) |
| Índice vetorial | `faiss-cpu` | 1.8+ | IndexFlatIP — busca exata |
| LLM local | `llama-cpp-python` | 0.2.85+ | Qwen2.5-7B GGUF inference |
| Backend | `fastapi` | 0.115+ | API REST + SSE streaming |
| Frontend | `streamlit` | 1.40+ | Interface web interativa |
| Validação | `pydantic` | 2.7+ | Schemas de request/response |
| Configuração | `pydantic-settings` | 2.3+ | Leitura de variáveis de ambiente |
| Dados tabulares | `pandas` | 2.2+ | Formatação de tabelas extraídas |

---

## 📊 Status do Projeto

| Componente | Status |
|-----------|--------|
| Backend (FastAPI + RAG Pipeline) | ✅ Production-Ready |
| Frontend (Streamlit + SSE) | ✅ Funcional |
| Busca híbrida (FAISS + BM25 + RRF) | ✅ Ativada |
| Extração de tabelas (Camelot) | ✅ Com fallback automático |
| Aceleração MPS (Apple Silicon) | ✅ Ativada |
| Guarda de falso negativo (LLM) | ✅ Ativada |
| Validação de assinatura do índice | ✅ Automática no startup |
| Testes unitários | ✅ 109/109 passing |
| Documentação técnica | ✅ 20+ diagramas Mermaid |
| Integrações ChromaDB/MongoDB | ⚙️ Opcionais (desativadas por padrão) |

---

## ✉️ Contato

**Autores:** Janio Lima · Arthur Damiao · Luis Felipe Rudnik · Gabryel Zanella

---

**Licença:** MIT  
**Versão:** 1.0.0  
**Última atualização:** Junho 2026
