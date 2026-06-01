# Persistent Memory System — Arquitetura e Setup

## Visão Geral

```
┌─────────────────────────────────────────────────────────────────────┐
│                          rag-chat-llm v3                            │
│                    Persistent AI Memory Stack                        │
└─────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐   embeddings    ┌──────────────────────────────┐
  │             │ ─────────────▶  │         ChromaDB             │
  │  FastAPI    │                 │   collection: pdf_embeddings  │
  │  RAG API    │ ◀─────────────  │   collection: chat_memory    │
  │  :8000      │  semantic search│   Docker volume: chroma_data  │
  │             │                 └──────────────────────────────┘
  │             │
  │             │   documents     ┌──────────────────────────────┐
  │             │ ─────────────▶  │           MongoDB            │
  │             │                 │   db: rag_knowledge           │
  │             │ ◀─────────────  │   col: documents (markdown)  │
  └─────────────┘  full-text /    │   col: sessions (turns)      │
         │         structured     │   col: tasks                 │
         │                        │   Docker volume: mongo_data   │
  ┌──────▼──────┐                 └──────────────────────────────┘
  │  Streamlit  │
  │  Frontend   │
  │  :8501      │
  └─────────────┘
```

---

## Separação de Responsabilidades

| Aspecto | ChromaDB | MongoDB |
|---|---|---|
| **O que armazena** | Vetores float32 (embeddings) | Documentos markdown estruturados |
| **Como acessa** | Similaridade semântica (cosseno) | ID, full-text, tags, tipo, sessão |
| **Coleções** | `pdf_embeddings`, `chat_memory` | `documents`, `sessions`, `tasks` |
| **Persistência** | Volume Docker `rag_chroma_data` | Volume Docker `rag_mongo_data` |
| **Porta** | 8200 (host) / 8000 (interno) | 27017 |
| **Analogia** | Motor de busca semântica | Obsidian / Notion |

---

## Estrutura de Arquivos

```
rag-chat-llm/
├── docker-compose.yml              ← serviços: chromadb, mongodb, rag-api
├── backend/
│   ├── .env.example                ← variáveis de ambiente (copiar para .env)
│   ├── Dockerfile
│   ├── requirements.txt            ← inclui chromadb, pymongo
│   ├── scripts/
│   │   └── mongo-init.js           ← cria collections, indexes, usuário app
│   └── app/
│       ├── main.py                 ← lifespan: conecta ambas as bases
│       ├── api/
│       │   ├── routes.py           ← /ingest, /query, /stats (inalterado)
│       │   └── memory_routes.py    ← /memory/* (sessions, notes, tasks, recall)
│       └── core/
│           ├── config.py           ← settings ChromaDB + MongoDB
│           ├── chroma_store.py     ← ChromaPDFStore + ChromaMemoryStore
│           ├── mongo_store.py      ← MongoDocumentStore + Session + Task
│           ├── memory.py           ← MemoryOrchestrator (orquestra ambos)
│           └── pipeline.py         ← RAGPipeline integrado à memória
└── frontend/
    └── ...                         ← sem alterações necessárias
```

---

## Schemas de Dados

### MongoDB — `documents`

```json
{
  "_id":         "ObjectId",
  "title":       "Cenários de crescimento do PIB Paraná",
  "content":     "## Pergunta\n\nQuais são...\n\n## Resposta\n\n...",
  "doc_type":    "conversation",
  "tags":        ["pib", "paraná", "economia"],
  "status":      "active",
  "source_file": "desenvolvimento_paranaense.pdf",
  "page_number": 2,
  "session_id":  "uuid-da-sessão",
  "chroma_ids":  ["chroma-id-da-pergunta", "chroma-id-da-resposta"],
  "created_at":  "2024-01-15T10:32:00Z",
  "updated_at":  "2024-01-15T10:32:00Z",
  "metadata":    {}
}
```

### MongoDB — `sessions`

```json
{
  "_id":        "ObjectId",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "title":      "Análise exportações 2024",
  "summary":    "Discussão sobre cenários de PIB e balança comercial",
  "turns": [
    {"role": "user",      "content": "Qual o percentual...", "timestamp": "...", "chroma_id": "..."},
    {"role": "assistant", "content": "Segundo [doc, p.2]...", "timestamp": "...", "chroma_id": "..."}
  ],
  "tags":       ["economia", "exportações"],
  "started_at": "2024-01-15T10:30:00Z",
  "ended_at":   "2024-01-15T11:45:00Z",
  "chroma_ids": ["..."],
  "doc_ids":    ["..."]
}
```

### ChromaDB — `pdf_embeddings`

```
id:        "desenvolvimento_paranaense_p2_t0"
embedding: [0.0234, -0.0891, ..., 0.1203]  ← 1024-dim
document:  "Cenário Tendencial: Poupança 17%..."
metadata: {
  source_file: "desenvolvimento_paranaense.pdf",
  page_number: 2,
  chunk_type:  "text",
  chunk_index: 0,
  token_estimate: 145,
  citation:    "[desenvolvimento_paranaense.pdf, p. 2]"
}
```

### ChromaDB — `chat_memory`

```
id:        "uuid-gerado"
embedding: [0.0123, ..., -0.0456]  ← 1024-dim
document:  "Quais são os cenários de crescimento do PIB do Paraná?"
metadata: {
  memory_type: "question",
  session_id:  "550e8400-...",
  mongo_id:    "65a3f1b2c4e5d6f7e8g9h0i1",
  tags:        "pib,paraná",
  timestamp:   "2024-01-15T10:32:00+00:00"
}
```

---

## Endpoints da API

### RAG (existentes, inalterados)

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/api/v1/ingest` | Ingere PDFs → ChromaDB + MongoDB |
| `POST` | `/api/v1/query` | Query RAG com memória de sessão |
| `GET`  | `/api/v1/stats` | Stats do vector store |
| `GET`  | `/health` | Health check |

### Memory (novos)

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/api/v1/memory/sessions` | Criar nova sessão |
| `GET`  | `/api/v1/memory/sessions` | Listar sessões recentes |
| `GET`  | `/api/v1/memory/sessions/{id}` | Obter sessão + histórico de turns |
| `POST` | `/api/v1/memory/sessions/{id}/close` | Fechar sessão com resumo |
| `POST` | `/api/v1/memory/notes` | Salvar nota markdown |
| `POST` | `/api/v1/memory/knowledge` | Salvar fragmento de conhecimento |
| `POST` | `/api/v1/memory/tasks` | Salvar tarefa estruturada |
| `GET`  | `/api/v1/memory/documents` | Listar / buscar documentos |
| `GET`  | `/api/v1/memory/documents/{id}` | Obter documento por ID |
| `POST` | `/api/v1/memory/recall` | Reconstruir contexto de sessões anteriores |
| `GET`  | `/api/v1/memory/stats` | Stats ChromaDB + MongoDB |

---

## Fluxo de Query com Memória

```
Nova pergunta do usuário
         │
         ▼
  embed_query(question)  ──▶  vetor 1024-dim
         │
         ├─▶  ChromaMemoryStore.recall()
         │    Busca em chat_memory as N memórias mais similares
         │    de sessões ANTERIORES  ──▶  contexto histórico
         │
         ├─▶  ChromaPDFStore.search()
         │    Busca em pdf_embeddings os chunks mais relevantes
         │    (com roteamento de tabelas para queries estatísticas)
         │
         ▼
  build_prompt(
      system_prompt,
      <memória_de_sessões_anteriores>...,   ← contexto histórico
      <contexto_documentos>...,             ← chunks PDF
      question
  )
         │
         ▼
     LLM.generate()
         │
         ▼
     save_turn(question, answer)
         ├─▶  ChromaDB chat_memory: embed Q + A → salvar
         ├─▶  MongoDB sessions.turns: append turn
         └─▶  MongoDB documents: criar doc "conversation"
```

---

## Setup — Subir o Stack

### 1. Pré-requisitos

```bash
docker --version    # Docker 24+
docker compose version  # Compose v2
```

### 2. Configurar variáveis

```bash
cp backend/.env.example backend/.env
# Editar backend/.env se necessário (credenciais, portas, etc.)
```

### 3. Colocar PDFs

```bash
# Os PDFs são montados como volume somente-leitura
mkdir -p backend/data/pdfs
cp /seus/documentos/*.pdf backend/data/pdfs/
```

### 4. Baixar o modelo LLM

```bash
python3 backend/scripts/download_model.py
# ou manual:
# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# → backend/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 5. Subir o stack completo

```bash
docker compose up --build
```

Sequência de inicialização:
1. `mongodb` inicia e executa `mongo-init.js` (cria collections e indexes)
2. `chromadb` inicia e aguarda health check
3. `rag-api` aguarda ambos (`depends_on: condition: service_healthy`)
4. `rag-api` conecta a ChromaDB e MongoDB, auto-ingesta PDFs

### 6. Verificar

```bash
# API
curl http://localhost:8000/health
# → {"status":"ok","version":"3.0.0"}

# ChromaDB
curl http://localhost:8200/api/v1/heartbeat

# Stats de memória
curl http://localhost:8000/api/v1/memory/stats
```

---

## Uso com sessões

### Criar sessão e fazer queries com contexto persistente

```bash
# 1. Criar sessão
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/memory/sessions \
  -H "Content-Type: application/json" \
  -d '{"title":"Análise PIB 2024","tags":["economia"]}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

echo "Session: $SESSION"

# 2. Query com session_id — respostas são salvas automaticamente
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Quais são os cenários de crescimento do PIB?\",\"session_id\":\"$SESSION\"}"

# 3. Segunda query — contexto da primeira é injetado automaticamente
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"E qual o papel da poupança nacional nesses cenários?\",\"session_id\":\"$SESSION\"}"

# 4. Ver histórico da sessão
curl http://localhost:8000/api/v1/memory/sessions/$SESSION

# 5. Fechar sessão com resumo
curl -X POST http://localhost:8000/api/v1/memory/sessions/$SESSION/close \
  -H "Content-Type: application/json" \
  -d '{"summary":"Análise dos cenários de PIB com foco em poupança e produtividade"}'
```

### Salvar notas e tarefas

```bash
# Nota markdown
curl -X POST http://localhost:8000/api/v1/memory/notes \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Insight sobre balança comercial",
    "content": "## Observação\n\nA balança comercial do Paraná...",
    "session_id": "'"$SESSION"'",
    "tags": ["balança-comercial","insight"]
  }'

# Tarefa
curl -X POST http://localhost:8000/api/v1/memory/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Comparar dados de 2023 vs 2024",
    "description": "Verificar evolução dos indicadores nas seções 3 e 4",
    "priority": "high",
    "tags": ["análise","pendente"]
  }'
```

### Recall de contexto (em nova sessão)

```bash
# Em uma nova sessão, buscar memórias relevantes de sessões anteriores
curl -X POST http://localhost:8000/api/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "question": "O que discutimos sobre o PIB do Paraná?",
    "session_id": "nova-session-uuid",
    "top_k": 5,
    "threshold": 0.45
  }'
```

---

## Volumes Docker

| Volume | Serviço | Dados |
|---|---|---|
| `rag_chroma_data` | chromadb | Embeddings HNSW, coleções |
| `rag_mongo_data` | mongodb | Documents, sessions, tasks |
| `rag_faiss_index` | rag-api | Índice FAISS legado (fallback) |
| `rag_pdf_documents` | rag-api | PDFs de origem (read-only) |
| `rag_llm_models` | rag-api | Pesos do modelo GGUF |

### Backup dos volumes

```bash
# ChromaDB
docker run --rm \
  -v rag_chroma_data:/source \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/chroma_$(date +%Y%m%d).tar.gz -C /source .

# MongoDB
docker exec rag_mongodb mongodump \
  --uri "mongodb://ragadmin:ragpassword@localhost:27017/rag_knowledge?authSource=admin" \
  --out /tmp/mongodump

docker cp rag_mongodb:/tmp/mongodump ./backups/mongo_$(date +%Y%m%d)
```

---

## Troubleshooting

### ChromaDB não conecta

```bash
# Verificar se o container está saudável
docker compose ps
docker logs rag_chromadb

# Testar heartbeat
curl http://localhost:8200/api/v1/heartbeat
```

### MongoDB não inicializa

```bash
docker logs rag_mongodb
# Se "mongo-init.js already ran", o volume existe — o init só roda uma vez

# Para re-inicializar do zero (APAGA TODOS OS DADOS):
docker compose down -v
docker compose up --build
```

### Resetar apenas o ChromaDB (re-ingest)

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'
```

### Inspeção direta do MongoDB

```bash
docker exec -it rag_mongodb mongosh \
  "mongodb://ragadmin:ragpassword@localhost:27017/rag_knowledge?authSource=admin"

# Dentro do mongosh:
db.documents.countDocuments()
db.sessions.find().limit(5).pretty()
db.documents.find({doc_type:"conversation"}).sort({created_at:-1}).limit(3).pretty()
```

### Inspeção direta do ChromaDB

```bash
# Via API REST
curl http://localhost:8200/api/v1/collections

# Contar vetores em cada coleção
curl http://localhost:8200/api/v1/collections/pdf_embeddings
curl http://localhost:8200/api/v1/collections/chat_memory
```
