# Documentação do Sistema RAG

Bem-vindo à documentação técnica do **Sistema de Retrieval-Augmented Generation (RAG)** offline.

## 📑 Índice

### Arquitetura e Design
- [**Arquitetura do Sistema**](./ARCHITECTURE.md) - Visão geral de componentes e fluxos
- [**Componentes**](./COMPONENTS.md) - Detalhamento de cada módulo
- [**Fluxos de Dados**](./DATA_FLOWS.md) - Processos end-to-end do sistema

### Configuração e Deploy
- [**Configuração**](./CONFIGURATION.md) - Variáveis de ambiente e settings
- [**Deployment**](./DEPLOYMENT.md) - Guia de instalação e execução

### Desenvolvimento
- [**API Reference**](./API_REFERENCE.md) - Documentação dos endpoints
- [**Testing**](./TESTING.md) - Guia de testes e cobertura
- [**Contributing**](./CONTRIBUTING.md) - Guia de contribuição

---

## 🎯 Quick Start

### Arquitetura em Alta Nível

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Chat UI  →  APIClient  →  HTTP/SSE  →  FastAPI Backend  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    RAGPipeline                             │  │
│  │  ┌──────────┬─────────┬──────────┬──────┬────┬──────────┐ │  │
│  │  │ Ingester │ Embedder │VectorStore│ MMR │LLM│ Chroma   │ │  │
│  │  └──────────┴─────────┴──────────┴──────┴────┴──────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        ↓                                           ↓
    PDFs          FAISS  +  MongoDB  +  ChromaDB
```

---

## 📊 Componentes Principais

| Componente | Função | Tecnologia |
|-----------|--------|-----------|
| **Frontend** | Interface de usuário | Streamlit |
| **Backend** | API e orquestração | FastAPI + Uvicorn |
| **Ingestion** | Extração de PDFs | pdfplumber + Camelot |
| **Embeddings** | Vetorização de texto | sentence-transformers (bge-m3) |
| **Vector Store** | Indexação de vetores | FAISS (IndexFlatIP) |
| **LLM** | Geração de respostas | llama-cpp-python (Mistral 7B) |
| **Optional Store** | Persistência avançada | ChromaDB + MongoDB |

---

## 🔄 Fluxo de Dados Principal

### Query (Pergunta)
```
User Query 
  → Frontend (Streamlit)
  → Backend API (/query)
  → Embed Query (bge-m3)
  → FAISS Search
  → MMR Re-rank
  → Build Prompt
  → LLM Generate
  → Stream Response
  → Frontend Display
```

### Ingest (Ingestão)
```
PDF Files
  → Load & Extract (pdfplumber + Camelot)
  → Clean Text
  → Split Chunks (recursive token-aware)
  → Embed Chunks (bge-m3)
  → FAISS Index
  → Optional: ChromaDB + MongoDB
```

---

## 🚀 Iniciando

### Desenvolvimento Local
```bash
python3 start.py --no-ingest
```

### Com Testes
```bash
PYTHONPATH=./backend:$PYTHONPATH pytest backend/tests/ -v
```

### Com Logging Detalhado
```bash
DEBUG=1 python3 start.py --no-ingest
```

---

## 📚 Documentos Detalhados

Para informações específicas sobre cada aspecto do sistema, consulte:

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Diagramas Mermaid de componentes e fluxos
- **[COMPONENTS.md](./COMPONENTS.md)** - Detalhamento técnico de cada módulo
- **[CONFIGURATION.md](./CONFIGURATION.md)** - Todas as variáveis configuráveis
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Instruções de setup e deploy
- **[API_REFERENCE.md](./API_REFERENCE.md)** - Endpoints e payloads

---

## 🔗 Estrutura de Diretórios

```
llm-rag/
├── docs/                    # Esta documentação
│   ├── README.md           # Você está aqui
│   ├── ARCHITECTURE.md     # Diagramas e visão geral
│   ├── COMPONENTS.md       # Detalhes técnicos
│   ├── DATA_FLOWS.md       # Fluxos de dados
│   ├── CONFIGURATION.md    # Configuração
│   ├── DEPLOYMENT.md       # Deploy & setup
│   ├── API_REFERENCE.md    # Endpoints
│   └── TESTING.md          # Testes e cobertura
├── backend/
│   ├── app/
│   │   ├── core/          # Lógica principal
│   │   ├── api/           # Endpoints
│   │   ├── models/        # Schemas Pydantic
│   │   └── utils/         # Utilitários
│   ├── tests/             # Suite de testes
│   └── requirements.txt
├── frontend/
│   ├── rag_chat.py        # UI principal
│   └── components/        # Componentes reutilizáveis
└── README.md              # Documentação raiz
```

---

## 💡 Conceitos Chave

### RAG (Retrieval-Augmented Generation)
Combina recuperação de documentos relevantes com geração de texto para responder perguntas baseadas em conhecimento local.

### Token-Aware Chunking
Divide textos respeitando limites de tokens (não caracteres), preservando contexto semântico.

### MMR (Maximal Marginal Relevance)
Re-classifica chunks recuperados para balancear relevância com diversidade, evitando duplicatas.

### Vector Embedding
Representa textos como vetores numéricos em espaço semântico, permitindo busca por similaridade.

---

## 📞 Suporte

Para dúvidas técnicas ou contribuições, consulte:
- [Guia de Contribuição](./CONTRIBUTING.md)
- [Issues do Projeto](https://github.com/seu-repo/issues)
- README principal do projeto

---

**Última atualização**: Junho 2026  
**Versão**: 1.0.0  
**Status**: Production-Ready ✅
