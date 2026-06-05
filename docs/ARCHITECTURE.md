# Arquitetura do Sistema RAG

Documentação detalhada da arquitetura do sistema de Retrieval-Augmented Generation offline.

## 📐 Visão Geral de Alto Nível

```mermaid
graph TB
    subgraph Client["🖥️ CLIENT LAYER"]
        Browser["Browser / Desktop"]
    end
    
    subgraph Frontend["🎨 FRONTEND LAYER - Streamlit"]
        UI["Chat UI<br/>rag_chat.py"]
        Components["Components<br/>chat, sidebar, etc"]
        APIClient["API Client<br/>api_client.py"]
        UI --> Components
        UI --> APIClient
    end
    
    subgraph Backend["⚙️ BACKEND LAYER - FastAPI"]
        Server["FastAPI Server<br/>uvicorn"]
        Routes["API Routes<br/>v1/query, /ingest"]
        Pipeline["RAG Pipeline<br/>Orchestration"]
        Server --> Routes
        Routes --> Pipeline
    end
    
    subgraph Core["🧠 CORE PROCESSING"]
        Ingester["PDF Ingester<br/>pdfplumber + Camelot"]
        Embedder["Embedder<br/>sentence-transformers"]
        VectorStore["Vector Store<br/>FAISS"]
        LLM["Local LLM<br/>llama-cpp-python"]
        Ingester --> Embedder
        Embedder --> VectorStore
        VectorStore --> LLM
    end
    
    subgraph Storage["💾 STORAGE LAYER"]
        FAISS["FAISS Index<br/>Vector embeddings"]
        PDF["PDF Files<br/>Source documents"]
        Models["Model Files<br/>GGUF format"]
        Optional["Optional<br/>ChromaDB + MongoDB"]
    end
    
    Browser --> UI
    APIClient -->|HTTP/SSE| Server
    Pipeline --> Ingester
    Pipeline --> VectorStore
    Pipeline --> LLM
    
    Ingester --> PDF
    LLM --> Models
    VectorStore --> FAISS
    VectorStore -.-> Optional
    
    style Client fill:#e1f5ff
    style Frontend fill:#f3e5f5
    style Backend fill:#fff3e0
    style Core fill:#e8f5e9
    style Storage fill:#fce4ec
```

---

## 🔄 Arquitetura de Componentes

```mermaid
graph LR
    subgraph Frontend["FRONTEND<br/>Streamlit"]
        FE_UI["UI Chat"]
        FE_CLIENT["API Client"]
        FE_UI --> FE_CLIENT
    end
    
    subgraph Backend["BACKEND<br/>FastAPI"]
        BE_ROUTE["Routes<br/>/query, /ingest"]
        BE_PIPELINE["Pipeline"]
        BE_ROUTE --> BE_PIPELINE
    end
    
    subgraph Processing["PROCESSING"]
        ING["Ingester"]
        EMB["Embedder"]
        VS["Vector Store"]
        MMR["MMR Ranker"]
        LLM["LLM"]
        
        ING --> EMB
        EMB --> VS
        VS --> MMR
        MMR --> LLM
    end
    
    subgraph Storage["STORAGE"]
        FAISS["FAISS<br/>Index"]
        CHROMA["ChromaDB<br/>Optional"]
        MONGO["MongoDB<br/>Optional"]
        FILES["Files<br/>Models, PDFs"]
    end
    
    FE_CLIENT -->|HTTP| BE_ROUTE
    BE_PIPELINE --> ING
    BE_PIPELINE --> VS
    BE_PIPELINE --> LLM
    
    VS --> FAISS
    VS --> CHROMA
    EMB --> MONGO
    LLM --> FILES
    
    classDef frontend fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef backend fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    classDef proc fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    classDef store fill:#FCE4EC,stroke:#C2185B,stroke-width:2px
    
    class Frontend frontend
    class Backend backend
    class Processing proc
    class Storage store
```

---

## 📊 Pipeline de Query (Pergunta)

```mermaid
sequenceDiagram
    actor User as User
    participant UI as Frontend<br/>Streamlit
    participant API as Backend<br/>FastAPI
    participant Pipeline as RAG<br/>Pipeline
    participant Embed as Embedder<br/>bge-m3
    participant VS as Vector Store<br/>FAISS
    participant LLM as Local LLM<br/>Mistral 7B

    User->>UI: Digita pergunta
    UI->>API: POST /api/v1/query
    API->>Pipeline: Processa query
    Pipeline->>Embed: Vetoriza pergunta
    Embed-->>Pipeline: Query vector (1024-dim)
    Pipeline->>VS: Busca similar (top-k)
    VS-->>Pipeline: Retrieved chunks + scores
    Pipeline->>Pipeline: MMR re-ranking
    Pipeline->>Pipeline: Build prompt<br/>sistema + context + query
    Pipeline->>LLM: Gera resposta
    LLM-->>Pipeline: Response tokens
    Pipeline-->>API: Stream SSE
    API-->>UI: Recebe tokens
    UI->>User: Exibe resposta

    Note over Pipeline,LLM: Tudo rodando localmente<br/>100% offline
```

---

## 📥 Pipeline de Ingestão (Ingesta)

```mermaid
flowchart TD
    Start["PDFs na pasta<br/>backend/data/pdfs"] --> Load["Carregar PDF<br/>pdfplumber"]
    Load --> Extract["Extrair texto por página"]
    Extract --> Tables["Extrair tabelas<br/>Camelot + fallback"]
    Tables --> Clean["Limpar texto<br/>remover headers/footers"]
    Clean --> Chunk["Dividir em chunks<br/>token-aware recursive"]
    Chunk --> Embed["Vetorizar chunks<br/>bge-m3 model"]
    Embed --> Index["Indexar no FAISS<br/>IndexFlatIP"]
    Index --> Optional["[Opcional]<br/>Persistir em ChromaDB<br/>e MongoDB"]
    Optional --> Done["✓ Índice pronto<br/>para queries"]
    
    style Start fill:#E3F2FD
    style Load fill:#BBDEFB
    style Extract fill:#90CAF9
    style Tables fill:#64B5F6
    style Clean fill:#42A5F5
    style Chunk fill:#2196F3
    style Embed fill:#1E88E5
    style Index fill:#1976D2
    style Optional fill:#1565C0
    style Done fill:#0D47A1,color:#fff
```

---

## 🏗️ Estrutura do Backend

```mermaid
graph TB
    subgraph AppLayer["APPLICATION LAYER"]
        MainApp["main.py<br/>FastAPI App"]
        Lifespan["Lifespan Context<br/>Load models"]
    end
    
    subgraph APILayer["API LAYER"]
        QueryRoute["POST /query<br/>Query endpoint"]
        IngestRoute["POST /ingest<br/>Ingest endpoint"]
        StatsRoute["GET /stats<br/>Index stats"]
        HealthRoute["GET /health<br/>Health check"]
        MemoryRoute["Memory Routes<br/>Chat history"]
    end
    
    subgraph CoreLayer["CORE LAYER"]
        Config["config.py<br/>Settings & defaults"]
        Pipeline["pipeline.py<br/>RAGPipeline class"]
        Ingestion["ingestion.py<br/>PDFIngester class"]
        Embedding["embedder.py<br/>Embedder class"]
        VectorStore["vector_store.py<br/>VectorStore class"]
        LLM["llm.py<br/>LocalLLM class"]
        Memory["memory.py<br/>MemoryOrchestrator"]
    end
    
    subgraph DataLayer["DATA LAYER"]
        Schemas["schemas.py<br/>Pydantic models"]
        TextUtils["text_utils.py<br/>Utilities"]
        Logging["logging.py<br/>Log setup"]
    end
    
    MainApp --> Lifespan
    Lifespan --> Pipeline
    
    QueryRoute --> Pipeline
    IngestRoute --> Pipeline
    StatsRoute --> Pipeline
    HealthRoute --> MainApp
    MemoryRoute --> Memory
    
    Pipeline --> Config
    Pipeline --> Ingestion
    Pipeline --> Embedding
    Pipeline --> VectorStore
    Pipeline --> LLM
    Pipeline --> Memory
    
    Ingestion --> Schemas
    Embedding --> Schemas
    VectorStore --> Schemas
    Memory --> Schemas
    
    TextUtils -.-> Ingestion
    Logging -.-> Pipeline
    
    style AppLayer fill:#E1F5FE
    style APILayer fill:#F3E5F5
    style CoreLayer fill:#E8F5E9
    style DataLayer fill:#FFF9C4
```

---

## 🎨 Arquitetura do Frontend

```mermaid
graph TB
    subgraph Main["MAIN APPLICATION"]
        RagChat["rag_chat.py<br/>Streamlit App"]
    end
    
    subgraph Components["UI COMPONENTS"]
        Chat["chat.py<br/>Chat interface"]
        Sidebar["sidebar.py<br/>Configuration panel"]
        Diagram["diagram.py<br/>Architecture viz"]
        Thinking["thinking.py<br/>LLM thinking display"]
        PromptViewer["prompt_viewer.py<br/>Prompt debug"]
    end
    
    subgraph Utils["UTILITIES"]
        APIClient["api_client.py<br/>Backend connector"]
        Formatting["formatting.py<br/>Text formatting"]
        DemoData["demo_data.py<br/>Sample data"]
    end
    
    RagChat --> Chat
    RagChat --> Sidebar
    RagChat --> Diagram
    RagChat --> Thinking
    RagChat --> PromptViewer
    
    Chat --> APIClient
    Sidebar --> APIClient
    Thinking --> APIClient
    PromptViewer --> APIClient
    
    Formatting -.-> Chat
    DemoData -.-> Sidebar
    
    style Main fill:#F3E5F5
    style Components fill:#E8F5E9
    style Utils fill:#FFF9C4
```

---

## 💾 Camada de Persistência

```mermaid
graph TB
    subgraph Primary["PRIMARY STORAGE"]
        FAISS_IDX["FAISS Index<br/>data/index/faiss.index"]
        METADATA["Metadata<br/>data/index/chunks_metadata.json"]
        TABLES["Tables<br/>data/index/tables/*.csv"]
    end
    
    subgraph Optional["OPTIONAL STORAGE"]
        CHROMA["ChromaDB<br/>Vector embeddings<br/>Chat memory"]
        MONGO["MongoDB<br/>PDF metadata<br/>Session data"]
    end
    
    subgraph Files["FILE SYSTEM"]
        PDFS["PDFs<br/>data/pdfs/*.pdf"]
        MODELS["Models<br/>models/*.gguf"]
        ENV["Environment<br/>.env"]
        CONFIG["Config<br/>.claude/settings.json"]
    end
    
    FAISS_IDX --> Primary
    METADATA --> Primary
    TABLES --> Primary
    
    CHROMA --> Optional
    MONGO --> Optional
    
    PDFS --> Files
    MODELS --> Files
    ENV --> Files
    CONFIG --> Files
    
    style Primary fill:#FCE4EC
    style Optional fill:#F3E5F5
    style Files fill:#FFF9C4
```

---

## 🔌 Fluxo de Comunicação

```mermaid
graph LR
    Browser["Browser<br/>Chrome/Safari/etc"]
    
    subgraph Streamlit["Streamlit Frontend<br/>localhost:8501"]
        ST["Streamlit App"]
    end
    
    subgraph Network["Network"]
        HTTP["HTTP REST<br/>Server-Sent Events"]
    end
    
    subgraph FastAPI["FastAPI Backend<br/>localhost:8000"]
        FA["FastAPI Server"]
        Proc["Processing"]
    end
    
    Browser -->|User Input| ST
    ST -->|HTTP Request| HTTP
    HTTP -->|Events| ST
    HTTP -->|Request| FA
    FA -->|Process| Proc
    Proc -->|Response| FA
    FA -->|SSE Stream| HTTP
    HTTP -->|Display| ST
    ST -->|Render| Browser
    
    style Streamlit fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    style Network fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    style FastAPI fill:#FFF3E0,stroke:#E65100,stroke-width:2px
```

---

## 🌐 Deployment Architecture

```mermaid
graph TB
    subgraph Development["DEVELOPMENT"]
        DevVenv["Virtual Env<br/>.venv"]
        DevRequire["requirements.txt<br/>pip install"]
    end
    
    subgraph Docker["DOCKER DEPLOYMENT"]
        DockerFile["Dockerfile<br/>Multi-stage build"]
        Compose["docker-compose.yml<br/>Services"]
        Services["Services<br/>Backend + Frontend<br/>+ Optional: MongoDB"]
    end
    
    subgraph Production["PRODUCTION"]
        Conda["Conda Env<br/>environment.yml"]
        Systemd["Systemd Service<br/>Auto-start"]
        Proxy["Reverse Proxy<br/>nginx/apache"]
        SSL["SSL/TLS<br/>certbot"]
    end
    
    Development -->|Build| DockerFile
    DockerFile --> Compose
    Compose --> Services
    
    Development -->|Setup| Conda
    Conda --> Systemd
    Systemd --> Proxy
    Proxy --> SSL
    
    style Development fill:#E3F2FD
    style Docker fill:#F3E5F5
    style Production fill:#FFF3E0
```

---

## 📈 Escalabilidade

### Limites Atuais
- **Documentos**: Testar até 1000 PDFs (depende de RAM)
- **Chunks**: ~100k chunks é limite prático
- **Throughput**: 1-2 queries/segundo (CPU-bound)

### Otimizações Possíveis
```mermaid
graph TB
    Current["System Atual<br/>Single-node CPU"]
    
    Optimize1["Otimização 1<br/>Usar GPU<br/>n_gpu_layers=-1"]
    Optimize2["Otimização 2<br/>Usar IVF Index<br/>Faster search"]
    Optimize3["Otimização 3<br/>Batch embedding<br/>Parallel processing"]
    Optimize4["Otimização 4<br/>Distributed FAISS<br/>Multiple nodes"]
    
    Current --> Optimize1
    Current --> Optimize2
    Current --> Optimize3
    Current --> Optimize4
    
    Optimize1 -->|10x faster| Output["Performance<br/>10-20x"]
    Optimize2 -->|5x faster| Output
    Optimize3 -->|4x faster| Output
    Optimize4 -->|Linear scaling| Output
    
    style Current fill:#FFEBEE
    style Output fill:#E8F5E9
```

---

## 🔐 Segurança & Privacy

```mermaid
graph TB
    Security["Security & Privacy"]
    
    Offline["✓ 100% Offline<br/>Zero external APIs<br/>All local processing"]
    NoNet["✓ No Network<br/>Optional: internal only<br/>No cloud storage"]
    OpenSource["✓ Open Source<br/>Transparent code<br/>Community audit"]
    DataOwn["✓ Data Ownership<br/>Full control<br/>No tracking"]
    
    Security --> Offline
    Security --> NoNet
    Security --> OpenSource
    Security --> DataOwn
    
    style Security fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style Offline fill:#A5D6A7
    style NoNet fill:#A5D6A7
    style OpenSource fill:#A5D6A7
    style DataOwn fill:#A5D6A7
```

---

## 📚 Diagrama de Dependências

```mermaid
graph TB
    FastAPI["FastAPI<br/>Web framework"]
    Pydantic["Pydantic<br/>Data validation"]
    Uvicorn["Uvicorn<br/>ASGI server"]
    
    PDFPlumber["pdfplumber<br/>PDF extraction"]
    Camelot["Camelot<br/>Table extraction"]
    
    SentenceTransformers["sentence-transformers<br/>Embeddings"]
    Torch["PyTorch<br/>Backend"]
    
    FAISS["FAISS<br/>Vector search"]
    NumPy["NumPy<br/>Numerical ops"]
    
    LlamaCPP["llama-cpp-python<br/>Local LLM"]
    
    Streamlit["Streamlit<br/>Frontend"]
    
    FastAPI --> Pydantic
    FastAPI --> Uvicorn
    
    PDFPlumber --> NumPy
    Camelot --> PDFPlumber
    
    SentenceTransformers --> Torch
    SentenceTransformers --> NumPy
    
    FAISS --> NumPy
    
    LlamaCPP --> NumPy
    
    Streamlit --> Requests
    Requests["Requests"]
    
    style FastAPI fill:#FFF3E0
    style Streamlit fill:#F3E5F5
    style SentenceTransformers fill:#E8F5E9
    style FAISS fill:#FCE4EC
    style LlamaCPP fill:#E1F5FE
```

---

## 📝 Notas Técnicas

### Performance
- **Embedding**: ~100 docs/segundo (depende do hardware)
- **Query**: 0.5-2 segundos (retrieval + LLM generation)
- **Memory**: ~6GB RAM (embedder + LLM + index)

### Configuração Padrão
```yaml
chunk_size: 512 tokens
chunk_overlap: 64 tokens
embedding_model: BAAI/bge-m3 (1024-dim)
embedding_device: auto (MPS > CUDA > CPU)
llm_model: Mistral 7B Instruct (4-bit)
faiss_index: IndexFlatIP (exact search)
retrieval_top_k: 5
retrieval_final_k: 3
similarity_threshold: 0.35
mmr_lambda: 0.6
```

### Limites
- Max PDF size: Limited by available RAM
- Max chunks: ~500k (practical limit with FAISS)
- Max context: LLM context window (4096 tokens default)

---

## 🔄 Próximos Passos

1. **GPU Support**: Offload embedding & LLM to GPU
2. **Distributed Index**: FAISS distributed search
3. **Advanced Retrieval**: Hybrid search (BM25 + vector)
4. **Fine-tuning**: Custom embeddings para domain
5. **Caching**: Semantic caching para queries frequentes

---

**Última atualização**: Junho 2026  
**Versão**: 1.0.0  
**Status**: Production-Ready ✅
