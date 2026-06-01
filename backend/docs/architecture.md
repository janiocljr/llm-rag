# Architecture Decision Records

## ADR-001 — Escolha do modelo de embedding

**Status:** Aceito  
**Data:** 2025-01-01

### Contexto
O sistema precisa converter chunks de texto em vetores densos para busca por similaridade. A qualidade semântica dos embeddings impacta diretamente a precisão do RAG.

### Decisão
Utilizar `BAAI/bge-m3` como modelo padrão de embedding via `sentence-transformers`.

### Alternativas consideradas
- `all-MiniLM-L6-v2` — menor precisão semântica
- `text-embedding-ada-002` (OpenAI) — requer API paga, viola requisito offline
- **TF-IDF + LSA (fallback)** — funciona offline sem dependências de rede; precisão semântica inferior

### Consequências
- Download automático na primeira execução (~90 MB)
- Em ambientes sem acesso ao HuggingFace Hub: fallback automático para TF-IDF+LSA

---

## ADR-002 — Escolha do índice vetorial (FAISS FlatIP)

**Status:** Aceito

### Contexto
O índice vetorial precisa suportar busca por similaridade coseno de forma eficiente.

### Decisão
Utilizar `faiss.IndexFlatIP` (Inner Product = coseno com vetores normalizados).

### Justificativa
- Busca exata (sem aproximação) — correta para volumes < 100k chunks
- Zero configuração de parâmetros de índice
- Persistência simples via `faiss.write_index` / `faiss.read_index`

### Alternativas consideradas
- `IndexIVFFlat` — mais rápido em escala, mas requer treinamento e tuning de `nlist`
- `IndexHNSWFlat` — busca aproximada, adequada para > 1M vetores

---

## ADR-003 — LLM local via llama-cpp-python

**Status:** Aceito

### Contexto
Requisito de operação totalmente offline, sem chamadas a APIs externas.

### Decisão
Utilizar `llama-cpp-python` para inferência de modelos GGUF quantizados localmente.

### Justificativa
- Suporta CPU e GPU (CUDA/Metal)
- Quantização Q4_K_M oferece boa relação qualidade/tamanho (~4 GB para 7B params)
- API compatível com OpenAI — fácil troca futura

---

## ADR-004 — MMR para re-ranking dos chunks

**Status:** Aceito

### Contexto
Recuperar apenas os top-K chunks por similaridade pode resultar em chunks redundantes, desperdiçando tokens do contexto do LLM.

### Decisão
Implementar Maximal Marginal Relevance (MMR) como etapa de re-ranking pós-recuperação.

### Parâmetros
- `lambda = 0.6` — favorece relevância sobre diversidade
- `retrieval_top_k = 5` — candidatos iniciais
- `retrieval_final_k = 3` — chunks injetados no prompt

---

## ADR-005 — Chunking por contagem de palavras

**Status:** Aceito

### Contexto
PDFs acadêmicos e relatórios possuem estrutura variável. Chunks muito grandes desperdiçam contexto; muito pequenos perdem coesão semântica.

### Decisão
Chunking por contagem de palavras com sobreposição: `chunk_size=512`, `chunk_overlap=64`.

### Justificativa
- Simples e determinístico
- Evita quebra de sentenças no meio por tokens (vs. tokenização BPE)
- Sobreposição garante continuidade semântica entre chunks consecutivos
