# rag-chat-llm

Sistema de **Retrieval-Augmented Generation (RAG)** totalmente offline para consulta de documentos PDF via linguagem natural. O objetivo é permitir ingestão, indexação e consulta de PDFs localmente sem depender de serviços externos.

---

## Sumário

- Visão Geral
- Arquitetura
- Requisitos
- Instalação e Execução
- Configuração
- API Reference
- Pipeline em Detalhe
- Docker
- Testes
- Troubleshooting

---

## Visão Geral

O sistema processa PDFs, extrai texto e tabelas, fragmenta em chunks, gera embeddings locais, indexa em FAISS e responde perguntas usando um LLM local.

Principais objetivos:
- Respostas com citação de fonte (arquivo + página).
- Extração robusta de tabelas (PoC com Camelot + fallback pdfplumber).
- Execução offline com dependências locais e opções Docker/Conda/venv.

---

## Arquitetura do projeto

```
rag-chat-llm/
├── start.py                  # Launcher: venv (ou usa Conda), instala deps, chama backend/start.sh
├── backend/
│   ├── start.sh              # Inicia backend + frontend, gerencia processos
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── api/routes.py
│       └── core/
│           ├── ingestion.py  # pdfplumber + Camelot PoC
│           ├── embedder.py
│           ├── vector_store.py
│           ├── llm.py
│           └── pipeline.py
├── frontend/
│   ├── rag_chat.py
│   └── components/
└── scripts/
	└── setup_backend_mac.sh  # Miniforge/Conda helper for macOS
```

Componentes:
- Backend: FastAPI + uvicorn
- Frontend: Streamlit
- Embeddings: `BAAI/bge-m3` (sentence-transformers wrapper)
- Vector DB: FAISS `IndexFlatIP`
- LLM: Mistral 7B Instruct (GGUF) via `llama-cpp-python`

---

## Requisitos

### Hardware

| Recurso | Mínimo | Recomendado |
|---|---:|---:|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8+ cores |
| Disco | 6 GB livres | 10 GB |

### Software

- macOS / Linux
- Python 3.10 — 3.12 (evite 3.13+ por compatibilidade de wheels)
- Homebrew (macOS) para dependências de sistema
- Miniforge/Conda recomendado no macOS Apple Silicon

---

## Instalação e Execução

Escolha entre Conda (recomendado no macOS) ou fallback com venv.

- Arquivos úteis:
	- [environment.yml](environment.yml) — ambiente Conda recomendado (`rag`).
	- [README-SETUP.md](README-SETUP.md) — instruções detalhadas para macOS usando `environment.yml`.

### 1) Recomendado (macOS) — Miniforge / Conda

```bash
# instala Miniforge e cria o env 'rag' com Python 3.11 (script cuidará disso)
bash scripts/setup_backend_mac.sh

# ative o conda na sessão (zsh)
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"
conda activate rag

# inicie o projeto (usa pip do Conda ativo)
python3 start.py --no-ingest
```

Observação: se preferir não ativar, use `conda run`:

```bash
$HOME/miniforge3/bin/conda run -n rag --no-capture-output python3 start.py --no-ingest
```

### 2) Fallback: venv local

```bash
python3 start.py --no-ingest
```

O `start.py` criará `.venv` se nenhum Conda ativo for detectado e instalará dependências. Em macOS, sem Conda, `numpy` pode compilar de origem e falhar.

---

## Quick Start — passo a passo (máquina nova)

Siga estes comandos se você acabou de clonar o repositório e quer rodar localmente hoje. Cobertura para macOS (recomendado) e alternativa com venv ou Docker.

1) Verifique o repositório e entre nele:

```bash
git clone <REPO_URL> your-folder
cd your-folder
```

2) (macOS) Instale Homebrew se não tiver:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ghostscript
```

3) (macOS recomendado) Prepare Conda + env `rag` com o script:

```bash
bash scripts/setup_backend_mac.sh
# Abra uma nova sessão do shell (ou rode o hook abaixo)
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"   # zsh
conda activate rag
```

4) Baixe o modelo LLM (~4.1 GB) e coloque em `backend/models/`:

```bash
# exemplo (substitua pela fonte que você usa)
mkdir -p backend/models
# coloque aqui o arquivo mistral-7b-instruct-v0.2.Q4_K_M.gguf
ls backend/models/
```

5) Copie seus PDFs para `backend/data/pdfs/`:

```bash
mkdir -p backend/data/pdfs
cp /caminho/para/seus/*.pdf backend/data/pdfs/
ls backend/data/pdfs/
```

6) Inicie a aplicação (usa o pip do Conda ativo):

```bash
python3 start.py --no-ingest
```

7) Verifique as URLs:

```
Chat UI → http://localhost:8501
API Docs → http://localhost:8000/docs
```

Alternativa (sem Conda) — criar venv local:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 start.py --no-ingest
```

Alternativa (Docker):

```bash
cd backend
docker compose up --build
```

Problemas comuns e checagens rápidas:
- Porta ocupada: `lsof -ti:8000 | xargs -r kill -9`
- `conda` não disponível: rode o `eval` hook mostrado acima
- `numpy` falhando: garanta que está no env Conda `rag` e que `numpy` foi instalado via conda-forge


---

## Configuração (variáveis importantes)

Defina em `backend/.env` ou via environment variables.

- `DATA_DIR` (default `data/pdfs`) — PDFs de entrada
- `INDEX_DIR` (default `data/index`) — persistência do índice
- `LLM_MODEL_PATH` — caminho relativo em `backend/` para o GGUF do LLM
- `EMBEDDING_MODEL` — `BAAI/bge-m3` por padrão
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `RETRIEVAL_FINAL_K`, `SIMILARITY_THRESHOLD`

---

## Endpoints principais

- `POST /api/v1/ingest` — processa PDFs e (re)constrói o índice
- `POST /api/v1/query` — consulta: input `question`, output `answer` + `retrieved_chunks`
- `GET /api/v1/stats` — estatísticas do índice
- `GET /health` — health check

Abra `http://localhost:8000/docs` para a documentação completa após iniciar.

---

## Pipeline em detalhe

1) Extração: `pdfplumber` por página; PoC `camelot` tenta `lattice` → `stream` por página para tabelas.
2) Normalização e chunking: `RecursiveCharSplitter` com consciência de página; estimativa tokens ~ len/4.
3) Embeddings: batches para `BAAI/bge-m3` (1024-dim).
4) Indexação: FAISS `IndexFlatIP` + `chunks_metadata.json`.
5) Retrieval: top-k, filtra por `SIMILARITY_THRESHOLD`, re-ranqueia com MMR e injeta final_k no prompt.
6) Geração: `llama-cpp-python` carrega GGUF e gera resposta com sentinel em caso de ausência de informação.

---

## Camelot PoC

- Camelot (`camelot-py[cv]`) foi integrado como PoC para extração de tabelas.
- Comportamento: por página tenta `camelot.read_pdf` (lattice then stream). Se falhar, usa `pdfplumber`.
- Tabelas detectadas são convertidas para CSV/markdown e adicionadas como chunks com `is_table=true`.

Nota: Camelot requer Ghostscript e libs de imagem — o `Dockerfile` e `scripts/setup_backend_mac.sh` instalam essas dependências.

---

## Docker

No diretório `backend`:

```bash
docker compose up --build
```

Volumes:

- `./data/pdfs` → `/app/data/pdfs`
- `./data/index` → `/app/data/index`
- `./models` → `/app/models`

---

## Testes

```bash
cd backend
pytest -m unit
pytest -m integration  # requer modelo e PDFs
```

---

## Troubleshooting

- `numpy` compilando e falhando no macOS: use o Conda `rag` (`scripts/setup_backend_mac.sh`) e instale `numpy`/`pandas` via conda-forge antes de `pip install -r`.
- `conda` não encontrado após script: execute o hook do shell:

```bash
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"  # zsh
```

- Portas ocupadas (8000 / 8501):

```bash
lsof -ti:8000 | xargs -r kill -9
lsof -ti:8501 | xargs -r kill -9
```

- Se `start.py` ainda tentar criar `.venv` mesmo com Conda: verifique se o shell tem `CONDA_PREFIX` exportado (ativação feita com `conda activate rag`).

---

## Arquivos e scripts úteis

- `start.py` — launcher que detecta Conda ativo e, caso contrário, cria `.venv`.
- `scripts/setup_backend_mac.sh` — instala Miniforge e cria `rag` com Python 3.11 e dependências binárias.
- `backend/app/core/ingestion.py` — extração e chunking (onde Camelot PoC está implementado).

---

## Próximos passos sugeridos

- Salvar CSVs extraídos pelo Camelot durante a ingestão e adicionar o caminho nos metadados.
- Adicionar testes de extração de tabelas comparando Camelot vs pdfplumber.
- Salvar CSVs extraídos pelo Camelot durante a ingestão e adicionar o caminho nos metadados — implementado. CSVs são salvos em `backend/data/index/tables/` e os chunks de tabela têm `is_table=true` e `table_csv_path` nos metadados.
- Adicionar testes de extração de tabelas comparando Camelot vs pdfplumber — implementado: veja `backend/tests/test_table_extraction.py`.

---

