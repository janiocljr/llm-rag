# Setup — macOS (Miniforge / Conda)

## Pré-requisitos obrigatórios

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

> Homebrew é necessário. O Ghostscript e o Miniforge são instalados automaticamente pelo `start.py` se ausentes.

---

## Fluxo recomendado (primeira execução)

### Sem Conda instalado

O `start.py` detecta a ausência do Conda, executa `scripts/setup_backend_mac.sh` (instala Miniforge + Ghostscript + cria o env `rag`) e reinicia dentro do ambiente automaticamente.

```bash
cp backend/.env.example .env
cp /seus/documentos/*.pdf backend/data/pdfs/
python3 start.py
```

### Com Conda já instalado

Se o Conda está instalado mas o ambiente `rag` ainda não existe, crie-o antes:

```bash
conda env create -f environment.yml -n rag
conda activate rag
```

Depois:

```bash
cp backend/.env.example .env
cp /seus/documentos/*.pdf backend/data/pdfs/
python3 start.py
```

---

## O que o `start.py` faz automaticamente

| Etapa | Responsável |
|---|---|
| Instala Ghostscript | `setup_backend_mac.sh` |
| Instala Miniforge e cria env `rag` | `setup_backend_mac.sh` |
| Instala dependências Python | `start.py` (pip no env ativo) |
| Baixa modelo de embedding `e5-small` (~118 MB) | `scripts/ensure_models.py` |
| Baixa modelo LLM Qwen2.5-7B (~4.7 GB) | `scripts/download_model.py` |
| Cria pastas `data/pdfs`, `data/index`, `models` | `start.sh` |
| Sobe FastAPI + Streamlit | `start.sh` |
| Executa ingest automático dos PDFs | `start.sh` → `POST /api/v1/ingest` |

---

## Configuração do `.env`

O arquivo `.env` na raíz do projeto é a fonte única de configuração. Crie a partir do exemplo e ajuste conforme o hardware:

```bash
cp backend/.env.example .env
```

Parâmetros mais importantes para ajustar:

```bash
EMBEDDING_DEVICE=mps    # mps (Apple Silicon) | cuda | cpu
LLM_N_GPU_LAYERS=-1     # -1 = GPU total | 0 = CPU
LLM_N_THREADS=12        # ajuste conforme número de cores
LLM_CONTEXT_LENGTH=8192 # reduza para 4096 se houver erros de memória
```

---

## Troubleshooting

- **`conda` não encontrado após instalação**: execute `eval "$(/opt/homebrew/bin/conda shell.zsh hook)"` ou abra um novo terminal.
- **Env `rag` não encontrado**: execute `conda env create -f environment.yml -n rag` antes do `start.py`.
- **Erro de compilação de `numpy`/`pandas`**: confirme que o ambiente foi criado via `environment.yml` antes do `pip install`.
- **Backend não sobe em 300s**: verifique o log em `/tmp/rag_api_*.log` para o erro real.
- **Download do modelo falhou**: execute `cd backend && python scripts/download_model.py` manualmente e verifique a conexão.
