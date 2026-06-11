# Setup — macOS (Miniforge / Conda)

## 1. Pré-requisitos

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ghostscript
```

## 2. Ambiente Conda

```bash
conda env create -f environment.yml -n rag || conda env update -f environment.yml -n rag
conda activate rag
```

## 3. Dependências Python

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r backend/requirements.txt
```

## 4. Modelos e dados

Copie o arquivo `.gguf` do Qwen2.5-7B para `backend/models/` e os PDFs para `backend/data/pdfs/`.

## 5. Iniciar

```bash
python3 start.py --no-ingest
```

---

**Troubleshooting**

- `conda` não encontrado: execute `eval "$(/opt/homebrew/bin/conda shell.zsh hook)"` ou abra um novo terminal.
- Erro de compilação de `numpy`/`pandas`: confirme que o ambiente foi criado via `environment.yml` antes do `pip install`.
