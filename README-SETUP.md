# macOS Detailed Setup (Miniforge / Conda)

Este documento descreve passos passo-a-passo para preparar uma máquina macOS (incluindo Apple Silicon) para rodar o projeto localmente, minimizando problemas de compilação de pacotes nativos como `numpy`.

1) Pré-requisitos

- Instale as ferramentas de linha de comando do Xcode (requerido por algumas libs nativas):

```bash
xcode-select --install
```

- Instale Homebrew (se já tiver, ignore):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2) Instalar Ghostscript (necessário para Camelot)

```bash
brew install ghostscript
```

3) Criar/atualizar o ambiente Conda `rag` usando `environment.yml`

```bash
# Se já tem Miniforge/Conda instalado:
conda env create -f environment.yml -n rag  || conda env update -f environment.yml -n rag

# Ative o ambiente (necessário para que `start.py` use o Python do Conda)
eval "$(/opt/homebrew/bin/conda shell.zsh hook)"   # ou use o caminho do seu Miniforge: $HOME/miniforge3/bin/conda
conda activate rag
```

4) Instalar dependências Python do projeto

```bash
# Usa o pip do ambiente conda (recomendado)
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r backend/requirements.txt
```

5) Colocar o modelo LLM e PDFs

```bash
mkdir -p backend/models
# copie o GGUF do Mistral (ex: mistral-7b-instruct-v0.2.Q4_K_M.gguf) para backend/models/
mkdir -p backend/data/pdfs
# copie seus PDFs para backend/data/pdfs/
```

6) Iniciar a aplicação

```bash
python3 start.py --no-ingest
```

Notas e troubleshooting
- Se `conda` não for encontrado após instalar Miniforge, verifique o hook do shell (passo `eval` acima) ou abra um novo terminal.
- Caso prefira que o `start.py` execute a instalação do Miniforge automaticamente, ele já contém essa lógica; veja `--no-conda-setup` para pular.
- Se `numpy` ou `pandas` tentarem compilar, garanta que o ambiente `rag` foi criado via `environment.yml` e que `numpy`/`pandas` foram instalados pelo `conda` antes do `pip install -r`.
