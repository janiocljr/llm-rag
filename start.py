#!/usr/bin/env python3
"""
start.py
========
Launcher para o RAG System (Linux / macOS).

1. Cria .venv na raiz do repositório (usa python3.12 ou python3).
2. Instala backend/requirements.txt e frontend/requirements.txt.
3. Delega o controle de processos ao backend/start.sh.

Uso:
    python3 start.py                   # inicia backend + frontend
    python3 start.py --port-api 8001   # porta customizada para a API
    python3 start.py --port-ui  8502   # porta customizada para o UI
    python3 start.py --no-ingest       # pula auto-ingest na inicialização
    python3 start.py --reload          # habilita hot-reload no uvicorn
    python3 start.py --help            # exibe esta mensagem
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import shutil
from pathlib import Path

IS_WIN = platform.system() == "Windows"

def _ansi(code: str) -> str:
    if IS_WIN and not os.environ.get("TERM"):
        return ""
    return f"\033[{code}m"

RESET  = _ansi("0")
BOLD   = _ansi("1")
RED    = _ansi("31")
GREEN  = _ansi("32")
YELLOW = _ansi("33")
BLUE   = _ansi("34")
CYAN   = _ansi("36")

SEP = CYAN + "─" * 70 + RESET

def log(msg: str)  -> None: print(f"{BOLD}{CYAN}[start]{RESET} {msg}", flush=True)
def ok(msg: str)   -> None: print(f"{BOLD}{GREEN}[  ok ]{RESET} {msg}", flush=True)
def warn(msg: str) -> None: print(f"{BOLD}{YELLOW}[ warn]{RESET} {msg}", flush=True)
def err(msg: str)  -> None: print(f"{BOLD}{RED}[error]{RESET} {msg}", file=sys.stderr, flush=True)
def sep()          -> None: print(SEP, flush=True)

ROOT        = Path(__file__).resolve().parent
VENV        = ROOT / ".venv"
BIN         = "Scripts" if IS_WIN else "bin"
VENV_PYTHON = VENV / BIN / ("python.exe" if IS_WIN else "python")
VENV_PIP    = VENV / BIN / ("pip.exe"    if IS_WIN else "pip")


def find_base_python() -> str:
    for candidate in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        try:
            r = subprocess.run([candidate, "--version"], capture_output=True, text=True)
            if r.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    err("Python 3.10+ não encontrado. Instale antes de executar este script.")
    sys.exit(1)


def setup_env() -> None:
    sep()
    print(f"{BOLD}{BLUE}  RAG System — Configuração do Ambiente{RESET}")
    sep()
    # If a Conda environment is active, prefer it and install using the
    # active Python (avoids building C extensions from source on macOS).
    conda_prefix = os.environ.get("CONDA_PREFIX")

    # If no Conda is active, try to install/configure Miniforge and
    # relançar este script dentro do env 'rag' usando `conda run`.
    # Use `--no-conda-setup` para pular este comportamento.
    if not conda_prefix and "--no-conda-setup" not in sys.argv:
        condacmd = shutil.which("conda")
        miniforge_conda = str(Path.home() / "miniforge3" / "bin" / "conda")
        if not condacmd and Path(miniforge_conda).exists():
            condacmd = miniforge_conda

        if not condacmd:
            warn("Conda não encontrado. Executando scripts/setup_backend_mac.sh para instalar Miniforge e criar env 'rag'...")
            setup_script = ROOT / "scripts" / "setup_backend_mac.sh"
            if setup_script.exists():
                try:
                    subprocess.run(["bash", str(setup_script)], check=True)
                except subprocess.CalledProcessError:
                    warn("Falha ao executar scripts/setup_backend_mac.sh. Verifique o script manualmente.")
            else:
                warn(f"Script não encontrado: {setup_script}.")

        # re-check for conda after running the setup script
        if not condacmd and Path(miniforge_conda).exists():
            condacmd = miniforge_conda
        if not condacmd:
            condacmd = shutil.which("conda")

        if condacmd:
            ok(f"Relançando este script dentro do env 'rag' usando {condacmd} (conda run -n rag python3)...")
            os.execvp(condacmd, [condacmd, "run", "-n", "rag", "--no-capture-output", "python3"] + sys.argv)
        else:
            warn("Conda não disponível após tentativa de instalação; continuando com fallback venv.")
    if conda_prefix:
        ok(f"Conda environment detected at {conda_prefix}; using active Python: {sys.executable}")
        for label, req in [
            ("backend",  ROOT / "backend"  / "requirements.txt"),
            ("frontend", ROOT / "frontend" / "requirements.txt"),
        ]:
            if not req.exists():
                warn(f"requirements.txt não encontrado: {req} — ignorando")
                continue
            log(f"Instalando dependências do {label} no ambiente Conda ativo…")
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(req)], check=True)
            ok(f"Dependências do {label} instaladas no Conda.")
        return

    # ── Criar venv ─────────────────────────────────────────────────────────
    if VENV_PYTHON.exists():
        ok(f"Venv já existe: {VENV}")
    else:
        base_py = find_base_python()
        log(f"Criando .venv com {base_py}…")
        subprocess.run([base_py, "-m", "venv", str(VENV)], check=True)
        ok(f"Venv criado em {VENV}")

    # ── Instalar requirements ──────────────────────────────────────────────
    for label, req in [
        ("backend",  ROOT / "backend"  / "requirements.txt"),
        ("frontend", ROOT / "frontend" / "requirements.txt"),
    ]:
        if not req.exists():
            warn(f"requirements.txt não encontrado: {req} — ignorando")
            continue
        log(f"Instalando dependências do {label}…")
        subprocess.run(
            [str(VENV_PIP), "install", "--quiet", "-r", str(req)],
            check=True,
        )
        ok(f"Dependências do {label} instaladas.")


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    if IS_WIN:
        err("Windows não é suportado pelo start.sh. Execute backend e frontend manualmente.")
        sys.exit(1)

    setup_env()

    script = ROOT / "backend" / "start.sh"
    if not script.exists():
        err(f"Script não encontrado: {script}")
        sys.exit(1)

    # Repassa todos os argumentos originais para o start.sh
    extra_args = sys.argv[1:]

    sep()
    log(f"Delegando para backend/start.sh …")

    # Substitui o processo Python pelo bash — sem overhead extra
    os.execvp("bash", ["bash", str(script)] + extra_args)


if __name__ == "__main__":
    main()