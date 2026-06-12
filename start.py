

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
    conda_prefix = os.environ.get("CONDA_PREFIX")

    if not conda_prefix and "--no-conda-setup" not in sys.argv:
        condacmd = shutil.which("conda")
        if IS_WIN:
            miniforge_conda = str(Path.home() / "miniforge3" / "Scripts" / "conda.exe")
        else:
            miniforge_conda = str(Path.home() / "miniforge3" / "bin" / "conda")
        if not condacmd and Path(miniforge_conda).exists():
            condacmd = miniforge_conda

        if not condacmd:
            if IS_WIN:
                warn("Conda não encontrado. Instale o Miniforge e crie o ambiente:")
                warn("  conda create -n rag python=3.11 && conda activate rag")
                warn("Continuando com venv como alternativa…")
            else:
                warn("Conda não encontrado. Executando scripts/setup_backend_mac.sh para instalar Miniforge e criar env 'rag'...")
                setup_script = ROOT / "scripts" / "setup_backend_mac.sh"
                if setup_script.exists():
                    try:
                        subprocess.run(["bash", str(setup_script)], check=True)
                    except subprocess.CalledProcessError:
                        warn("Falha ao executar scripts/setup_backend_mac.sh. Verifique o script manualmente.")
                else:
                    warn(f"Script não encontrado: {setup_script}.")

        if not condacmd and Path(miniforge_conda).exists():
            condacmd = miniforge_conda
        if not condacmd:
            condacmd = shutil.which("conda")

        if condacmd:
            if IS_WIN:
                ok(f"Relançando este script dentro do env 'rag' usando {condacmd}…")
                result = subprocess.run([condacmd, "run", "-n", "rag", "--no-capture-output", "python"] + sys.argv)
                sys.exit(result.returncode)
            else:
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

    if VENV_PYTHON.exists():
        ok(f"Venv já existe: {VENV}")
    else:
        base_py = find_base_python()
        log(f"Criando .venv com {base_py}…")
        subprocess.run([base_py, "-m", "venv", str(VENV)], check=True)
        ok(f"Venv criado em {VENV}")

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


def _to_ps_args(args: list[str]) -> list[str]:
    valued = {"--port-api": "-PortApi", "--port-ui": "-PortUi"}
    flags  = {"--no-ingest": "-NoIngest", "--reload": "-Reload"}
    ps: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in valued and i + 1 < len(args):
            ps += [valued[a], args[i + 1]]; i += 2
        elif a in flags:
            ps.append(flags[a]); i += 1
        else:
            i += 1
    return ps


def _launch_windows(extra_args: list[str]) -> None:
    script = ROOT / "start.ps1"
    if not script.exists():
        err(f"Script não encontrado: {script}")
        sys.exit(1)
    sep()
    log("Delegando para start.ps1 …")
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)] + _to_ps_args(extra_args)
    )
    sys.exit(result.returncode)


def _launch_unix(extra_args: list[str]) -> None:
    script = ROOT / "backend" / "start.sh"
    if not script.exists():
        err(f"Script não encontrado: {script}")
        sys.exit(1)
    sep()
    log("Delegando para backend/start.sh …")
    os.execvp("bash", ["bash", str(script)] + extra_args)


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    setup_env()

    extra_args = [a for a in sys.argv[1:] if a != "--no-conda-setup"]

    if IS_WIN:
        _launch_windows(extra_args)
    else:
        _launch_unix(extra_args)


if __name__ == "__main__":
    main()
