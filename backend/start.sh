#!/usr/bin/env bash
set -euo pipefail

RESET="\033[0m"
BOLD="\033[1m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
BLUE="\033[0;34m"
MAGENTA="\033[0;35m"

log()   { echo -e "${BOLD}${CYAN}[start]${RESET} $*"; }
ok()    { echo -e "${BOLD}${GREEN}[  ok ]${RESET} $*"; }
warn()  { echo -e "${BOLD}${YELLOW}[ warn]${RESET} $*"; }
err()   { echo -e "${BOLD}${RED}[error]${RESET} $*" >&2; }
sep()   { echo -e "${CYAN}$(printf '─%.0s' {1..70})${RESET}"; }

PORT_API=8000
PORT_UI=8501
AUTO_INGEST=true
HOT_RELOAD=false
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-api)   PORT_API="$2";    shift 2 ;;
    --port-ui)    PORT_UI="$2";     shift 2 ;;
    --no-ingest)  AUTO_INGEST=false; shift ;;
    --reload)     HOT_RELOAD=true;  shift ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) err "Argumento desconhecido: $1"; exit 1 ;;
  esac
done

sep
echo -e "${BOLD}${BLUE}  RAG System — Launcher${RESET}"
echo -e "  Backend  → http://localhost:${PORT_API}"
echo -e "  Frontend → http://localhost:${PORT_UI}"
echo -e "  Docs     → http://localhost:${PORT_API}/docs"
sep

cd "$ROOT_DIR"
log "Diretório de trabalho: $ROOT_DIR"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON="$(command -v python3 2>/dev/null || echo "${CONDA_PREFIX}/bin/python3")"
  ok "Usando Conda env: $PYTHON (CONDA_PREFIX=$CONDA_PREFIX)"
elif [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
  ok "Usando venv: $PYTHON"
else
  err "Nenhum ambiente Python encontrado (.venv ou Conda)."
  err "Execute primeiro: python3 start.py"
  exit 1
fi

mkdir -p backend/data/pdfs backend/data/index backend/models
if [[ -z "$(ls -A backend/data/pdfs 2>/dev/null)" ]]; then
  warn "backend/data/pdfs/ está vazio — adicione arquivos PDF antes de ingerir."
fi

log "Verificando e pre-cachendo modelos de embedding…"
export HF_HUB_DISABLE_TELEMETRY=1
(cd "$ROOT_DIR/backend" && "$PYTHON" scripts/ensure_models.py)
EMBED_OK=$?
if [[ $EMBED_OK -eq 0 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ok "Modelos em cache — backend iniciará em modo offline"
else
  warn "Modelos não verificados — aplicando SSL bypass para o backend"
  export REQUESTS_CA_BUNDLE=""
  export CURL_CA_BUNDLE=""
fi

MODEL_PATH="${LLM_MODEL_PATH:-backend/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf}"
if [[ ! -f "$MODEL_PATH" ]]; then
  warn "Modelo LLM não encontrado em: $MODEL_PATH"
  warn "O backend iniciará em STUB MODE (sem inferência real)."
  warn "Download: $PYTHON backend/scripts/download_model.py"
fi

if [[ -f "backend/.env" ]]; then
  ok "backend/.env encontrado — será lido pelo Pydantic-Settings no startup."
  _env_model=$(grep -E '^[[:space:]]*LLM_MODEL_PATH[[:space:]]*=' backend/.env \
    | tail -1 | cut -d'=' -f2- | tr -d '"'"'" | xargs 2>/dev/null || true)
  [[ -n "$_env_model" ]] && export LLM_MODEL_PATH="$_env_model"
else
  warn "backend/.env não encontrado — usando configurações padrão."
  warn "Execute: cp backend/.env.example backend/.env"
fi

check_port() {
  local port=$1
  if lsof -Pi ":$port" -sTCP:LISTEN -t &>/dev/null 2>&1; then
    return 1
  fi
  return 0
}

if ! check_port "$PORT_API"; then
  err "Porta $PORT_API já está em uso. Use --port-api para escolher outra."
  exit 1
fi
if ! check_port "$PORT_UI"; then
  err "Porta $PORT_UI já está em uso. Use --port-ui para escolher outra."
  exit 1
fi
ok "Portas $PORT_API (API) e $PORT_UI (UI) disponíveis."

API_PID=""
UI_PID=""
LOG_API="/tmp/rag_api_$$.log"
LOG_UI="/tmp/rag_ui_$$.log"

cleanup() {
  echo ""
  sep
  log "Encerrando…"
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null && ok "Backend encerrado (PID $API_PID)"
  [[ -n "$UI_PID"  ]] && kill "$UI_PID"  2>/dev/null && ok "Frontend encerrado (PID $UI_PID)"
  rm -f "$LOG_API" "$LOG_UI"
  sep
  echo -e "${BOLD}${CYAN}  Até logo.${RESET}"
}
trap cleanup EXIT INT TERM

sep
log "Iniciando FastAPI backend na porta ${PORT_API}…"

UVICORN_ARGS="app.main:app --host 0.0.0.0 --port $PORT_API --workers 1"
[[ "$HOT_RELOAD" == true ]] && UVICORN_ARGS="$UVICORN_ARGS --reload"

(cd "$ROOT_DIR/backend" && exec "$PYTHON" -m uvicorn $UVICORN_ARGS) > "$LOG_API" 2>&1 &
API_PID=$!
ok "Backend iniciado (PID $API_PID)"

log "Aguardando backend ficar pronto…"
WAIT_SECS=0
MAX_WAIT=300
until curl -sf --max-time 2 "http://localhost:$PORT_API/health" > /dev/null 2>&1; do
  sleep 2
  WAIT_SECS=$((WAIT_SECS + 2))
  if ! kill -0 "$API_PID" 2>/dev/null; then
    err "Backend encerrou inesperadamente. Últimas linhas do log:"
    tail -20 "$LOG_API" >&2
    exit 1
  fi
  if [[ $WAIT_SECS -ge $MAX_WAIT ]]; then
    err "Backend não ficou saudável em ${MAX_WAIT}s."
    tail -20 "$LOG_API" >&2
    exit 1
  fi
  echo -n "."
done
echo ""
ok "Backend saudável em http://localhost:${PORT_API} (aguardou ${WAIT_SECS}s)"

if [[ "$AUTO_INGEST" == true ]]; then
  log "Executando auto-ingest (force_reindex=false)…"
  INGEST_RESULT=$(curl -sf --max-time 600 \
    -X POST "http://localhost:$PORT_API/api/v1/ingest" \
    -H "Content-Type: application/json" \
    -d '{"force_reindex": false}' 2>&1 || echo '{"error": "timeout or failed"}')
  ok "Ingest: $INGEST_RESULT"
fi

sep
log "Iniciando Streamlit frontend na porta ${PORT_UI}…"

STREAMLIT_ARGS="frontend/rag_chat.py \
  --server.port $PORT_UI \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --theme.base dark"

"$PYTHON" -m streamlit run $STREAMLIT_ARGS > "$LOG_UI" 2>&1 &
UI_PID=$!
ok "Frontend iniciado (PID $UI_PID)"

sleep 3
if ! kill -0 "$UI_PID" 2>/dev/null; then
  err "Frontend encerrou inesperadamente. Últimas linhas do log:"
  tail -20 "$LOG_UI" >&2
  exit 1
fi
ok "Frontend disponível em http://localhost:${PORT_UI}"

sep
echo -e "${BOLD}${GREEN}  ✅  RAG System está rodando!${RESET}"
echo ""
echo -e "  ${BOLD}Chat UI${RESET}      → ${BLUE}http://localhost:${PORT_UI}${RESET}"
echo -e "  ${BOLD}API Docs${RESET}     → ${BLUE}http://localhost:${PORT_API}/docs${RESET}"
echo -e "  ${BOLD}Health${RESET}       → ${BLUE}http://localhost:${PORT_API}/health${RESET}"
echo ""
echo -e "  ${YELLOW}Pressione Ctrl+C para encerrar ambos os serviços.${RESET}"
sep

stream_log() {
  local prefix="$1" file="$2" color="$3"
  tail -f "$file" 2>/dev/null | while IFS= read -r line; do
    echo -e "${color}${prefix}${RESET} $line"
  done &
}

stream_log "[API]" "$LOG_API" "$MAGENTA"
stream_log "[UI] " "$LOG_UI"  "$BLUE"

wait "$API_PID" "$UI_PID" 2>/dev/null || true