#Requires -Version 5.1
<#
.SYNOPSIS
.PARAMETER PortApi
.PARAMETER PortUi
.PARAMETER NoIngest
.PARAMETER Reload
#>
param(
    [int]   $PortApi  = 8000,
    [int]   $PortUi   = 8501,
    [switch]$NoIngest,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$AutoIngest = -not $NoIngest.IsPresent
$HotReload  = $Reload.IsPresent

function Log  ($m) { Write-Host "[start] $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "[  ok ] $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "[ warn] $m" -ForegroundColor Yellow }
function Fail ($m) { Write-Host "[error] $m" -ForegroundColor Red }
function Sep       { Write-Host ("─" * 70) -ForegroundColor Cyan }

$RootDir = Split-Path -Parent $PSCommandPath
Set-Location $RootDir

Sep
Write-Host "  RAG System — Launcher (Windows)" -ForegroundColor Blue
Write-Host "  Backend  → http://localhost:$PortApi"
Write-Host "  Frontend → http://localhost:$PortUi"
Write-Host "  Docs     → http://localhost:${PortApi}/docs"
Sep
Log "Diretório de trabalho: $RootDir"

# ── Python ────────────────────────────────────────────────────────────────────
$PYTHON = $null
if ($env:CONDA_PREFIX) {
    $PYTHON = (Get-Command python -ErrorAction SilentlyContinue)?.Source
    if (-not $PYTHON) { $PYTHON = Join-Path $env:CONDA_PREFIX "python.exe" }
    Ok "Usando Conda env: $PYTHON"
} elseif (Test-Path "$RootDir\.venv\Scripts\python.exe") {
    $PYTHON = "$RootDir\.venv\Scripts\python.exe"
    Ok "Usando venv: $PYTHON"
} else {
    Fail "Nenhum ambiente Python encontrado (.venv ou Conda)."
    Fail "Execute primeiro: python start.py"
    exit 1
}

# ── Diretórios ────────────────────────────────────────────────────────────────
foreach ($d in @("backend\data\pdfs","backend\data\index","backend\models")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RootDir $d) | Out-Null
}
if ((Get-ChildItem (Join-Path $RootDir "backend\data\pdfs") -ErrorAction SilentlyContinue).Count -eq 0) {
    Warn "backend\data\pdfs\ está vazio — adicione PDFs antes de ingerir."
}

# ── .env ─────────────────────────────────────────────────────────────────────
$EnvFile = Join-Path $RootDir ".env"
if (Test-Path $EnvFile) {
    Ok ".env encontrado — será lido pelo Pydantic-Settings."
    $line = Get-Content $EnvFile |
        Where-Object { $_ -match "^\s*LLM_MODEL_PATH\s*=" } |
        Select-Object -Last 1
    if ($line) {
        $env:LLM_MODEL_PATH = ($line -split "=",2)[1].Trim().Trim("'`"")
    }
} else {
    Warn ".env não encontrado — usando configurações padrão."
    Warn "Execute: copy backend\.env.example .env"
}

# ── Modelos de embedding ──────────────────────────────────────────────────────
Log "Verificando modelos de embedding…"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
Push-Location (Join-Path $RootDir "backend")
& $PYTHON scripts\ensure_models.py
$embedOk = $LASTEXITCODE
Pop-Location
if ($embedOk -eq 0) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    Ok "Modelos em cache — backend em modo offline."
} else {
    Warn "Modelos não verificados — SSL bypass ativo."
    $env:REQUESTS_CA_BUNDLE = ""
    $env:CURL_CA_BUNDLE = ""
}

# ── Modelo LLM ────────────────────────────────────────────────────────────────
$rawModel = if ($env:LLM_MODEL_PATH) { $env:LLM_MODEL_PATH } else { "models\qwen2.5-7b-instruct-q4_k_m.gguf" }
$rawModel  = $rawModel -replace "/","\"
if ($rawModel -match "^[A-Za-z]:\\" -or $rawModel -match "^backend\\") {
    $ModelPath = Join-Path $RootDir $rawModel
} else {
    $ModelPath = Join-Path $RootDir "backend\$rawModel"
}
if (-not (Test-Path $ModelPath)) {
    Log "Modelo LLM não encontrado ($ModelPath) — iniciando download (~4.7 GB)…"
    Push-Location (Join-Path $RootDir "backend")
    & $PYTHON scripts\download_model.py
    $dlOk = $LASTEXITCODE
    Pop-Location
    if ($dlOk -ne 0) { Warn "Download falhou — backend iniciará em STUB MODE." }
    else              { Ok "Modelo LLM baixado com sucesso." }
}

# ── Verificação de portas ─────────────────────────────────────────────────────
function Test-PortFree([int]$Port) {
    return (netstat -ano 2>$null | Select-String ":$Port\s+.*LISTENING").Count -eq 0
}
if (-not (Test-PortFree $PortApi)) { Fail "Porta $PortApi já está em uso. Use -PortApi para escolher outra."; exit 1 }
if (-not (Test-PortFree $PortUi))  { Fail "Porta $PortUi já está em uso. Use -PortUi para escolher outra.";  exit 1 }
Ok "Portas $PortApi (API) e $PortUi (UI) disponíveis."

# Arquivos de log temporários
$LogApi    = [IO.Path]::GetTempFileName()
$LogApiErr = [IO.Path]::GetTempFileName()
$LogUi     = [IO.Path]::GetTempFileName()
$LogUiErr  = [IO.Path]::GetTempFileName()

# ── Backend ───────────────────────────────────────────────────────────────────
Sep
Log "Iniciando FastAPI backend na porta $PortApi…"
$uvArgs = @("-m","uvicorn","app.main:app","--host","0.0.0.0","--port","$PortApi","--workers","1")
if ($HotReload) { $uvArgs += "--reload" }

$ApiProc = Start-Process $PYTHON -ArgumentList $uvArgs `
    -WorkingDirectory (Join-Path $RootDir "backend") `
    -RedirectStandardOutput $LogApi -RedirectStandardError $LogApiErr `
    -PassThru -NoNewWindow
Ok "Backend iniciado (PID $($ApiProc.Id))"

# ── Aguardar health ───────────────────────────────────────────────────────────
Log "Aguardando backend ficar pronto…"
$waited = 0
while ($true) {
    try {
        if ((Invoke-WebRequest "http://localhost:$PortApi/health" -TimeoutSec 2 -UseBasicParsing -EA Stop).StatusCode -eq 200) { break }
    } catch {}
    Start-Sleep 2; $waited += 2
    if ($ApiProc.HasExited) {
        Fail "Backend encerrou inesperadamente. Últimas linhas do log:"
        Get-Content $LogApi,$LogApiErr -Tail 20 -EA SilentlyContinue | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        exit 1
    }
    if ($waited -ge 300) { Fail "Backend não ficou saudável em 300s."; exit 1 }
    Write-Host "." -NoNewline
}
Write-Host ""
Ok "Backend saudável em http://localhost:$PortApi (aguardou ${waited}s)"

# ── Auto-ingest ───────────────────────────────────────────────────────────────
if ($AutoIngest) {
    Log "Executando auto-ingest (force_reindex=false)…"
    try {
        $r = Invoke-RestMethod -Method Post `
            -Uri "http://localhost:$PortApi/api/v1/ingest" `
            -Body '{"force_reindex":false}' -ContentType "application/json" -TimeoutSec 600
        Ok "Ingest concluído: $($r | ConvertTo-Json -Compress)"
    } catch {
        Warn "Ingest retornou erro: $($_.Exception.Message)"
    }
}

# ── Frontend ──────────────────────────────────────────────────────────────────
Sep
Log "Iniciando Streamlit frontend na porta $PortUi…"
$stArgs = @("-m","streamlit","run","frontend\rag_chat.py",
    "--server.port","$PortUi","--server.address","0.0.0.0",
    "--server.headless","true","--browser.gatherUsageStats","false",
    "--theme.base","dark")

$UiProc = Start-Process $PYTHON -ArgumentList $stArgs `
    -WorkingDirectory $RootDir `
    -RedirectStandardOutput $LogUi -RedirectStandardError $LogUiErr `
    -PassThru -NoNewWindow
Ok "Frontend iniciado (PID $($UiProc.Id))"
Start-Sleep 3
if ($UiProc.HasExited) {
    Fail "Frontend encerrou inesperadamente. Últimas linhas do log:"
    Get-Content $LogUi,$LogUiErr -Tail 20 -EA SilentlyContinue | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    exit 1
}
Ok "Frontend disponível em http://localhost:$PortUi"

Sep
Write-Host "  RAG System está rodando!" -ForegroundColor Green
Write-Host ""
Write-Host "  Chat UI  → http://localhost:$PortUi"         -ForegroundColor Blue
Write-Host "  API Docs → http://localhost:${PortApi}/docs"  -ForegroundColor Blue
Write-Host "  Health   → http://localhost:${PortApi}/health" -ForegroundColor Blue
Write-Host ""
Write-Host "  Pressione Ctrl+C para encerrar ambos os serviços." -ForegroundColor Yellow
Sep

# ── Stream de logs (polling) ──────────────────────────────────────────────────
$apiOff = 0; $uiOff = 0
try {
    while (-not $ApiProc.HasExited -and -not $UiProc.HasExited) {
        $apiLines = @(Get-Content $LogApi,$LogApiErr -EA SilentlyContinue)
        if ($apiLines.Count -gt $apiOff) {
            $apiLines[$apiOff..($apiLines.Count-1)] | ForEach-Object { Write-Host "[API] $_" -ForegroundColor Magenta }
            $apiOff = $apiLines.Count
        }
        $uiLines = @(Get-Content $LogUi,$LogUiErr -EA SilentlyContinue)
        if ($uiLines.Count -gt $uiOff) {
            $uiLines[$uiOff..($uiLines.Count-1)] | ForEach-Object { Write-Host "[UI]  $_" -ForegroundColor Blue }
            $uiOff = $uiLines.Count
        }
        Start-Sleep 1
    }
} finally {
    Write-Host ""; Sep; Log "Encerrando…"
    Stop-Process -Id $ApiProc.Id -Force -EA SilentlyContinue; Ok "Backend encerrado (PID $($ApiProc.Id))."
    Stop-Process -Id $UiProc.Id  -Force -EA SilentlyContinue; Ok "Frontend encerrado (PID $($UiProc.Id))."
    Remove-Item $LogApi,$LogApiErr,$LogUi,$LogUiErr -EA SilentlyContinue
    Sep; Write-Host "  Até logo." -ForegroundColor Cyan
}
