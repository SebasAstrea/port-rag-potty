<#
  RAG portable · setup.ps1  (v2)
  ----------------------------------------------------------------------------
  Monta el RAG en un proyecto con UNA sola ejecución: copia el motor, genera
  la config, arranca Ollama + servicio, autoindexa hasta drift=0 y valida con
  un smoke test (/health + /methodology + /query).

  Uso:
    .\setup.ps1 [<proyecto-destino>] [opciones]

  Opciones:
    <proyecto-destino>   ruta al proyecto a indexar         (default: ".")
    -Force               reindex force=true (re-embebe todo; LENTO en CPU)
    -DryRun              prepara archivos y valida el compose SIN arrancar
                         nada ni descargar imágenes
    -Local               sin Docker de servicio: venv + python -m rag.server
                         (Ollama: el del sistema o un contenedor auxiliar)
    -Docker              usar Docker (default): compose con ollama + rag
    -Down                para el servicio (no borra volúmenes)
    -Status              informa del estado (sale con 1 si está caído)
    -Port N              puerto inicial; si está ocupado autoincrementa (≤20)
    -NoAgents            no copiar la plantilla AGENTS.md
    -Help                esta ayuda

  Variables opcionales (env):
    RAG_REPO             URL git del repo RAG (solo si este script está fuera)
    RAG_PORT             puerto inicial                       (default 8765)
    RAG_EMBED_MODEL      modelo de embeddings                 (default bge-m3)
    RAG_EMBED_DIM        dimensión del vector                 (default 1024)
    RAG_COLLECTION       override de la colección Chroma (default <slug>_chunks)
    RAG_INDEX_TIMEOUT    seg. máx. esperando el reindex       (default 1800)
    RAG_OLLAMA_VOLUME    volumen Docker ya poblado con el modelo (p.ej.
                         rag-visionrt_ollama_data); ahorra el pull de bge-m3
    OLLAMA_HOST          URL de un Ollama existente           (default autodetecta)

  Comportamiento clave (v2):
    * Puerto: si algo responde y su config.collection ES la de este proyecto
      → se reutiliza; si es otro servicio → autoincrementa (hasta +20) y
      re-valida la identidad tras arrancar.
    * Ollama: reutiliza uno accesible con el modelo; si no, contenedor
      (reusando RAG_OLLAMA_VOLUME si se indica). Con -Local: contenedor
      auxiliar o el CLI `ollama` del sistema.
    * Reindex: espera con timeout, tolera 409 (reindex en curso), exige
      drift=0; ante error imprime los últimos logs.
    * Copia limpia (robocopy): nunca arrastra data\ ni __pycache__ ni .venv.
#>

param(
  [Parameter(Position = 0)][string]$Dest = ".",
  [switch]$Force,
  [switch]$DryRun,
  [switch]$Local,
  [switch]$Docker,
  [switch]$Down,
  [switch]$Status,
  [int]$Port = 0,
  [switch]$NoAgents,
  [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "aviso: $msg" -ForegroundColor Yellow }
function Stop-Rag($msg) {
  Write-Host "error: $msg" -ForegroundColor Red
  exit 1
}
function ConvertTo-Slug([string]$s) {
  $b = [Regex]::Replace((Split-Path -Leaf $s), '[^a-zA-Z0-9_-]', '_')
  if ([string]::IsNullOrWhiteSpace($b)) { return 'proyecto' }
  return $b.ToLower()
}
function Read-Utf8([string]$path) {
  [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
}
function Write-Utf8NoBom([string]$path, [string]$text) {
  [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding $false))
}
function Test-PortOpen([int]$p) {
  $c = New-Object System.Net.Sockets.TcpClient
  try {
    $task = $c.ConnectAsync('127.0.0.1', $p)
    return $task.Wait(800) -and $c.Connected
  } catch {
    return $false
  } finally {
    $c.Close()
  }
}
function Get-HealthObject([int]$p) {
  try {
    return Invoke-RestMethod -Uri "http://127.0.0.1:$p/health" -TimeoutSec 3 -ErrorAction Stop
  } catch {
    return $null
  }
}
function Get-PortStatus([int]$p, [string]$expected) {
  if (-not (Test-PortOpen $p)) { return 'free' }
  $h = Get-HealthObject $p
  if ($h -and $h.config -and $h.config.collection -eq $expected) { return 'ours' }
  return 'other'
}
function Select-Port([int]$base, [string]$expected) {
  for ($i = 0; $i -lt 20; $i++) {
    $p = $base + $i
    $st = Get-PortStatus $p $expected
    if ($st -eq 'free') { return @{ Port = $p; Reused = $false } }
    if ($st -eq 'ours') {
      Write-Step "puerto $p ya sirve ESTE proyecto (collection=$expected) - reutilizando"
      return @{ Port = $p; Reused = $true }
    }
    Write-Step "puerto $p ocupado por otro servicio -> probando $($p + 1)"
  }
  Stop-Rag "ningun puerto libre en $base..$($base + 19). Usa -Port N o libera puertos"
}
function Get-OllamaProbe([string]$base) {
  $base = $base.TrimEnd('/')
  try {
    $tags = Invoke-RestMethod -Uri "$base/api/tags" -TimeoutSec 3 -ErrorAction Stop
  } catch {
    return 'down'
  }
  foreach ($m in @($tags.models)) {
    $n = [string]$m.name
    if ($n -eq $script:Model -or $n.StartsWith("$($script:Model):")) { return 'model' }
  }
  return 'nomodel'
}
function Test-ModelInNames([string[]]$names) {
  foreach ($n in $names) {
    if ($n -eq $script:Model -or $n.StartsWith("$($script:Model):")) { return $true }
  }
  return $false
}
function Get-FreePortFrom([int]$start) {
  for ($i = 0; $i -lt 10; $i++) {
    if (-not (Test-PortOpen ($start + $i))) { return ($start + $i) }
  }
  return $null
}
function Read-State([string]$key) {
  $f = Join-Path $script:Dest "rag\data\setup.state"
  if (-not (Test-Path $f)) { return $null }
  foreach ($line in Get-Content $f) {
    if ($line -like "$key=*") { return $line.Substring($key.Length + 1) }
  }
  return $null
}
function Write-State([string]$mode, [int]$p) {
  $d = Join-Path $script:Dest "rag\data"
  if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
  $prevAux = Read-State 'AUX_PORT'
  $auxVal = if ($script:AuxPort) { $script:AuxPort } elseif ($prevAux) { $prevAux } else { '' }
  $lines = @(
    "MODE=$mode",
    "PORT=$p",
    "COLLECTION=$($script:Expected)",
    "SLUG=$($script:Slug)",
    "AUX_PORT=$auxVal",
    "OLLAMA_BASE=$($script:OllamaBase)"
  )
  Write-Utf8NoBom (Join-Path $d "setup.state") (($lines -join "`n") + "`n")
}
function Show-Logs {
  Write-Host "--- logs ---" -ForegroundColor Yellow
  if ($script:Mode -eq 'docker' -and $script:Slug) {
    $lg = docker logs --tail 40 "rag-$($script:Slug)-rag" 2>&1 | Select-Object -Last 40
    $lg | ForEach-Object { Write-Host $_ }
  }
  $serverLog = Join-Path $script:Dest "rag\data\server.log"
  if (Test-Path $serverLog) {
    Get-Content $serverLog -Tail 40 | ForEach-Object { Write-Host $_ }
  }
  Write-Host "-----------" -ForegroundColor Yellow
}
function Assert-Identity([int]$p) {
  $h = Get-HealthObject $p
  $coll = if ($h -and $h.config) { $h.config.collection } else { $null }
  if ($coll -ne $script:Expected) {
    Stop-Rag "identidad no coincide: servidor='$coll' esperada='$($script:Expected)' (otro RAG en el puerto $p?)"
  }
}
function Invoke-ReindexAndWait([int]$p, [bool]$force) {
  $url = "http://127.0.0.1:$p/reindex"
  if ($force) {
    Write-Step "force=true - re-embebiendo todo (lento en CPU)"
    $url += "?force=true"
  }
  $code = 0
  try {
    Invoke-WebRequest -Uri $url -Method Post -UseBasicParsing -TimeoutSec 15 -ErrorAction Stop | Out-Null
    $code = 200
  } catch {
    $resp = $_.Exception.Response
    if ($resp) { $code = [int]$resp.StatusCode } else { $code = 0 }
  }
  switch ($code) {
    200 { Write-Step "reindex lanzado (incremental)" }
    409 { Write-Warn "ya hay un reindex en curso (409) - esperando a que termine" }
    default {
      Show-Logs
      Stop-Rag "POST /reindex -> HTTP $code"
    }
  }

  $timeout = 1800
  if ($env:RAG_INDEX_TIMEOUT) {
    $timeout = 0
    if (-not [int]::TryParse($env:RAG_INDEX_TIMEOUT, [ref]$timeout)) { $timeout = 1800 }
  }
  $deadline = (Get-Date).AddSeconds($timeout)
  $fails = 0
  Write-Host -NoNewline "esperando el indice "
  $st = $null
  while ($true) {
    try {
      $st = Invoke-RestMethod -Uri "http://127.0.0.1:$p/stats" -TimeoutSec 5 -ErrorAction Stop
      $fails = 0
    } catch {
      $fails++
      if ($fails -ge 20) { Write-Host ""; Show-Logs; Stop-Rag "/stats dejo de responder" }
      Write-Host -NoNewline "."
      Start-Sleep -Seconds 3
      continue
    }
    if (-not $st.reindex.running) { break }
    if ($st.reindex.status -eq 'error') {
      Write-Host ""
      Show-Logs
      Stop-Rag "la indexacion fallo: $($st.reindex.error)"
    }
    if ((Get-Date) -gt $deadline) {
      Write-Host ""
      Show-Logs
      Stop-Rag "timeout de ${timeout}s esperando el reindex (sube RAG_INDEX_TIMEOUT)"
    }
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 3
  }
  Write-Host ""

  if ($st.reindex.status -eq 'error') {
    Show-Logs
    Stop-Rag "la indexacion fallo: $($st.reindex.error)"
  }
  $drift = $st.drift
  Write-Step "indice: manifest=$($st.manifest_chunks) chroma=$($st.chroma_chunks) drift=$drift status=$($st.reindex.status)"
  if ("$drift" -ne '0') {
    Stop-Rag "drift=$drift != 0 - repite: Invoke-RestMethod -Method Post http://127.0.0.1:$p/reindex"
  }
}
function Invoke-Smoke([int]$p) {
  $h = Get-HealthObject $p
  if (-not $h) { Stop-Rag "smoke: /health no responde" }
  if (-not $h.ok) { Stop-Rag "smoke: health.ok=false" }
  if (-not $h.ollama.ok) { Stop-Rag "smoke: ollama no disponible ($($h.ollama.reason))" }
  if ($h.config.collection -ne $script:Expected) { Stop-Rag "smoke: identidad incorrecta ($($h.config.collection))" }
  Write-Step "smoke /health OK (coleccion=$($h.config.collection), ollama ok, chunks=$($h.collection.count))"

  try {
    $meth = Invoke-RestMethod -Uri "http://127.0.0.1:$p/methodology" -TimeoutSec 5 -ErrorAction Stop
  } catch { $meth = $null }
  if (-not $meth -or -not $meth.count) {
    Write-Warn "/methodology vacio (sin AGENTS.md/README indexados?)"
  } else {
    Write-Step "smoke /methodology OK ($($meth.count) chunks del alma del proyecto)"
  }

  $chroma = $null
  try { $chroma = (Invoke-RestMethod -Uri "http://127.0.0.1:$p/stats" -TimeoutSec 5).chroma_chunks } catch { }
  if (-not $chroma) {
    Write-Warn "indice vacio (proyecto sin archivos indexables) - omito el smoke de /query"
    return
  }
  try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:$p/query" -Method Post `
      -ContentType 'application/json' -Body '{"q":"README principal del proyecto","k":1}' `
      -TimeoutSec 60 -ErrorAction Stop
  } catch {
    Stop-Rag "smoke: /query fallo: $($_.Exception.Message)"
  }
  if (-not $q.results -or $q.results.Count -eq 0) {
    Stop-Rag "smoke: /query devolvio 0 resultados con un indice de $chroma chunks"
  }
  $top = $q.results[0]
  $score = if ($null -ne $top.score) { $top.score } else { 'n/a' }
  Write-Step "smoke /query OK top-1: $($top.file) (score=$score)"
}

if ($Help) {
  Get-Content $MyInvocation.MyCommand.Path | Select-Object -Skip 1 -First 60 |
    ForEach-Object { $_ -replace '^  ?#?', '' }
  exit 0
}

if ($Down -and $Status) { Stop-Rag "-Down y -Status no se pueden combinar" }
if ($Local -and $Docker) { Stop-Rag "-Local y -Docker no se pueden combinar" }
$Mode = if ($Local) { 'local' } else { 'docker' }

$PortBase = $Port
if ($PortBase -le 0) {
  $PortBase = if ($env:RAG_PORT) { [int]$env:RAG_PORT } else { 8765 }
}
$Model = if ($env:RAG_EMBED_MODEL) { $env:RAG_EMBED_MODEL } else { 'bge-m3' }
$EmbedDim = if ($env:RAG_EMBED_DIM) { $env:RAG_EMBED_DIM } else { '1024' }
$ComposeName = 'docker-compose.rag.yml'
$Expected = $null
$Slug = ''
$AuxPort = ''
$OllamaBase = ''
$Strategy = ''
$PortChosen = 0
$PortReused = $false
$TmpRag = $null

# --- modo -Status --------------------------------------------------------------
if ($Status) {
  if (-not (Test-Path $Dest)) { Stop-Rag "no existe el proyecto $Dest" }
  $Dest = (Resolve-Path $Dest).Path
  $sp = Read-State 'PORT'
  $sm = Read-State 'MODE'
  if (-not $sp) {
    $cf = Join-Path $Dest $ComposeName
    if (Test-Path $cf) {
      $m = [Regex]::Match((Read-Utf8 $cf), '"(\d+):8765"')
      if ($m.Success) { $sp = $m.Groups[1].Value }
    }
  }
  if (-not $sp) { $sp = '8765' }
  if (-not $sm) { $sm = 'docker' }
  Write-Step "proyecto: $Dest (modo=$sm, puerto=$sp)"
  $h = Get-HealthObject ([int]$sp)
  if (-not $h) {
    Write-Host "servicio:  caido (no responde en :$sp)"
    if ($sm -eq 'local') {
      Write-Host "arranque:  cd $Dest; .\rag\.venv\Scripts\python.exe -m rag.server"
    } else {
      Write-Host "arranque:  docker compose -f $Dest\$ComposeName up -d"
    }
    exit 1
  }
  $st = $null
  try { $st = Invoke-RestMethod -Uri "http://127.0.0.1:$sp/stats" -TimeoutSec 5 -ErrorAction Stop } catch { }
  Write-Host "servicio:  activo"
  Write-Host "proyecto:  $($h.config.project)"
  Write-Host "coleccion: $($h.config.collection)"
  Write-Host "ollama:    $($h.ollama.ok) ($($h.ollama.ollama_host))"
  if ($st) {
    Write-Host "chunks:    chroma=$($st.chroma_chunks) manifest=$($st.manifest_chunks) drift=$($st.drift)"
    Write-Host "archivos:  $($st.manifest_files)"
    Write-Host "reindex:   running=$($st.reindex.running) status=$($st.reindex.status)"
  }
  Write-Host "grafo:     themes=$($h.graph.categories) tokens=$($h.graph.tokens) files=$($h.graph.files) edges=$($h.graph.edges)"
  exit 0
}

# --- modo -Down ----------------------------------------------------------------
if ($Down) {
  if (-not (Test-Path $Dest)) { Stop-Rag "no existe el proyecto $Dest" }
  $Dest = (Resolve-Path $Dest).Path
  $dm = Read-State 'MODE'
  if (-not $dm) {
    if (Test-Path (Join-Path $Dest $ComposeName)) { $dm = 'docker' } else { $dm = 'local' }
  }
  $stopped = $false
  if ($dm -eq 'local') {
    $pf = Join-Path $Dest "rag\data\server.pid"
    if (Test-Path $pf) {
      $pid_ = 0
      if ([int]::TryParse(([string](Get-Content $pf -Raw)).Trim(), [ref]$pid_)) {
        $proc = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
        if ($proc) {
          Stop-Process -Id $pid_ -Force -ErrorAction SilentlyContinue
          Write-Step "servidor local (pid $pid_) detenido"
          $stopped = $true
        }
      }
      Remove-Item $pf -Force -ErrorAction SilentlyContinue
    }
    if (Get-Command docker -ErrorAction SilentlyContinue) {
      $aux = docker ps -a --format '{{.Names}}' 2>$null
      if (@($aux) -contains 'rag-ollama-aux' -and (Read-State 'AUX_PORT')) {
        docker stop rag-ollama-aux | Out-Null
        Write-Step "contenedor auxiliar rag-ollama-aux detenido (volumen intacto)"
        $stopped = $true
      }
    }
  } else {
    $cf = Join-Path $Dest $ComposeName
    if (Test-Path $cf) {
      if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Stop-Rag "docker no esta disponible" }
      docker compose version *> $null
      if ($LASTEXITCODE -ne 0) { Stop-Rag "docker compose no esta disponible" }
      docker compose -f $cf down --remove-orphans
      $global:LASTEXITCODE = 0
      $stopped = $true
    }
  }
  if (-not $stopped) { Write-Warn "nada que parar en $Dest" }
  Write-Step "volumenes y datos de indice NO se borran (borra a mano si lo necesitas)"
  exit 0
}

# --- prerequisitos -------------------------------------------------------------
if ($Mode -eq 'docker') {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Stop-Rag "docker no esta instalado (o usa -Local)" }
  docker compose version *> $null
  if ($LASTEXITCODE -ne 0) { Stop-Rag "docker compose (v2) no esta disponible (o usa -Local)" }
} else {
  $PythonExe = $null
  if (Get-Command python -ErrorAction SilentlyContinue) { $PythonExe = 'python' }
  elseif (Get-Command py -ErrorAction SilentlyContinue) { $PythonExe = 'py' }
  else { Stop-Rag "python no esta instalado (modo -Local)" }
}

# --- localizar el repo del RAG -------------------------------------------------
$RagSrc = $PSScriptRoot
if ((Test-Path "$RagSrc\rag\server.py") -and (Test-Path "$RagSrc\rag\config.py")) {
  Write-Step "repositorio del RAG: $RagSrc"
} elseif ($env:RAG_REPO) {
  $TmpRag = Join-Path ([System.IO.Path]::GetTempPath()) ("rag_portable_" + [guid]::NewGuid().ToString('n'))
  Write-Step "clonando $($env:RAG_REPO) ..."
  git clone --quiet --depth 1 $env:RAG_REPO $TmpRag
  if ($LASTEXITCODE -ne 0) { Stop-Rag "no se pudo clonar $($env:RAG_REPO)" }
  $RagSrc = $TmpRag
} else {
  Stop-Rag "no encuentro el repo del RAG. Copia setup.ps1 dentro de RAG_portable, o setea RAG_REPO=<url-git>"
}

try {
  # --- proyecto destino ---------------------------------------------------------
  if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest -Force | Out-Null }
  $Dest = (Resolve-Path $Dest).Path
  $script:Dest = $Dest
  $Slug = ConvertTo-Slug $Dest
  $script:Slug = $Slug
  $Project = Split-Path -Leaf $Dest
  $Expected = if ($env:RAG_COLLECTION) { $env:RAG_COLLECTION } else { "${Slug}_chunks" }
  $script:Expected = $Expected
  $script:Model = $Model
  $script:Mode = $Mode

  Write-Step "proyecto: $Dest (slug=$Slug, modo=$Mode, coleccion=$Expected)"

  # --- reutilizar servidor local previo (-Local) --------------------------------
  if ($Mode -eq 'local') {
    $oldPort = Read-State 'PORT'
    $pf = Join-Path $Dest "rag\data\server.pid"
    if (Test-Path $pf) {
      $oldPid = 0
      if ([int]::TryParse(([string](Get-Content $pf -Raw)).Trim(), [ref]$oldPid)) {
        $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($proc -and $oldPort -and ((Get-PortStatus ([int]$oldPort) $Expected) -eq 'ours')) {
          $PortChosen = [int]$oldPort
          $PortReused = $true
          Write-Step "servidor local previo (pid $oldPid) sigue activo en :$PortChosen - reutilizando"
        } elseif ($proc) {
          Write-Warn "matando servidor local previo (pid $oldPid) que ya no responde"
          Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
          Start-Sleep -Seconds 2
          Remove-Item $pf -Force -ErrorAction SilentlyContinue
        } else {
          Remove-Item $pf -Force -ErrorAction SilentlyContinue
        }
      }
    }
  }

  # --- seleccion de puerto -------------------------------------------------------
  if (-not $PortReused) {
    $sel = Select-Port $PortBase $Expected
    $PortChosen = [int]$sel.Port
    $PortReused = [bool]$sel.Reused
    if ($PortChosen -ne $PortBase) { Write-Step "puerto elegido: $PortChosen" }
  }
  $Port = $PortChosen

  # --- estrategia Ollama -----------------------------------------------------------
  function Resolve-OllamaDocker {
    $script:Strategy = 'container'
    $script:OllamaBase = ''
    $cands = @()
    if ($env:OLLAMA_HOST) { $cands += $env:OLLAMA_HOST }
    $cands += 'http://127.0.0.1:11434'
    foreach ($cand in $cands) {
      $probe = Get-OllamaProbe $cand
      if ($probe -eq 'model') {
        $script:Strategy = 'host'
        $script:OllamaBase = $cand
        Write-Step "reutilizando Ollama del host: $cand (modelo $Model ya presente)"
        return
      }
      if ($probe -eq 'nomodel') {
        Write-Warn "Ollama en $cand accesible pero SIN modelo $Model - uso un contenedor aparte"
      }
    }
    if ($env:RAG_OLLAMA_VOLUME) {
      Write-Step "Ollama en contenedor con volumen externo: $($env:RAG_OLLAMA_VOLUME) (modelo ya descargado)"
    } else {
      Write-Step "Ollama en contenedor (primer arranque descarga la imagen + modelo)"
    }
  }
  function Resolve-OllamaLocal {
    $script:OllamaBase = ''
    $cands = @()
    if ($env:OLLAMA_HOST) { $cands += $env:OLLAMA_HOST }
    $cands += 'http://127.0.0.1:11434'
    foreach ($cand in $cands) {
      $probe = Get-OllamaProbe $cand
      if ($probe -eq 'model') {
        $script:OllamaBase = $cand
        $script:Strategy = 'host'
        Write-Step "usando Ollama existente: $cand"
        return
      }
      if ($probe -eq 'nomodel') {
        if (Get-Command ollama -ErrorAction SilentlyContinue) {
          Write-Step "descargando modelo $Model en $cand (CLI ollama)..."
          $env:OLLAMA_HOST = $cand
          ollama pull $Model 2>$null | Out-Null
          if ((Get-OllamaProbe $cand) -eq 'model') {
            $script:OllamaBase = $cand
            $script:Strategy = 'host'
            return
          }
        }
      }
    }
    if (Get-Command docker -ErrorAction SilentlyContinue) {
      $script:Strategy = 'aux'
      return
    }
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
      Write-Step "iniciando el daemon del sistema (ollama serve)..."
      Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden
      Start-Sleep -Seconds 3
      if ((Get-OllamaProbe 'http://127.0.0.1:11434') -ne 'down') {
        $env:OLLAMA_HOST = 'http://127.0.0.1:11434'
        ollama pull $Model 2>$null | Out-Null
        $script:OllamaBase = 'http://127.0.0.1:11434'
        $script:Strategy = 'host'
        return
      }
    }
    Stop-Rag ("no hay Ollama accesible ni Docker. Instala uno de ellos:`n" +
      "  Ollama: https://ollama.com/download  (luego: ollama pull $Model)`n" +
      "  Docker: https://docs.docker.com/get-docker/")
  }

  $dataDir = Join-Path $Dest 'rag\data'
  if (-not (Test-Path $dataDir)) { New-Item -ItemType Directory -Path $dataDir -Force | Out-Null }
  if ($Mode -eq 'docker') { Resolve-OllamaDocker } else { Resolve-OllamaLocal }
  $script:AuxPort = $AuxPort

  # --- copiar el motor (copia limpia: sin data\ ni cache) -------------------------
  Write-Step "copiando motor -> $Dest\rag"
  $ragDst = Join-Path $Dest 'rag'
  if (-not (Test-Path $ragDst)) { New-Item -ItemType Directory -Path $ragDst -Force | Out-Null }
  $srcRag = Join-Path $RagSrc 'rag'
  $srcFull = [System.IO.Path]::GetFullPath($srcRag).TrimEnd('\')
  $dstFull = [System.IO.Path]::GetFullPath($ragDst).TrimEnd('\')
  if ($srcFull -eq $dstFull) {
    Write-Step "el motor ya esta en el destino (proyecto == repo RAG) - omito la copia"
  } else {
    robocopy $srcRag $ragDst /E /XD data __pycache__ .venv /XF *.pyc /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { Stop-Rag "robocopy fallo (codigo $LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
  }
  if (-not (Test-Path $dataDir)) { New-Item -ItemType Directory -Path $dataDir -Force | Out-Null }

  if (-not $NoAgents -and -not (Test-Path (Join-Path $Dest 'AGENTS.md')) -and (Test-Path (Join-Path $RagSrc 'AGENTS.md'))) {
    $t = Read-Utf8 (Join-Path $RagSrc 'AGENTS.md')
    $t = $t -replace '8765', "$Port"
    Write-Utf8NoBom (Join-Path $Dest 'AGENTS.md') $t
    Write-Step "plantilla AGENTS.md creada con el puerto $Port"
  }

  # --- generar docker-compose.rag.yml (solo modo docker) --------------------------
  if ($Mode -eq 'docker') {
    $projectQ = $Project -replace '"', '\"'
    $collectionQ = $Expected -replace '"', '\"'
    $ollamaSvc = ''
    $depBlock = ''
    $volOllama = ''
    $servicesExtra = ''
    $ollamaEnv = ''

    if ($Strategy -eq 'host') {
      $ollamaUrl = $OllamaBase
      if ($ollamaUrl -match '^http://(127\.0\.0\.1|localhost):(\d+)$') {
        $ollamaUrl = "http://host.docker.internal:$($Matches[2])"
      }
      $servicesExtra = "    extra_hosts:`n      - `"host.docker.internal:host-gateway`"`n"
      $ollamaEnv = $ollamaUrl
      if ($env:RAG_OLLAMA_VOLUME) { Write-Warn "RAG_OLLAMA_VOLUME se ignora: se usa el Ollama del host" }
    } else {
      $ollamaEnv = 'http://ollama:11434'
      $depBlock = "    depends_on:`n      ollama:`n        condition: `"service_healthy`"`n"
      $ollamaSvc = @"
  ollama:
    image: "ollama/ollama:latest"
    container_name: "rag-${Slug}-ollama"
    restart: "unless-stopped"
    healthcheck:
      test: ["CMD", "/bin/sh", "-c", "ollama list >/dev/null 2>&1 || exit 1"]
      interval: "10s"
      timeout: "5s"
      retries: "5"
    volumes:
      - "ollama_data:/root/.ollama"

"@
      if ($env:RAG_OLLAMA_VOLUME) {
        $volName = $env:RAG_OLLAMA_VOLUME -replace '"', '\"'
        $volOllama = "  ollama_data:`n    external: true`n    name: `"$volName`""
      } else {
        $volOllama = '  ollama_data:'
      }
    }

    $compose = @"
name: "rag-${Slug}"
services:
${ollamaSvc}  rag:
    build: "./rag"
    image: "rag-${Slug}-rag"
    container_name: "rag-${Slug}-rag"
    restart: "unless-stopped"
    ports:
      - "${Port}:8765"
    environment:
      OLLAMA_HOST: "${ollamaEnv}"
      RAG_HOST: "0.0.0.0"
      RAG_PORT: "8765"
      RAG_EMBED_MODEL: "${Model}"
      RAG_EMBED_DIM: "${EmbedDim}"
      RAG_PROJECT: "${projectQ}"
      RAG_COLLECTION: "${collectionQ}"
${servicesExtra}    volumes:
      - ".:/app"
      - "rag_data:/app/rag/data"
    working_dir: "/app"
${depBlock}
volumes:
${volOllama}
  rag_data:
"@
    Write-Utf8NoBom (Join-Path $Dest $ComposeName) ($compose + "`n")
    Write-Step "generado $ComposeName (puerto $Port, ollama=$Strategy)"
  }

  # --- modo -DryRun ----------------------------------------------------------------
  if ($DryRun) {
    Write-Step "dry-run - nada se ha arrancado ni descargado"
    if ($Mode -eq 'docker') {
      docker compose -f (Join-Path $Dest $ComposeName) config *> $null
      if ($LASTEXITCODE -ne 0) { Stop-Rag "compose invalido" }
      Write-Step "compose valido: $Dest\$ComposeName"
      if ($env:RAG_OLLAMA_VOLUME -and $Strategy -eq 'container') {
        docker volume inspect $env:RAG_OLLAMA_VOLUME *> $null
        if ($LASTEXITCODE -ne 0) {
          Write-Warn "el volumen externo '$($env:RAG_OLLAMA_VOLUME)' no existe: crealo antes del up"
        }
        $global:LASTEXITCODE = 0
      }
    } else {
      Write-Step "modo local: venv en rag\.venv + python -m rag.server en :$Port"
    }
    $planTimeout = if ($env:RAG_INDEX_TIMEOUT) { $env:RAG_INDEX_TIMEOUT } else { '1800' }
    $planOllama = if ($OllamaBase) { "$Strategy en $OllamaBase" } else { $Strategy }
    Write-Host @"
  plan (no ejecutado):
    proyecto:    $Dest
    modo:        $Mode
    puerto:      $Port
    coleccion:   $Expected
    ollama:      $planOllama
    force:       $Force
    timeout:     ${planTimeout}s
"@
    exit 0
    exit 0
  }

  # --- arrancar servicios ------------------------------------------------------------
  $containerOllama = "rag-${Slug}-ollama"
  $containerRag = "rag-${Slug}-rag"
  if ($Mode -eq 'docker') {
    if ($Strategy -eq 'container') {
      if ($env:RAG_OLLAMA_VOLUME) {
        docker volume inspect $env:RAG_OLLAMA_VOLUME *> $null
        if ($LASTEXITCODE -ne 0) { Stop-Rag "el volumen externo '$($env:RAG_OLLAMA_VOLUME)' no existe (crealo o usa otro)" }
        $global:LASTEXITCODE = 0
      }
      Write-Step "arrancando Ollama (contenedor)..."
      docker compose -f (Join-Path $Dest $ComposeName) up -d ollama
      if ($LASTEXITCODE -ne 0) { Stop-Rag "no se pudo arrancar ollama" }
      $healthy = $false
      for ($i = 0; $i -lt 60; $i++) {
        $st = docker inspect --format '{{.State.Health.Status}}' $containerOllama 2>$null
        if (-not $st) { $st = 'starting' }
        if ("$st" -eq 'healthy') { $healthy = $true; break }
        if ("$st" -eq 'unhealthy') { Stop-Rag "el contenedor $containerOllama quedo unhealthy" }
        Start-Sleep -Seconds 2
      }
      if (-not $healthy) { Stop-Rag "Ollama no arranco a tiempo" }
      $names = @(docker exec $containerOllama ollama list 2>$null | Select-Object -Skip 1 |
        ForEach-Object { ($_ -split '\s+')[0] })
      if (Test-ModelInNames $names) {
        Write-Step "modelo $Model ya disponible en el volumen"
      } else {
        Write-Step "descargando modelo $Model (puede tardar)..."
        docker exec $containerOllama ollama pull $Model
      }
    }
    Write-Step "arrancando el servicio RAG en :$Port ..."
    docker compose -f (Join-Path $Dest $ComposeName) up -d --build --force-recreate rag
    if ($LASTEXITCODE -ne 0) { Show-Logs; Stop-Rag "no se pudo arrancar el RAG" }
    $up = $false
    for ($i = 0; $i -lt 90; $i++) {
      if (Get-HealthObject $Port) { $up = $true; break }
      $run = docker inspect --format '{{.State.Running}} {{.State.ExitCode}}' $containerRag 2>$null
      if ("$run" -like 'false*') { Show-Logs; Stop-Rag "el contenedor $containerRag termino ($run)" }
      Start-Sleep -Seconds 2
    }
    if (-not $up) { Show-Logs; Stop-Rag "el RAG no responde en :$Port (180s)" }
  } else {
    # --- modo local: venv + python -m rag.server ------------------------------------
    $venv = Join-Path $Dest 'rag\.venv'
    $py = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $py)) {
      Write-Step "creando venv en rag\.venv ..."
      if ($PythonExe -eq 'py') { py -3 -m venv $venv } else { python -m venv $venv }
      if ($LASTEXITCODE -ne 0 -or -not (Test-Path $py)) { Stop-Rag "no pude crear el venv" }
    }
    Write-Step "instalando dependencias (requirements.txt)..."
    $pipLog = Join-Path $dataDir 'pip-install.log'
    & $py -m pip install --disable-pip-version-check -q -r (Join-Path $Dest 'rag\requirements.txt') *> $pipLog
    if ($LASTEXITCODE -ne 0) {
      Get-Content $pipLog -Tail 30 | ForEach-Object { Write-Host $_ }
      Stop-Rag "pip install fallo (log: $pipLog)"
    }

    if ($Strategy -eq 'aux') {
      Write-Step "lanzando Ollama auxiliar en contenedor..."
      $auxName = 'rag-ollama-aux'
      $existing = @(docker ps -a --format '{{.Names}}' 2>$null)
      if ($existing -contains $auxName) {
        $cur = $null
        $portLine = docker port $auxName 11434/tcp 2>$null | Select-Object -First 1
        if ($portLine -match ':(\d+)\s*$') { $cur = $Matches[1] }
        if ($cur -and (Get-OllamaProbe "http://127.0.0.1:$cur") -eq 'model') {
          $AuxPort = $cur
          docker start $auxName | Out-Null
          Write-Step "contenedor auxiliar ya existente en :$AuxPort con el modelo - reutilizado"
        } else {
          docker rm -f $auxName | Out-Null
          $AuxPort = ''
        }
      }
      if (-not $AuxPort) {
        $AuxPort = Get-FreePortFrom 11434
        if (-not $AuxPort) { $AuxPort = 11434 }
        $vol = if ($env:RAG_OLLAMA_VOLUME) { $env:RAG_OLLAMA_VOLUME } else { 'rag_ollama_aux_data' }
        docker run -d --name $auxName -p "${AuxPort}:11434" -v "${vol}:/root/.ollama" `
          --restart unless-stopped ollama/ollama:latest | Out-Null
        if ($LASTEXITCODE -ne 0) { Stop-Rag "no pude lanzar el contenedor Ollama auxiliar" }
      }
      $ok = $false
      for ($i = 0; $i -lt 60; $i++) {
        if ((Get-OllamaProbe "http://127.0.0.1:$AuxPort") -ne 'down') { $ok = $true; break }
        Start-Sleep -Seconds 2
      }
      if (-not $ok) { Stop-Rag "el Ollama auxiliar no respondio" }
      $names = @(docker exec $auxName ollama list 2>$null | Select-Object -Skip 1 |
        ForEach-Object { ($_ -split '\s+')[0] })
      if (Test-ModelInNames $names) {
        Write-Step "modelo $Model ya disponible en el auxiliar"
      } else {
        Write-Step "descargando modelo $Model en el auxiliar (puede tardar)..."
        docker exec $auxName ollama pull $Model
      }
      $OllamaBase = "http://127.0.0.1:$AuxPort"
    }

    $logF = Join-Path $dataDir 'server.log'
    $errF = Join-Path $dataDir 'server.err.log'
    $pidF = Join-Path $dataDir 'server.pid'
    if (-not $PortReused) {
      Write-Step "arrancando el servidor local (python -m rag.server) en :$Port ..."
      $env:OLLAMA_HOST = $OllamaBase
      $env:RAG_HOST = '127.0.0.1'
      $env:RAG_PORT = "$Port"
      $env:RAG_PROJECT = $Project
      $env:RAG_COLLECTION = $Expected
      $p = Start-Process -FilePath $py -ArgumentList @('-m', 'rag.server') -WorkingDirectory $Dest `
        -RedirectStandardOutput $logF -RedirectStandardError $errF -PassThru -NoNewWindow
      "$($p.Id)" | Set-Content $pidF
      $up = $false
      for ($i = 0; $i -lt 60; $i++) {
        if (Get-HealthObject $Port) { $up = $true; break }
        Start-Sleep -Seconds 2
      }
      if (-not $up) {
        Show-Logs
        if (Test-Path $errF) { Get-Content $errF -Tail 40 | ForEach-Object { Write-Host $_ } }
        Stop-Rag "el servidor local no respondio (log: $logF)"
      }
    } else {
      Write-Step "servidor local ya activo - omito el arranque"
    }
  }

  Assert-Identity $Port
  Write-Step "identidad OK en :$Port (collection=$Expected)"

  Invoke-ReindexAndWait $Port $Force
  Write-State $Mode $Port
  Invoke-Smoke $Port

  Write-Host @"
========================================================================
  RAG listo en $Dest
  Modo:        $Mode
  Puerto:      $Port
  Coleccion:   $Expected
  Ollama:      $Strategy $(if ($OllamaBase) { "($OllamaBase)" })
  Indice:      drift=0 y smoke test superados
  Estado:      .\setup.ps1 $Dest -Status
  Parar:       .\setup.ps1 $Dest -Down
  Health:      Invoke-RestMethod http://localhost:$Port/health
  Indexar:     Invoke-RestMethod -Method Post http://localhost:$Port/reindex
  Stats:       Invoke-RestMethod http://localhost:$Port/stats
  (Re)config:  editar $Dest\rag.config.json y reiniciar
========================================================================
"@ -ForegroundColor Green
} finally {
  if ($TmpRag -and (Test-Path $TmpRag)) { Remove-Item -Recurse -Force $TmpRag }
}
