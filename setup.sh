#!/usr/bin/env bash
# =============================================================================
# RAG portable · setup.sh  (v2)
#
# Monta el RAG en un proyecto con UNA sola ejecución: copia el motor, genera
# la config, arranca Ollama + servicio, autoindexa hasta drift=0 y valida con
# un smoke test (/health + /methodology + /query).
#
# Uso:
#   ./setup.sh [<proyecto-destino>] [opciones]
#
# Opciones:
#   <proyecto-destino>   ruta al proyecto a indexar          (default: ".")
#   --force              reindex force=true (re-embebe todo; LENTO en CPU)
#   --dry-run            prepara archivos y valida el compose SIN arrancar
#                        nada ni descargar imágenes
#   --docker             usar Docker (default): compose con ollama + rag
#   --local              sin Docker de servicio: venv + `python -m rag.server`
#                        (Ollama: el del sistema, o un contenedor auxiliar)
#   --down               para el servicio del proyecto (no borra volúmenes)
#   --status             informa del estado (salta si el servicio está caído)
#   --port N             puerto inicial; si está ocupado autoincrementa (≤20)
#   --no-agents          no copiar la plantilla AGENTS.md
#   -h, --help           esta ayuda
#
# Variables opcionales (env):
#   RAG_REPO             URL git del repo RAG (solo si este script está fuera)
#   RAG_PORT             puerto inicial                        (default 8765)
#   RAG_EMBED_MODEL      modelo de embeddings                  (default bge-m3)
#   RAG_EMBED_DIM        dimensión del vector                  (default 1024)
#   RAG_COLLECTION       override de la colección Chroma       (default <slug>_chunks)
#   RAG_INDEX_TIMEOUT    seg. máx. esperando el reindex        (default 1800)
#   RAG_OLLAMA_VOLUME    volumen Docker ya poblado con el modelo (p.ej.
#                        rag-visionrt_ollama_data); ahorra el pull de bge-m3
#   OLLAMA_HOST          URL de un Ollama existente            (default autodetecta)
#
# Comportamiento clave (v2):
#   * Puerto: si algo responde en el puerto y su config.collection ES la de
#     este proyecto → se reutiliza; si es otro servicio → se incrementa el
#     puerto (hasta +20) y se re-valida la identidad tras arrancar.
#   * Ollama: se reutiliza uno accesible con el modelo; si no, contenedor
#     (reusando RAG_OLLAMA_VOLUME si se indica). Modo --local: mismo criterio
#     con contenedor auxiliar en 11434 o el CLI `ollama` del sistema.
#   * Reindex: espera con timeout, tolera 409 (reindex en curso) y exige
#     drift=0; ante error imprime los últimos logs.
#   * Copia limpia: nunca arrastra data/ (Chroma), __pycache__ ni .venv.
# =============================================================================
set -euo pipefail

# --- helpers -----------------------------------------------------------------
C_B=$'\033[1;34m'; C_Y=$'\033[1;33m'; C_R=$'\033[1;31m'; C_G=$'\033[1;32m'; C_0=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$C_B" "$C_0" "$*"; }
warn() { printf '%saviso:%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$C_R" "$C_0" "$*" >&2; exit 1; }

HAS_PY=0
command -v python3 >/dev/null 2>&1 && HAS_PY=1

jget() { # dotted.path — lee JSON de stdin (python3 si existe; si no, heurística)
  local path="$1"
  if [ "$HAS_PY" = 1 ]; then
    python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    for k in sys.argv[1].split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    if d is True: print("true")
    elif d is False: print("false")
    elif d is None: print("null")
    elif isinstance(d, (dict, list)): pass
    else: print(d)
except Exception:
    pass' "$path"
  else
    local key="${path##*.}"
    grep -oE "\"$key\"[[:space:]]*:[[:space:]]*(true|false|null|\"[^\"]*\"|[0-9][0-9.]*)" \
      | head -1 | sed -E 's/^[^:]*:[[:space:]]*//; s/^"(.*)"$/\1/' || true
  fi
}

url_slug() {
  local b
  b="$(basename "$1")"
  b="${b// /-}"
  b="$(printf '%s' "$b" | tr -c 'a-zA-Z0-9_-' '_' | tr 'A-Z' 'a-z')"
  b="${b#_}"; b="${b%_}"
  [ -n "$b" ] && printf '%s' "$b" || printf 'proyecto'
}

yaml_q() { printf '%s' "${1//\"/\\\"}"; }

probe_port() { # puerto → free | ours | other
  local p="$1" out rc coll
  out="$(curl -s -m 2 "http://127.0.0.1:${p}/health" 2>/dev/null)" && rc=0 || rc=$?
  if [ "$rc" = 7 ] || [ "$rc" = 6 ]; then printf 'free\n'; return 0; fi
  if [ "$rc" != 0 ]; then printf 'other\n'; return 0; fi
  coll="$(printf '%s' "$out" | jget config.collection)"
  if [ -n "$coll" ] && [ "$coll" = "$EXPECTED_COLLECTION" ]; then
    printf 'ours\n'
  else
    printf 'other\n'
  fi
}

pick_port() { # base → PORT / PORT_REUSED (autoincrementa hasta 20)
  local base="$1" p st i
  PORT_REUSED=0
  for i in $(seq 0 19); do
    p=$((base + i))
    st="$(probe_port "$p")"
    case "$st" in
      free)  PORT="$p"; return 0 ;;
      ours)  PORT="$p"; PORT_REUSED=1
             log "puerto $p ya sirve ESTE proyecto (collection=$EXPECTED_COLLECTION) — reutilizando"; return 0 ;;
      other) log "puerto $p ocupado por otro servicio → probando $((p + 1))" ;;
    esac
  done
  die "ningún puerto libre en $base..$((base + 19)). Usa --port N o libera puertos"
}

wait_http() { # url intentos intervalo
  local url="$1" n="${2:-60}" iv="${3:-2}" i
  for i in $(seq 1 "$n"); do
    curl -fsS -m 3 "$url" >/dev/null 2>&1 && return 0
    sleep "$iv"
  done
  return 1
}

wait_healthy() { # contenedor intentos intervalo
  local c="$1" n="${2:-60}" iv="${3:-2}" st i
  for i in $(seq 1 "$n"); do
    st="$(docker inspect --format '{{.State.Health.Status}}' "$c" 2>/dev/null || printf starting)"
    [ "$st" = healthy ] && return 0
    [ "$st" = unhealthy ] && die "el contenedor $c quedó unhealthy"
    sleep "$iv"
  done
  return 1
}

wait_rag_up() { # puerto contenedor — espera /health y detecta crash rápido
  local port="$1" c="$2" i st
  for i in $(seq 1 90); do
    curl -fsS -m 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
    if [ -n "$c" ]; then
      st="$(docker inspect --format '{{.State.Running}} {{.State.ExitCode}}' "$c" 2>/dev/null || printf 'true 0')"
      if [ "${st%% *}" = "false" ]; then
        dump_logs
        die "el contenedor $c terminó (exit ${st##* })"
      fi
    fi
    sleep 2
  done
  dump_logs
  die "el RAG no responde en :${port} (180s)"
}

dump_logs() {
  printf '%s--- logs ---%s\n' "$C_Y" "$C_0" >&2
  if [ "${MODE:-docker}" = "docker" ] && [ -n "${SLUG:-}" ]; then
    docker logs --tail 40 "rag-${SLUG}-rag" 2>&1 | tail -40 >&2 || true
  fi
  if [ -n "${DEST:-}" ] && [ -f "$DEST/rag/data/server.log" ]; then
    tail -40 "$DEST/rag/data/server.log" >&2 || true
  fi
  printf '%s------------%s\n' "$C_Y" "$C_0" >&2
}

ollama_probe() { # base_url → down | nomodel | model
  local base="${1%/}" tags
  tags="$(curl -s -m 3 "$base/api/tags" 2>/dev/null)" || true
  [ -n "$tags" ] || { printf 'down\n'; return 0; }
  if [ "$HAS_PY" = 1 ]; then
    printf '%s' "$tags" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    names = [m.get("name", "") for m in d.get("models", [])]
    w = sys.argv[1]
    print("model" if any(n == w or n.startswith(w + ":") for n in names) else "nomodel")
except Exception:
    print("down")' "$MODEL"
  else
    if printf '%s' "$tags" | grep -Eq "\"${MODEL}(:[^\"]*)?\""; then
      printf 'model\n'
    else
      printf 'nomodel\n'
    fi
  fi
}

model_in_names() { # stdin: nombres de modelos → 0 si está el modelo
  if [ "$HAS_PY" = 1 ]; then
    python3 -c '
import sys
w = sys.argv[1]
names = [l.strip() for l in sys.stdin if l.strip()]
sys.exit(0 if any(n == w or n.startswith(w + ":") for n in names) else 1)' "$MODEL"
  else
    grep -F "$MODEL" >/dev/null
  fi
}

free_port_from() { # inicio rango → primer puerto libre (TCP rechazado)
  local start="$1" p i rc
  for i in $(seq 0 9); do
    p=$((start + i))
    curl -s -m 1 "http://127.0.0.1:${p}/" >/dev/null 2>&1
    rc=$?
    if [ "$rc" = 7 ] || [ "$rc" = 6 ]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

check_identity() { # puerto
  local h coll
  h="$(curl -s -m 5 "http://127.0.0.1:$1/health" 2>/dev/null || true)"
  coll="$(printf '%s' "$h" | jget config.collection)"
  [ "$coll" = "$EXPECTED_COLLECTION" ] && return 0
  die "identidad no coincide: servidor='${coll:-ninguna}' esperada='$EXPECTED_COLLECTION' (¿otro RAG en el puerto $1?)"
}

write_state() { # modo puerto
  mkdir -p "$DEST/rag/data"
  local prev_aux=""
  prev_aux="$(read_state AUX_PORT 2>/dev/null || true)"
  {
    printf 'MODE=%s\n' "$1"
    printf 'PORT=%s\n' "$2"
    printf 'COLLECTION=%s\n' "$EXPECTED_COLLECTION"
    printf 'SLUG=%s\n' "$SLUG"
    printf 'AUX_PORT=%s\n' "${AUX_PORT:-${prev_aux:-}}"
    printf 'OLLAMA_BASE=%s\n' "${OLLAMA_BASE:-}"
  } > "$DEST/rag/data/setup.state"
}

read_state() { # clave → valor (si existe)
  [ -f "$DEST/rag/data/setup.state" ] || return 1
  grep -E "^$1=" "$DEST/rag/data/setup.state" | head -1 | cut -d= -f2-
}

usage() { sed -n '3,47p' "$0" | sed 's/^# \{0,1\}//'; }

# --- argumentos --------------------------------------------------------------
DEST="." ; FORCE=0 ; DRY=0 ; MODE="docker" ; DO_DOWN=0 ; DO_STATUS=0
NO_AGENTS=0 ; PORT_BASE="" ; EXPECTED_COLLECTION="" ; PORT_REUSED=0
SLUG="" ; PORT="" ; AUX_PORT="" ; OLLAMA_BASE="" ; STRATEGY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force)       FORCE=1 ;;
    --dry-run)     DRY=1 ;;
    --local)       MODE="local" ;;
    --docker)      MODE="docker" ;;
    --down)        DO_DOWN=1 ;;
    --status)      DO_STATUS=1 ;;
    --port)        [ $# -ge 2 ] || die "--port requiere un valor"; PORT_BASE="$2"; shift ;;
    --port=*)      PORT_BASE="${1#--port=}" ;;
    --no-agents)   NO_AGENTS=1 ;;
    -h|--help)     usage; exit 0 ;;
    -*)            die "flag desconocido: $1 (ver --help)" ;;
    *)             DEST="$1" ;;
  esac
  shift
done
[ "$DO_DOWN" = 1 ] && [ "$DO_STATUS" = 1 ] && die "--down y --status no se pueden combinar"

command -v curl >/dev/null 2>&1 || die "curl no está instalado"

# --- modo --status -----------------------------------------------------------
if [ "$DO_STATUS" = 1 ]; then
  [ -d "$DEST" ] || die "no existe el proyecto $DEST"
  DEST="$(cd "$DEST" && pwd)"
  local_port="$(read_state PORT 2>/dev/null || true)"
  local_mode="$(read_state MODE 2>/dev/null || true)"
  if [ -z "${local_port:-}" ] && [ -f "$DEST/docker-compose.rag.yml" ]; then
    local_port="$(grep -oE '"[0-9]+:8765"' "$DEST/docker-compose.rag.yml" | head -1 | tr -d '"' | cut -d: -f1 || true)"
  fi
  local_port="${local_port:-8765}"; local_mode="${local_mode:-docker}"
  log "proyecto: $DEST (modo=$local_mode, puerto=$local_port)"
  h="$(curl -s -m 5 "http://127.0.0.1:${local_port}/health" 2>/dev/null || true)"
  if [ -z "$h" ]; then
    printf 'servicio:  caído (no responde en :%s)\n' "$local_port"
    if [ "$local_mode" = "local" ]; then
      printf 'arranque:  cd %s && rag/.venv/bin/python -m rag.server\n' "$DEST"
    else
      printf 'arranque:  docker compose -f %s/docker-compose.rag.yml up -d\n' "$DEST"
    fi
    exit 1
  fi
  s="$(curl -s -m 5 "http://127.0.0.1:${local_port}/stats" 2>/dev/null || true)"
  printf 'servicio:  activo\n'
  printf 'proyecto:  %s\n' "$(printf '%s' "$h" | jget config.project)"
  printf 'coleccion: %s\n' "$(printf '%s' "$h" | jget config.collection)"
  printf 'ollama:    %s (%s)\n' "$(printf '%s' "$h" | jget ollama.ok)" "$(printf '%s' "$h" | jget ollama.ollama_host)"
  printf 'chunks:    chroma=%s manifest=%s drift=%s\n' \
    "$(printf '%s' "$s" | jget chroma_chunks)" \
    "$(printf '%s' "$s" | jget manifest_chunks)" \
    "$(printf '%s' "$s" | jget drift)"
  printf 'archivos:  %s\n' "$(printf '%s' "$s" | jget manifest_files)"
  printf 'reindex:   running=%s status=%s\n' \
    "$(printf '%s' "$s" | jget reindex.running)" "$(printf '%s' "$s" | jget reindex.status)"
  printf 'grafo:     themes=%s tokens=%s files=%s edges=%s\n' \
    "$(printf '%s' "$h" | jget graph.categories)" "$(printf '%s' "$h" | jget graph.tokens)" \
    "$(printf '%s' "$h" | jget graph.files)" "$(printf '%s' "$h" | jget graph.edges)"
  exit 0
fi

# --- modo --down -------------------------------------------------------------
if [ "$DO_DOWN" = 1 ]; then
  [ -d "$DEST" ] || die "no existe el proyecto $DEST"
  DEST="$(cd "$DEST" && pwd)"
  local_mode="$(read_state MODE 2>/dev/null || true)"
  if [ -z "${local_mode:-}" ]; then
    if [ -f "$DEST/docker-compose.rag.yml" ]; then local_mode=docker; else local_mode=local; fi
  fi
  stopped=0
  if [ "$local_mode" = "local" ]; then
    if [ -f "$DEST/rag/data/server.pid" ]; then
      pid="$(cat "$DEST/rag/data/server.pid")"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null || true
        log "servidor local (pid $pid) detenido"
        stopped=1
      fi
      rm -f "$DEST/rag/data/server.pid"
    fi
    if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx 'rag-ollama-aux'; then
      if [ "$(read_state AUX_PORT 2>/dev/null || true)" != "" ]; then
        docker stop rag-ollama-aux >/dev/null 2>&1 || true
        log "contenedor auxiliar rag-ollama-aux detenido (volumen intacto)"
        stopped=1
      fi
    fi
  else
    if [ -f "$DEST/docker-compose.rag.yml" ]; then
      if docker compose version >/dev/null 2>&1; then DC=(docker compose)
      elif command -v docker-compose >/dev/null 2>&1; then DC=(docker-compose)
      else die "docker compose no está disponible"; fi
      "${DC[@]}" -f "$DEST/docker-compose.rag.yml" down --remove-orphans
      stopped=1
    fi
  fi
  [ "$stopped" = 1 ] || warn "nada que parar en $DEST"
  log "volumenes y datos de índice NO se borran (borra a mano si lo necesitas)"
  exit 0
fi

# --- prerequisitos según modo ------------------------------------------------
PORT_BASE="${PORT_BASE:-${RAG_PORT:-8765}}"
[[ "$PORT_BASE" =~ ^[0-9]+$ ]] || die "--port/RAG_PORT debe ser numérico"
MODEL="${RAG_EMBED_MODEL:-bge-m3}"
EMBED_DIM="${RAG_EMBED_DIM:-1024}"
RAG_INDEX_TIMEOUT="${RAG_INDEX_TIMEOUT:-1800}"
[[ "$RAG_INDEX_TIMEOUT" =~ ^[0-9]+$ ]] || die "RAG_INDEX_TIMEOUT debe ser numérico (segundos)"
COMPOSE_NAME="docker-compose.rag.yml"

if [ "$MODE" = "docker" ]; then
  command -v docker >/dev/null 2>&1 || die "docker no está instalado (o usa --local)"
  if docker compose version >/dev/null 2>&1; then DC=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then DC=(docker-compose)
  else die "docker compose (v2) no está disponible (o usa --local)"; fi
else
  command -v python3 >/dev/null 2>&1 || die "python3 no está instalado (modo --local)"
fi

# --- localizar el repo del RAG -----------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/rag/server.py" ] && [ -f "$SCRIPT_DIR/rag/config.py" ]; then
  RAG_SRC="$SCRIPT_DIR"
  TMP_RAG=""
  log "repositorio del RAG: $RAG_SRC"
elif [ -n "${RAG_REPO:-}" ]; then
  TMP_RAG="$(mktemp -d)"
  trap 'rm -rf "$TMP_RAG"' EXIT
  log "clonando $RAG_REPO ..."
  git clone --quiet --depth 1 "$RAG_REPO" "$TMP_RAG/rag-portable"
  RAG_SRC="$TMP_RAG/rag-portable"
else
  die "no encuentro el repo del RAG. Copia setup.sh dentro de RAG_portable, o exporta RAG_REPO=<url-git>"
fi

# --- proyecto destino ---------------------------------------------------------
[ -d "$DEST" ] || mkdir -p "$DEST" || die "no se pudo crear $DEST"
DEST="$(cd "$DEST" && pwd)"
SLUG="$(url_slug "$DEST")"
PROJECT="$(basename "$DEST")"
EXPECTED_COLLECTION="${RAG_COLLECTION:-${SLUG}_chunks}"

log "proyecto: $DEST (slug=$SLUG, modo=$MODE, colección=$EXPECTED_COLLECTION)"

# --- reutilizar servidor local previo (modo --local) --------------------------
if [ "$MODE" = "local" ]; then
  OLD_PORT="$(read_state PORT 2>/dev/null || true)"
  PIDF="$DEST/rag/data/server.pid"
  if [ -f "$PIDF" ]; then
    old_pid="$(cat "$PIDF" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
      old_st="$(probe_port "${OLD_PORT:-0}" 2>/dev/null || printf other)"
      if [ "$old_st" = ours ] && [ -n "$OLD_PORT" ]; then
        PORT="$OLD_PORT"; PORT_REUSED=1
        log "servidor local previo (pid $old_pid) sigue activo en :$PORT — reutilizando"
      else
        warn "matando servidor local previo (pid $old_pid) que ya no responde"
        kill "$old_pid" 2>/dev/null || true
        sleep 2
        rm -f "$PIDF"
      fi
    else
      rm -f "$PIDF"
    fi
  fi
fi

# --- selección de puerto ------------------------------------------------------
if [ "$PORT_REUSED" != 1 ]; then
  pick_port "$PORT_BASE"
  [ "$PORT" != "$PORT_BASE" ] && log "puerto elegido: $PORT"
fi

# --- estrategia Ollama --------------------------------------------------------
resolve_ollama_docker() {
  STRATEGY="container"; OLLAMA_BASE=""
  local cand probe
  for cand in ${OLLAMA_HOST:+"$OLLAMA_HOST"} "http://127.0.0.1:11434"; do
    probe="$(ollama_probe "$cand")"
    if [ "$probe" = model ]; then
      STRATEGY="host"; OLLAMA_BASE="$cand"
      log "reutilizando Ollama del host: $cand (modelo $MODEL ya presente)"
      return 0
    elif [ "$probe" = nomodel ]; then
      warn "Ollama en $cand accesible pero SIN modelo $MODEL — uso un contenedor aparte"
    fi
  done
  if [ -n "${RAG_OLLAMA_VOLUME:-}" ]; then
    log "Ollama en contenedor con volumen externo: $RAG_OLLAMA_VOLUME (modelo ya descargado)"
  else
    log "Ollama en contenedor (primer arranque descarga la imagen + modelo)"
  fi
}

resolve_ollama_local() {
  local cand probe
  OLLAMA_BASE=""
  for cand in ${OLLAMA_HOST:+"$OLLAMA_HOST"} "http://127.0.0.1:11434"; do
    probe="$(ollama_probe "$cand")"
    if [ "$probe" = model ]; then
      OLLAMA_BASE="$cand"; STRATEGY="host"
      log "usando Ollama existente: $cand"; return 0
    elif [ "$probe" = nomodel ]; then
      if command -v ollama >/dev/null 2>&1; then
        log "descargando modelo $MODEL en $cand (CLI ollama)…"
        OLLAMA_HOST="$cand" ollama pull "$MODEL" || warn "no pude descargar $MODEL vía CLI"
        if [ "$(ollama_probe "$cand")" = model ]; then
          OLLAMA_BASE="$cand"; STRATEGY="host"; return 0
        fi
      fi
    fi
  done
  if command -v docker >/dev/null 2>&1; then
    STRATEGY="aux"; return 0
  fi
  if command -v ollama >/dev/null 2>&1; then
    log "iniciando el daemon del sistema (ollama serve)…"
    nohup ollama serve >"$DEST/rag/data/ollama-serve.log" 2>&1 &
    if wait_http "http://127.0.0.1:11434/api/tags" 15 2; then
      OLLAMA_HOST="http://127.0.0.1:11434" ollama pull "$MODEL" || warn "no pude descargar $MODEL"
      OLLAMA_BASE="http://127.0.0.1:11434"; STRATEGY="host"; return 0
    fi
  fi
  die "no hay Ollama accesible ni Docker. Instala uno de ellos:
  · Ollama:  curl -fsSL https://ollama.com/install.sh | sh   (luego: ollama pull $MODEL)
  · Docker:  https://docs.docker.com/get-docker/             (el script lanza un Ollama contenedor)"
}

mkdir -p "$DEST/rag/data"
if [ "$MODE" = "docker" ]; then
  resolve_ollama_docker
else
  resolve_ollama_local
fi

# --- copiar el motor (copia limpia: sin data/ ni caché) -----------------------
log "copiando motor → $DEST/rag/"
mkdir -p "$DEST/rag"
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude 'data/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.venv/' \
    "$RAG_SRC/rag/" "$DEST/rag/"
else
  (cd "$RAG_SRC/rag" && tar cf - \
    --exclude='./data' --exclude='./__pycache__' --exclude='*.pyc' --exclude='./.venv' .
  ) | (cd "$DEST/rag" && tar xf -)
fi
mkdir -p "$DEST/rag/data"

if [ "$NO_AGENTS" != 1 ] && [ ! -f "$DEST/AGENTS.md" ] && [ -f "$RAG_SRC/AGENTS.md" ]; then
  sed "s/8765/${PORT}/g" "$RAG_SRC/AGENTS.md" > "$DEST/AGENTS.md"
  log "plantilla AGENTS.md creada con el puerto $PORT"
fi

# --- generar docker-compose.rag.yml (solo modo docker) ------------------------
if [ "$MODE" = "docker" ]; then
  PROJECT_Q="$(yaml_q "$PROJECT")"
  COLLECTION_Q="$(yaml_q "$EXPECTED_COLLECTION")"

  if [ "$STRATEGY" = "host" ]; then
    OLLAMA_URL="$OLLAMA_BASE"
    case "$OLLAMA_URL" in
      http://127.0.0.1:*|http://localhost:*)
        OLLAMA_URL="http://host.docker.internal:${OLLAMA_URL##*:}" ;;
    esac
    SERVICES_EXTRA="    extra_hosts:
      - \"host.docker.internal:host-gateway\"
"
    OLLAMA_ENV="$OLLAMA_URL"
    DEP_BLOCK=""
    OLLAMA_SVC=""
    VOL_OLLAMA=""
    [ -n "${RAG_OLLAMA_VOLUME:-}" ] && warn "RAG_OLLAMA_VOLUME se ignora: se usa el Ollama del host"
  else
    SERVICES_EXTRA=""
    OLLAMA_ENV="http://ollama:11434"
    DEP_BLOCK="    depends_on:
      ollama:
        condition: \"service_healthy\"
"
    OLLAMA_SVC="  ollama:
    image: \"ollama/ollama:latest\"
    container_name: \"rag-${SLUG}-ollama\"
    restart: \"unless-stopped\"
    healthcheck:
      test: [\"CMD\", \"/bin/sh\", \"-c\", \"ollama list >/dev/null 2>&1 || exit 1\"]
      interval: \"10s\"
      timeout: \"5s\"
      retries: \"5\"
    volumes:
      - \"ollama_data:/root/.ollama\"

"
    if [ -n "${RAG_OLLAMA_VOLUME:-}" ]; then
      VOL_OLLAMA="  ollama_data:
    external: true
    name: \"$(yaml_q "$RAG_OLLAMA_VOLUME")\""
    else
      VOL_OLLAMA="  ollama_data:"
    fi
  fi

  cat > "$DEST/$COMPOSE_NAME" <<YAML
name: "rag-${SLUG}"
services:
${OLLAMA_SVC}  rag:
    build: "./rag"
    image: "rag-${SLUG}-rag"
    container_name: "rag-${SLUG}-rag"
    restart: "unless-stopped"
    ports:
      - "${PORT}:8765"
    environment:
      OLLAMA_HOST: "${OLLAMA_ENV}"
      RAG_HOST: "0.0.0.0"
      RAG_PORT: "8765"
      RAG_EMBED_MODEL: "${MODEL}"
      RAG_EMBED_DIM: "${EMBED_DIM}"
      RAG_PROJECT: "${PROJECT_Q}"
      RAG_COLLECTION: "${COLLECTION_Q}"
${SERVICES_EXTRA}    volumes:
      - ".:/app"
      - "rag_data:/app/rag/data"
    working_dir: "/app"
${DEP_BLOCK}
volumes:
${VOL_OLLAMA}
  rag_data:
YAML
  log "generado $COMPOSE_NAME (puerto $PORT, ollama=$STRATEGY)"
fi

# --- modo dry-run --------------------------------------------------------------
if [ "$DRY" = 1 ]; then
  log "dry-run — nada se ha arrancado ni descargado"
  if [ "$MODE" = "docker" ]; then
    "${DC[@]}" -f "$DEST/$COMPOSE_NAME" config >/dev/null || die "compose inválido"
    log "compose válido: $DEST/$COMPOSE_NAME"
    if [ -n "${RAG_OLLAMA_VOLUME:-}" ] && [ "$STRATEGY" = container ]; then
      docker volume inspect "$RAG_OLLAMA_VOLUME" >/dev/null 2>&1 \
        || warn "el volumen externo '$RAG_OLLAMA_VOLUME' no existe: créalo antes del up (docker volume create ...)"
    fi
  else
    python3 -c 'import venv' 2>/dev/null || die "python3 sin módulo venv"
    log "modo local: venv en rag/.venv + python -m rag.server en :$PORT"
  fi
  cat <<PLAN
  plan (no ejecutado):
    proyecto:    $DEST
    modo:        $MODE
    puerto:      $PORT
    coleccion:   $EXPECTED_COLLECTION
    ollama:      ${STRATEGY}${OLLAMA_BASE:+ en $OLLAMA_BASE}
    force:       $FORCE
    timeout:     ${RAG_INDEX_TIMEOUT}s
PLAN
  exit 0
fi

# --- arrancar servicios --------------------------------------------------------
if [ "$MODE" = "docker" ]; then
  if [ "$STRATEGY" = "container" ]; then
    if [ -n "${RAG_OLLAMA_VOLUME:-}" ]; then
      docker volume inspect "$RAG_OLLAMA_VOLUME" >/dev/null 2>&1 \
        || die "el volumen externo '$RAG_OLLAMA_VOLUME' no existe ( créalo o usa otro )"
    fi
    log "arrancando Ollama (contenedor)…"
    "${DC[@]}" -f "$DEST/$COMPOSE_NAME" up -d ollama
    wait_healthy "rag-${SLUG}-ollama" || die "Ollama no arrancó a tiempo"
    if docker exec "rag-${SLUG}-ollama" ollama list 2>/dev/null | model_in_names; then
      log "modelo $MODEL ya disponible en el volumen"
    else
      log "descargando modelo $MODEL (puede tardar)…"
      docker exec "rag-${SLUG}-ollama" ollama pull "$MODEL"
    fi
  fi
  log "arrancando el servicio RAG en :$PORT …"
  "${DC[@]}" -f "$DEST/$COMPOSE_NAME" up -d --build --force-recreate rag
  wait_rag_up "$PORT" "rag-${SLUG}-rag"
else
  # --- modo local: venv + python -m rag.server --------------------------------
  VENV="$DEST/rag/.venv"
  PY="$VENV/bin/python"
  if [ ! -x "$PY" ]; then
    log "creando venv en rag/.venv …"
    python3 -m venv "$VENV" || die "no pude crear el venv (¿python3-venv instalado?)"
  fi
  log "instalando dependencias (requirements.txt)…"
  if ! "$VENV/bin/pip" install --disable-pip-version-check -q \
      -r "$DEST/rag/requirements.txt" >"$DEST/rag/data/pip-install.log" 2>&1; then
    tail -30 "$DEST/rag/data/pip-install.log" >&2 || true
    die "pip install falló (log: $DEST/rag/data/pip-install.log)"
  fi

  if [ "$STRATEGY" = "aux" ]; then
    log "lanzando Ollama auxiliar en contenedor…"
    AUX_PORT=""
    if docker ps -a --format '{{.Names}}' | grep -qx 'rag-ollama-aux'; then
      cur="$(docker port rag-ollama-aux 11434/tcp 2>/dev/null | head -1 | awk -F: '{print $NF}' || true)"
      if [ -n "$cur" ] && [ "$(ollama_probe "http://127.0.0.1:$cur")" = model ]; then
        AUX_PORT="$cur"
        docker start rag-ollama-aux >/dev/null 2>&1 || true
        log "contenedor auxiliar ya existente en :$AUX_PORT con el modelo — reutilizado"
      else
        docker rm -f rag-ollama-aux >/dev/null 2>&1 || true
      fi
    fi
    if [ -z "$AUX_PORT" ]; then
      AUX_PORT="$(free_port_from 11434 || printf 11434)"
      docker run -d --name rag-ollama-aux \
        -p "${AUX_PORT}:11434" \
        -v "${RAG_OLLAMA_VOLUME:-rag_ollama_aux_data}:/root/.ollama" \
        --restart unless-stopped \
        ollama/ollama:latest >/dev/null
    fi
    wait_http "http://127.0.0.1:${AUX_PORT}/api/tags" 60 2 || die "el Ollama auxiliar no respondió"
    if [ "$(ollama_probe "http://127.0.0.1:${AUX_PORT}")" = model ]; then
      log "modelo $MODEL ya disponible en el auxiliar"
    else
      log "descargando modelo $MODEL en el auxiliar (puede tardar)…"
      docker exec rag-ollama-aux ollama pull "$MODEL"
    fi
    OLLAMA_BASE="http://127.0.0.1:${AUX_PORT}"
  fi

  PIDF="$DEST/rag/data/server.pid"
  LOGF="$DEST/rag/data/server.log"
  if [ "$PORT_REUSED" != 1 ]; then
    log "arrancando el servidor local (python -m rag.server) en :$PORT …"
    (
      cd "$DEST" || exit 1
      env OLLAMA_HOST="$OLLAMA_BASE" \
          RAG_HOST=127.0.0.1 \
          RAG_PORT="$PORT" \
          RAG_PROJECT="$PROJECT" \
          RAG_COLLECTION="$EXPECTED_COLLECTION" \
          nohup "$PY" -m rag.server >"$LOGF" 2>&1 &
      echo $! > "$PIDF"
    )
    wait_http "http://127.0.0.1:${PORT}/health" 60 2 || { dump_logs; die "el servidor local no respondió (log: $LOGF)"; }
  else
    log "servidor local ya activo — omito el arranque"
  fi
fi

check_identity "$PORT"
log "identidad OK en :$PORT (collection=$EXPECTED_COLLECTION)"

# --- indexar y esperar (timeout + 409 + drift=0) ------------------------------
do_reindex() {
  local url resp code body deadline stats running status drift err fails=0
  url="http://127.0.0.1:${PORT}/reindex"
  [ "$FORCE" = 1 ] && { url="${url}?force=true"; log "force=true — re-embebiendo todo (lento en CPU)"; }
  resp="$(curl -s -m 15 -w $'\n%{http_code}' -X POST "$url" 2>/dev/null || true)"
  code="$(printf '%s' "$resp" | tail -n1)"
  body="$(printf '%s' "$resp" | sed '$d')"
  case "$code" in
    200) log "reindex lanzado (incremental)" ;;
    409) warn "ya hay un reindex en curso (409) — esperando a que termine" ;;
    *)   dump_logs; die "POST /reindex → HTTP ${code:-sin respuesta}: ${body}" ;;
  esac

  deadline=$((SECONDS + RAG_INDEX_TIMEOUT))
  printf 'esperando el índice '
  while :; do
    stats="$(curl -s -m 5 "http://127.0.0.1:${PORT}/stats" 2>/dev/null || true)"
    if [ -z "$stats" ]; then
      fails=$((fails + 1))
      [ "$fails" -ge 20 ] && { printf '\n'; dump_logs; die "/stats dejó de responder"; }
      printf '.'; sleep 3; continue
    fi
    fails=0
    running="$(printf '%s' "$stats" | jget reindex.running)"
    status="$(printf '%s' "$stats" | jget reindex.status)"
    [ "$running" = false ] && break
    if [ "$status" = error ]; then
      err="$(printf '%s' "$stats" | jget reindex.error)"
      printf '\n'; dump_logs; die "la indexación falló: ${err:-error desconocido}"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      printf '\n'; dump_logs
      die "timeout de ${RAG_INDEX_TIMEOUT}s esperando el reindex (sube RAG_INDEX_TIMEOUT)"
    fi
    printf '.'
    sleep 3
  done
  printf '\n'

  status="$(printf '%s' "$stats" | jget reindex.status)"
  if [ "$status" = error ]; then
    err="$(printf '%s' "$stats" | jget reindex.error)"
    dump_logs; die "la indexación falló: ${err:-error desconocido}"
  fi
  drift="$(printf '%s' "$stats" | jget drift)"
  log "índice: manifest=$(printf '%s' "$stats" | jget manifest_chunks) chroma=$(printf '%s' "$stats" | jget chroma_chunks) drift=$drift status=${status:-ok}"
  [ "${drift:-x}" = "0" ] || die "drift=$drift ≠ 0 — repite: curl -X POST http://localhost:${PORT}/reindex"
}

do_reindex
write_state "$MODE" "$PORT"

# --- smoke test ----------------------------------------------------------------
do_smoke() {
  local h ok oll coll meth mc chroma q file score
  h="$(curl -s -m 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)"
  [ -n "$h" ] || die "smoke: /health no responde"
  ok="$(printf '%s' "$h" | jget ok)"
  oll="$(printf '%s' "$h" | jget ollama.ok)"
  coll="$(printf '%s' "$h" | jget config.collection)"
  [ "$ok" = true ] || die "smoke: health.ok=false"
  [ "$oll" = true ] || die "smoke: ollama no disponible ($(printf '%s' "$h" | jget ollama.reason))"
  [ "$coll" = "$EXPECTED_COLLECTION" ] || die "smoke: identidad incorrecta ($coll)"
  log "smoke /health ✓ (colección=$coll, ollama ok, chunks=$(printf '%s' "$h" | jget collection.count))"

  meth="$(curl -s -m 5 "http://127.0.0.1:${PORT}/methodology" 2>/dev/null || true)"
  mc="$(printf '%s' "$meth" | jget count)"
  if [ "${mc:-0}" = 0 ]; then
    warn "/methodology vacío (¿sin AGENTS.md/README indexados?)"
  else
    log "smoke /methodology ✓ ($mc chunks del alma del proyecto)"
  fi

  chroma="$(curl -s -m 5 "http://127.0.0.1:${PORT}/stats" 2>/dev/null | jget chroma_chunks || true)"
  if [ "${chroma:-0}" = 0 ]; then
    warn "índice vacío (proyecto sin archivos indexables) — omito el smoke de /query"
    return 0
  fi
  q="$(curl -s -m 60 -X POST "http://127.0.0.1:${PORT}/query" \
        -H 'Content-Type: application/json' \
        -d '{"q":"README principal del proyecto","k":1}' 2>/dev/null || true)"
  [ -n "$q" ] || die "smoke: /query no responde"
  if ! printf '%s' "$q" | grep -q '"results"'; then
    die "smoke: /query → $(printf '%s' "$q" | jget detail)"
  fi
  file="$(printf '%s' "$q" | jget results.0.file)"
  score="$(printf '%s' "$q" | jget results.0.score)"
  [ -n "$file" ] || die "smoke: /query devolvió 0 resultados con un índice de $chroma chunks"
  log "smoke /query ✓ top-1: $file (score=${score:-n/a})"
}

do_smoke

# --- resumen --------------------------------------------------------------------
cat <<SUMMARY
${C_G}========================================================================
  RAG listo en $DEST
  Modo:        $MODE
  Puerto:      $PORT
  Colección:   $EXPECTED_COLLECTION
  Ollama:      $STRATEGY${OLLAMA_BASE:+ ($OLLAMA_BASE)}
  Índice:      drift=0 y smoke test superados
  Estado:      $0 $DEST --status
  Parar:       $0 $DEST --down
  Health:      curl http://localhost:$PORT/health
  Indexar:     curl -X POST http://localhost:$PORT/reindex
  Stats:       curl http://localhost:$PORT/stats
  (Re)config:  editar $DEST/rag.config.json y reiniciar
========================================================================${C_0}
SUMMARY
