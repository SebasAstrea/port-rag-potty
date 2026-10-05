# port-rag-potty

# RAG portable

> **RAG con ChromaDB + Ollama, «cópialo y funciona».** Sirve a tus agentes de
> IA (opencode, Claude, Cursor, Cline…) contexto curado de cualquier proyecto:
> medido sobre despliegues reales, reduce la sesión de agente de **~18-42k
> tokens** (documentación de onboarding) a **~6-9k**, con consultas sueltas de
> **~1,4-2,1k tokens** (ver [Benchmarks](#benchmarks-medido-con-datos-reales)).

Un directorio `rag/` + una plantilla `AGENTS.md` que se copian dentro de
**cualquier** proyecto. No está atado a ningún tipo de código (frontend, Python,
docs…) — la configuración por proyecto vive en un único `rag.config.json`
auto-generado.

---

## Qué resuelve

Un agente nuevo que entra a un repo desconocido hoy se lee README + docs al
completo (medido: **23k-43k tokens** de onboarding; el corpus íntegro llega a
**63k-264k**). Con el RAG hace llamadas HTTP de ~1,5-2k tokens y tiene:

- **`/methodology`** — el «alma» del proyecto: propósito, arquitectura,
  convenciones (chunks de README/AGENTS marcados como methodology).
- **`/query`** — búsqueda semántica sobre todo el código (embeddings bge-m3).
- **`/categories`** — grupos (temas CSS, módulos…) y sus tokens/archivos (opcional).

---

## Benchmarks (medido con datos reales)

Ejecutado el **2026-09-26** sobre los dos despliegues vivos — **KGuard** (:8766,
216 chunks / 26 archivos) y **VisionRT** (:8767, 727 chunks / 157 archivos) —
con 33 consultas reales curadas con ground-truth verificado. **Todas las
pruebas estadísticas** (Wilcoxon, test de signo, bootstrap, permutaciones,
Mann-Whitney) están en **[`docs/benchmarks/REPORT.md`](docs/benchmarks/REPORT.md)**
; los scripts para reproducirlo, en [`benchmarks/`](benchmarks/).

| Métrica | Resultado |
|---|---|
| **Ahorro de tokens por consulta** (mediana, n=33) | **+801 tok** (IC95 [−1, 1071]) — Wilcoxon **p = 0,0169** |
| Condición de rentabilidad | destino **≥ 2k tok → +1 121** (p = 2·10⁻⁶); destino **< 2k tok → −786** (conviene leer el archivo) |
| **Calidad de recuperación** | **hit@5 = 96,97 %**, MRR = 0,733 vs 0,06-0,11 del azar — permutación **p = 0,0002** |
| Sesión completa (alma + 1 consulta) | KGuard 18 214 → **6 258 tok (×2,9)** · VisionRT 41 767 → **9 328 (×4,5)** |
| Latencia `/query` (n=100/proyecto) | p50 ≈ 42 ms, **p95 ≤ 50 ms** |
| Embeddings bge-m3 (566,7M, F16, dim 1024) | **~30 ms/consulta**, 9,1 txt/s en CPU (batch sin beneficio) |

![Distribución del ahorro de tokens](docs/benchmarks/figs/fig02_savings_hist.png)

**Veredicto:** ahorra de forma **significativa y con calidad alta**, pero su
ahorro es **condicional al tamaño del destino** (cruza por cero en ~1,7-2k
tokens) —12 de33 consultas medidas gastan más que la lectura directa. El
antiguo claim «80-150k → 3-5k» estaba desviado en ambas direcciones (detalle
en el informe).

---

## Requisitos

- **Docker** (para la instalación automatizada) **o** Python 3.10+ (instalación manual)
- Ollama con el modelo de embeddings, por ejemplo `bge-m3`:
  ```bash
  ollama pull bge-m3
  ```
  La instalación automatizada usa un contenedor `ollama/ollama` y pullea el
  modelo por ti.

---

## Instalación

El repo trae scripts de automatización equivalente en **bash** (`setup.sh`) y
**PowerShell** (`setup.ps1`): clonan/buscan el repositorio del RAG, lo copian
al proyecto destino, montan los contenedores (Ollama + RAG), pullean el modelo,
crean la base Chroma e indexan — todo hasta que `/stats` da `drift = 0`.

### Opción A — Automatizada (recomendada, Docker)

```bash
# 1) Clona el repositorio del RAG (solo la primera vez)
git clone <url-de-este-repo> rag-portable
cd rag-portable

# 2) Instala el RAG en tu proyecto y déjalo indexado
./setup.sh /ruta/al/proyecto        # bash (Linux/macOS/WSL)
# o en Windows PowerShell:
.\setup.ps1 C:\ruta\al\proyecto
```

Parámetros y variables (en ambos scripts; flags en bash con `--`, en PowerShell con `-`):

| Flag / variable | Descripción | Default |
|---|---|---|
| `<proyecto>` | Directorio a indexar | `.` |
| `--force` / `-Force` | Reindex `force=true` (re-embebe todo; **lento** en CPU) | off |
| `--dry-run` / `-DryRun` | Prepara archivos y valida el compose sin arrancar nada | off |
| `--local` / `-Local` | Sin Docker de servicio: venv + `python -m rag.server` | off |
| `--down` / `-Down` | Para el servicio del proyecto (**no** borra volúmenes) | — |
| `--status` / `-Status` | Informa del estado (exit 1 si el servicio está caído) | — |
| `--port N` / `-Port N` | Puerto inicial; si está ocupado autoincrementa (≤20) | `8765` |
| `--no-agents` / `-NoAgents` | No copiar la plantilla `AGENTS.md` | off |
| `RAG_PORT` | Puerto inicial (equivale a `--port`) | `8765` |
| `RAG_EMBED_MODEL` | Modelo de embeddings a pullear/usar | `bge-m3` |
| `RAG_EMBED_DIM` | Dimensión del vector | `1024` |
| `RAG_COLLECTION` | Override del nombre de colección Chroma | `<slug>_chunks` |
| `RAG_INDEX_TIMEOUT` | Segundos máx. esperando el reindex | `1800` |
| `RAG_OLLAMA_VOLUME` | Volumen Docker ya poblado con el modelo (ahorra el pull; p.ej. `rag-visionrt_ollama_data`) | — |
| `RAG_REPO` | URL git del RAG (solo si el script está fuera del repo) | — |
| `OLLAMA_HOST` | URL de un Ollama existente a reutilizar | autodetecta |

Qué hace exactamente:

1. **Puerto seguro**: si en el puerto ya responde **este** proyecto (compara
   `config.collection`) lo reutiliza; si lo ocupa otro servicio, incrementa el
   puerto (hasta +20) y re-valida la identidad tras arrancar — nunca mandas un
   `/reindex` al RAG de otro proyecto.
2. Copia `rag/` de forma **limpia** (sin `data/`, `__pycache__` ni `.venv`;
   si destino == repo del RAG, omite la copia) + plantilla `AGENTS.md` con el
   puerto real (solo si el proyecto aún no tiene una). Conviene ignorar en tu
   `.gitignore` los artefactos de runtime: `rag/data/`, `rag.config.json` y
   `docker-compose.rag.yml`.
3. Genera `docker-compose.rag.yml` con todos los scalars comillados. **Reutiliza
   un Ollama accesible con el modelo** (host o `OLLAMA_HOST`); si no, levanta un
   contenedor — opcionalmente montando `RAG_OLLAMA_VOLUME` con el modelo ya
   descargado.
4. Arranca el RAG y espera a que `/health` responda (detecta crashes del
   contenedor en vez de esperar en vano).
5. `POST /reindex` con **timeout** (`RAG_INDEX_TIMEOUT`), tolerando `409`
   (reindex en curso) y exigiendo `drift == 0`; ante error imprime los últimos
   logs.
6. **Smoke test**: `/health` (identidad + Ollama), `/methodology` (chunks del
   alma) y `/query` de prueba con top-1.
7. Resumen final con comandos de arranque/parada/estado.

Gestión del servicio ya montado:

```bash
./setup.sh /ruta/al/proyecto --status   # estado (exit 1 si caído)
./setup.sh /ruta/al/proyecto --down     # para; conserva volúmenes e índice
```

Modo **sin Docker**: `--local` / `-Local` crea un venv en `rag/.venv` y arranca
`python -m rag.server`. Si no hay Ollama accesible lanza un contenedor
auxiliar en `:11434` (o usa tu CLI `ollama`); sin Docker ni Ollama aborta con
instrucciones.

Re-indexa con:

```bash
curl -X POST 'http://localhost:8765/reindex'   # o el puerto que indique --status
```

### Opción B — Manual (sin Docker)

```bash
cd /ruta/al/proyecto
cp -r <este-repo>/rag ./rag                     # copia el motor
# opcional: plantilla de instrucciones para agentes
cp <este-repo>/AGENTS.md ./AGENTS.md            # solo si no existe la tuya

pip install -r rag/requirements.txt
python -m rag.server &                          # sirve en :8765
curl -X POST 'http://localhost:8765/reindex'    # indexa en background
curl -s http://localhost:8765/stats             # drift debe ser 0
```

La primera indexación tarda 1-5 min (depende del repo y de si Ollama tiene el
modelo en memoria). Un `force=true` re-embebe todo y en CPU puede tardar >1h.

---

## Configuración (opcional)

En el primer arranque se genera `rag.config.json` en la raíz del proyecto con
valores detectados automáticamente (nombre de colección, globs, exclusiones,
methodology y si hay que activar categorías). Se recarga reiniciando el servidor.

```jsonc
{
  "project_name": "mi-proyecto",
  "collection": "mi_proyecto_chunks",
  "index_globs": ["**/*.md", "**/*.css", "**/*.js", "**/*.py", "..."],
  "exclude_dirs": ["node_modules", ".git", "venv", "...", "data", "rag/data"],
  "methodology": { "entire_files": ["AGENTS.md"], "heading_patterns": [] },
  "graph": { "block_regex": "[data-theme='...'] {...}", "token_regex": "--x:", "..." }
}
```

Documentación completa de cada campo: `rag/README.md`.

Variables de entorno: `OLLAMA_HOST`, `RAG_EMBED_MODEL`, `RAG_EMBED_DIM`,
`RAG_HOST`, `RAG_PORT`, `RAG_LOG_LEVEL`, `RAG_PROJECT`, `RAG_COLLECTION`.

---

## Endpoints

| Método | Endpoint | Uso |
|---|---|---|
| `GET` | `/health` | Estado Ollama + colección + grafo |
| `GET` | `/methodology` | «Alma» del proyecto (inyectar SIEMPRE al agente) |
| `GET` | `/categories` / `/category/{id}` | Categorías y su ficha (tokens+archivos) |
| `POST` | `/query` | Búsqueda semántica `{q, k, category?, file_glob?}` |
| `GET` | `/related/{kind}/{id}` | Vecinos en el grafo (`category`/`file`/`token`) |
| `POST` | `/reindex` | Re-indexar (incremental + reconciliación) |
| `GET` | `/stats` | Estadísticas + `drift` + estado del reindex |

---

## Estructura del repo

```
AGENTS.md             ← instrucciones obligatorias para agentes de IA (plantilla)
README.md             ← este archivo
setup.sh / setup.ps1  ← instalación automatizada vía Docker
benchmarks/           ← suite de benchmark (33 consultas, tests de hipótesis)
docs/benchmarks/      ← informe estadístico + figuras + resultados crudos
rag/
  server.py           ← API HTTP (FastAPI)
  indexer.py          ← chunk + embed + upsert + reconciliación
  chunker.py          ← parseo semántico por tipo de archivo
  graph.py            ← grafo categorías↔tokens↔archivos (JSON)
  embeddings.py       ← cliente Ollama (bge-m3)
  config.py           ← configuración + bootstrap de rag.config.json
  requirements.txt    ← dependencias Python
  Dockerfile          ← imagen del servicio
  README.md           ← documentación técnica del motor
```

---

## Licencia privada de uso

Uso personal. Sujétalo a los términos de tu organización si lo compartes.