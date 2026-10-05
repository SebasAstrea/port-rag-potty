# RAG portable · Retrieval-Augmented Generation

Servicio RAG local **portable**: copias la carpeta `rag/` dentro de **cualquier
proyecto** y sirve contexto curado del proyecto a agentes de IA, reduciendo
~80-150k tokens de input por sesión a ~3-5k de respuestas HTTP precisas.

Motor: **bge-m3** (embeddings vía Ollama) + **ChromaDB** (vector store
embebido) + **grafo ligero JSON** (relaciones categoría↔token↔archivo,
configurable por regex).

No sabe nada de "temas CSS" ni de ningún proyecto concreto: todo lo específico
se declara en **`<PROJECT_ROOT>/rag.config.json`**, auto-generado en el primer
arranque (detecta si el proyecto tiene CSS para activar el grafo de categorías).

---

## 1. Quickstart

### 1.1. Pre-requisitos

- Python 3.10+
- Ollama con `bge-m3` ya cargado (ajustar `OLLAMA_HOST` si no está en
  `http://localhost:11434`)

```bash
ollama pull bge-m3
```

### 1.2. Copiar el RAG al proyecto

> **Atajo de 1 ejecución** (recomendado): desde la raíz de este repo,
> `./setup.sh <tu-proyecto>` (o `.\setup.ps1 <tu-proyecto>`) — copia el motor,
> monta Ollama + servicio, autoindexa hasta `drift = 0` y valida con un smoke
> test. Ver la tabla de flags en el README raíz («Instalación»).

```bash
cp -r <ruta-al-rag>/rag  <tu-proyecto>/rag
cp <ruta-al-rag>/AGENTS.md       <tu-proyecto>/AGENTS.md   # opcional
```

### 1.3. Instalar y arrancar

```bash
cd <tu-proyecto>
pip install -r rag/requirements.txt
python -m rag.server          # sirve en http://localhost:8765
```

### 1.4. Indexar y probar

```bash
# Indexación inicial (background)
curl -X POST 'http://localhost:8765/reindex'

# Estado
curl -s http://localhost:8765/stats | python -m json.tool   # drift debe ser 0

# "Alma" del proyecto
curl -s http://localhost:8765/methodology | python -m json.tool | head -30

# Búsqueda semántica
curl -s -X POST http://localhost:8765/query \
  -H 'Content-Type: application/json' \
  -d '{"q": "cómo se configura el despliegue", "k": 5}' \
  | python -m json.tool
```

`POST /reindex?force=true` solo si `/stats` deja `drift != 0` tras un reindex
normal: re-embebe todo y en CPU puede tardar más de una hora.

---

## 2. Configuración por proyecto: `rag.config.json`

Se genera en la raíz del proyecto la primera vez que arranca el servidor. Si no
existe, el bootstrap:

1. Detecta el tipo de proyecto: si hay `*.css` **y** `index.html`, activa el
   grafo de categorías estilo CSS; si no, lo deja desactivado.
2. Deriva `project_name` y `collection` del nombre del directorio.
3. Rellena globs de índice, exclusiones, methodology y chunkers con defaults
   genéricos.

El archivo se recarga **solo al reiniciar el servidor**. Cualquier clave se
puede dejar fuera para usar su default.

### Campos

| Clave | Default | Descripción |
|---|---|---|
| `project_name` | nombre del directorio | Nombre visible del proyecto |
| `collection` | `<project_name>_chunks` | Nombre de la colección Chroma |
| `index_globs` | muchos `**/*.<ext>` | Patrones a indexar (md, html, css, js, ts, py, json, yaml, toml, sh…) |
| `exclude_dirs` | `node_modules, .git, .venv, build, data, rag/data…` | Carpetas excluidas |
| `exclude_files` | `package-lock.json, *.lock, rag.config.json…` | Archivos excluidos |
| `chunkers` | ext → chunker | Dispatch: `css, js, md, html, json, py, yaml, toml, raw` |
| `chunk_sizes` | por chunker | `{target, max}` en caracteres |
| `chunk_overlap` | `120` | Solapamiento entre chunks partidos |
| `embed_text_max_chars` | `4000` | Límite absoluto por chunk (bge-m3) |
| `methodology.entire_files` | `["AGENTS.md", "CLAUDE.md", ".cursorrules"]` | Documentos completos "alma" |
| `methodology.heading_patterns` | `[]` | Prefijos explícitos de secciones methodology (si vacío, se infieren) |
| `methodology.infer_keywords` | lista es/en | Keywords de inferencia de secciones |
| `graph.block_regex` | CSS `data-theme` | Regex categoría+bloque (grupo 1=cat, 2=cuerpo) |
| `graph.token_regex` | CSS `--token:` | Tokens dentro de bloques |
| `graph.mention_regex` | `data-theme` | Menciones sueltas de categoría |
| `graph.js_category_regex` | heurística JS | Chequeos JS → categorías |
| `graph.block_scan_exts` / `token_scan_exts` / `mention_scan_exts` | css / css / html+js… | Extensiones donde se aplica cada scan |
| `graph.summary_file` / `summary_regex` | `js/theme.js` | Resúmenes 1-línea por categoría (`{id}` se sustituye) |
| `graph.html_entry` | `index.html` | HTML que importa scripts/links (edges `imported_by`) |

> **Ejemplo genérico**: en un backend Python sin UI, borra las de `graph`
> (regex vacías) para desactivar categorías, o define una regex propia — p.ej.
> agrupar por módulo: `"block_regex": "^# =+\\s+([\\w ]+) =+\\s*$"` sobre `.py`.

### Variables de entorno (entorno, no proyecto)

| Variable | Default | Descripción |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | URL del servidor Ollama |
| `RAG_EMBED_MODEL` | `bge-m3` | Modelo de embeddings |
| `RAG_EMBED_DIM` | `1024` | Dimensión del vector |
| `RAG_HOST` | `0.0.0.0` | Host del servidor FastAPI |
| `RAG_PORT` | `8765` | Puerto del servidor |
| `RAG_LOG_LEVEL` | `INFO` | Nivel de log |
| `RAG_PROJECT` | — | Override de `project_name` |
| `RAG_COLLECTION` | — | Override de `collection` |

---

## 3. Arquitectura

```
┌──────────────────────────────────────────────────────────────────┐
│                       AGENTE (Claude/Cursor/opencode)            │
│                                                                  │
│   4 llamadas HTTP → ~3-5k tokens de contexto curado              │
└──────────────────────┬───────────────────────────────────────────┘
                       │  :8765
┌──────────────────────▼───────────────────────────────────────────┐
│                    rag.server (FastAPI)                          │
│                                                                  │
│   /methodology  /categories  /category/<id>                      │
│   /query        /related/<kind>/<id>                             │
│   /reindex      /health  /stats                                 │
└────────┬─────────────────────────────────────────┬───────────────┘
         │                                         │
┌────────▼──────────────────┐         ┌─────────────▼───────────────┐
│   rag.indexer             │         │   rag.graph (graph.json)     │
│   (chunk + embed + upsert)│         │   {categories, tokens, files,│
│                           │         │    edges}                    │
└────────┬──────────────────┘         └─────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│   chroma (PersistentClient en ./rag/data/chroma)                  │
│   + Ollama bge-m3 (1024-dim)                                      │
└───────────────────────────────────────────────────────────────────┘
```

### Capas de indexación

| Capa | Qué indexa | Cómo se trocea | Metadata |
|---|---|---|---|
| **Vectores** | Chunks de código + docs | Semántico por tipo (CSS regla, JS función, Python def, MD sección, YAML key…) | `file, kind, category, section, tokens, methodology` |
| **Grafo JSON** | Relaciones explícitas categoría↔token↔archivo | Análisis estático configurable por regex | `{categories, tokens, files, edges}` |
| **Methodology** | README + AGENTS.md + `entire_files` | Por heading, con flag `methodology=true` | Siempre se inyecta en `/query` |

### Tipos de chunk por archivo

- **CSS**: cada regla top-level = 1 chunk (con `category` si el selector la
  define). Comentarios sueltos también.
- **JS/TS/TSX**: funciones, arrow functions, clases, IIFEs, cabecera y bloques
  top-level residuales.
- **Python**: `def`/`class` con decoradores, docstring/cabecera del módulo y
  bloques top-level (imports, constantes).
- **MD**: secciones por heading H2/H3 + cabecera; methodology según config.
- **HTML**: comentarios y bloques top-level (`<div>`, `<main>`, `<section>`…).
- **JSON**: cada top-level key (o el documento entero).
- **YAML / TOML**: top-level keys / tablas `[sección]`.
- **resto**: ventanas por líneas (`raw`) — cualquier extensión es indexable sin
  configuración.

---

## 4. Endpoints

### `GET /health`

```json
{
  "ok": true,
  "ollama": { "ok": true, "ollama_host": "...", "embed_model": "bge-m3", "model_loaded": true, "available_models": [...] },
  "collection": { "ok": true, "count": 412, "model": "bge-m3" },
  "graph": { "categories": 3, "tokens": 85, "files": 22, "edges": 422 },
  "config": { "project": "...", "collection": "...", "embed_model": "...", "embed_dim": 1024, "ollama_host": "...", "chroma_dir": ".../rag/data/chroma" }
}
```

### `GET /methodology`

Devuelve hasta 50 chunks con `methodology=true`. **Úsalo SIEMPRE** al arrancar
una sesión de agente.

### `GET /categories` y `GET /category/{id}`

```json
{ "count": 3, "categories": [ { "id": "cauldron", "summary": "...", "n_tokens": 24, "n_files": 2, "n_scripts": 1, "files": ["css/cauldron.css"] } ] }
```

`/category/{id}` da tokens, archivos, scripts y hasta 30 chunks de muestra.
Si el proyecto no declara categorías, `/categories` devuelve lista vacía.

### `POST /query`

```json
{ "q": "cómo se hace que un widget reaccione al escribir", "k": 5, "category": "cauldron", "file_glob": "js", "include_methodology": true }
```

Respuesta: `results` (chunks rankeados con score cosine) + `methodology_chunks`
(inyección automática de contexto del proyecto).

### `GET /related/{kind}/{id}`

`kind ∈ {category, file, token}`. Vecinos a 1 salto en el grafo.

### `POST /reindex?force=true`

Lanza la reindexación en **background**, un solo job a la vez. Incremental por
defecto: mtime + reconciliación manifest↔Chroma. `force=true` rechunkea todo
(puede tardar >1h en CPU). `409` si ya hay una pasada en curso.

### `GET /stats`

`chroma_chunks`, `manifest_files`, `manifest_chunks`, `drift` y estado del
reindex. `drift = chroma_chunks - manifest_chunks` **debe ser 0**.

---

## 5. Docker (opcional)

```yaml
# docker-compose.yml (añadir a tu compose existente)
services:
  rag:
    build: ./rag
    ports: ["8765:8765"]
    environment:
      OLLAMA_HOST: http://ollama:11434
      RAG_PORT: 8765
    volumes:
      - .:/app
      - rag_data:/app/rag/data
    working_dir: /app
    command: python -m rag.server

volumes:
  rag_data:
```

```dockerfile
# rag/Dockerfile (genérico: no copia archivos de ningún proyecto)
FROM python:3.11-slim
WORKDIR /app
COPY rag/requirements.txt ./rag/
RUN pip install --no-cache-dir -r rag/requirements.txt
COPY rag ./rag
EXPOSE 8765
CMD ["python", "-m", "rag.server"]
```

Ejecución: el volumen monta el código real del proyecto en `/app`; el
`rag.config.json` se genera solo si falta.

---

## 6. Mantenimiento

- **`/reindex` periódico**: tras commits grandes o cada día si hay mucho churn.
- **Cambio de modelo**: actualiza `RAG_EMBED_MODEL` y ejecuta
  `POST /reindex?force=true` (los embeddings viejos son incompatibles).
- **Limpiar**: borra `rag/data/` y `rag.config.json` y reindexa desde cero.

### Invariantes del indexador (no romper)

1. **Nunca se borra antes de insertar.** `run_index()` chunkea → embebe →
   `upsert()` → y sólo entonces retira los ids viejos que no se reutilizaron.
2. **El manifest sólo registra lo que está en Chroma.** `Manifest.set_file()`
   se llama con los ids realmente upserteados.
3. **Un archivo incompleto se marca para reintento** (`mtime = 0.0`).
4. **Reconciliación en cada pasada.** `reconcile()` compara los `chunk_ids` del
   manifest contra Chroma y reindexa lo que falte, así el índice se auto-cura
   sin `force=true`.
5. **Ningún chunk supera `EMBED_TEXT_MAX_CHARS`.** `_enforce_embed_limit()`
   parte los que se pasan; `embed_text()` sólo trunca como último recurso.

### Troubleshooting

| Síntoma | Causa / arreglo |
|---|---|
| `/stats` con `drift != 0` | Chunks declarados que no están en Chroma → `POST /reindex` (la reconciliación lo cura) |
| `409` en `/reindex` | Ya hay una pasada en curso → mira `/stats → reindex.running` |
| `chroma_chunks` muy bajo durante un `force=true` | Normal: espera a `reindex.running == false` |
| `500 the input length exceeds the context length` | Chunk por encima del presupuesto → revisa `embed_text_max_chars` |
| Los cambios en `rag/*.py` no surten efecto | Uvicorn corre sin `reload`: reinicia el servicio/`docker restart` |
| `rag.config.json` no refleja mis cambios | Editar archivo + reiniciar el servidor |

---

## 7. Roadmap (próximas fases)

- **Fase 3 — Watcher + caché**: daemon con `watchdog` que re-indexa
  automáticamente al cambiar archivos; caché SQLite de queries frecuentes.
- **Fase 4 — MCP server**: wrapper MCP para que Claude/Cursor/Cline consuman
  el RAG como tools nativos.
- **Re-ranking**: añadir bge-reranker-v2-m3 para mejorar top-1 de las queries.
- **Query expansion**: expansión con sinónimos del dominio.

---

Si encuentras formas de mejorar el RAG, edita este archivo y `../AGENTS.md`
(plantilla portable que se copia a cada proyecto).