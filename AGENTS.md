# 🤖 AGENTS.md — Instrucciones obligatorias para agentes de IA

> **Si eres un agente de IA (Claude Code, Cursor, Cline, opencode, etc.) y vas
> a modificar, crear o analizar cualquier archivo de este repositorio, lee
> este documento ANTES de hacer nada.**

Este proyecto tiene un **servicio RAG local** que te da contexto curado sobre
el proyecto. **Usarlo es obligatorio** — gastarás muchos menos tokens y tus
respuestas serán más precisas.

---

## 1. Verifica que el servicio RAG está levantado

```bash
curl -s http://localhost:8766/health | head -50
```

Si responde, sigue. Si no, levántalo:

```bash
cd <repo-root>
pip install -r rag/requirements.txt
python -m rag.server &          # arranca en :8766
# Espera ~2s a que arranque, luego indexa:
curl -X POST 'http://localhost:8766/reindex'
# y comprueba que cuadra (drift debe ser 0):
curl -s http://localhost:8766/stats | python -m json.tool
```

La primera indexación tarda 1-5 minutos (depende del tamaño de los chunks y de
si Ollama tiene `bge-m3` ya en memoria).

> **No uses `?force=true` por defecto.** Re-embebe el proyecto entero y en CPU
> puede tardar **más de una hora** (medido en este repo: `elapsed_s` 3931 y
> 3964 para ~1.7k chunks). El reindex incremental basta el 99% de las veces:
> detecta cambios por mtime **y** reconcilia el manifest contra Chroma.
> Resérvalo para cuando un reindex normal deje `drift != 0` en `/stats`.

---

## 2. Antes de CUALQUIER modificación, invoca estos endpoints

### 2.1. Lee el "alma" del proyecto (siempre)

```bash
curl -s http://localhost:8766/methodology | python -m json.tool | head -120
```

Te devuelve los chunks marcados como **methodology**: propósito, arquitectura,
estructura, convenciones y la guía de contribución (de `README.md`, `AGENTS.md`
y de los archivos `methodology.entire_files` de `rag.config.json`). **No
modifiques nada sin haberlo leído al menos una vez en la sesión.**

### 2.2. Si el proyecto declara categorías, pide su ficha

```bash
curl -s http://localhost:8766/categories | python -m json.tool     # qué categorías hay
curl -s http://localhost:8766/category/<id> | python -m json.tool  # ficha de una
```

Las categorías (activo solo si `rag.config.json → graph` lo define, p.ej.
temas CSS vía `data-theme`) agrupan archivos, scripts y tokens. Si
`/categories` devuelve lista vacía, el proyecto no las usa: busca directamente
con `/query`.

### 2.3. Si necesitas buscar un patrón o entender "cómo se hace X"

```bash
curl -s -X POST http://localhost:8766/query \
  -H 'Content-Type: application/json' \
  -d '{"q": "cómo se configura el despliegue", "k": 5}'
```

Parámetros útiles:
- `category`: filtra a una categoría (`"category": "<id>"`)
- `file_glob`: filtra por patrón de archivo (substring) (`"file_glob": "rag/config"`)
- `include_methodology`: `true` por defecto; inyecta el contexto del proyecto
- `k`: 1-20 chunks (default 5)

### 2.4. Para entender relaciones del grafo

```bash
# ¿Qué archivos y tokens componen una categoría?
curl -s http://localhost:8766/related/category/<id> | python -m json.tool

# ¿Qué archivos importa la página de entrada?
curl -s http://localhost:8766/related/file/index.html | python -m json.tool

# ¿Qué categorías definen este token? (solo si hay tokens, p.ej. CSS)
curl -s http://localhost:8766/related/token/--bg | python -m json.tool
```

`/related/{kind}/<id>` con `kind ∈ {category, file, token}` devuelve los vecinos
a 1 salto en el grafo.

---

## 3. Reglas de comportamiento

1. **Lee los archivos completos solo cuando sea estrictamente necesario.** Si el
   RAG te da el chunk relevante, lee solo ese + su contexto adyacente, no el
   archivo entero.

2. **Tras modificar un archivo, re-indexa (incremental):**
   ```bash
   curl -X POST 'http://localhost:8766/reindex'
   curl -s http://localhost:8766/stats | python -m json.tool   # drift debe ser 0
   ```
   El indexador detecta cambios por mtime y re-embebe solo lo necesario.

   - **Un solo reindex a la vez.** Si ya hay uno en curso el endpoint responde
     `409`; consulta `/stats → reindex` (`running`, `status`, `started_at`) y
     espera. No lances otro «por si acaso».
   - **Verifica `drift`.** En `/stats`, `drift = chroma_chunks - manifest_chunks`
     debe ser `0`. Si no lo es, hay chunks declarados que no están en Chroma: un
     reindex incremental (sin `force`) lo auto-cura por reconciliación.
   - Si `/stats → reindex.status` es `"error"`, la pasada falló (p.ej. Ollama
     caído). El índice queda **intacto** y se reintenta en la siguiente pasada.
   - No midas el progreso por `chroma_chunks` a mitad de una pasada `force=true`:
     durante el embedding puede parecer que hay 0 chunks. Usa
     `/stats → reindex.running`.

3. **Si añades una categoría nueva** (temas CSS, módulos, etc.):
   - Declara su detección en `rag.config.json → graph` (regex de bloques y
     menciones; ver `rag/README.md`).
   - `curl -X POST 'http://localhost:8766/reindex'` (incremental; no hace falta
     `force`).

4. **Si añades un token nuevo** que deba estar en el grafo (CSS `--token`, etc.):
   el indexador lo detecta automáticamente al re-indexar.

5. **Si modificas el README o AGENTS.md:** las secciones con `methodology=true`
   se inyectan automáticamente en cada `/query`. La detección es configurable en
   `rag.config.json → methodology` (`entire_files` para documentos completos,
   `heading_patterns` para secciones concretas; si está vacío se infiere).

---

## 4. Endpoints de referencia rápida

| Método | Endpoint | Uso |
|---|---|---|
| `GET`  | `/health` | Estado de Ollama + colección + grafo |
| `GET`  | `/methodology` | Chunks "alma" del proyecto (README + AGENTS.md + entire_files) |
| `GET`  | `/categories` | Lista de categorías con resumen 1-línea (si hay) |
| `GET`  | `/category/{id}` | Ficha completa de una categoría (tokens + archivos + chunks) |
| `POST` | `/query` | Búsqueda semántica `{q, k, category?, file_glob?}` |
| `GET`  | `/related/{kind}/{id}` | Vecinos en el grafo (`kind ∈ {category, file, token}`) |
| `POST` | `/reindex` | Reindexar (incremental + reconciliación; `409` si ya hay uno) |
| `GET`  | `/stats` | Estadísticas + `drift` + estado del reindex |

---

## 5. Configuración del RAG para este proyecto

El comportamiento se declara en **`rag.config.json`** (en la raíz del
proyecto). Se genera automáticamente en el primer arranque con valores
razonables y detecta el tipo de proyecto (si hay CSS, activa el grafo de
categorías/tokens; si no, lo deja desactivado). Edítalo cuando necesites:

- `collection` / `project_name` — nombre de la colección Chroma y del proyecto
- `index_globs` / `exclude_dirs` / `exclude_files` — qué se indexa y qué no
- `chunkers` — ext → chunker (css, js, py, md, yaml, toml, raw…)
- `methodology` — `entire_files` y `heading_patterns` ("alma" del proyecto)
- `graph` — regex de categorías/tokens y archivo de resúmenes (p.ej. temas CSS)
- `chunk_sizes` / `chunk_overlap` / `embed_text_max_chars` — presupuestos

No lo borres sin más: regenéralo desde `rag/config.py` (borra `rag.config.json`
y reinicia) o edítalo a mano; se recarga al reiniciar el servidor.

Variables de entorno adicionales (cosas que dependen del entorno):

```bash
export OLLAMA_HOST=http://localhost:11434    # URL de tu contenedor Ollama
export RAG_EMBED_MODEL=bge-m3                # modelo de embeddings
export RAG_EMBED_DIM=1024                    # dimensión (default de bge-m3)
export RAG_HOST=0.0.0.0
export RAG_PORT=8766
export RAG_PROJECT=mi-proyecto               # override de project_name
export RAG_COLLECTION=mi_proyecto_chunks     # override de collection
```

Si tu Ollama está en otro host (ej. contenedor Docker), ajusta `OLLAMA_HOST`
y reinicia el servicio RAG.

---

## 6. Por qué existe este archivo

- **Antes**: un agente nuevo gastaba ~80-150k tokens de input solo en leer
  README + archivos fuente para entender el contexto.
- **Después**: con 4-5 llamadas HTTP (~3-5k tokens de respuesta) tiene el mismo
  conocimiento + la "alma" del proyecto inyectada automáticamente.
- **Bonus**: el grafo permite responder "¿qué archivos componen X categoría?" o
  "¿qué categorías definen este token?" sin tener que parsearlo tú mismo.

Si encuentras formas de mejorar el RAG, edita `rag/README.md` y este archivo.