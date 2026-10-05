# 🔍 Informe de mejoras — motor RAG (`rag/`)

> **Alcance**: este documento recoge la revisión completa del motor Python
> (chunker, graph, config, robustez, rendimiento) realizada el 2026-10-05.
> **No se ha modificado código del motor**: es un informe priorizado con
> archivo:línea, impacto y propuesta de fix para cada hallazgo.
>
> Método: revisión estática asistida (búsqueda exhaustiva + verificación de
> líneas con grep) sobre `rag/chunker.py` (771), `rag/graph.py` (293),
> `rag/config.py` (350) y su interacción con `rag/indexer.py` / `rag/server.py`.

**Leyenda de prioridad**

| Prioridad | Significado |
|---|---|
| **P0** | Corrección: produce datos incorrectos, errores en runtime o filtrado inválido. |
| **P1** | Importante: chunking degradado, rendimiento malo en repos grandes, robustez frágil. |
| **P2** | Coste/deuda: config muerta, inconsistencias, duplicidad, determinismo. |

**Resumen**: 7 P0 · 17 P1 · 20 P2 = **44 hallazgos**.

---

## P0 — Corrección

### P0-1 · Residuos Python indexados con `kind: js-block`
- **Dónde**: `rag/chunker.py:485` llama a `_js_residual()`; ésta hardcodea
  `kind="js-block"` y `section="código top-level"` en `rag/chunker.py:286`.
- **Impacto**: imports, constantes y bloques `if __name__ == "__main__"` de
  archivos `.py` se indexan como `js-block`. Cualquier filtro por `kind`
  (`file_glob`/metadatos) devuelve basura para Python y el contexto compuesto
  (`compose_embed_text`) muestra `[js-block:…]` para código Python.
- **Fix**: parametrizar `_js_residual(kind=…, section=…)` y pasar
  `kind="py-block"` (o `py-residual`) desde `chunk_py`.

### P0-2 · `lstrip("./")` elimina *caracteres*, no un prefijo
- **Dónde**: `rag/config.py:316` (`_rel_key`), `rag/graph.py:223` y
  `rag/graph.py:227` (imports `<script src>` / `<link href>`).
- **Impacto**: `"../lib/x.js".lstrip("./")` → `"lib/x.js"`; `".gitignore"` →
  `"gitignore"`. Las rutas de los nodos/edges del grafo se empanan y las
  relaciones importan archivos equivocados o huérfanos.
- **Fix**: helper `_strip_prefix(p, "./")` que elimine solo el prefijo
  literal (o `p.removeprefix("./")` tras normalizar `../` con
  `posixpath.normpath`).

### P0-3 · La extensión real del archivo se pierde en metadata
- **Dónde**: cada chunker fija la extensión canónica en cada chunk:
  css `rag/chunker.py:174,187,205` (`.scss/.sass/.less` → `.css`),
  py `463,477` (`.pyi` → `.py`), md `500,516,537` (`.markdown` → `.md`),
  json `601,613,622`, yaml `651,666` (`.yml` → `.yaml`),
  html `562,584` (`.htm` → `.html`).
  `chunk_js` y `chunk_raw` sí lo hacen bien con `Path(file_rel).suffix`
  (`286`, `687`).
- **Impacto**: filtros y agrupaciones por `ext` engañan (un `.scss` aparece
  como `.css`); imposible distinguir variantes en `/query` con `file_glob`.
- **Fix**: usar `Path(file_rel).suffix` en todos los chunkers (ya existe el
  patrón correcto en js/raw).

### P0-4 · BOM UTF-8 rompe las regex de línea 1
- **Dónde**: `rag/chunker.py:750` lee con `encoding="utf-8"`.
- **Impacto**: un `\ufeff` inicial hace fallar `^#` (md, `490`) y
  `^(?:async )?def` (py, `403`) en la primera línea: el header/sección
  primera del archivo no se detecta.
- **Fix**: `encoding="utf-8-sig"` en `chunk_file` (y en los `read_text` de
  `graph.py:166`).

### P0-5 · `AttributeError` en reindex con `mention_regex` de usuario
- **Dónde**: `rag/graph.py:188` — `cid = (m.group(1) if m.groups() else m.group(0))`.
- **Impacto**: si el patrón *tiene* grupos pero el grupo 1 no participó
  (alternancia `(a)|(b)`), `m.group(1)` es `None` → `None.strip()` lanza
  `AttributeError` y rompe `/reindex` completo.
- **Fix**: `m.group(1) or m.group(0)` (o `next((g for g in m.groups() if g), m.group(0))`).

### P0-6 · Regex de usuario sin validar rompen `/reindex` sin error amigable
- **Dónde**: `block_regex` exige ≥2 grupos — `IndexError` en `rag/graph.py:172,175`;
  `re.error` sin capturar en `rag/graph.py:153,171,187,203`
  (y `99/105` de chunker para `category_regex`/`token_regex`).
- **Impacto**: una `rag.config.json` con regex mal escrita tumba la
  indexación con un stacktrace críptico en vez de un mensaje de config.
- **Fix**: validar al cargar config (`re.compile` + comprobar
  `compiled.groups >= 2` en `block_regex`) y envolver los `finditer` con
  `except re.error as e: log.warning(...)`.

### P0-7 · `ensure_category` trunca el id pero los edges usan el id completo
- **Dónde**: `rag/graph.py:49` guarda `cid.strip()[:80]`; los callers usan el
  `cid` sin truncar en edges/membership (`rag/graph.py:177-179, 192-193`).
- **Impacto**: con ids >80 chars, los edges y `files` apuntan a una categoría
  inexistente → nodos huérfanos y fallos silenciosos en
  `related_entity` (`graph.py:272`).
- **Fix**: normalizar el id una sola vez (`cid = _norm(cid)`) al inicio de
  `build_graph` y usar ese valor en todas partes; o truncar dentro de
  `ensure_category` devolviendo el id normalizado.

---

## P1 — Correctness de chunking y rendimiento

### P1-1 · IIFE: se busca el `{` *después* del que ya se consumió
- **Dónde**: `rag/chunker.py:363` — `_JS_IIFE` ya consumió `{`, pero
  `content.find("{", m.end())` busca el siguiente.
- **Impacto**: IIFE sin llaves internas → `brace == -1` → chunk descartado
  (`364-365`); con llaves internas → se tritura solo el primer bloque y el
  resto queda como residual solapado.
- **Fix**: buscar `}` balanced starting at `m.end()-1` (o reutilizar
  `_slice_balanced` desde el `{` que ya consumió el regex).

### P1-2 · `async function` / `export function` / `export const` no matchean
- **Dónde**: `rag/chunker.py:215-218` exige `function`/`const` justo tras
  inicio de línea.
- **Impacto**: módulos ES/TS modernos enteros caen a residual `js-block`
  genérico: se pierde nombre de símbolo y `section`.
- **Fix**: alternativas `(?:async\s+|export\s+(?:default\s+)?)?` en `_JS_FUNCTION`.

### P1-3 · Flecha `async` mal cortada
- **Dónde**: `rag/chunker.py:304` busca `=>` *después* de `async`
  (`body_start < arrow_body_start` nunca se cumple) → cae en la rama
  `308-322` que corta en el próximo `;`.
- **Impacto**: `const f = async () => { … }` se trunca en la primera
  sentencia interna.
- **Fix**: localizar `=>` desde el cuerpo del match, no desde `m.end()`.

### P1-4 · `find("{")` puede pillar la llave de *otra* función
- **Dónde**: `rag/chunker.py:303`.
- **Impacto**: una arrow con cuerpo expresión (`() => 1;`) + una función
  debajo produce un chunk fusionado que marca como consumido al vecino;
  también `function f(a = {})` / tipos TS `(): {a:number}` truncan el slice.
- **Fix**: limitar la búsqueda al rango hasta el `;` de la declaración y
  validar balanceo con `_slice_balanced`.

### P1-5 · Header JS off-by-one
- **Dónde**: `rag/chunker.py:381-384` — `if m.start() > 0` saltea el match
  en posición 0 y `header_end` termina siendo el inicio de la *segunda* línea.
- **Impacto**: archivos que empiezan por código (≥30 chars) generan un
  `js-header` falso que además se excluye del residual (texto perdido).
- **Fix**: aceptar `m.start() >= 0` y cortar en el primer match real.

### P1-6 · Metadatos de línea falsos
- **Dónde**:
  - JSON: todos los chunks con `start_line=1` (`rag/chunker.py:611-616`) y
    texto re-serializado con `indent=2` (`609`) que no coincide con el fuente.
  - `_enforce_embed_limit` re-deriva líneas desde el *texto del chunk*
    (`716-727`), no desde el archivo: inválido para CSS reconstruido
    (`selector {\n…}`) y JSON re-serializado.
  - CSS: `end_line = start_line + body.count("\n")` (`159`) ignora selectores
    multilínea; reglas cuyo selector (sin comentarios) no se encuentra en el
    original se **descartan en silencio** (`155-156`).
- **Impacto**: los hints `file:line` del contexto de embedding mienten →
  el modelo cita líneas equivocadas al usuario.
- **Fix**: propagar `start_line` real por segmento; en `_enforce_embed_limit`
  conservar el `start_line` original y solo incrementarlo por líneas nuevas
  del split; en CSS calcular desde el offset del match original.

### P1-7 · CSS `@media` / anidados no matcheados
- **Dónde**: `_CSS_RULE` usa cuerpo `[^{}]*` (`rag/chunker.py:133`).
- **Impacto**: `@media … { .a {…} }` solo matchea el `.a` interior: se pierde
  el contexto del media en selector y texto (igual `@supports`, anidados SCSS).
- **Fix**: parser de llaves balanceadas para bloques de nivel superior, o
  regex con cuerpo balanceado (contar `{`/`}` en lugar de negarlos).

### P1-8 · HTML anidado truncado + comentarios duplicados
- **Dónde**: `rag/chunker.py:570` corta en el primer `</tag>`;
  comentarios dentro de un bloque se emiten también aparte (`554-566`).
- **Impacto**: `<div><div>…</div></div>` se corta antes de tiempo;
  contenido duplicado en el índice (bloque + comentario).
- **Fix**: balanceo de tags (contador por tag) en lugar de `find("</tag>")`;
  emitir comentarios solo si no están dentro de un bloque ya emitido.

### P1-9 · Decoradores Python duplicados (header vs def)
- **Dónde**: el header termina en `matches[0].start()` (`rag/chunker.py:471`)
  pero el primer def empieza en `_py_dec_start` (`454`).
- **Impacto**: la racha de decoradores aparece en `py-header` **y** en el
  primer `py-def` → doble indexación del mismo texto.
- **Fix**: el header debe terminar en `_py_dec_start(matches[0])`.

### P1-10 · `related_entity(max_hops=…)` ignora el parámetro
- **Dónde**: `rag/graph.py:264` — el argumento nunca se usa; solo devuelve
  vecinos de 1 salto pese a la docstring «1 o 2 saltos».
- **Impacto**: API engañosa (hoy sin crash porque `server.py:306` no lo pasa).
- **Fix**: implementar BFS acotado a `max_hops` (o eliminar el parámetro y
  la promesa de la docstring).

### P1-11 · `build_graph(chunks=…)` no se usa y relee todo el árbol
- **Dónde**: `rag/graph.py:158` (param muerto), `166` re-lee cada archivo ya
  leído por `chunk_file`; caller `rag/indexer.py:309` no pasa chunks.
- **Impacto**: I/O duplicado completo en cada reindex.
- **Fix**: derivar el grafo de los chunks/metadata ya en memoria cuando se
  pasan; si no, mantener el scan actual.

### P1-12 · El árbol se recorre ~30 globs × 3 veces por reindex, sin dedup
- **Dónde**: `iter_project_files` (`rag/chunker.py:761`) ejecuta cada glob de
  `INDEX_GLOBS` (≈30 patrones con `**`, `config.py:47-57`) contra todo el
  árbol, **sin set de dedup**; llamado desde `rag/indexer.py:172`,
  `rag/indexer.py:298` y `rag/graph.py:162`.
- **Impacto**: coste O(globs × árbol) × 3 por reindex; globs solapados (p.ej.
  un `**/*` añadido por el usuario) producen ficheros duplicados → chunks
  duplicados y embeddings repetidos.
- **Fix**: cachear el listado (1× por reindex) en un `set` compartido
  indexer→graph; dedup obligatorio.

### P1-13 · `_line_of` O(offset) por chunk → cuadrático
- **Dónde**: `rag/chunker.py:92-93` (`text.count("\n", 0, offset)`;
  peor en `_split_long`, `124`, 2 llamadas por segmento).
- **Impacto**: archivos grandes/minificados con muchos chunks → coste
  cuadrático en CPU del indexador.
- **Fix**: precalcular offsets de `\n` una vez (lista + `bisect`).

### P1-14 · `in_consumed` barridos lineales O(k²)
- **Dónde**: `rag/chunker.py:271-272, 298-299, 440-441` —
  `any(a <= p < b for a, b in consumed)` por cada match; el bucle interno
  de `chunk_py` (`449-453`) recalcula indent por cada def posterior.
- **Impacto**: degrada en ficheros generados o minificados con cientos de
  símbolos.
- **Fix**: estructura indexada (sorted list + `bisect`, o marcas por offset).

### P1-15 · HTML: `content[m.end():]` copia el resto del documento por tag
- **Dónde**: `rag/chunker.py:570` (slice + `re.search` sin `pos`).
- **Impacto**: O(n²) de memoria en HTML con mucha etiqueta.
- **Fix**: `pat.compile(...).search(content, m.end())` (sin copia).

### P1-16 · `Graph.add_edge` dedup O(E) por inserción → O(E²)
- **Dónde**: `rag/graph.py:64`.
- **Impacto**: con grafos grandes (miles de edges) el reindex se encoge.
- **Fix**: mantener `set` de claves `(src, kind, dst)` paralelo a la lista.

### P1-17 · `summary_file` re-leído y re-compilado por cada categoría
- **Dónde**: `rag/graph.py:140-155` llamado en bucle (`232-235`); patrón
  recompilado con `.format` por categoría.
- **Impacto**: I/O + regex redundantes (proporcional al nº de categorías).
- **Fix**: leer/compilar una vez fuera del bucle.

---

## P1 — Robustez

### P1-18 · Archivos no decodificables/binarios desaparecen sin rastro
- **Dónde**: `rag/chunker.py:749-752` y `rag/graph.py:165-168` —
  `UnicodeDecodeError`/`OSError` → `return []` / `continue` sin log.
- **Impacto**: un fichero en latin-1/cp1252 o binario accidental desaparece
  del índice sin diagnóstico; también no hay límite de tamaño de fichero
  (un JS minificado de 100 MB se lee entero y `_slice_balanced` lo recorre
  carácter a carácter en Python).
- **Fix**: `logger.warning("skip %s: %s", file, e)` + umbral de tamaño
  configurable (`max_file_bytes`) con log.

### P1-19 · Archivos vacíos → chunks vacíos; archivos diminutos → 0 chunks
- **Dónde**: `rag/chunker.py:111-112` y `496-506` emiten `text=""`;
  los umbrales de header/residual (`387`, `473`, `282`) pueden dejar archivos
  cortos sin ningún chunk.
- **Impacto**: vectores de solo prefijo de contexto (ruido en el índice);
  ficheros pequeños útiles (p.ej. un `.env.example`) no indexados.
- **Fix**: no emitir chunks con texto vacío; para archivos < umbral emitir
  un chunk `raw` con el contenido completo.

### P1-20 · Margen fijo `-200` del presupuesto de embedding
- **Dónde**: `rag/chunker.py:705` (`budget = max_chars - 200`) vs prefijo real
  de `compose_embed_text` (`79-80`): categoría + `section[:200]` + `file:line`
  puede superar 200 chars.
- **Impacto**: el texto compuesto aún puede exceder `EMBED_TEXT_MAX_CHARS` y
  sufrir truncado silencioso al final del chunk (pérdida de datos).
- **Fix**: restar la longitud real del prefijo calculado por chunk.

### P1-21 · `Graph.load` / `from_dict` frágiles
- **Dónde**: `rag/graph.py:127-130` captura solo `JSONDecodeError` (un
  `OSError` propaga); `from_dict` (`100-114`) asume tipos en cada valor.
- **Impacto**: un `graph.json` editado a mano o con permisos raros tumbe el
  `/reindex` con stacktrace.
- **Fix**: capturar `Exception` en load (log + grafo vacío) y validar tipos en
  `from_dict` con defaults.

### P1-22 · `relative_to` / `ensure_category` pueden lanzar excepciones no vistas
- **Dónde**: `rag/graph.py:216` (`rel_entry.relative_to(project_root)` con
  `html_entry` absoluto o fuera del root → `ValueError`);
  `rag/graph.py:51` (`ensure_category` → `ValueError` con id vacío).
- **Impacto**: config legítima (entrada HTML fuera del árbol) rompe el
  reindex.
- **Fix**: envolver en try/except con warning; en `ensure_category` devolver
  `None` o pre-normalizar en el caller central.

---

## P2 — Config muerta, inconsistencias y deuda

### P2-1 · `CHUNK_TARGET_CHARS` es config muerta
- **Dónde**: único consumidor asigna a un local no usado
  (`rag/chunker.py:162`); `chunk_sizes.*.target` de `rag.config.json` no hace
  nada.
- **Impacto**: el usuario «tunea» `target` y no ve efecto (falsa
  expectativa documentada en `config.py:144-154`).
- **Fix**: eliminar del schema o aplicar de verdad en los chunkers.

### P2-2 · `chunk_sizes.max` solo lo usan css/md/raw
- **Dónde**: `rag/chunker.py:161` (css), `530` (md), `680` (raw); `js`, `py`,
  `json`, `html`, `yaml`, `toml` lo ignoran — su tamaño lo decide
  `_enforce_embed_limit` a secas.
- **Impacto**: inconsistencia entre la superficie de config documentada y el
  comportamiento real.
- **Fix**: respetar `CHUNK_MAX_CHARS[ext]` en todos (o documentar la
  excepción).

### P2-3 · Token ids con prefijo `--` hardcodeado
- **Dónde**: `rag/graph.py:182,184,199-200` (`"--" + tok`).
- **Impacto**: un `token_regex` que ya capture nombres completos
  (`$brand`, `--var`) los duplica/empaña (`----var`); asume CSS.
- **Fix**: hacer el prefijo configurable (`graph.token_prefix`) o capturarlo
  en el grupo del regex.

### P2-4 · Casing de categorías inconsistente
- **Dónde**: `rag/graph.py:207` — `candidate.lower()` solo si la primera
  letra es mayúscula: `MyTheme` → `mytheme` pero `myTheme` se queda.
- **Impacto**: la misma categoría lógica se fragmenta en dos ids según qué
  scan la vea primero.
- **Fix**: normalizar siempre con `.lower()` (o nunca) en un único punto.

### P2-5 · Defaults frontend persisten si hay 1 CSS suelto
- **Dónde**: defaults `summary_file: "js/theme.js"`, `html_entry:
  "index.html"`, regex `data-theme` (`config.py:114-140`) solo se neutralizan
  si `_looks_frontend` no encuentra **ningún** `*.css` (`166-183`); y
  `_looks_frontend` hace `rglob("*.css")` sobre todo el árbol al importar
  (`config.py:169`).
- **Impacto**: un proyecto Python con un CSS de docs/vendor obtiene la config
  de grafo de themes; el `rglob` inicial recorre también `node_modules`.
- **Fix**: contar CSS fuera de `EXCLUDE_DIRS` y exigir umbral (p.ej. ≥3);
  mover el walk al arranque del indexado, no al import; reutilizar
  `iter_project_files` ya filtrado.

### P2-6 · Entrada muerta `EXCLUDE_DIRS = "rag/data"`
- **Dónde**: `rag/config.py:67` (valor con dos partes) vs filtro que compara
  partes simples (`rag/chunker.py:768`).
- **Impacto**: nunca matchea (el `data` simple ya cubre el caso) — confuso al
  leer.
- **Fix**: quitar la entrada o soportar rutas relativas completas en el filtro.

### P2-7 · Orden no determinista de `tokens`/metadatos
- **Dónde**: `rag/chunker.py:99,105` — `list({m for m in re.findall(…)})`
  itera un `set` (orden por hash randomizado).
- **Impacto**: el CSV `tokens` del chunk puede cambiar entre procesos para el
  mismo chunk → churn de metadata si algo compara stored vs computed.
- **Fix**: `sorted(set(...))`.

### P2-8 · Contenido duplicado entre kinds
- **Dónde**: comentarios CSS dentro de regla también emitidos como
  `css-comment` (`196-209`); HTML comments vs bloque (P1-8); decoradores
  (P1-9).
- **Impacto**: índice inflado y respuestas casi idénticas para una query.
- **Fix**: excluir rangos ya consumidos al emitir comentarios sueltos.

### P2-9 · `chunk_json` re-serializa el texto indexado
- **Dónde**: `rag/chunker.py:609` — `json.dumps(..., indent=2)`.
- **Impacto**: tabs/escapes/key-spacing distintos al fuente → el usuario no
  encuentra por búsqueda el texto que la RAG le devolvió.
- **Fix**: cortar el texto original por offsets de keys (reutilizando
  `json.JSONDecoder.raw_decode` sobre el texto) en vez de re-serializar.

### P2-10 · Regex de usuario sin límite de complejidad
- **Dónde**: `graph.py:171/187/203`, `chunker.py:99/105` — patrones de
  `rag.config.json` compilados en cada uso, sin validación de coste.
- **Impacto**: un patrón con backtracking catastrófico cuelga el reindex.
- **Fix**: `re.compile` al cargar config + `try/except re.error` (ver P0-6);
  opcionalmente límite de longitud.

### P2-11 · `_CSS_RULE` con cuantificadores anidados
- **Dónde**: `rag/chunker.py:132-135` — `(?:[^{}/]|/(?!\*)[^/])+?` reescanea
  por cada posición de inicio.
- **Impacto**: O(n²) en CSS minificado con miles de reglas.
- **Fix**: escaneo carácter a carácter (autómata) o grupo atómico/possessive.

### P2-12 · `hashlib` importado dentro del helper más caliente
- **Dónde**: `rag/chunker.py:86` (`_stable_id`).
- **Impacto**: overhead menor por chunk (ya hay lookup de módulo cacheado,
  coste bajo) — deuda cosmética.
- **Fix**: import a nivel de módulo.

### P2-13 · `META_KEYS` de config desactualizado
- **Dónde**: `rag/config.py:343` documenta
  `css-rule | js-fn | py-def | md-section | html-block | json | raw` y omite
  `js-arrow/js-class/js-iife/js-header/js-block/py-class/py-header/
  css-comment/yaml-block/toml-section` y no menciona la fuga `js-block` en
  Python (P0-1).
- **Fix**: regenerar la lista desde los kinds emitidos (o quitarla).

### P2-14 · `section`/`tokens` truncados con caps no expuestos
- **Dónde**: `rag/chunker.py:53` (`section[:200]`), `56` (`tokens[:32]`).
- **Impacto**: caps arbitrarias fuera de config; 32 tokens largos pierden
  edges desde metadata de chunk (el grafo no, va por scan aparte).
- **Fix**: mover a `config` o documentar.

### P2-15 · Extensiones mapeadas a `raw` sin estructura
- **Dónde**: `.go .rs .java .rb .lua .tf .sh .sql …` → `chunk_raw`
  (`config.py:76-89`).
- **Impacto**: no es un bug (cobertura completa: los 9 keys existen en
  `_CHUNKER_BY_KEY`, `chunker.py:735-745`; desconocidos → raw `755-756`),
  pero esos lenguajes no reciben chunking semántico.
- **Fix** (futuro): añadir chunkers o documentar la limitación.

### P2-16 · Errores de sintaxis en `graph.json`/manifest no tipados
- Ver P1-21. Además: `indexer.py` no distingue manifest corrupto de
  desactualizado más allá de reconstruir — sin métrica de cuántos entries
  se descartaron.
- **Fix**: log con conteo de entries inválidos.

### P2-17 · Sin tests unitarios del motor
- **Dónde**: no existe `tests/` en el repo; `benchmarks/` mide calidad, no
  regresiones.
- **Impacto**: cualquier fix de este informe (P0-1…P1-9 son táctiles) puede
  regresionar sin red de seguridad.
- **Fix**: añadir `tests/test_chunker.py` con casos por kind (py residual,
  IIFE, BOM, decoradores, `@media`, HTML anidado) — prioridad alta si se
  implementan los fixes.

### P2-18 · `index_globs` con `**` redundantes no declarados
- Ver P1-12: además del coste, falta dedup a nivel de ficheros.
- **Fix**: normalizar/dedup globs al cargar config.

### P2-19 · `chunk_file` devuelve `[]` también en `OSError` (permisos)
- Ver P1-18: un fichero sin permisos se salta sin log.
- **Fix**: mismo log unificado.

### P2-20 · El prefijo de contexto incluye `section` ya truncada dos veces
- **Dónde**: `section[:200]` en `to_meta` (53) y de nuevo al componer (79-80).
- **Impacto**: solo cosmético/duplicación de lógica.
- **Fix**: truncar una vez, en el origen.

---

## Cobertura de `kind` (estado actual)

| Chunker | `kind` emitidos |
|---|---|
| css | `css-rule`, `css-comment` |
| js | `js-fn`, `js-arrow`, `js-class`, `js-iife`, `js-header`, `js-block` |
| py | `py-def`, `py-class`, `py-header`, **+ `js-block` (fuga P0-1)** |
| md | `md-section` |
| html | `html-block` (bloques y comentarios) |
| json | `json` |
| yaml | `yaml-block` |
| toml | `toml-section` |
| raw | `raw` |

Los 9 valores de `CHUNKERS_DEFAULT` (`config.py:76-89`) tienen chunker en
`_CHUNKER_BY_KEY` (`chunker.py:735-745`); extensiones desconocidas → `raw`.

---

## Orden recomendado de implementación

1. **P0-1 … P0-7** (corrección, cambios localizados y de bajo riesgo).
2. **P2-17** tests de regresión básicos → habilita el resto con seguridad.
3. **P1-1 … P1-9** (calidad de chunking) — empezando por P1-1/P1-2/P1-5
   (JS moderno) y P1-6 (líneas falsas, afecta a todos los hints).
4. **P1-12, P1-13, P1-16, P1-17** (rendimiento de reindex, alto ROI).
5. **P1-18 … P1-22** (robustez) y **P2** (deuda) según toque.

> Fuera de este informe (y ya en marcha): la revisión y el hardening de
> `setup.sh`/`setup.ps1` — ver plan en la conversación de trabajo.
