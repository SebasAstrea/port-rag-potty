# Router de decisión: ¿RAG o lectura directa?

> **Respuesta corta:** el RAG no siempre ahorra. En el benchmark de 33 consultas,
> **12/33** costaban más que leer el archivo directamente, y el cruce estaba en
> **~1,7-2k tokens de destino**. Este router decide *antes de leer* y elige la
> vía más barata: `POST /route`.

---

## 1. El problema, medido

`docs/benchmarks/REPORT.md` mide dos condiciones emparejadas por consulta:

| Condición | Qué cuesta |
|---|---|
| **A — directa** | Leer completos los archivos ground-truth con tool_calling |
| **B — RAG** | Payload real de `POST /query` (k=5 + methodology) |

Resultados que motivan el router:

- Ahorro mediano **+801 tok/consulta** (Wilcoxon p=0,0169), pero **12/33
  consultas negativas**.
- Destino **≥ 2 000 tok** → **+1 121 tok** medianos (p=2·10⁻⁶).
- Destino **< 2 000 tok** → **−786 tok** medianos: leer el archivo gana.
- El «alma» (`methodology`) cuesta **438-566 tok** por consulta; si ya está en
  contexto, no hay que pagarla otra vez.

Un router útil tiene que estimar A y B **antes** de que el agente lea nada.
El problema es que A (tamaño del destino) normalmente no se conoce hasta
descubrir los archivos. Ahí entra el *probe* interno.

---

## 2. Cómo decide

```text
                         ┌──────────────────────────────┐
consulta ───────────────▶│  /route                      │
                         │  1. probe interno (sin coste │
                         │     de tokens para el modelo)│
                         │  2. estima B (payload exacto)│
                         │  3. estima A (archivos)      │
                         │  4. aplica riesgo de fallo   │
                         └──────────────┬───────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
                  action=rag        action=direct      action=hybrid
                  /query            read(path)          /query + read(top-1)
```

### 2.1. Probe interno

Si el agente no pasa `candidate_files`, el router hace una consulta semántica
interna a Chroma + Ollama (k=12 por defecto, ~40-50 ms) y **no devuelve los
chunks**: sólo los usa para descubrir archivos probables y estimar el payload
de `/query`. Ese probe **no consume tokens del modelo**.

Si Ollama/Chroma no están disponibles, cae a un probe lexical (términos de la
consulta contra rutas y categorías del grafo), que es peor pero no bloquea la
decisión.

### 2.2. Estimación de costes

- **B (RAG):** si hay probe semántico, se construye el JSON que devolvería
  `/query` (snippets de 600 chars, methodology de 500) y se tokeniza. Si no,
  se usa el modelo calibrado:
  `B ≈ base + per_result·k + methodology·[0|1]`.
- **A (directa):** se leen los tamaños reales de los archivos candidatos y se
  tokenizan (tiktoken si está instalado; si no, `chars/3.6`). Se añade un
  envoltorio por archivo (`direct_overhead_tokens`, 25 por defecto).
- **Riesgo de que RAG no baste:** el top-k puede no contener la respuesta
  (hit@5 medido = 96,97 %, pero la cobertura exacta no se midió). Se modela:

  ```text
  E[RAG] = B + p_miss · (A + fallback_penalty)
  ```

  donde `p_miss` parte de 0,05 y sube si el probe es lexical o el top score es
  bajo.

### 2.3. Regla de decisión

```text
RAG gana si:      B + margen < A          (ventaja bruta)
            y     E[RAG] + margen < A    (ventaja esperada con riesgo)
Directo gana en el resto (tie-break conservador: leer no falla).
Híbrido:          B + top1 + margen < A  y RAG esperado no gana claramente.
```

El margen por defecto es **60 tok**; el coste de la propia respuesta de
`/route` (**80 tok** estimados) se descuenta del ahorro que se reporta en
`save`. El cruce teórico es:

```text
A* = B / (1 - p_miss)
```

Con `B ≈ 1 680` y `p_miss = 0,05` → **A\* ≈ 1 768 tok**, coherente con el cruce
medido de ~1,7-2k.

### 2.4. Conciencia de sesión

El agente puede pasar:

- `methodology_loaded=true` → no se paga otra vez el «alma» (~470 tok).
- `loaded_files=["..."]` → esos archivos ya están en contexto y su coste es 0.
- `token_budget=N` → si la vía elegida no cabe, se busca la más barata que quepa.

Esto evita el error clásico de contar dos veces el mismo contexto.

---

## 3. API

### `POST /route`

```bash
curl -s -X POST http://localhost:8766/route \
  -H 'Content-Type: application/json' \
  -d '{"q":"¿Cómo funciona el frame gate latest-only?","k":5}'
```

Respuesta compacta (por defecto, ~40-80 tok):

```json
{
  "id": "rt_4f6a2c91ab",
  "action": "direct",
  "conf": 0.86,
  "save": 1180,
  "why": "small_target",
  "read": [{"path": "perception/.../LatestOnlyFrameGate.kt", "tok": 420}]
}
```

```json
{
  "id": "rt_9b1e77d0c3",
  "action": "rag",
  "conf": 0.84,
  "save": 2110,
  "why": "large_target",
  "rag": {"k": 5, "methodology": true}
}
```

Campos de la respuesta compacta:

| Campo | Significado |
|---|---|
| `id` | Identificador para enviar feedback |
| `action` | `rag` → llamar `/query`; `direct` → leer `read`; `hybrid` → ambas |
| `conf` | Confianza de la decisión (0-0,97) |
| `save` | Ahorro neto estimado (ya descuenta el coste de `/route`) |
| `why` | Código corto: `small_target`, `large_target`, `hybrid`, `no_candidates` |
| `read` | Archivos a leer (sólo en `direct`/`hybrid`) |
| `rag` | Plan para `/query` (sólo en `rag`/`hybrid`) |
| `warn` | Aviso (sin candidatos, presupuesto, etc.) |

Con `"compact": false` se añaden `rag_tok`, `direct_tok`, `est` (costes,
cruce, `p_miss`), `probe`, `candidates`, `calibration` y `detail`.

Parámetros útiles:

```jsonc
{
  "q": "...",
  "k": 5,
  "candidate_files": ["src/foo.py"],   // si el agente ya sabe dónde mirar
  "loaded_files": ["README.md"],       // ya en contexto → coste 0
  "methodology_loaded": false,
  "probe": true,                       // probe semántico interno
  "read_ratio": 1.0,                   // 1.0 = leer archivo completo
  "token_budget": 8000,
  "compact": true
}
```

### `POST /route/feedback`

Cierra el bucle: el agente reporta el coste real y si RAG falló.

```bash
curl -s -X POST http://localhost:8766/route/feedback \
  -H 'Content-Type: application/json' \
  -d '{"decision_id":"rt_4f6a2c91ab","actual_rag_tokens":1712,
       "actual_direct_tokens":430,"actual_total_tokens":430,
       "needed_followup":false,"answer_ok":true}'
```

Con `decision_id`, el router recupera `k`, methodology y el coste directo
completo de la decisión original. Aprende:

- coeficientes de `B` (OLS con las últimas 100 muestras),
- ratio real de lectura directa (`actual_direct / full_file_tokens`),
- tasa de fallo (`needed_followup`).

La calibración se guarda en `rag/data/router_calibration.json` (local,
ignorado por git).

### `GET /route/stats`

```bash
curl -s http://localhost:8766/route/stats | python -m json.tool
```

Devuelve decisiones por acción, confianza media, calibración actual y, cuando
hay feedback con ambas vías, `accuracy`, `regret` y ahorro frente a
«siempre RAG» / «siempre directo».

### CLI

```bash
python -m rag.router route --q "¿cómo funciona X?" --port 8766
python -m rag.router route --q "..." --files src/foo.py --verbose --pretty
python -m rag.router feedback --id rt_... --rag-tokens 1712 --direct-tokens 430
python -m rag.router stats --port 8766
```

---

## 4. Resultados del replay (33 consultas)

`benchmarks/run_router.py` reconstruye la decisión con los costes **reales**
A/B ya medidos (no vuelve a llamar a los servicios) e imputa el coste de la
vía elegida + 80 tok de decisión:

| Estrategia | Tokens totales | Ahorro vs siempre RAG | Regret vs oráculo |
|---|---:|---:|---:|
| Siempre RAG | 59 006 | — | +8 950 |
| Siempre directo | 109 283 | −50 277 | +59 227 |
| **Router (modelo)** | **52 824** | **+6 182 (10,5 %)** | **+2 768** |
| Router (B exacto, cota) | 52 872 | +6 134 | +2 816 |
| Oráculo (`min(A,B)`) | 50 056 | — | — |

- **Aciertos de vía:** 90,9 % (22 RAG, 11 directo); con B exacto, 93,9 %
  (19 RAG, 14 directo).
- **Captura el 69,1 %** del ahorro que conseguiría el oráculo.
- **Subgrupo A < 2k:** router 16 468 vs siempre RAG 24 170 (**+7 702**);
  frente a siempre directo 15 505 pierde **963 tok**, que son casi
  exactamente el coste de las 14 decisiones (14 × 80 = 1 120). Es el precio
  de preguntar; aun así evita la mayor parte del desperdicio del RAG.
- **Subgrupo A ≥ 2k:** router 36 356 vs siempre RAG 34 836; pierde sólo el
  coste de decisión (19 × 80 = 1 520), porque en ese rango RAG gana casi
  siempre.

> **Honestidad estadística:** la calibración por defecto se sembró con este
> mismo benchmark. El replay es **in-sample**: sirve para validar la lógica y
> la magnitud, no como inferencia sobre proyectos nuevos. El feedback real
> (`/route/feedback`) es el que adapta los coeficientes a cada proyecto.

Datos crudos: [`docs/benchmarks/results/router.json`](benchmarks/results/router.json).
Reproducir:

```bash
python benchmarks/run_router.py   # requiere haber corrido antes run_tokens.py
```

---

## 5. Integración con agentes

El flujo recomendado (y el que está en `AGENTS.md`) es:

1. **Antes de leer archivos**, si la tarea no es trivial:
   ```bash
   curl -s -X POST http://localhost:8766/route -H 'Content-Type: application/json' \
     -d '{"q":"<pregunta>","k":5,"methodology_loaded":false}'
   ```
2. Seguir `action`:
   - `direct` → leer sólo `read[].path`.
   - `rag` → `POST /query` con `rag.k` y `rag.methodology`.
   - `hybrid` → `/query` y después leer el top-1.
3. Si se usó el router, enviar feedback con los tokens reales y
   `needed_followup` para que el sistema aprenda.

**Cuándo NO llamar al router:** si el agente ya sabe que el archivo es diminuto
(<1k tok) o ya lo tiene cargado, leerlo directo es más barato que cualquier
decisión. El router está pensado para el caso incierto: no sabes si el destino
es de 300 o de 12 000 tokens.

---

## 6. Configuración (`rag.config.json → router`)

| Clave | Default | Uso |
|---|---|---|
| `enabled` | `true` | Activa `/route` |
| `probe_k` | `12` | k del probe interno |
| `probe_timeout_s` | `5` | Si Ollama no responde, se cae a probe lexical |
| `probe_cooldown_s` | `30` | Tras un fallo del probe, no reintentar Ollama durante este tiempo |
| `lexical_cache_s` | `30` | TTL de la lista de archivos del probe lexical |
| `max_direct_files` | `3` | Máximo de archivos a leer en la estimación |
| `direct_score_floor` | `0.45` | Corte de score para añadir candidatos |
| `min_savings_tokens` | `60` | Margen mínimo para elegir RAG |
| `decision_cost_tokens` | `80` | Coste estimado de `/route` (se resta del ahorro) |
| `base_tokens` | `110` | Intercepto del coste RAG |
| `per_result_tokens` | `220` | Coste por resultado de `/query` |
| `methodology_tokens` | `470` | Coste del «alma» |
| `direct_overhead_tokens` | `25` | Envoltorio por archivo leído |
| `direct_read_ratio` | `1.0` | Fracción leída (1.0 = completa) |
| `rag_miss_rate` | `0.05` | Probabilidad de que RAG no baste |
| `fallback_penalty_tokens` | `80` | Llamada extra si RAG falla |
| `allow_hybrid` | `true` | Permitir `rag + top-1` |
| `log_decisions` | `true` | Guarda decisiones/feedback locales |
| `max_file_bytes` | `2 000 000` | No leer completos archivos enormes |
| `tokenizer` | `auto` | `tiktoken` si está instalado; si no, heurística |

---

## 7. Limitaciones

1. **Tokenizador:** sin `tiktoken`, `chars/3.6` puede desviar ±10-15 %. El
   orden de magnitud y la decisión se mantienen; para precisión exacta:
   `pip install tiktoken`.
2. **Probe:** hit@5 = 96,97 %, pero la cobertura exacta del contenido no se
   midió. El término `p_miss` lo aproxima; el feedback lo corrige.
3. **Calibración in-sample:** los defaults vienen del benchmark de 33
   consultas. En un proyecto nuevo, los primeros feedbacks son los que
   importan.
4. **El coste de decidir existe:** ~40-80 tok. Para archivos <1k tokens, leer
   directo suele ser mejor que preguntar.
5. **Ground-truth del benchmark:** A = leer los archivos gt completos. Un
   agente que lea sólo fragmentos con `grep`/`read` parcial reduciría A; eso
   se modela con `read_ratio` y se aprende con feedback.
6. **Privacidad:** `router_decisions.jsonl` guarda la consulta truncada y
   paths locales en `rag/data/` (ignorado por git). Se puede desactivar con
   `log_decisions: false`.
