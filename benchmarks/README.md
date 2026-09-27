# Benchmarks de RAG_portable

Mediciones reales del servicio sobre los dos despliegues vivos
(**KGuard** `:8766`, **VisionRT** `:8767`) con sus dos índices reales.

- **Resultados completos + todas las pruebas estadísticas:**
  [`../docs/benchmarks/REPORT.md`](../docs/benchmarks/REPORT.md)
- **Cifras clave en el README raíz.**

## Requisitos

- Los servicios del proyecto benchmarked levantados y con `drift=0`
  (`curl -s localhost:8766/stats`).
- Docker accesible (se lee el manifest de los contenedores y se embebe
  dentro de ellos).
- Python 3.10+ y:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python3 run_all.py                    # todo: tokens + retrieval + latency + embedding + figuras (~4 min)
python3 run_all.py --skip-embedding   # omitir el bench de Ollama (~1 min)
python3 run_tokens.py                 # pasos sueltos (mismos flags que arriba)
python3 run_retrieval.py
python3 run_latency.py
python3 run_embedding.py
python3 figures.py                    # regenera figs/ desde results/*.json
```

Todo es determinista (SEED=42 en bootstrap/permutaciones); las latencias
varían con la carga del host.

## Estructura

```
benchmarks/
├── common.py          # tokens (tiktoken), HTTP, tests estadísticos, I/O
├── datasets/
│   ├── kguard.json    # 16 consultas con ground-truth verificado
│   └── visionrt.json  # 17 consultas con ground-truth verificado
├── run_tokens.py      # H1/H3: ahorro de tokens + Wilcoxon/signo/bootstrap/ablación
├── run_retrieval.py   # H2: MRR/P@5/R@5/nDCG + test de permutación vs azar
├── run_latency.py     # percentiles /query|/methodology|/stats + Mann-Whitney
├── run_embedding.py   # bge-m3: latencia vs longitud, throughput, metadatos
├── run_all.py         # orquestador
└── figures.py         # figuras del informe
```

Salidas: `../docs/benchmarks/results/*.json` (crudos) y
`../docs/benchmarks/figs/*.png` (figuras).

## Diseño de las condiciones (resumen)

| Condición | Qué mide |
|---|---|
| **A** (sin RAG) | tokens de leer completos los archivos ground-truth de la consulta |
| **B** (con RAG) | tokens del payload HTTP real de `POST /query` (k=5, methodology on) |
| ablación B′ | igual sin `include_methodology` (coste del «alma») |
| azar (null) | rankings aleatorios sobre el corpus indexado real (B=5000) |

Tests: Wilcoxon un/bilateral, test de signo exacto, bootstrap percentílico
IC95 (10k), Cohen's dz, Shapiro, Spearman, test de permutación,
Mann-Whitney U. Detalle y valores en el informe.
