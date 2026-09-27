#!/usr/bin/env python3
"""Benchmark del modelo de embeddings (bge-m3) corriendo DENTRO del contenedor.

Mide vía Ollama /api/embed (red del contenedor, sin publicar al host):
  - latencia por longitud de texto (64/256/1024/4000 chars), n=15 + 2 warmup
  - latencia con consultas reales del dataset
  - throughput: batch 32 en una llamada vs 32 llamadas secuenciales
  - metadatos del modelo (/api/tags): tamaño en disco, parámetros, dimensión

Salida: docs/benchmarks/results/embedding.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

PY_SCRIPT = r'''
import json, os, statistics, time, urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").split(",")[0].rstrip("/")
MODEL = os.environ.get("RAG_EMBED_MODEL", "bge-m3")

def post(path, payload, timeout=180):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return (time.perf_counter() - t0) * 1000.0, body

def get(path, timeout=30):
    with urllib.request.urlopen(HOST + path, timeout=timeout) as r:
        return json.load(r)

def embed_one(text):
    ms, body = post("/api/embed", {"model": MODEL, "input": text})
    ev = body.get("embeddings", [[]])[0]
    toks = body.get("prompt_eval_count")
    return ms, len(ev), toks

out = {"host": HOST, "model": MODEL}

# ── metadatos del modelo ──
tags = get("/api/tags")
m = next((x for x in tags.get("models", []) if x.get("name", "").startswith(MODEL)), None)
if m:
    out["model_meta"] = {
        "name": m.get("name"), "size_bytes": m.get("size"),
        "details": m.get("details", {}),
        "modified_at": m.get("modified_at"),
    }

# ── latencia por longitud ──
def mk(n):  # texto de n chars (mezcla es/en, típico de docs)
    base = "El sistema detecta obstáculos y personas en tiempo real con la cámara. "
    return (base * (n // len(base) + 1))[:n]

by_len = {}
for n in (64, 256, 1024, 4000):
    for _ in range(2):
        embed_one(mk(n))
    ms_list, dims, toks = [], set(), []
    for _ in range(15):
        ms, d, t = embed_one(mk(n))
        ms_list.append(ms); dims.add(d)
        if t: toks.append(t)
    ms_list.sort()
    by_len[str(n)] = {
        "n": len(ms_list), "mean_ms": statistics.fmean(ms_list),
        "p50_ms": statistics.median(ms_list),
        "p95_ms": ms_list[max(0, int(round(0.95 * (len(ms_list) - 1))))],
        "min_ms": ms_list[0], "max_ms": ms_list[-1],
        "dim": dims.pop() if len(dims) == 1 else sorted(dims),
        "prompt_eval_count": statistics.fmean(toks) if toks else None,
    }
out["latency_by_chars"] = by_len

# ── consultas reales del dataset (via stdin JSON) ──
queries = json.loads(os.environ["BENCH_QUERIES"])
ms_q = []
for q in queries:
    ms, d, t = embed_one(q)
    ms_q.append(ms)
ms_q.sort()
out["latency_real_queries"] = {
    "n": len(ms_q), "mean_ms": statistics.fmean(ms_q),
    "p50_ms": statistics.median(ms_q), "p95_ms": ms_q[int(0.95 * (len(ms_q) - 1))],
    "min_ms": ms_q[0], "max_ms": ms_q[-1],
}

# ── throughput batch vs secuencial ──
batch = [mk(512) for _ in range(32)]
ms_batch, body = post("/api/embed", {"model": MODEL, "input": batch})
seq = []
for t in batch:
    ms, _ = post("/api/embed", {"model": MODEL, "input": t})
    seq.append(ms)
out["throughput_32x512chars"] = {
    "batch_total_ms": ms_batch, "batch_per_text_ms": ms_batch / 32,
    "seq_total_ms": sum(seq), "seq_per_text_ms": statistics.fmean(seq),
    "speedup_batch": sum(seq) / ms_batch,
    "batch_texts_per_s": 32000.0 / ms_batch,
    "seq_texts_per_s": 32000.0 / sum(seq),
}

print(json.dumps(out))
'''


def run_in_container(container: str, queries: list[str]) -> dict:
    env = {"BENCH_QUERIES": json.dumps(queries)}
    cmd = ["docker", "exec", "-i", "-e", f"BENCH_QUERIES={env['BENCH_QUERIES']}",
           container, "python", "-"]
    p = subprocess.run(cmd, input=PY_SCRIPT, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"{container}: {p.stderr.strip()}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> None:
    out = {}
    for project in C.SERVICES:
        ds = C.load_dataset(project)
        queries = [q["q"] for q in ds["queries"]]
        print(f"[embedding] {project} (contenedor rag-{project}-rag)…", flush=True)
        out[project] = run_in_container(f"rag-{project}-rag", queries)

    path = C.save_result("embedding.json", out)
    print(f"guardado → {path}")
    for p, r in out.items():
        mm = r.get("model_meta", {})
        det = mm.get("details", {})
        lat = r["latency_by_chars"]
        rq = r["latency_real_queries"]
        th = r["throughput_32x512chars"]
        print(f"  {p}: params={det.get('parameter_size')} size={mm.get('size_bytes', 0)/1e9:.2f}GB "
              f"dim={lat['256']['dim']} | query embed p50={rq['p50_ms']:.1f}ms "
              f"| batch {th['batch_texts_per_s']:.1f} txt/s (speedup x{th['speedup_batch']:.1f})")


if __name__ == "__main__":
    main()
