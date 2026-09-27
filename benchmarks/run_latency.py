#!/usr/bin/env python3
"""Latencia de los endpoints (cliente + servidor) — n=100 /query, n=50 /methodology.

Registra:
  - wall-time del cliente (incluye HTTP + serialización)
  - `elapsed_ms` reportado por el servidor (embed + búsqueda Chroma + inyección)
Comparación entre proyectos: Mann-Whitney U (¿el índice más grande es más lento?).

Salida: docs/benchmarks/results/latency.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import common as C

N_QUERY = 100
N_METH = 50
WARMUP = 5


def _pct(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()), "std": float(a.std(ddof=1)),
        **{f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 95, 99)},
        "min": float(a.min()), "max": float(a.max()),
    }


def run_project(project: str) -> dict:
    ds = C.load_dataset(project)
    qs = [q["q"] for q in ds["queries"]]

    for i in range(WARMUP):
        C.post_query(project, qs[i % len(qs)])

    q_client, q_server = [], []
    for i in range(N_QUERY):
        t0 = time.perf_counter()
        resp = C.post_query(project, qs[i % len(qs)], k=5, include_methodology=True)
        q_client.append((time.perf_counter() - t0) * 1000)
        q_server.append(resp.json()["elapsed_ms"])

    for i in range(3):
        C.get_raw(project, "/methodology")
    m_client = []
    for _ in range(N_METH):
        t0 = time.perf_counter()
        C.get_raw(project, "/methodology")
        m_client.append((time.perf_counter() - t0) * 1000)

    # /stats (también lo usa el flujo de agente)
    s_client = []
    for _ in range(20):
        t0 = time.perf_counter()
        C.get_json(project, "/stats")
        s_client.append((time.perf_counter() - t0) * 1000)

    return {
        "project": project,
        "query_client_ms": _pct(q_client),
        "query_server_ms": _pct(q_server),
        "methodology_client_ms": _pct(m_client),
        "stats_client_ms": _pct(s_client),
        "raw": {"query_client_ms": q_client, "query_server_ms": q_server,
                "methodology_client_ms": m_client},
    }


def main() -> None:
    out = {}
    for project in C.SERVICES:
        print(f"[latency] {project}…", flush=True)
        out[project] = run_project(project)

    from scipy import stats as sps

    a = out["kguard"]["raw"]["query_client_ms"]
    b = out["visionrt"]["raw"]["query_client_ms"]
    u = sps.mannwhitneyu(a, b, alternative="two-sided")
    out["_mwu_kguard_vs_visionrt"] = {
        "statistic": float(u.statistic), "p_two_sided": float(u.pvalue),
        "median_kguard": float(np.median(a)), "median_visionrt": float(np.median(b)),
        "cliffs_delta": float((u.statistic / (len(a) * len(b))) * 2 - 1),
    }
    path = C.save_result("latency.json", out)
    print(f"guardado → {path}")
    for p in C.SERVICES:
        qc, qs = out[p]["query_client_ms"], out[p]["query_server_ms"]
        print(f"  {p}: client p50={qc['p50']:.1f}ms p95={qc['p95']:.1f}ms p99={qc['p99']:.1f}ms | "
              f"server p50={qs['p50']:.1f}ms p95={qs['p95']:.1f}ms")


if __name__ == "__main__":
    main()
