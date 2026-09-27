"""Utilidades comunes del benchmark de RAG_portable.

Requiere: requests, scipy, numpy, tiktoken (pip install -r requirements-dev.txt).

Convenciones:
  - Reproducibilidad: SEED=42, rng por módulo.
  - Tokens: tiktoken `cl100k_base` como proxy de contexto de agente LLM
    (ver limitaciones en docs/benchmarks/REPORT.md).
  - Los resultados crudos se guardan como JSON en docs/benchmarks/results/.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import requests

SEED = 42
BENCH_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
RESULTS_DIR = REPO_DIR / "docs" / "benchmarks" / "results"
FIGS_DIR = REPO_DIR / "docs" / "benchmarks" / "figs"

SERVICES = {
    "kguard": "http://localhost:8766",
    "visionrt": "http://localhost:8767",
}
ROOTS = {
    "kguard": "/home/sebastian/Descargas/Android/KGuard",
    "visionrt": "/home/sebastian/Descargas/Android/VisionRT",
}

TIMEOUT = 60

# ── tokens ─────────────────────────────────────────────────────────────────
_ENC = None


def _enc_mod():
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def count_tokens(text: str) -> int:
    """Tokens de `text` con cl100k_base (proxy del contexto que ve el agente)."""
    if not text:
        return 0
    return len(_enc_mod().encode(text))


# ── HTTP ───────────────────────────────────────────────────────────────────
def get_json(project: str, path: str) -> Any:
    r = requests.get(f"{SERVICES[project]}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_query(project: str, q: str, k: int = 5,
               include_methodology: bool = True) -> requests.Response:
    r = requests.post(
        f"{SERVICES[project]}/query",
        json={"q": q, "k": k, "include_methodology": include_methodology},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r


def get_raw(project: str, path: str) -> requests.Response:
    """GET devolviendo la respuesta cruda (para medir los tokens reales del payload)."""
    r = requests.get(f"{SERVICES[project]}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r


# ── datasets ───────────────────────────────────────────────────────────────
def load_dataset(project: str) -> dict:
    with open(BENCH_DIR / "datasets" / f"{project}.json", encoding="utf-8") as f:
        return json.load(f)


def read_file(root: str, rel: str) -> str:
    p = Path(root) / rel
    if not p.is_file():
        raise FileNotFoundError(f"{p}")
    return p.read_text(encoding="utf-8", errors="replace")


def manifest_files(project: str) -> dict[str, int]:
    """{archivo: n_chunks} del índice vivo, leyendo el manifest dentro del contenedor.
    Requiere Docker (los servicios benchmarked corren en contenedores)."""
    import subprocess

    code = (
        "from rag import indexer; import json;"
        "m = indexer.Manifest.load();"
        "print(json.dumps({f: v.get('n', 0) for f, v in m.files.items()}))"
    )
    out = subprocess.run(
        ["docker", "exec", f"rag-{project}-rag", "python", "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"manifest_files({project}) failed: {out.stderr.strip()}")
    return json.loads(out.stdout.strip().splitlines()[-1])


# ── persistencia ───────────────────────────────────────────────────────────
def save_result(name: str, payload: Any) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def load_result(name: str) -> Any:
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def result_exists(name: str) -> bool:
    return (RESULTS_DIR / name).is_file()


# ── estadística ────────────────────────────────────────────────────────────
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def bootstrap_ci(x: Sequence[float], stat: Callable[[np.ndarray], float] = np.median,
                 n_boot: int = 10_000, ci: float = 0.95, seed: int = SEED) -> tuple[float, float]:
    """IC percentílico bootstrap (BCa no necesario para el reporte)."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    gen = np.random.default_rng(seed)
    idx = gen.integers(0, a.size, size=(n_boot, a.size))
    vals = np.array([stat(a[i]) for i in idx]) if a.size < 5000 else \
        np.array([stat(a[r]) for r in idx])
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def wilcoxon(diffs: Sequence[float], alternative: str = "greater") -> dict:
    """Wilcoxon signed-rank sobre diferencias emparejadas (H0: mediana = 0)."""
    from scipy import stats

    d = np.asarray([v for v in diffs if v != 0], dtype=float)
    n = len(d)
    if n == 0:
        return {"n_nonzero": 0, "stat": float("nan"), "p": float("nan"),
                "alternative": alternative}
    if n < 10:  # scipy >=1.9 permite zero_method; con pocos datos aproximamos
        zero = len([v for v in diffs if v == 0])
    res = stats.wilcoxon(d, alternative=alternative, method="auto")
    return {"n_nonzero": n, "stat": float(res.statistic), "p": float(res.pvalue),
            "alternative": alternative}


def sign_test(diffs: Sequence[float]) -> dict:
    """Test de signo exacto (binomial) sobre las diferencias ≠ 0."""
    from scipy import stats

    pos = sum(1 for v in diffs if v > 0)
    neg = sum(1 for v in diffs if v < 0)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p": float("nan")}
    res = stats.binomtest(pos, n, p=0.5, alternative="greater")
    return {"pos": pos, "neg": neg, "n": n, "p": float(res.pvalue)}


def shapiro_p(x: Sequence[float]) -> float:
    from scipy import stats

    a = np.asarray(x, dtype=float)
    if a.size < 3:
        return float("nan")
    return float(stats.shapiro(a).pvalue)


def cohen_dz(diffs: Sequence[float]) -> float:
    a = np.asarray(diffs, dtype=float)
    sd = a.std(ddof=1)
    return float(a.mean() / sd) if sd > 0 else float("nan")


def describe(x: Sequence[float]) -> dict:
    a = np.asarray(x, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()) if a.size else float("nan"),
        "median": float(np.median(a)) if a.size else float("nan"),
        "std": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        "min": float(a.min()) if a.size else float("nan"),
        "max": float(a.max()) if a.size else float("nan"),
        "p25": float(np.quantile(a, 0.25)) if a.size else float("nan"),
        "p75": float(np.quantile(a, 0.75)) if a.size else float("nan"),
    }


def percentiles(x: Sequence[float], ps: Iterable[int] = (50, 95, 99)) -> dict:
    a = np.asarray(x, dtype=float)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps} if a.size else {}


def permutation_pvalue(observed: float, null_samples: Sequence[float],
                       alternative: str = "greater") -> float:
    """p = (#{null ≥ obs} + 1) / (B + 1)  — corrección estándar +1."""
    null = np.asarray(null_samples, dtype=float)
    if alternative == "greater":
        ext = int(np.sum(null >= observed))
    else:
        ext = int(np.sum(null <= observed))
    return float((ext + 1) / (null.size + 1))


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}"
