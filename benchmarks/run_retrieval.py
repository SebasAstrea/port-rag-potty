#!/usr/bin/env python3
"""H2 — Calidad de recuperación: métricas reales vs línea base aleatoria.

Por consulta (k=5, file-level sobre los chunks devueltos):
  hit@1, hit@5, P@5, R@5 (cobertura de archivos gt), RR (recíproco del primer
  acierto) → MRR, nDCG@5.

Línea base aleatoria: rankings sintéticos sobre el corpus indexado real
(tamaños de chunk incluidos) → distribución nula de MRR/P@5/R@5 con B=5000
permutaciones y test de permutación (p = P(null ≥ observado)).

Salida: docs/benchmarks/results/retrieval.json
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import common as C

B_PERM = 5000
K = 5


def eval_ranking(ranked_files: list[str], gt: set[str]) -> dict:
    y = [1 if f in gt else 0 for f in ranked_files]
    first = next((i + 1 for i, v in enumerate(y) if v), None)
    rr = 1.0 / first if first else 0.0
    # nDCG@5 con relevancia DEDUPLICADA por archivo: cada archivo gt cuenta una
    # sola vez (varios chunks del mismo archivo no suman relevancia repetida) →
    # garantiza nDCG ∈ [0, 1].
    seen: set[str] = set()
    y_dedup = []
    for f in ranked_files:
        if f in gt and f not in seen:
            seen.add(f)
            y_dedup.append(1)
        else:
            y_dedup.append(0)
    dcg = sum(v / math.log2(i + 2) for i, v in enumerate(y_dedup))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(K, len(gt))))
    return {
        "first_rank": first,
        "hit_at_1": bool(y[0]) if y else False,
        "hit_at_5": any(y),
        "precision_at_5": sum(y) / K,
        "recall_at_5": len(set(ranked_files) & gt) / len(gt),
        "reciprocal_rank": rr,
        "ndcg_at_5": dcg / idcg if idcg > 0 else 0.0,
    }


def run_project(project: str) -> dict:
    ds = C.load_dataset(project)
    manifest = C.manifest_files(project)

    # corpus virtual de chunks (archivo repetido n veces = n chunks indexados)
    corpus = np.array([f for f, n in manifest.items() for _ in range(n)])
    n_chunks = int(corpus.size)

    rows, gt_sets = [], []
    for q in ds["queries"]:
        resp = C.post_query(project, q["q"], k=K, include_methodology=True)
        ranked = [r["file"] for r in resp.json()["results"]]
        gt = set(q["gt"])
        m = eval_ranking(ranked, gt)
        m.update({"id": q["id"], "q": q["q"], "ranked_files": ranked, "gt": q["gt"]})
        rows.append(m)
        gt_sets.append(gt)

    # ── null aleatorio (permutaciones) ──
    gen = np.random.default_rng(C.SEED)
    null_mrr = np.empty(B_PERM)
    null_p5 = np.empty(B_PERM)
    null_r5 = np.empty(B_PERM)
    for b in range(B_PERM):
        rrs, p5s, r5s = [], [], []
        for gt in gt_sets:
            idx = gen.choice(n_chunks, size=K, replace=False)
            ranked = corpus[idx].tolist()
            m = eval_ranking(ranked, gt)
            rrs.append(m["reciprocal_rank"])
            p5s.append(m["precision_at_5"])
            r5s.append(m["recall_at_5"])
        null_mrr[b] = np.mean(rrs)
        null_p5[b] = np.mean(p5s)
        null_r5[b] = np.mean(r5s)

    obs = {
        "hit_at_1": float(np.mean([r["hit_at_1"] for r in rows])),
        "hit_at_5": float(np.mean([r["hit_at_5"] for r in rows])),
        "precision_at_5": float(np.mean([r["precision_at_5"] for r in rows])),
        "recall_at_5": float(np.mean([r["recall_at_5"] for r in rows])),
        "mrr": float(np.mean([r["reciprocal_rank"] for r in rows])),
        "ndcg_at_5": float(np.mean([r["ndcg_at_5"] for r in rows])),
    }
    ci_mrr = C.bootstrap_ci([r["reciprocal_rank"] for r in rows])
    ci_r5 = C.bootstrap_ci([r["recall_at_5"] for r in rows])

    stats = {
        "observed": obs,
        "mrr_bootstrap_ci95": list(ci_mrr),
        "recall_at_5_bootstrap_ci95": list(ci_r5),
        "random_baseline": {
            "B": B_PERM,
            "mrr_mean": float(null_mrr.mean()), "mrr_p95": float(np.percentile(null_mrr, 95)),
            "precision_at_5_mean": float(null_p5.mean()),
            "recall_at_5_mean": float(null_r5.mean()),
        },
        "permutation_p": {
            "mrr": C.permutation_pvalue(obs["mrr"], null_mrr),
            "precision_at_5": C.permutation_pvalue(obs["precision_at_5"], null_p5),
            "recall_at_5": C.permutation_pvalue(obs["recall_at_5"], null_r5),
        },
        "n_chunks_corpus": n_chunks,
        "n_files_corpus": len(manifest),
    }
    return {"project": project, "rows": rows, "stats": stats}


def main() -> None:
    out = {}
    for project in C.SERVICES:
        print(f"[retrieval] {project}…", flush=True)
        out[project] = run_project(project)

    # pooled
    all_rows = [r for p in C.SERVICES for r in out[p]["rows"]]
    pooled = {
        "n": len(all_rows),
        "hit_at_1": float(np.mean([r["hit_at_1"] for r in all_rows])),
        "hit_at_5": float(np.mean([r["hit_at_5"] for r in all_rows])),
        "mrr": float(np.mean([r["reciprocal_rank"] for r in all_rows])),
        "precision_at_5": float(np.mean([r["precision_at_5"] for r in all_rows])),
        "recall_at_5": float(np.mean([r["recall_at_5"] for r in all_rows])),
        "ndcg_at_5": float(np.mean([r["ndcg_at_5"] for r in all_rows])),
        "mrr_bootstrap_ci95": list(C.bootstrap_ci([r["reciprocal_rank"] for r in all_rows])),
        "misses": [r["id"] for r in all_rows if not r["hit_at_5"]],
    }
    out["_pooled"] = pooled
    path = C.save_result("retrieval.json", out)
    print(f"guardado → {path}")
    print(f"  pooled: hit@5={pooled['hit_at_5']:.2%} MRR={pooled['mrr']:.3f} "
          f"R@5={pooled['recall_at_5']:.3f} misses={pooled['misses']}")
    for p in C.SERVICES:
        s = out[p]["stats"]
        print(f"  {p}: MRR={s['observed']['mrr']:.3f} vs random={s['random_baseline']['mrr_mean']:.3f} "
              f"(p={C.fmt_p(s['permutation_p']['mrr'])})")


if __name__ == "__main__":
    main()
