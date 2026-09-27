#!/usr/bin/env python3
"""H1 + H3 — Ahorro de tokens: RAG vs lectura directa de archivos.

Por consulta:
  A = tokens de los archivos ground-truth leídos completos  (sin RAG)
  B = tokens del cuerpo HTTP real de POST /query            (con RAG)
  ahorro = A - B ; ratio = B/A

Sesión completa (H3, claims README):
  A_soul = ficheros completos de los que salen los chunks /methodology
  B_soul = payload real de GET /methodology
  corpus = tokens de la documentación de onboarding (README+AGENTS+docs)

Tests: Wilcoxon (one-sided), sign test, bootstrap CI, Cohen's dz, Shapiro.
Salida: docs/benchmarks/results/tokens.json
"""
from __future__ import annotations

import sys
import time

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def corpus_onboarding_tokens(root: str) -> dict:
    """Tokens de README raíz + AGENTS.md + docs/**.md (lectura típica sin RAG)."""
    rootp = Path(root)
    files = []
    for pat in ("README.md", "AGENTS.md", "docs/**/*.md", "docs/**/*.markdown"):
        files.extend(p for p in rootp.glob(pat) if p.is_file())
    seen, uniq = set(), []
    for p in sorted(files):
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    per_file = {str(p.relative_to(rootp)): C.count_tokens(
        p.read_text(encoding="utf-8", errors="replace")) for p in uniq}
    return {"files": per_file, "total_tokens": sum(per_file.values()), "n_files": len(uniq)}


def run_project(project: str) -> dict:
    ds = C.load_dataset(project)
    root = ds["root"]
    rows = []
    for q in ds["queries"]:
        t0 = time.perf_counter()
        resp = C.post_query(project, q["q"], k=5, include_methodology=True)
        client_ms = (time.perf_counter() - t0) * 1000
        body = resp.text
        data = resp.json()

        a_tok = 0
        missing = []
        for rel in q["gt"]:
            try:
                a_tok += C.count_tokens(C.read_file(root, rel))
            except FileNotFoundError:
                missing.append(rel)
        b_tok = C.count_tokens(body)

        # ablación: misma consulta SIN inyectar methodology (coste del "alma")
        resp_nm = C.post_query(project, q["q"], k=5, include_methodology=False)
        b_nometh = C.count_tokens(resp_nm.text)

        top_files = [r["file"] for r in data["results"]]
        hit_files = sorted({f for f in top_files if f in q["gt"]})
        rows.append({
            "id": q["id"], "q": q["q"], "gt": q["gt"], "missing_gt": missing,
            "A_targeted_tokens": a_tok, "B_query_tokens": b_tok,
            "B_query_no_methodology_tokens": b_nometh,
            "saving_tokens": a_tok - b_tok,
            "saving_no_methodology_tokens": a_tok - b_nometh,
            "ratio_B_over_A": (b_tok / a_tok) if a_tok else None,
            "hit_at_5": bool(hit_files),
            "top_files": top_files,
            "server_elapsed_ms": data.get("elapsed_ms"),
            "client_ms": round(client_ms, 1),
            "B_chars": len(body),
        })

    # ── soul / methodology (una vez por sesión) ──
    m_resp = C.get_raw(project, "/methodology")
    m_data = m_resp.json()
    meth_files = sorted({c["file"] for c in m_data["chunks"]})
    a_soul, soul_missing = 0, []
    soul_per_file = {}
    for rel in meth_files:
        try:
            t = C.count_tokens(C.read_file(root, rel))
        except FileNotFoundError:
            soul_missing.append(rel)
            continue
        soul_per_file[rel] = t
        a_soul += t
    b_soul = C.count_tokens(m_resp.text)

    onboarding = corpus_onboarding_tokens(root)
    manifest = C.manifest_files(project)
    indexed_tokens = sum(
        C.count_tokens(C.read_file(root, f)) for f in manifest if (Path(root) / f).is_file()
    )

    return {
        "project": project,
        "n_queries": len(rows),
        "rows": rows,
        "soul": {
            "methodology_files": meth_files,
            "methodology_missing": soul_missing,
            "A_soul_full_tokens": a_soul,
            "B_soul_payload_tokens": b_soul,
            "per_file_tokens": soul_per_file,
            "B_session_tokens": b_soul + (rows[0]["B_query_tokens"] if rows else 0),
        },
        "corpus": {
            "onboarding_docs_tokens": onboarding["total_tokens"],
            "onboarding_n_files": onboarding["n_files"],
            "onboarding_per_file": onboarding["files"],
            "indexed_files": len(manifest),
            "indexed_corpus_tokens": indexed_tokens,
        },
    }


def main() -> None:
    out = {}
    for project in C.SERVICES:
        print(f"[tokens] {project}…", flush=True)
        out[project] = run_project(project)

    # ── tests estadísticos (pooled + por proyecto) ──
    def tests(rows: list) -> dict:
        from scipy import stats as sps

        diffs = [r["saving_tokens"] for r in rows]
        diffs_nm = [r["saving_no_methodology_tokens"] for r in rows]
        ratios = [r["ratio_B_over_A"] for r in rows if r["ratio_B_over_A"] is not None]
        hits = [r for r in rows if r["hit_at_5"]]
        d_hits = [r["saving_tokens"] for r in hits]
        big = [r for r in rows if r["A_targeted_tokens"] >= 2000]
        small = [r for r in rows if r["A_targeted_tokens"] < 2000]
        a_arr = np.array([r["A_targeted_tokens"] for r in rows], dtype=float)
        s_arr = np.array([r["saving_tokens"] for r in rows], dtype=float)
        spear = sps.spearmanr(a_arr, s_arr)
        med = float(np.median(diffs))
        lo, hi = C.bootstrap_ci(diffs, n_boot=10_000)
        rlo, rhi = C.bootstrap_ci(ratios, n_boot=10_000)
        return {
            "n": len(diffs),
            "wilcoxon_one-sided": C.wilcoxon(diffs, "greater"),
            "wilcoxon_two-sided": C.wilcoxon(diffs, "two-sided"),
            "sign_test": C.sign_test(diffs),
            "shapiro_diffs_p": C.shapiro_p(diffs),
            "cohen_dz": C.cohen_dz(diffs),
            "median_saving": med,
            "mean_saving": float(np.mean(diffs)),
            "median_saving_bootstrap_ci95": [lo, hi],
            "median_ratio_B_over_A": float(np.median(ratios)),
            "median_ratio_bootstrap_ci95": [rlo, rhi],
            "saving_desc": C.describe(diffs),
            "ratio_desc": C.describe(ratios),
            "n_B_lt_5000": sum(1 for r in rows if r["B_query_tokens"] < 5000),
            "n_negative_savings": sum(1 for v in diffs if v < 0),
            "spearman_A_vs_saving": {"rho": float(spear.statistic), "p": float(spear.pvalue)},
            "subgroup_A_ge_2000": {
                "n": len(big),
                "median_saving": float(np.median([r["saving_tokens"] for r in big])) if big else None,
                "wilcoxon_one-sided": C.wilcoxon([r["saving_tokens"] for r in big], "greater") if big else None,
            },
            "subgroup_A_lt_2000": {
                "n": len(small),
                "median_saving": float(np.median([r["saving_tokens"] for r in small])) if small else None,
                "wilcoxon_one-sided": C.wilcoxon([r["saving_tokens"] for r in small], "greater") if small else None,
            },
            "ablation_no_methodology": {
                "median_saving": float(np.median(diffs_nm)),
                "wilcoxon_one-sided": C.wilcoxon(diffs_nm, "greater"),
                "mean_methodology_overhead_tokens": float(np.mean(
                    [r["B_query_tokens"] - r["B_query_no_methodology_tokens"] for r in rows])),
            },
            "sensitivity_hits_only": {
                "n": len(d_hits),
                "wilcoxon_one-sided": C.wilcoxon(d_hits, "greater") if d_hits else None,
                "median_saving": float(np.median(d_hits)) if d_hits else None,
            },
        }

    pooled = [r for p in C.SERVICES for r in out[p]["rows"]]
    out["_stats"] = {
        "pooled": tests(pooled),
        "per_project": {p: tests(out[p]["rows"]) for p in out},
    }
    path = C.save_result("tokens.json", out)
    s = out["_stats"]["pooled"]
    print(f"guardado → {path}")
    print(f"  n={s['n']}  mediana ahorro={s['median_saving']:.0f} tok  "
          f"IC95=[{s['median_saving_bootstrap_ci95'][0]:.0f}, "
          f"{s['median_saving_bootstrap_ci95'][1]:.0f}]  "
          f"Wilcoxon p={C.fmt_p(s['wilcoxon_one-sided']['p'])}  "
          f"dz={s['cohen_dz']:.2f}")


if __name__ == "__main__":
    main()
