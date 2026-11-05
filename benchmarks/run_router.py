#!/usr/bin/env python3
"""Replay del router sobre las 33 consultas del benchmark (sin red).

Usa los costes ya medidos en `docs/benchmarks/results/tokens.json`:
  A = tokens de leer completos los archivos ground-truth
  B = tokens del payload real de POST /query

Simula la decisión del router con su modelo de coste por defecto y con el
coste RAG exacto (cota superior). El coste que se imputa es el **real** de la
vía elegida (A o B) + el coste de la propia respuesta de /route. Así se puede
comparar contra:
  · siempre RAG          = Σ B
  · siempre directo      = Σ A
  · oráculo              = Σ min(A, B)

Salida: docs/benchmarks/results/router.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from rag import config  # noqa: E402
from rag.router import Calibration  # noqa: E402

RESULTS = REPO_DIR / "docs" / "benchmarks" / "results"
TOKENS_FILE = RESULTS / "tokens.json"
OUT_FILE = RESULTS / "router.json"
K = 5


def _sum(rows, key):
    return sum(float(r[key]) for r in rows)


def _stats(values):
    a = sorted(values)
    if not a:
        return {}
    n = len(a)
    return {
        "n": n,
        "mean": round(sum(a) / n, 1),
        "median": round(a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2, 1),
        "min": round(a[0], 1),
        "max": round(a[-1], 1),
    }


def simulate(rows: list[dict], cal: Calibration, cfg: dict, *, exact_rag: bool) -> dict:
    out = []
    for r in rows:
        a = float(r["A_targeted_tokens"])
        b = float(r["B_query_tokens"])
        rag_est = b if exact_rag else cal.estimate_rag(K, True)
        direct_est = a + cal.direct_overhead_tokens
        p_miss = cal.rag_miss_rate
        rag_expected = rag_est + p_miss * (direct_est + cal.fallback_penalty_tokens)
        min_savings = float(cfg["min_savings_tokens"])
        action = "direct" if direct_est <= rag_expected + min_savings else "rag"
        actual = b if action == "rag" else a
        cheaper = "rag" if b < a else "direct"
        out.append({
            "id": r["id"],
            "A": a,
            "B": b,
            "rag_estimate": round(rag_est, 1),
            "rag_expected": round(rag_expected, 1),
            "direct_estimate": round(direct_est, 1),
            "action": action,
            "actual_cost": round(actual, 1),
            "oracle_cost": round(min(a, b), 1),
            "regret": round(actual - min(a, b), 1),
            "correct": action == cheaper,
            "cheaper": cheaper,
        })
    return {"rows": out}


def summarize(rows: list[dict], sim: dict, cfg: dict) -> dict:
    decision_cost = float(cfg["decision_cost_tokens"])
    n = len(rows)
    always_rag = _sum(rows, "B_query_tokens")
    always_direct = _sum(rows, "A_targeted_tokens")
    oracle = sum(min(float(r["A_targeted_tokens"]), float(r["B_query_tokens"])) for r in rows)
    router_total = sum(r["actual_cost"] + decision_cost for r in sim["rows"])
    exact_sim = simulate(rows, Calibration.from_config(cfg), cfg, exact_rag=True)
    exact_total = sum(r["actual_cost"] + decision_cost for r in exact_sim["rows"])

    def block(rows_subset, sim_rows):
        n_sub = len(rows_subset)
        if not n_sub:
            return {}
        a = sum(float(r["A_targeted_tokens"]) for r in rows_subset)
        b = sum(float(r["B_query_tokens"]) for r in rows_subset)
        total = sum(r["actual_cost"] + decision_cost for r in sim_rows)
        return {
            "n": n_sub,
            "always_rag": round(b, 1),
            "always_direct": round(a, 1),
            "router": round(total, 1),
            "saving_vs_rag": round(b - total, 1),
            "saving_vs_direct": round(a - total, 1),
        }

    small = [(r, s) for r, s in zip(rows, sim["rows"]) if float(r["A_targeted_tokens"]) < 2000]
    big = [(r, s) for r, s in zip(rows, sim["rows"]) if float(r["A_targeted_tokens"]) >= 2000]
    false_rag = sum(1 for r in sim["rows"] if r["action"] == "rag" and r["cheaper"] == "direct")
    false_direct = sum(1 for r in sim["rows"] if r["action"] == "direct" and r["cheaper"] == "rag")

    return {
        "n": n,
        "k": K,
        "decision_cost_tokens": decision_cost,
        "calibration": cal_summary(cfg),
        "always_rag_tokens": round(always_rag, 1),
        "always_direct_tokens": round(always_direct, 1),
        "oracle_tokens": round(oracle, 1),
        "router_estimated_tokens": round(router_total, 1),
        "router_exact_rag_tokens": round(exact_total, 1),
        "saving_vs_always_rag": round(always_rag - router_total, 1),
        "saving_vs_always_direct": round(always_direct - router_total, 1),
        "regret_vs_oracle": round(router_total - oracle, 1),
        "regret_vs_oracle_exact_rag": round(exact_total - oracle, 1),
        "pct_of_oracle_saving_captured": round(
            (always_rag - router_total) / max(1e-9, always_rag - oracle), 4
        ),
        "actions": {
            "rag": sum(1 for r in sim["rows"] if r["action"] == "rag"),
            "direct": sum(1 for r in sim["rows"] if r["action"] == "direct"),
        },
        "errors": {
            "false_rag": false_rag,
            "false_direct": false_direct,
            "accuracy": round(sum(1 for r in sim["rows"] if r["correct"]) / n, 4),
        },
        "per_query_saving": _stats([
            max(float(r["B"]) - decision_cost - r["actual_cost"], 0)
            if r["action"] == "direct"
            else max(float(r["A"]) - decision_cost - r["actual_cost"], 0)
            for r in sim["rows"]
        ]),
        "subgroups": {
            "A_lt_2000": block([r for r, _ in small], [s for _, s in small]),
            "A_ge_2000": block([r for r, _ in big], [s for _, s in big]),
        },
        "exact_rag_actions": {
            "rag": sum(1 for r in exact_sim["rows"] if r["action"] == "rag"),
            "direct": sum(1 for r in exact_sim["rows"] if r["action"] == "direct"),
        },
    }


def cal_summary(cfg: dict) -> dict:
    cal = Calibration.from_config(cfg)
    return cal.summary()


def main() -> None:
    if not TOKENS_FILE.exists():
        raise SystemExit(
            f"Falta {TOKENS_FILE}. Ejecuta antes `python benchmarks/run_tokens.py`."
        )
    raw = json.loads(TOKENS_FILE.read_text("utf-8"))
    rows = [r for p in ("kguard", "visionrt") for r in raw[p]["rows"]]
    cfg = dict(config.ROUTER_DEFAULT)
    cfg.update(getattr(config, "ROUTER", {}))
    cal = Calibration.from_config(cfg)

    sim = simulate(rows, cal, cfg, exact_rag=False)
    summary = summarize(rows, sim, cfg)
    exact = simulate(rows, cal, cfg, exact_rag=True)
    exact_summary = summarize(rows, exact, cfg)

    payload = {
        "note": (
            "Replay sobre los costes reales A/B de tokens.json. La calibración "
            "por defecto se sembró con este mismo benchmark: es una evaluación "
            "in-sample, útil para validar la lógica, no para inferencia externa."
        ),
        "summary": summary,
        "exact_rag_summary": {
            "router_tokens": exact_summary["router_estimated_tokens"],
            "saving_vs_always_rag": exact_summary["saving_vs_always_rag"],
            "regret_vs_oracle": exact_summary["regret_vs_oracle"],
            "actions": exact_summary["actions"],
            "errors": exact_summary["errors"],
        },
        "rows": sim["rows"],
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    s = summary
    print(f"guardado -> {OUT_FILE}")
    print(
        f"  n={s['n']}  siempre RAG={s['always_rag_tokens']:.0f}  "
        f"siempre directo={s['always_direct_tokens']:.0f}  "
        f"oraculo={s['oracle_tokens']:.0f}"
    )
    print(
        f"  router={s['router_estimated_tokens']:.0f}  "
        f"ahorro vs RAG={s['saving_vs_always_rag']:.0f}  "
        f"regret vs oráculo={s['regret_vs_oracle']:.0f}  "
        f"aciertos={s['errors']['accuracy'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
