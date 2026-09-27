#!/usr/bin/env python3
"""Genera las figuras del benchmark a partir de docs/benchmarks/results/*.json."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C

C_A = "#c44e52"   # baseline sin RAG
C_B = "#4c72b0"   # con RAG
C_R = "#dd8452"   # random
C_G = "#55a868"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


def fig_token_bars(tok: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for ax, proj, title in zip(axes, ("kguard", "visionrt"),
                               ("KGuard (216 chunks)", "VisionRT (727 chunks)")):
        rows = sorted(tok[proj]["rows"], key=lambda r: -r["A_targeted_tokens"])
        x = np.arange(len(rows))
        w = 0.4
        ax.bar(x - w / 2, [r["A_targeted_tokens"] for r in rows], w,
               label="A · leer archivos gt (sin RAG)", color=C_A)
        ax.bar(x + w / 2, [r["B_query_tokens"] for r in rows], w,
               label="B · payload /query (con RAG)", color=C_B)
        ax.set_xticks(x)
        ax.set_xticklabels([r["id"] for r in rows], rotation=90, fontsize=8)
        ax.set_ylabel("tokens (cl100k_base)")
        ax.set_title(f"{title} — tokens por consulta")
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Coste de contexto por consulta: lectura directa vs RAG (H1)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "fig01_tokens_paired.png", bbox_inches="tight")
    plt.close(fig)


def fig_savings_hist(tok: dict) -> None:
    pooled = [r["saving_tokens"] for p in C.SERVICES for r in tok[p]["rows"]]
    kg = [r["saving_tokens"] for r in tok["kguard"]["rows"]]
    vr = [r["saving_tokens"] for r in tok["visionrt"]["rows"]]
    st = tok["_stats"]["pooled"]
    lo, hi = st["median_saving_bootstrap_ci95"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    bins = np.arange(-2000, 12001, 1000)
    axes[0].hist(pooled, bins=bins, color="#8c8c8c", edgecolor="white")
    axes[0].axvline(0, color="black", lw=1.2, ls="--", label="0 (sin ahorro)")
    axes[0].axvline(st["median_saving"], color=C_B, lw=2,
                    label=f"mediana={st['median_saving']:.0f} tok")
    axes[0].axvspan(lo, hi, color=C_B, alpha=0.15,
                    label=f"IC95 bootstrap [{lo:.0f}, {hi:.0f}]")
    axes[0].set_title("Ahorro por consulta — conjunto (n=33)")
    axes[0].set_xlabel("A − B (tokens)")
    axes[0].set_ylabel("nº de consultas")
    axes[0].legend(fontsize=8)

    axes[1].hist(kg, bins=bins, alpha=0.65, color=C_G, edgecolor="white", label="KGuard")
    axes[1].hist(vr, bins=bins, alpha=0.65, color=C_R, edgecolor="white", label="VisionRT")
    axes[1].axvline(0, color="black", lw=1.2, ls="--")
    axes[1].set_title("Desglose por proyecto")
    axes[1].set_xlabel("A − B (tokens)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Distribución del ahorro de tokens (positivo = el RAG ahorra)", fontsize=12)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "fig02_savings_hist.png", bbox_inches="tight")
    plt.close(fig)


def fig_crossover(tok: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for proj, color, marker in (("kguard", C_G, "o"), ("visionrt", C_R, "s")):
        rows = tok[proj]["rows"]
        ax.scatter([r["A_targeted_tokens"] for r in rows],
                   [r["saving_tokens"] for r in rows],
                   s=55, color=color, marker=marker, alpha=0.85, label=proj.capitalize(),
                   edgecolor="white")
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.axvline(2000, color="#555555", lw=1, ls=":", label="A = 2000 tok (corte subgrupo)")
    ax.set_xlabel("A · tokens del archivo objetivo (lectura directa)")
    ax.set_ylabel("Ahorro (A − B)")
    ax.set_title("El ahorro crece con el tamaño del destino (Spearman)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "fig03_crossover.png", bbox_inches="tight")
    plt.close(fig)


def fig_retrieval(ret: dict) -> None:
    projects = list(C.SERVICES)
    obs_mrr = [ret[p]["stats"]["observed"]["mrr"] for p in projects]
    rnd_mrr = [ret[p]["stats"]["random_baseline"]["mrr_mean"] for p in projects]
    obs_r5 = [ret[p]["stats"]["observed"]["recall_at_5"] for p in projects]
    rnd_r5 = [ret[p]["stats"]["random_baseline"]["recall_at_5_mean"] for p in projects]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    x = np.arange(len(projects))
    w = 0.35
    for ax, obs, rnd, title in (
        (axes[0], obs_mrr, rnd_mrr, "MRR"),
        (axes[1], obs_r5, rnd_r5, "Recall@5"),
        (axes[2],
         [ret[p]["stats"]["observed"]["hit_at_5"] for p in projects],
         [np.nan] * len(projects), "Hit@5 (aciertos en top-5)"),
    ):
        ax.bar(x - w / 2, obs, w, label="RAG semántico", color=C_B)
        if not np.all(np.isnan(rnd)):
            ax.bar(x + w / 2, rnd, w, label="aleatorio (null)", color=C_R)
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in projects])
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        for i, v in enumerate(obs):
            ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].legend(fontsize=8)
    p_k = ret["kguard"]["stats"]["permutation_p"]["mrr"]
    p_v = ret["visionrt"]["stats"]["permutation_p"]["mrr"]
    fig.suptitle(f"Calidad de recuperación vs azar — test de permutación "
                 f"(p_mrr: KGuard={p_k:.4f}, VisionRT={p_v:.4f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "fig04_retrieval.png", bbox_inches="tight")
    plt.close(fig)


def fig_latency_cdf(lat: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for proj, color in (("kguard", C_G), ("visionrt", C_R)):
        xs = np.sort(lat[proj]["raw"]["query_client_ms"])
        ys = np.arange(1, len(xs) + 1) / len(xs)
        axes[0].plot(xs, ys, lw=1.8, color=color, label=proj.capitalize())
    axes[0].axvline(50, color="#555555", ls=":", lw=1)
    axes[0].set_xlabel("latencia /query — cliente (ms)")
    axes[0].set_ylabel("proporción acumulada")
    axes[0].set_title("CDF de latencia /query (k=5, n=100)")
    axes[0].legend(fontsize=9)

    x = np.arange(len(C.SERVICES))
    w = 0.35
    clients = [lat[p]["query_client_ms"]["p95"] for p in C.SERVICES]
    servers = [lat[p]["query_server_ms"]["p95"] for p in C.SERVICES]
    axes[1].bar(x - w / 2, clients, w, color=C_B, label="cliente (HTTP completo)")
    axes[1].bar(x + w / 2, servers, w, color="#a0b8d8", label="servidor (embed+busca)")
    for i, (c, s) in enumerate(zip(clients, servers)):
        axes[1].text(i - w / 2, c + 1, f"{c:.0f}", ha="center", fontsize=8)
        axes[1].text(i + w / 2, s + 1, f"{s:.0f}", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.capitalize() for p in C.SERVICES], fontsize=9)
    axes[1].set_ylabel("p95 (ms)")
    axes[1].set_title("p95 /query por capa")
    axes[1].legend(fontsize=8)
    fig.suptitle("Latencia de los endpoints del RAG", fontsize=12)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "fig05_latency.png", bbox_inches="tight")
    plt.close(fig)


def fig_embedding(emb: dict) -> None:
    k = emb["kguard"]
    lens = sorted(int(n) for n in k["latency_by_chars"])
    p50 = [k["latency_by_chars"][str(n)]["p50_ms"] for n in lens]
    p95 = [k["latency_by_chars"][str(n)]["p95_ms"] for n in lens]
    toks = [k["latency_by_chars"][str(n)]["prompt_eval_count"] for n in lens]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(lens, p50, "o-", color=C_B, label="p50")
    axes[0].plot(lens, p95, "s--", color=C_A, label="p95")
    for n, t in zip(lens, toks):
        axes[0].annotate(f"{t:.0f} tok", (n, p50[lens.index(n)]),
                         textcoords="offset points", xytext=(6, -12), fontsize=8)
    axes[0].set_xlabel("longitud del texto (chars)")
    axes[0].set_ylabel("latencia /api/embed (ms)")
    axes[0].set_title("bge-m3: latencia vs longitud (n=15)")
    axes[0].legend(fontsize=9)

    rq = [emb[p]["latency_real_queries"] for p in C.SERVICES]
    axes[1].bar([p.capitalize() for p in C.SERVICES],
                [r["p50_ms"] for r in rq], 0.5, color=C_B, label="p50")
    axes[1].bar([p.capitalize() for p in C.SERVICES],
                [r["p95_ms"] for r in rq], 0.3, color=C_G, label="p95")
    for i, r in enumerate(rq):
        axes[1].text(i, r["p95_ms"] + 1, f"p95={r['p95_ms']:.0f}ms",
                     ha="center", fontsize=9)
    axes[1].set_ylabel("ms")
    axes[1].set_ylim(0, max(r["p95_ms"] for r in rq) * 1.3)
    axes[1].set_title("Latencia con consultas reales (n=33)")
    axes[1].legend(fontsize=9)
    th = k["throughput_32x512chars"]
    fig.suptitle(f"Modelo de embeddings bge-m3 (566.7M · F16 · dim 1024 · 1.16 GB) — "
                 f"throughput {th['seq_texts_per_s']:.1f} txt/s (batch ×{th['speedup_batch']:.1f})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "fig06_embedding.png", bbox_inches="tight")
    plt.close(fig)


def fig_session_soul(tok: dict) -> None:
    projects = list(C.SERVICES)
    a_soul = [tok[p]["soul"]["A_soul_full_tokens"] for p in projects]
    b_ses = [tok[p]["soul"]["B_soul_payload_tokens"] + tok[p]["rows"][0]["B_query_tokens"]
             for p in projects]
    x = np.arange(len(projects))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - w / 2, a_soul, w, label="A · ficheros 'alma' completos", color=C_A)
    ax.bar(x + w / 2, b_ses, w, label="B · /methodology + 1 consulta", color=C_B)
    for i, (a, b) in enumerate(zip(a_soul, b_ses)):
        ax.text(i - w / 2, a + 400, f"{a:,}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 400, f"{b:,}", ha="center", fontsize=9)
        ax.text(i, max(a, b) * 1.12, f"×{a / b:.1f} más barato", ha="center",
                fontsize=10, color=C_G, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in projects])
    ax.set_ylabel("tokens")
    ax.set_ylim(0, max(a_soul) * 1.28)
    ax.set_title("Sesión de agente: contexto del 'alma' — RAG vs lectura completa")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIGS_DIR / "fig07_session_soul.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    C.FIGS_DIR.mkdir(parents=True, exist_ok=True)
    tok = C.load_result("tokens.json")
    ret = C.load_result("retrieval.json")
    lat = C.load_result("latency.json")
    emb = C.load_result("embedding.json")

    fig_token_bars(tok)
    fig_savings_hist(tok)
    fig_crossover(tok)
    fig_retrieval(ret)
    fig_latency_cdf(lat)
    fig_embedding(emb)
    fig_session_soul(tok)
    print(f"figuras → {C.FIGS_DIR}")


if __name__ == "__main__":
    main()
