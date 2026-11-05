"""RAG portable · router de decisión: ¿RAG o lectura directa?

Motivación medida (n=33, docs/benchmarks/REPORT.md):
  · El RAG ahorra de mediana +801 tok/consulta, pero **12/33** consultas
    cuestan más que leer el archivo directamente.
  · El cruce está en ~1,7-2k tokens de destino: por debajo, leer el archivo
    gana; por encima, el RAG gana.
  · El «alma» (methodology) añade ~438-566 tok por consulta.

Este módulo decide la vía ANTES de que el agente lea nada:

  1. Si el agente ya conoce candidatos (`candidate_files`), se usan.
     Si no, se hace un *probe* semántico interno (Chroma + Ollama) para
     descubrir archivos probables. El probe no se devuelve: no cuesta tokens
     al agente, sólo ~30-50 ms de servidor.
  2. Estima el coste real de `/query` (si hay probe, tokeniza el payload
     exacto que se devolvería) y el de leer los archivos completos.
  3. Incorpora el riesgo de fallo de recuperación: si el top-k no basta,
     el coste esperado del RAG es B + p_miss·(A + penalización).
  4. Elige la vía más barata con un margen (`min_savings_tokens`) y devuelve
     una respuesta **compacta** (~100-150 tok) con el plan: `rag`, `direct`
     o `hybrid`.

Además aprende de feedback (`POST /route/feedback`): recalibra el coste del
RAG, el ratio de lectura directa y la tasa de fallo con datos reales. La
calibración vive en `rag/data/router_calibration.json` (local, gitignored).

La respuesta de `/route` también cuesta tokens. Por eso se descuenta
`decision_cost_tokens` del ahorro y se devuelve `save` ya neto.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from . import config

log = logging.getLogger("rag.router")

PRIOR_MISS_RATE = 0.05
PRIOR_MISS_N = 20.0

_STOPWORDS = frozenset({
    # español
    "que", "como", "cual", "cuales", "donde", "cuando", "por", "para", "con",
    "sin", "sobre", "entre", "este", "esta", "estos", "estas", "ese", "esa",
    "esos", "esas", "del", "las", "los", "una", "unos", "unas", "porque",
    "sirve", "funciona", "hace", "hacer", "tiene", "tienen", "puedo", "puede",
    "debe", "deben", "desde", "hasta", "todo", "toda", "todos", "todas",
    "hay", "más", "mas", "muy", "sin", "según", "segun", "cómo", "como",
    # inglés
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "what", "which", "where", "when", "how", "does", "doesn", "are", "was",
    "were", "have", "has", "had", "can", "could", "should", "would", "into",
    "about", "your", "you", "its", "our", "their", "there", "here", "not",
    "all", "any", "but", "use", "used", "using", "get", "gets", "make",
})

_WORD_RE = re.compile(r"[a-z0-9_áéíóúüñ]{3,}", re.IGNORECASE)


# ── utilidades numéricas ─────────────────────────────────────────────────────
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _median(xs: Sequence[float]) -> float:
    a = sorted(float(x) for x in xs)
    if not a:
        return 0.0
    n = len(a)
    return a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2.0


def _normalize(text: str) -> str:
    """Minúsculas + sin acentos, para matching lexical robusto."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _terms(text: str) -> list[str]:
    out: list[str] = []
    for t in _WORD_RE.findall(_normalize(text)):
        if t not in _STOPWORDS and t not in out:
            out.append(t)
    return out


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gauss con pivote parcial para sistemas pequeños; None si es singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-9:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f == 0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= f * m[col][j]
    return [m[i][n] for i in range(n)]


# ── calibración ────────────────────────────────────────────────────────────
@dataclass
class Calibration:
    """Coeficientes de coste + muestras para recalibrar con feedback real."""

    base_tokens: float = 110.0
    per_result_tokens: float = 220.0
    methodology_tokens: float = 470.0
    direct_overhead_tokens: float = 25.0
    direct_read_ratio: float = 1.0
    rag_miss_rate: float = PRIOR_MISS_RATE
    fallback_penalty_tokens: float = 80.0
    rag_samples: list[list[float]] = field(default_factory=list)
    direct_samples: list[float] = field(default_factory=list)
    miss_samples: list[int] = field(default_factory=list)
    updated_at: float = 0.0

    # ── construcción ──
    @classmethod
    def from_config(cls, cfg: dict) -> "Calibration":
        return cls(
            base_tokens=float(cfg.get("base_tokens", 110)),
            per_result_tokens=float(cfg.get("per_result_tokens", 220)),
            methodology_tokens=float(cfg.get("methodology_tokens", 470)),
            direct_overhead_tokens=float(cfg.get("direct_overhead_tokens", 25)),
            direct_read_ratio=float(cfg.get("direct_read_ratio", 1.0)),
            rag_miss_rate=float(cfg.get("rag_miss_rate", PRIOR_MISS_RATE)),
            fallback_penalty_tokens=float(cfg.get("fallback_penalty_tokens", 80)),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        cal = cls()
        for key in (
            "base_tokens", "per_result_tokens", "methodology_tokens",
            "direct_overhead_tokens", "direct_read_ratio", "rag_miss_rate",
            "fallback_penalty_tokens", "updated_at",
        ):
            if key in data and data[key] is not None:
                setattr(cal, key, float(data[key]))
        for key in ("rag_samples", "direct_samples", "miss_samples"):
            if isinstance(data.get(key), list):
                setattr(cal, key, data[key])
        cal.rag_samples = [
            [float(v) for v in s]
            for s in cal.rag_samples[-200:]
            if isinstance(s, (list, tuple)) and len(s) >= 3
        ]
        cal.direct_samples = [
            float(v) for v in cal.direct_samples[-100:]
            if isinstance(v, (int, float))
        ]
        cal.miss_samples = [int(v) for v in cal.miss_samples[-200:]]
        cal._clamp()
        return cal

    def to_dict(self) -> dict:
        return {
            "base_tokens": round(self.base_tokens, 2),
            "per_result_tokens": round(self.per_result_tokens, 2),
            "methodology_tokens": round(self.methodology_tokens, 2),
            "direct_overhead_tokens": round(self.direct_overhead_tokens, 2),
            "direct_read_ratio": round(self.direct_read_ratio, 4),
            "rag_miss_rate": round(self.rag_miss_rate, 4),
            "fallback_penalty_tokens": round(self.fallback_penalty_tokens, 2),
            "rag_samples": [[round(float(v), 3) for v in s] for s in self.rag_samples],
            "direct_samples": [round(float(v), 4) for v in self.direct_samples],
            "miss_samples": list(self.miss_samples),
            "updated_at": self.updated_at,
        }

    def _clamp(self) -> None:
        self.base_tokens = _clamp(self.base_tokens, 0.0, 1500.0)
        self.per_result_tokens = _clamp(self.per_result_tokens, 20.0, 1000.0)
        self.methodology_tokens = _clamp(self.methodology_tokens, 0.0, 3000.0)
        self.direct_overhead_tokens = _clamp(self.direct_overhead_tokens, 0.0, 300.0)
        self.direct_read_ratio = _clamp(self.direct_read_ratio, 0.05, 3.0)
        self.rag_miss_rate = _clamp(self.rag_miss_rate, 0.0, 0.9)
        self.fallback_penalty_tokens = _clamp(self.fallback_penalty_tokens, 0.0, 1000.0)

    # ── estimación ──
    def estimate_rag(self, k: int, methodology: bool) -> float:
        est = self.base_tokens + self.per_result_tokens * max(0, int(k))
        if methodology:
            est += self.methodology_tokens
        return max(1.0, est)

    # ── aprendizaje ──
    def observe_rag(self, k: int, methodology: bool, actual_tokens: float) -> None:
        if actual_tokens <= 0:
            return
        self.rag_samples.append([float(k), 1.0 if methodology else 0.0, float(actual_tokens)])
        self.rag_samples = self.rag_samples[-200:]
        if len(self.rag_samples) >= 8:
            self._fit_rag()
        else:
            # Con pocas muestras, ajuste suave del intercepto.
            pred = self.estimate_rag(k, methodology)
            self.base_tokens = _clamp(
                self.base_tokens + 0.35 * (actual_tokens - pred), 0.0, 1500.0
            )
        self.updated_at = time.time()
        self._clamp()

    def _fit_rag(self) -> None:
        """OLS con features [1, k, methodology]; fallback robusto si es singular."""
        xs = self.rag_samples[-100:]
        # 3x3
        s1 = float(len(xs))
        sk = sum(x[0] for x in xs)
        sm = sum(x[1] for x in xs)
        skk = sum(x[0] * x[0] for x in xs)
        skm = sum(x[0] * x[1] for x in xs)
        smm = sm  # m ∈ {0,1} → m² = m
        sy = sum(x[2] for x in xs)
        sky = sum(x[0] * x[2] for x in xs)
        smy = sum(x[1] * x[2] for x in xs)
        beta = _solve_linear(
            [[s1, sk, sm], [sk, skk, skm], [sm, skm, smm]],
            [sy, sky, smy],
        )
        if beta and all(math.isfinite(v) for v in beta):
            self.base_tokens, self.per_result_tokens, self.methodology_tokens = beta
            self._clamp()
            return
        # Fallback: mantener base/methodology y estimar per_result por mediana.
        ratios = [
            (y - self.base_tokens - self.methodology_tokens * m) / max(k, 1.0)
            for k, m, y in xs
        ]
        self.per_result_tokens = _clamp(_median(ratios), 20.0, 1000.0)
        residuals = [
            y - self.base_tokens - self.per_result_tokens * k - self.methodology_tokens * m
            for k, m, y in xs
        ]
        self.base_tokens = _clamp(self.base_tokens + _median(residuals), 0.0, 1500.0)

    def observe_direct_ratio(self, ratio: float) -> None:
        if not (0.01 <= ratio <= 5.0):
            return
        self.direct_samples.append(float(ratio))
        self.direct_samples = self.direct_samples[-100:]
        if len(self.direct_samples) >= 3:
            self.direct_read_ratio = _median(self.direct_samples)
        else:
            self.direct_read_ratio = 0.5 * self.direct_read_ratio + 0.5 * ratio
        self.direct_read_ratio = _clamp(self.direct_read_ratio, 0.05, 3.0)
        self.updated_at = time.time()

    def observe_miss(self, missed: bool) -> None:
        self.miss_samples.append(1 if missed else 0)
        self.miss_samples = self.miss_samples[-200:]
        total = PRIOR_MISS_RATE * PRIOR_MISS_N + sum(self.miss_samples)
        n = PRIOR_MISS_N + len(self.miss_samples)
        self.rag_miss_rate = _clamp(total / n, 0.0, 0.9)
        self.updated_at = time.time()

    def summary(self) -> dict:
        return {
            "base_tokens": round(self.base_tokens, 1),
            "per_result_tokens": round(self.per_result_tokens, 1),
            "methodology_tokens": round(self.methodology_tokens, 1),
            "direct_read_ratio": round(self.direct_read_ratio, 3),
            "rag_miss_rate": round(self.rag_miss_rate, 4),
            "fallback_penalty_tokens": round(self.fallback_penalty_tokens, 1),
            "n_rag_samples": len(self.rag_samples),
            "n_direct_samples": len(self.direct_samples),
            "n_miss_samples": len(self.miss_samples),
            "updated_at": self.updated_at,
        }


# ── tokenizador ────────────────────────────────────────────────────────────
class TokenCounter:
    """tiktoken cl100k_base si está disponible; si no, aproximación por chars."""

    def __init__(self, mode: str = "auto") -> None:
        self.mode = "heuristic"
        self._enc = None
        self._cache: dict[tuple[str, int, int], int] = {}
        if mode in ("auto", "tiktoken"):
            try:
                import tiktoken  # type: ignore

                self._enc = tiktoken.get_encoding("cl100k_base")
                self.mode = "tiktoken:cl100k_base"
            except Exception:  # noqa: BLE001
                if mode == "tiktoken":
                    log.warning("tiktoken no disponible; usando heurística chars/3.6")

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, math.ceil(len(text) / 3.6))

    def count_file(self, path: Path) -> tuple[int, int, bool]:
        """(tokens, n_bytes, truncated). Cachea por mtime+size."""
        try:
            st = path.stat()
        except OSError:
            return (0, 0, False)
        key = (str(path), st.st_mtime_ns, st.st_size)
        cached = self._cache.get(key)
        if cached is not None:
            return (cached, st.st_size, False)
        if st.st_size > int(config.ROUTER.get("max_file_bytes", 2_000_000)):
            # No leer archivos enormes: estimación por tamaño, marcada.
            return (max(1, math.ceil(st.st_size / 3.6)), st.st_size, True)
        try:
            raw = path.read_bytes()
        except OSError:
            return (0, st.st_size, False)
        if b"\x00" in raw[:8192]:
            # Binario: no tiene sentido leerlo como contexto de texto.
            return (max(1, math.ceil(st.st_size / 3.6)), st.st_size, True)
        text = raw.decode("utf-8", errors="replace")
        n = self.count(text)
        self._cache[key] = n
        if len(self._cache) > 4096:
            self._cache.clear()
        return (n, st.st_size, False)


# ── probe ──────────────────────────────────────────────────────────────────
@dataclass
class ProbeChunk:
    id: str
    file: str
    section: str = ""
    score: float | None = None
    doc: str = ""
    meta: dict = field(default_factory=dict)

    def payload(self, max_chars: int = 600) -> dict:
        meta = self.meta or {}
        return {
            "id": self.id,
            "score": round(self.score, 4) if self.score is not None else None,
            "file": meta.get("file", self.file),
            "kind": meta.get("kind", ""),
            "section": meta.get("section", self.section),
            "category": meta.get("category", ""),
            "start_line": int(meta.get("start_line", 0) or 0),
            "end_line": int(meta.get("end_line", 0) or 0),
            "snippet": (self.doc or "")[:max_chars],
            "methodology": bool(meta.get("methodology", False)),
        }


@dataclass
class ProbeResult:
    source: str = "none"          # semantic | lexical | none
    chunks: list[ProbeChunk] = field(default_factory=list)
    methodology: list[ProbeChunk] = field(default_factory=list)
    top_score: float = 0.0


@dataclass
class FileInfo:
    path: str
    tokens: int
    n_bytes: int
    lines: int = 0
    readable: bool = True
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "tokens": self.tokens,
            "bytes": self.n_bytes,
            "lines": self.lines,
            "readable": self.readable,
            "truncated": self.truncated,
        }


# ── router ─────────────────────────────────────────────────────────────────
class QueryRouter:
    """Decide RAG vs lectura directa con estimación de coste y feedback."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        config_overrides: dict | None = None,
        state_dir: str | Path | None = None,
        counter: TokenCounter | None = None,
        calibration: Calibration | None = None,
        collection: Any = None,
        graph: Any = None,
        embed: Callable[[Sequence[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self.root = Path(project_root or config.PROJECT_ROOT).resolve()
        self.cfg = {
            **config.ROUTER_DEFAULT,
            **getattr(config, "ROUTER", {}),
            **(config_overrides or {}),
        }
        self.state_dir = Path(state_dir or config.DATA_DIR)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_file = self.state_dir / "router_calibration.json"
        self.decisions_file = self.state_dir / "router_decisions.jsonl"
        self.feedback_file = self.state_dir / "router_feedback.jsonl"
        self.counter = counter or TokenCounter(str(self.cfg.get("tokenizer", "auto")))
        self.cal = calibration or self._load_calibration()
        self.collection = collection
        self.graph = graph
        self.embed = embed
        self._file_cache: dict[tuple[str, int, int], FileInfo] = {}
        # Circuit breaker del probe semántico: si Ollama falla, no reintentar
        # en cada decisión (evita 5 s de espera repetidos).
        self._embed_down_until: float = 0.0
        # Caché de la lista de archivos del fallback lexical (TTL configurable).
        self._lex_files_cache: tuple[float, list[str]] | None = None

    # ── estado / calibración ──
    def _load_calibration(self) -> Calibration:
        try:
            if self.calibration_file.exists():
                data = json.loads(self.calibration_file.read_text("utf-8"))
                if isinstance(data, dict):
                    return Calibration.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            log.warning("calibración del router ilegible (%s); usando defaults", e)
        return Calibration.from_config(self.cfg)

    def save_calibration(self) -> None:
        try:
            tmp = self.calibration_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.cal.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.calibration_file)
        except OSError as e:
            log.warning("no se pudo guardar la calibración del router: %s", e)

    # ── resolución segura de paths ──
    def _inside(self, p: Path) -> bool:
        try:
            p.relative_to(self.root)
            return True
        except ValueError:
            return False

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def resolve_files(self, patterns: Sequence[str] | None) -> list[Path]:
        """Resuelve paths/globs relativos al proyecto; ignora escapes del root."""
        out: list[Path] = []
        seen: set[str] = set()
        for raw in patterns or []:
            raw = str(raw).strip().replace("\\", "/")
            if not raw:
                continue
            if Path(raw).is_absolute() or re.match(r"^[a-zA-Z]:/", raw):
                log.debug("router: path absoluto ignorado (%s)", raw)
                continue
            if any(ch in raw for ch in "*?[]"):
                try:
                    matches: Iterable[Path] = sorted(self.root.glob(raw))
                except (OSError, ValueError):
                    continue
            else:
                matches = [self.root / raw]
            for m in matches:
                try:
                    rp = m.resolve()
                except OSError:
                    continue
                if not rp.is_file() or not self._inside(rp):
                    continue
                key = str(rp)
                if key in seen:
                    continue
                seen.add(key)
                out.append(rp)
                if len(out) >= 200:
                    return out
        return out

    def _normalize_loaded(self, loaded_files: Sequence[str] | None) -> set[str]:
        out: set[str] = set()
        for raw in loaded_files or []:
            raw = str(raw).strip().replace("\\", "/")
            if not raw or Path(raw).is_absolute():
                continue
            try:
                rp = (self.root / raw).resolve()
            except OSError:
                continue
            if self._inside(rp):
                out.add(self._rel(rp))
        return out

    def _file_info(self, rp: Path) -> FileInfo | None:
        try:
            st = rp.stat()
        except OSError:
            return None
        key = (str(rp), st.st_mtime_ns, st.st_size)
        cached = self._file_cache.get(key)
        if cached is not None:
            return cached
        tokens, nbytes, truncated = self.counter.count_file(rp)
        info = FileInfo(
            path=self._rel(rp),
            tokens=tokens,
            n_bytes=nbytes,
            lines=-1 if truncated else self._count_lines(rp),
            readable=not truncated,
            truncated=truncated,
        )
        self._file_cache[key] = info
        return info

    @staticmethod
    def _count_lines(path: Path) -> int:
        try:
            with open(path, "rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def describe_files(self, paths: Sequence[Path]) -> list[FileInfo]:
        out: list[FileInfo] = []
        for p in paths:
            info = self._file_info(p)
            if info is not None:
                out.append(info)
        return out

    # ── probe semántico / lexical ──
    def _get_collection(self):
        c = self.collection
        return c() if callable(c) else c

    def _get_graph(self):
        g = self.graph
        return g() if callable(g) else g

    async def probe(
        self,
        q: str,
        *,
        k: int | None = None,
        include_methodology: bool = True,
        category: str | None = None,
        file_glob: str | None = None,
    ) -> ProbeResult:
        k = int(k or self.cfg.get("probe_k", 12))
        col = self._get_collection()
        if col is None or self.embed is None:
            return self.lexical_probe(q, limit=k)
        # Circuit breaker: si el último probe semántico falló, no volver a
        # esperar el timeout en cada decisión mientras Ollama siga caído.
        if self._embed_down_until > time.monotonic():
            return self.lexical_probe(q, limit=k)
        try:
            vecs = await asyncio.wait_for(
                self.embed([q]),
                timeout=float(self.cfg.get("probe_timeout_s", 5.0)),
            )
        except Exception as e:  # noqa: BLE001
            self._embed_down_until = time.monotonic() + float(
                self.cfg.get("probe_cooldown_s", 30.0)
            )
            log.warning("probe semántico falló (%s); usando lexical", e)
            return self.lexical_probe(q, limit=k)
        if not vecs:
            return self.lexical_probe(q, limit=k)
        try:
            where: dict | None = None
            conds: list[dict] = []
            if category:
                conds.append({"category": category})
            if file_glob:
                g = self._get_graph()
                files = sorted(
                    f for f in (getattr(g, "files", {}) or {}) if file_glob in f
                )
                if not files:
                    return self.lexical_probe(q, limit=k)
                conds.append({"file": {"$in": files}})
            if len(conds) == 1:
                where = conds[0]
            elif len(conds) > 1:
                where = {"$and": conds}
            res = col.query(
                query_embeddings=[vecs[0]],
                n_results=max(1, k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("consulta a Chroma falló (%s); usando lexical", e)
            return self.lexical_probe(q, limit=k)

        chunks: list[ProbeChunk] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, cid in enumerate(ids):
            meta = metas[i] or {}
            dist = dists[i] if i < len(dists) else None
            chunks.append(ProbeChunk(
                id=str(cid),
                file=str(meta.get("file", "")),
                section=str(meta.get("section", "")),
                score=(1 - float(dist)) if dist is not None else None,
                doc=str(docs[i] or "") if i < len(docs) else "",
                meta=meta,
            ))
        top = max((c.score or 0.0) for c in chunks) if chunks else 0.0

        methodology: list[ProbeChunk] = []
        if include_methodology:
            try:
                m = col.get(
                    where={"methodology": True},
                    include=["documents", "metadatas"],
                    limit=3,
                )
                for i, cid in enumerate(m.get("ids") or []):
                    meta = (m.get("metadatas") or [])[i] or {}
                    methodology.append(ProbeChunk(
                        id=str(cid),
                        file=str(meta.get("file", "")),
                        section=str(meta.get("section", "")),
                        score=None,
                        doc=str((m.get("documents") or [])[i] or ""),
                        meta=meta,
                    ))
            except Exception as e:  # noqa: BLE001
                log.debug("probe methodology falló: %s", e)

        return ProbeResult(
            source="semantic" if chunks else "lexical",
            chunks=chunks,
            methodology=methodology,
            top_score=float(top),
        )

    def lexical_probe(self, q: str, *, limit: int = 12) -> ProbeResult:
        """Fallback barato si no hay Chroma/Ollama: score por términos en rutas."""
        terms = _terms(q)
        g = self._get_graph()
        file_categories: dict[str, list[str]] = {}
        if g is not None:
            for cid, data in (getattr(g, "categories", {}) or {}).items():
                for f in (data.get("files", set()) or set()):
                    file_categories.setdefault(str(f), []).append(str(cid))

        files: list[str] = []
        if g is not None and getattr(g, "files", None):
            files = [str(f) for f in g.files]
        if not files:
            # El recorrido del árbol es lo más caro del fallback; se cachea
            # para que decisiones seguidas no lo repitan.
            now = time.monotonic()
            ttl = float(self.cfg.get("lexical_cache_s", 30.0))
            cached = self._lex_files_cache
            if cached is not None and (now - cached[0]) < ttl:
                files = cached[1]
            else:
                try:
                    from . import chunker

                    files = [
                        self._rel(p) for p in chunker.iter_project_files(self.root)
                    ]
                except Exception:  # noqa: BLE001
                    files = []
                if not files:
                    try:
                        files = [
                            p.relative_to(self.root).as_posix()
                            for p in sorted(self.root.rglob("*"))
                            if p.is_file()
                        ]
                    except OSError:
                        files = []
                self._lex_files_cache = (now, files)

        scored: list[tuple[float, str]] = []
        for rel in files:
            rel_n = _normalize(rel)
            base = rel_n.rsplit("/", 1)[-1]
            cats = " ".join(file_categories.get(rel, []))
            score = 0.0
            for t in terms:
                if t in base:
                    score += 4.0
                elif t in rel_n:
                    score += 2.0
                if t in cats:
                    score += 1.5
            if score > 0:
                scored.append((score, rel))
        scored.sort(key=lambda x: (-x[0], x[1]))
        chunks = [
            ProbeChunk(id=f"lex:{rel}", file=rel, score=score)
            for score, rel in scored[: max(1, limit)]
        ]
        return ProbeResult(
            source="lexical" if chunks else "none",
            chunks=chunks,
            top_score=float(scored[0][0]) if scored else 0.0,
        )

    # ── estimación del payload RAG ──
    def estimate_payload(
        self,
        q: str,
        probe: ProbeResult,
        *,
        k: int,
        include_methodology: bool,
    ) -> float | None:
        """Tokeniza el JSON que /query devolvería con los chunks del probe."""
        if probe.source != "semantic" or not probe.chunks:
            return None
        payload = {
            "q": q,
            "k": k,
            "results": [c.payload(600) for c in probe.chunks[:k]],
            "methodology_chunks": (
                [c.payload(500) for c in probe.methodology]
                if include_methodology else None
            ),
            "elapsed_ms": 0.0,
        }
        return float(self.counter.count(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ))

    # ── selección de candidatos directos ──
    def _select_probe_files(self, probe: ProbeResult) -> list[FileInfo]:
        if not probe.chunks:
            return []
        best: dict[str, float] = {}
        for ch in probe.chunks:
            if not ch.file:
                continue
            s = float(ch.score or 0.0)
            best[ch.file] = max(best.get(ch.file, 0.0), s)
        if not best:
            return []
        ordered = sorted(best, key=lambda f: (-best[f], f))
        top = max(best.values())
        floor = float(self.cfg.get("direct_score_floor", 0.45)) * top
        max_files = max(1, int(self.cfg.get("max_direct_files", 3)))
        selected: list[str] = []
        for rel in ordered:
            if len(selected) >= max_files:
                break
            if selected and best[rel] < floor:
                break
            selected.append(rel)
        infos: list[FileInfo] = []
        for rel in selected:
            try:
                rp = (self.root / rel).resolve()
            except OSError:
                continue
            if not self._inside(rp) or not rp.is_file():
                continue
            info = self._file_info(rp)
            if info is not None:
                infos.append(info)
        return infos

    # ── decisión ──
    def decide(
        self,
        q: str,
        *,
        k: int = 5,
        include_methodology: bool | None = None,
        methodology_loaded: bool = False,
        candidate_files: Sequence[str] | None = None,
        loaded_files: Sequence[str] | None = None,
        read_ratio: float | None = None,
        token_budget: int | None = None,
        probe: ProbeResult | None = None,
        allow_hybrid: bool | None = None,
        min_savings: int | None = None,
        compact: bool = True,
        category: str | None = None,
        file_glob: str | None = None,
    ) -> dict:
        k = max(1, int(k))
        if methodology_loaded:
            include_methodology = False
        elif include_methodology is None:
            include_methodology = True
        if allow_hybrid is None:
            allow_hybrid = bool(self.cfg.get("allow_hybrid", True))
        if min_savings is None:
            min_savings = int(self.cfg.get("min_savings_tokens", 60))
        if read_ratio is None:
            read_ratio = float(self.cal.direct_read_ratio)
        read_ratio = _clamp(float(read_ratio), 0.05, 3.0)
        decision_cost = int(self.cfg.get("decision_cost_tokens", 110))

        # Probe: el servidor lo pasa ya hecho; en uso síncrono, lexical.
        if probe is None:
            probe = self.lexical_probe(q, limit=int(self.cfg.get("probe_k", 12)))

        # Candidatos para lectura directa.
        explicit = self.resolve_files(candidate_files)
        explicit_infos = self.describe_files(explicit) if explicit else []
        candidate_infos = explicit_infos or self._select_probe_files(probe)
        loaded = self._normalize_loaded(loaded_files)
        read_infos = [i for i in candidate_infos if i.path not in loaded]
        direct_unknown = not candidate_infos
        direct_full_tokens = sum(i.tokens for i in candidate_infos)
        direct_tokens = (
            sum(i.tokens for i in read_infos) * read_ratio
            + self.cal.direct_overhead_tokens * len(read_infos)
        )

        # Coste RAG: payload exacto si el probe es semántico; si no, modelo.
        exact_rag = self.estimate_payload(
            q, probe, k=k, include_methodology=bool(include_methodology)
        )
        rag_tokens = (
            exact_rag if exact_rag is not None
            else self.cal.estimate_rag(k, bool(include_methodology))
        )

        # Riesgo de que RAG no baste (top-k insuficiente).
        p_miss = float(self.cal.rag_miss_rate)
        if probe.source == "semantic":
            top = max(0.0, float(probe.top_score))
            p_miss *= 1.0 + max(0.0, 0.5 - top) * 0.8
        elif probe.source == "lexical":
            p_miss = max(p_miss, 0.15)
        elif probe.source == "none" and not candidate_infos:
            # Sin probe ni candidatos no hay evidencia: riesgo alto. Si el
            # agente ya pasó candidatos, se mantiene el prior del benchmark.
            p_miss = max(p_miss, 0.50)
        p_miss = _clamp(p_miss, 0.0, 0.75)

        if direct_unknown:
            rag_expected = rag_tokens
        else:
            rag_expected = rag_tokens + p_miss * (
                direct_tokens + self.cal.fallback_penalty_tokens
            )

        # Híbrido: RAG para localizar + leer solo el top-1.
        top_info = read_infos[0] if read_infos else None
        hybrid_tokens = (
            rag_tokens + top_info.tokens * read_ratio + self.cal.direct_overhead_tokens
            if top_info is not None else None
        )

        raw_rag_wins = (not direct_unknown) and (
            rag_tokens + min_savings < direct_tokens
        )
        exp_rag_wins = (not direct_unknown) and (
            rag_expected + min_savings < direct_tokens
        )
        hybrid_wins = (
            bool(allow_hybrid) and top_info is not None and hybrid_tokens is not None
            and hybrid_tokens + min_savings < direct_tokens
        )

        warn: str | None = None
        if direct_unknown:
            action = "rag" if rag_tokens > 0 else "direct"
            warn = "sin candidatos: pasa candidate_files o usa /query"
        elif raw_rag_wins and exp_rag_wins:
            action = "rag"
        elif raw_rag_wins and hybrid_wins:
            action = "hybrid"
        else:
            action = "direct"

        # Presupuesto: si la vía elegida no cabe, intentar la más barata.
        chosen_est = {
            "rag": rag_expected,
            "direct": direct_tokens,
            "hybrid": hybrid_tokens if hybrid_tokens is not None else direct_tokens,
        }[action]
        if token_budget is not None:
            if chosen_est + decision_cost > token_budget:
                options = [("rag", rag_expected), ("direct", direct_tokens)]
                if hybrid_tokens is not None:
                    options.append(("hybrid", hybrid_tokens))
                fitting = [(a, v) for a, v in options if v + decision_cost <= token_budget]
                if fitting:
                    action, chosen_est = min(fitting, key=lambda x: x[1])
                    warn = "presupuesto: se eligió la vía más barata que cabe"
                else:
                    action, chosen_est = min(options, key=lambda x: x[1])
                    warn = "presupuesto insuficiente para ambas vías"

        # Ahorro neto (descontando la propia llamada a /route).
        if action == "direct":
            saving = rag_expected - direct_tokens
        elif action == "hybrid" and hybrid_tokens is not None:
            saving = direct_tokens - hybrid_tokens
        else:
            saving = direct_tokens - rag_expected if not direct_unknown else 0.0
        net_saving = saving - decision_cost

        crossover = (
            rag_tokens / max(1e-6, 1.0 - p_miss) if not direct_unknown else None
        )

        # Confianza y riesgo.
        if direct_unknown:
            confidence = 0.30
        else:
            gap = abs(direct_tokens - rag_expected)
            denom = max(200.0, 0.25 * (direct_tokens + rag_expected))
            confidence = 0.55 + 0.40 * min(1.0, gap / denom)
            if probe.source != "semantic":
                confidence *= 0.90
        confidence = round(_clamp(confidence, 0.30, 0.97), 2)
        risk = "low" if confidence >= 0.75 and p_miss < 0.10 else (
            "medium" if confidence >= 0.55 else "high"
        )

        if direct_unknown:
            why = "sin candidatos: usa /query; pasa candidate_files para decidir por coste"
        elif action == "direct":
            why = (
                f"destino ~{direct_tokens:.0f} tok < cruce ~{crossover or 0:.0f}: "
                "leer directo"
            )
        elif action == "hybrid":
            why = (
                f"RAG + top-1 ({hybrid_tokens or 0:.0f} tok) < leer "
                f"{len(read_infos)} archivo(s)"
            )
        else:
            why = (
                f"destino ~{direct_tokens:.0f} tok > cruce ~{crossover or 0:.0f}: "
                f"RAG ahorra ~{saving:.0f}"
            )

        decision_id = "rt_" + uuid.uuid4().hex[:10]
        reason_code = (
            "no_candidates" if direct_unknown
            else "hybrid" if action == "hybrid"
            else "small_target" if action == "direct"
            else "large_target"
        )
        # Respuesta compacta mínima: la decisión también cuesta tokens.
        out: dict[str, Any] = {
            "id": decision_id,
            "action": action,
            "conf": confidence,
            "save": int(round(net_saving)),
            "why": reason_code,
        }
        if action in ("direct", "hybrid"):
            out["read"] = [
                {
                    "path": i.path,
                    "tok": int(round(i.tokens * read_ratio)),
                }
                for i in read_infos
            ]
        if action in ("rag", "hybrid"):
            out["rag"] = {
                "k": k,
                "methodology": bool(include_methodology),
            }
            if category:
                out["rag"]["category"] = category
            if file_glob:
                out["rag"]["file_glob"] = file_glob
        if warn:
            out["warn"] = warn

        if not compact:
            out["rag_tok"] = int(round(rag_tokens))
            if not direct_unknown:
                out["direct_tok"] = int(round(direct_tokens))
            out["decision_tok"] = decision_cost
            out["detail"] = why
            out["est"] = {
                "rag_raw": round(rag_tokens, 1),
                "rag_expected": round(rag_expected, 1),
                "direct": round(direct_tokens, 1) if not direct_unknown else None,
                "hybrid": round(hybrid_tokens, 1) if hybrid_tokens is not None else None,
                "crossover": round(crossover, 1) if crossover is not None else None,
                "p_miss": round(p_miss, 3),
                "read_ratio": round(read_ratio, 3),
                "decision_cost": decision_cost,
            }
            out["probe"] = {
                "source": probe.source,
                "top_score": round(float(probe.top_score), 3),
                "n_chunks": len(probe.chunks),
                "n_methodology": len(probe.methodology),
            }
            out["candidates"] = [i.to_dict() for i in candidate_infos]
            if "read" in out:
                for entry, info in zip(out["read"], read_infos):
                    entry["lines"] = info.lines
            out["calibration"] = self.cal.summary()

        # Log local para /route/stats y feedback.
        if self.cfg.get("log_decisions", True):
            self._log_jsonl(self.decisions_file, {
                "ts": time.time(),
                "id": decision_id,
                "q": q[:500],
                "k": k,
                "methodology": bool(include_methodology),
                "action": action,
                "conf": confidence,
                "rag_tok": round(rag_tokens, 1),
                "rag_expected": round(rag_expected, 1),
                "direct_tok": round(direct_tokens, 1) if not direct_unknown else None,
                "direct_full_tokens": direct_full_tokens,
                "hybrid_tok": round(hybrid_tokens, 1) if hybrid_tokens is not None else None,
                "chosen_est": round(chosen_est, 1),
                "saving": round(saving, 1),
                "net_save": round(net_saving, 1),
                "crossover": round(crossover, 1) if crossover is not None else None,
                "p_miss": round(p_miss, 3),
                "probe": probe.source,
                "files": [i.path for i in read_infos],
                "candidates": [i.path for i in candidate_infos],
            })
        return out

    # ── feedback / estadísticas ──
    def record_feedback(
        self,
        *,
        decision_id: str | None = None,
        chosen_action: str | None = None,
        actual_rag_tokens: float | None = None,
        actual_direct_tokens: float | None = None,
        actual_total_tokens: float | None = None,
        needed_followup: bool | None = None,
        answer_ok: bool | None = None,
        notes: str | None = None,
    ) -> dict:
        rec = self._find_decision(decision_id) if decision_id else None
        action = chosen_action or (rec or {}).get("action")
        k = int((rec or {}).get("k", 5))
        methodology = bool((rec or {}).get("methodology", True))

        if actual_rag_tokens is not None:
            self.cal.observe_rag(k, methodology, float(actual_rag_tokens))
        if actual_direct_tokens is not None:
            full = float((rec or {}).get("direct_full_tokens") or 0)
            if full > 0:
                self.cal.observe_direct_ratio(float(actual_direct_tokens) / full)
        if needed_followup is not None:
            self.cal.observe_miss(bool(needed_followup))

        self.save_calibration()
        payload = {
            "ts": time.time(),
            "decision_id": decision_id,
            "action": action,
            "actual_rag_tokens": actual_rag_tokens,
            "actual_direct_tokens": actual_direct_tokens,
            "actual_total_tokens": actual_total_tokens,
            "needed_followup": needed_followup,
            "answer_ok": answer_ok,
            "notes": (notes or "")[:500] or None,
        }
        self._log_jsonl(self.feedback_file, payload)
        return {
            "ok": True,
            "decision_id": decision_id,
            "action": action,
            "calibration": self.cal.summary(),
        }

    def stats(self) -> dict:
        decisions = self._read_jsonl(self.decisions_file, limit=2000)
        feedback = self._read_jsonl(self.feedback_file, limit=2000)
        by_action: dict[str, int] = {}
        for d in decisions:
            a = str(d.get("action", "?"))
            by_action[a] = by_action.get(a, 0) + 1

        by_id = {str(d.get("id")): d for d in decisions if d.get("id")}
        evaluable: list[dict] = []
        for f in feedback:
            rec = by_id.get(str(f.get("decision_id")))
            arag = f.get("actual_rag_tokens")
            adir = f.get("actual_direct_tokens")
            if arag is None or adir is None:
                continue
            action = f.get("action") or (rec or {}).get("action")
            total = f.get("actual_total_tokens")
            if total is None:
                if action == "rag":
                    total = arag
                elif action == "direct":
                    total = adir
                elif action == "hybrid":
                    total = float(arag) + float(adir)
                else:
                    total = float(arag)
            cheaper = "rag" if float(arag) <= float(adir) else "direct"
            correct = (
                action == cheaper
                or (action == "hybrid" and float(total) <= min(float(arag), float(adir)) + 1)
            )
            evaluable.append({
                "rag": float(arag),
                "direct": float(adir),
                "total": float(total),
                "action": action,
                "correct": bool(correct),
                "regret": float(total) - min(float(arag), float(adir)),
            })

        summary: dict[str, Any] = {
            "decisions": len(decisions),
            "feedback": len(feedback),
            "by_action": by_action,
            "calibration": self.cal.summary(),
        }
        if decisions:
            summary["last_decision_at"] = max(
                float(d.get("ts") or 0) for d in decisions
            )
            confs = [float(d.get("conf") or 0) for d in decisions if d.get("conf") is not None]
            if confs:
                summary["mean_confidence"] = round(sum(confs) / len(confs), 3)
        if evaluable:
            n = len(evaluable)
            summary["evaluated"] = {
                "n": n,
                "accuracy": round(sum(1 for e in evaluable if e["correct"]) / n, 4),
                "mean_regret_tokens": round(sum(e["regret"] for e in evaluable) / n, 1),
                "total_regret_tokens": round(sum(e["regret"] for e in evaluable), 1),
                "saving_vs_always_rag": round(
                    sum(e["rag"] - e["total"] for e in evaluable), 1
                ),
                "saving_vs_always_direct": round(
                    sum(e["direct"] - e["total"] for e in evaluable), 1
                ),
            }
        # Error de estimación RAG (si el feedback trae el coste real).
        errs = []
        for f in feedback:
            rec = by_id.get(str(f.get("decision_id")))
            if rec and f.get("actual_rag_tokens") is not None:
                errs.append(float(f["actual_rag_tokens"]) - float(rec.get("rag_tok", 0)))
        if errs:
            summary["rag_estimation_error"] = {
                "median": round(_median(errs), 1),
                "mean": round(sum(errs) / len(errs), 1),
                "n": len(errs),
            }
        return summary

    def _find_decision(self, decision_id: str | None) -> dict | None:
        if not decision_id:
            return None
        # Lee del final: las decisiones recientes son las relevantes.
        rows = self._read_jsonl(self.decisions_file, limit=2000)
        for rec in reversed(rows):
            if str(rec.get("id")) == str(decision_id):
                return rec
        return None

    # ── JSONL ──
    @staticmethod
    def _log_jsonl(path: Path, payload: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > 4_000_000:
                old = path.with_suffix(path.suffix + ".1")
                try:
                    if old.exists():
                        old.unlink()
                    path.replace(old)
                except OSError:
                    pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("no se pudo escribir %s: %s", path, e)

    @staticmethod
    def _read_jsonl(path: Path, *, limit: int = 2000) -> list[dict]:
        if not path.exists():
            return []
        try:
            lines = path.read_text("utf-8", errors="replace").splitlines()
        except OSError:
            return []
        out: list[dict] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out


# ── CLI (cliente HTTP del servicio) ────────────────────────────────────────
def _http_json(base: str, method: str, path: str, payload: dict | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urlrequest.Request(
        base.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urlrequest.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urlerror.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code} en {path}: {body}") from e
    except urlerror.URLError as e:
        raise SystemExit(
            f"No se pudo conectar con {base}: {e}. ¿Está el RAG levantado?"
        ) from e


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.router",
        description="Router RAG vs lectura directa (cliente del servicio HTTP).",
    )
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("RAG_PORT", "8765"))
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_route = sub.add_parser("route", help="Decide la vía para una consulta")
    p_route.add_argument("--q", required=True)
    p_route.add_argument("--k", type=int, default=5)
    p_route.add_argument("--files", nargs="*", default=None,
                         help="Candidatos conocidos (paths/globs relativos)")
    p_route.add_argument("--loaded", nargs="*", default=None,
                         help="Archivos ya leídos en la sesión")
    p_route.add_argument("--methodology-loaded", action="store_true",
                         help="El alma ya está en contexto (no se paga otra vez)")
    p_route.add_argument("--no-methodology", action="store_true")
    p_route.add_argument("--no-probe", action="store_true",
                         help="No hacer probe semántico interno")
    p_route.add_argument("--no-hybrid", action="store_true")
    p_route.add_argument("--read-ratio", type=float, default=None)
    p_route.add_argument("--budget", type=int, default=None)
    p_route.add_argument("--verbose", action="store_true")
    p_route.add_argument("--pretty", action="store_true")

    p_fb = sub.add_parser("feedback", help="Registra el coste real observado")
    p_fb.add_argument("--id", default=None, dest="decision_id")
    p_fb.add_argument("--action", default=None,
                      choices=["rag", "direct", "hybrid"])
    p_fb.add_argument("--rag-tokens", type=float, default=None)
    p_fb.add_argument("--direct-tokens", type=float, default=None)
    p_fb.add_argument("--total-tokens", type=float, default=None)
    p_fb.add_argument("--followup", action="store_true",
                      help="RAG no bastó y hubo que leer el archivo")
    p_fb.add_argument("--ok", action="store_true", help="La respuesta fue correcta")
    p_fb.add_argument("--notes", default=None)
    p_fb.add_argument("--pretty", action="store_true")

    p_stats = sub.add_parser("stats", help="Calibración y resultados observados")
    p_stats.add_argument("--pretty", action="store_true")

    args = parser.parse_args(argv)
    host = args.host.rstrip("/")
    if "://" not in host:
        host = "http://" + host
    base = f"{host}:{args.port}"
    indent = 2 if getattr(args, "pretty", False) else None

    if args.cmd == "route":
        body: dict[str, Any] = {
            "q": args.q,
            "k": args.k,
            "compact": not args.verbose,
            "candidate_files": args.files,
            "loaded_files": args.loaded,
            "methodology_loaded": args.methodology_loaded,
            "read_ratio": args.read_ratio,
            "token_budget": args.budget,
            "probe": not args.no_probe,
            "allow_hybrid": not args.no_hybrid,
        }
        if args.no_methodology:
            body["include_methodology"] = False
        out = _http_json(base, "POST", "/route", body)
    elif args.cmd == "feedback":
        out = _http_json(base, "POST", "/route/feedback", {
            "decision_id": args.decision_id,
            "chosen_action": args.action,
            "actual_rag_tokens": args.rag_tokens,
            "actual_direct_tokens": args.direct_tokens,
            "actual_total_tokens": args.total_tokens,
            "needed_followup": True if args.followup else None,
            "answer_ok": True if args.ok else None,
            "notes": args.notes,
        })
    else:
        out = _http_json(base, "GET", "/route/stats")

    print(json.dumps(out, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
