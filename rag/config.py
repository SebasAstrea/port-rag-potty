"""RAG portable · configuración del servicio RAG (genérico, por proyecto).

El RAG es portable: se copia la carpeta `rag/` dentro de CUALQUIER proyecto y
se indexa la raíz de ese proyecto (`PROJECT_ROOT` = el directorio padre de
`rag/`). No sabe nada de "temas CSS" ni de frontend-style-lab: todo lo que
depende del proyecto se declara en su `rag.config.json`.

Configuración por proyecto (auto-generada en el primer arranque si falta):
  <PROJECT_ROOT>/rag.config.json

El bootstrap detecta el tipo de proyecto (si hay *.css → activa un grafo de
categorías estilo CSS; si no, lo deja desactivado) y rellena el resto con
valores genéricos sensatos. Puedes editar el JSON cuando quieras; se recarga
reiniciando el servidor.

Variables de entorno (para lo que depende del ENTORNO, no del proyecto) pueden
forzar cualquier valor:
  OLLAMA_HOST  RAG_EMBED_MODEL  RAG_EMBED_DIM
  RAG_HOST     RAG_PORT         RAG_LOG_LEVEL
  RAG_PROJECT  RAG_COLLECTION
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("rag.config")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAG_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = RAG_DIR / "data"
CHROMA_DIR: Path = DATA_DIR / "chroma"
GRAPH_FILE: Path = DATA_DIR / "graph.json"
MANIFEST_FILE: Path = DATA_DIR / "manifest.json"
ROUTER_CALIBRATION_FILE: Path = DATA_DIR / "router_calibration.json"
ROUTER_DECISIONS_FILE: Path = DATA_DIR / "router_decisions.jsonl"
ROUTER_FEEDBACK_FILE: Path = DATA_DIR / "router_feedback.jsonl"

CONFIG_FILE: Path = PROJECT_ROOT / "rag.config.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults (genéricos) ─────────────────────────────────────────────────────
INDEX_GLOBS_DEFAULT: tuple[str, ...] = (
    "**/*.md", "**/*.markdown",
    "**/*.html", "**/*.htm",
    "**/*.css", "**/*.scss", "**/*.sass", "**/*.less",
    "**/*.js", "**/*.mjs", "**/*.cjs", "**/*.jsx", "**/*.ts", "**/*.tsx",
    "**/*.json", "**/*.jsonc", "**/*.py", "**/*.pyi",
    "**/*.toml", "**/*.yml", "**/*.yaml", "**/*.ini", "**/*.cfg", "**/*.conf",
    "**/*.sh", "**/*.bash", "**/*.zsh", "**/*.env", "**/*.txt",
    "**/*.sql", "**/*.go", "**/*.rs", "**/*.java", "**/*.rb", "**/*.lua",
    "**/*.tf",
)

EXCLUDE_DIRS_DEFAULT: frozenset[str] = frozenset({
    "node_modules", ".git", ".svn", ".hg",
    ".venv", "venv", "env", "__pycache__", ".tox", ".nox",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage_cache",
    ".next", ".nuxt", ".cache", ".parcel-cache", ".webpack",
    "vendor", "build", "dist", "target", ".egg-info", "htmlcov",
    ".gradle", ".idea", ".vscode", ".terraform", "site-packages",
    "data",      # chroma + graph + descartes por defecto
    "rag/data",
})

EXCLUDE_FILES_DEFAULT: frozenset[str] = frozenset({
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock",
    "uv.lock", "pip.lock", ".DS_Store", "rag.config.json",
})

# Chunker: ext → clave de chunker (ver rag/chunker.py → CHUNKERS)
CHUNKERS_DEFAULT: dict[str, str] = {
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".js": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "js", ".tsx": "js", ".jsx": "js",
    ".md": "md", ".markdown": "md",
    ".html": "html", ".htm": "html",
    ".json": "json", ".jsonc": "raw",
    ".py": "py", ".pyi": "py",
    ".yml": "yaml", ".yaml": "yaml",
    ".toml": "toml", ".ini": "raw", ".cfg": "raw", ".conf": "raw",
    ".sh": "raw", ".bash": "raw", ".zsh": "raw", ".env": "raw", ".txt": "raw",
    ".sql": "raw", ".go": "raw", ".rs": "raw", ".java": "raw",
    ".rb": "raw", ".lua": "raw", ".tf": "raw",
}

# Methodology: toda una categoría de entradas se elimina, no con datos por
# defecto peligrosos.
METHODOLOGY_ENTIRE_FILES_DEFAULT: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md", ".cursorrules")

# Inferencia de secciones "alma" cuando heading_patterns está vacío.
# Coincidencia por substring contra el heading en minúsculas.
METHODOLOGY_INFER_KEYWORDS_DEFAULT: tuple[str, ...] = (
    "qué es este proyecto", "qué es este", "que es este proyecto", "sobre este proyecto",
    "propósito", "proposito", "purpose", "overview", "resumen", "introducción", "introduccion",
    "cómo funciona", "como funciona", "how it works", "how does", "funcionamiento",
    "arquitectura", "architecture", "estructura", "structure", "organización", "organizacion",
    "instalación", "instalacion", "install", "installation", "getting started",
    "quickstart", "setup", "requisitos", "requirements", "prerequisitos",
    "cómo ejecutar", "como ejecutar", "how to run", "ejecución", "ejecucion", "arrancar",
    "cómo usar", "como usar", "usage", "uso rápido",
    "cómo añadir", "como añadir", "how to add", "contribut", "cómo crear", "como crear",
    "convenciones", "conventions", "buenas prácticas", "buenas practicas", "estándares", "estandares",
    "roadmap", "endpoints", "servicio", "configuración", "configuracion", "configuration",
)

# ── Grafo (configurable; defaults replican detección al estilo CSS) ─────────
# Si `graph.category_regex` (o `graph.block_regex`) queda vacío, el grafo de
# categorías se desactiva y /categories devuelve lista vacía.
GRAPH_DEFAULT: dict = {
    # Categoría + bloque (grupo 1 = categoría, grupo 2 = cuerpo). Si el cuerpo
    # define tokens (group 2 de token_regex), se crea edge category→token.
    "block_regex": r"\[data-theme=['\"]([^'\"]+)['\"]\]\s*\{([^}]*)\}",
    "token_regex": r"--([a-zA-Z0-9_-]+)\s*:",
    # Menciones de categoría en cualquier archivo (script src, data-theme, ...)
    "mention_regex": r"\[data-theme=['\"]([^'\"]+)['\"]\]",
    # Heurística JS → categorías (vacío = desactivada)
    "js_category_regex": (
        r"(?:html|documentElement|root)\.?(?:dataset\.theme|getAttribute\(['\"]data-theme['\"]\))"
        r"\s*(?:===|==|!==|!=|includes|startsWith)?\s*['\"]([^'\"]+)['\"]"
        r"|(?:^|[^\w])(?:currentTheme|current_theme|theme|tema)\s*(?:===|==|!==|!=)"
        r"\s*['\"]([^'\"]+)['\"]"
        r"|\b(is|on|handle|setup)([A-Z][a-z]+)Theme\s*\("
        r"|\.([a-z][a-z0-9-]*)-scene\b"
    ),
    # Extensiones donde se aplica cada scan ("" / lista vacía → desactivado)
    "block_scan_exts": ["css", "scss", "sass", "less"],
    "token_scan_exts": ["css", "scss", "sass", "less"],
    "mention_scan_exts": ["html", "htm", "js", "mjs", "cjs", "ts", "tsx", "jsx", "xhtml"],
    # Resúmenes de categoría desde un archivo declarativo. {id} se sustituye
    # por el id de categoría en la regex.
    "summary_file": "js/theme.js",
    "summary_regex": r"\{\s*id\s*:\s*['\"]{id}['\"][^}]*?tagline\s*:\s*['\"]([^'\"]+)['\"]",
    # Página que importa recursos (script/link) → edges file importados por HTML
    "html_entry": "index.html",
}

# ── Router de decisión RAG vs lectura directa ────────────────────────────────
# El router estima el coste (tokens) de cada vía antes de que el agente lea
# nada. Los defaults están calibrados con el benchmark de 33 consultas
# (docs/benchmarks/results/tokens.json); el feedback real los ajusta.
ROUTER_DEFAULT: dict = {
    "enabled": True,
    # Descubrimiento de candidatos: k del probe semántico (no cuesta tokens al
    # agente; sólo una consulta interna a Chroma + Ollama).
    "probe_k": 12,
    # Si Ollama/Chroma no responden en este tiempo, el probe cae a lexical.
    # (Una consulta real tarda ~30-50 ms; 5 s cubre incluso el arranque en frío.)
    "probe_timeout_s": 5.0,
    # Tras un fallo/timeout del probe semántico, no se reintenta durante este
    # tiempo: evita pagar 5 s en cada decisión cuando Ollama está caído.
    "probe_cooldown_s": 30.0,
    # TTL de la lista de archivos usada por el probe lexical (evita recorrer
    # el árbol del proyecto en cada decisión si no hay grafo).
    "lexical_cache_s": 30.0,
    # Como máximo se estima la lectura directa de estos archivos.
    "max_direct_files": 3,
    # Se para de añadir archivos cuando el score cae por debajo de este
    # porcentaje del mejor candidato.
    "direct_score_floor": 0.45,
    # RAG debe ganar por este margen para elegirse (evita decisiones ruidosas).
    "min_savings_tokens": 60,
    # Coste estimado de la propia respuesta compacta de /route (se descuenta
    # del ahorro). Medido: ~40-80 tok según longitud de paths.
    "decision_cost_tokens": 80,
    # Estimación global (se sustituye por el payload exacto si hay probe).
    "base_tokens": 110,
    "per_result_tokens": 220,
    "methodology_tokens": 470,
    # Coste de abrir un archivo con la tool de lectura (envoltorio/metadata).
    "direct_overhead_tokens": 25,
    # 1.0 = leer el archivo completo (baseline del benchmark). El feedback lo
    # ajusta si el agente suele leer sólo fragmentos.
    "direct_read_ratio": 1.0,
    # Probabilidad de que el top-k no baste y haya que leer el archivo igual.
    "rag_miss_rate": 0.05,
    # Llamada extra de fallback cuando RAG falla.
    "fallback_penalty_tokens": 80,
    # Permite recomendar "RAG + leer el top-1" cuando sale mejor que leer todos.
    "allow_hybrid": True,
    # Guarda decisiones/feedback en rag/data/router_*.jsonl (privado, local).
    "log_decisions": True,
    # No leer completos archivos enormes al estimar (se estima por tamaño).
    "max_file_bytes": 2_000_000,
    # tiktoken es opcional; sin él se aproxima por caracteres.
    "tokenizer": "auto",  # auto | tiktoken | heuristic
}


# ── Configuración del proyecto (rag.config.json) ─────────────────────────────
def _default_chunk_sizes() -> dict:
    return {
        "css": {"target": 900, "max": 1800},
        "js": {"target": 1100, "max": 2400},
        "md": {"target": 1000, "max": 2000},
        "html": {"target": 1200, "max": 2500},
        "json": {"target": 2000, "max": 4000},
        "py": {"target": 1100, "max": 2400},
        "yaml": {"target": 1200, "max": 2400},
        "toml": {"target": 1200, "max": 2400},
        "raw": {"target": 1500, "max": 3000},
    }


def _project_name_detected() -> str:
    return PROJECT_ROOT.name.strip() or "proyecto"


def _sanitize(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
    return s or "rag"


def _looks_frontend(project_root: Path) -> bool:
    """Heurística de bootstrap: si hay CSS, activamos el grafo CSS de categorías."""
    try:
        return any(project_root.rglob("*.css"))
    except OSError:
        return False


def _default_graph(package: dict) -> dict:
    g = dict(GRAPH_DEFAULT)
    if not _looks_frontend(PROJECT_ROOT):
        # Proyecto sin CSS: nada de categorías/tokens CSS por defecto.
        for k in ("block_regex", "token_regex", "mention_regex", "js_category_regex",
                  "block_scan_exts", "token_scan_exts", "mention_scan_exts"):
            g[k] = "" if k.endswith("_regex") else []
        g["summary_file"] = ""
        g["html_entry"] = "index.html" if (PROJECT_ROOT / "index.html").exists() else ""
    return g


def bootstrap_config() -> dict:
    """Genera (si no existe) <PROJECT_ROOT>/rag.config.json y lo devuelve."""
    data: dict = {
        "_comment": "Configuración del RAG portable para este proyecto. "
                    "Regenerar: borra este archivo y reinicia el servidor.",
        "project_name": _project_name_detected(),
        "collection": _sanitize(_project_name_detected()) + "_chunks",
        "index_globs": list(INDEX_GLOBS_DEFAULT),
        "exclude_dirs": sorted(EXCLUDE_DIRS_DEFAULT),
        "exclude_files": sorted(EXCLUDE_FILES_DEFAULT),
        "chunkers": {k: v for k, v in sorted(CHUNKERS_DEFAULT.items())},
        "chunk_sizes": {k: {"target": v["target"], "max": v["max"]}
                        for k, v in _default_chunk_sizes().items()},
        "chunk_overlap": 120,
        "embed_text_max_chars": 4000,
        "methodology": {
            "entire_files": list(METHODOLOGY_ENTIRE_FILES_DEFAULT),
            "heading_patterns": [],
            "infer_keywords": list(METHODOLOGY_INFER_KEYWORDS_DEFAULT),
        },
        "router": dict(ROUTER_DEFAULT),
        "graph": _default_graph({}),
    }
    try:
        if not CONFIG_FILE.exists():
            CONFIG_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("config generada: %s", CONFIG_FILE)
    except OSError as e:
        log.warning("no se pudo escribir %s (%s); usando defaults", CONFIG_FILE, e)
    return data


def _load_project_config() -> dict:
    if not CONFIG_FILE.exists():
        return bootstrap_config()
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("rag.config.json debe ser un objeto JSON")
        return raw
    except (json.JSONDecodeError, ValueError, OSError) as e:
        log.warning("rag.config.json inválido (%s); usando defaults", e)
        return bootstrap_config()


_PROJECT = _load_project_config()


def _get(path: str, default):
    """Acceso _comment-simple a valores anidados del config del proyecto."""
    cur = _PROJECT
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ── Ollama / embeddings (entorno) ────────────────────────────────────────────
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST") or _get("ollama_host", "http://localhost:11434")
EMBED_MODEL: str = os.environ.get("RAG_EMBED_MODEL") or _get("embed_model", "bge-m3")
EMBED_DIM: int = int(os.environ.get("RAG_EMBED_DIM") or _get("embed_dim", "1024"))

# ── HTTP ─────────────────────────────────────────────────────────────────────
RAG_HOST: str = os.environ.get("RAG_HOST", "0.0.0.0")
RAG_PORT: int = int(os.environ.get("RAG_PORT", "8765"))

# ── Colección ────────────────────────────────────────────────────────────────
PROJECT_NAME: str = os.environ.get("RAG_PROJECT") or _get("project_name", _project_name_detected())
CHROMA_COLLECTION: str = os.environ.get("RAG_COLLECTION") or _get("collection", _sanitize(PROJECT_NAME) + "_chunks")

# ── Indexación ───────────────────────────────────────────────────────────────
INDEX_GLOBS: tuple[str, ...] = tuple(_get("index_globs", INDEX_GLOBS_DEFAULT))
EXCLUDE_DIRS: frozenset[str] = frozenset(_get("exclude_dirs", EXCLUDE_DIRS_DEFAULT))
EXCLUDE_FILES: frozenset[str] = frozenset(_get("exclude_files", EXCLUDE_FILES_DEFAULT))

# ── Chunker ──────────────────────────────────────────────────────────────────
CHUNKERS: dict[str, str] = {**CHUNKERS_DEFAULT, **_get("chunkers", {})}

_default_sizes = _default_chunk_sizes()
_configured_sizes = _get("chunk_sizes", {})
CHUNK_TARGET_CHARS: dict[str, int] = {
    k: _configured_sizes.get(k, {}).get("target", v["target"])
    for k, v in _default_sizes.items()
}
CHUNK_MAX_CHARS: dict[str, int] = {
    k: _configured_sizes.get(k, {}).get("max", v["max"])
    for k, v in _default_sizes.items()
}
CHUNK_OVERLAP: int = int(_get("chunk_overlap", 120))

# Safety net: límite absoluto por chunk antes de enviar a bge-m3 (4000 medido
# en este entorno: 5547 chars OK, 5957 → HTTP 500 de Ollama).
EMBED_TEXT_MAX_CHARS: int = int(_get("embed_text_max_chars", 4000))

# ── Categorías / tokens (grafo) ──────────────────────────────────────────────
GRAPH = {**GRAPH_DEFAULT, **_get("graph", {})}
CATEGORY_REGEX: str = GRAPH.get("mention_regex", "")
TOKEN_REGEX: str = GRAPH.get("token_regex", "")
GRAPH_BLOCK_REGEX: str = GRAPH.get("block_regex", "")
JS_CATEGORY_CHECK_REGEX: str = GRAPH.get("js_category_regex", "")


def _norm_exts(vals) -> frozenset[str]:
    """Extensiones sin punto, en minúsculas ({"css"} == {".CSS", "CSS"})."""
    return frozenset(str(v).lstrip(".").lower() for v in (vals or []))


GRAPH_BLOCK_EXTS: frozenset[str] = _norm_exts(GRAPH.get("block_scan_exts", []))
GRAPH_TOKEN_EXTS: frozenset[str] = _norm_exts(GRAPH.get("token_scan_exts", []))
GRAPH_MENTION_EXTS: frozenset[str] = _norm_exts(GRAPH.get("mention_scan_exts", []))
GRAPH_SUMMARY_FILE: str = GRAPH.get("summary_file", "") or ""
GRAPH_SUMMARY_REGEX: str = GRAPH.get("summary_regex", "") or ""
GRAPH_HTML_ENTRY: str = GRAPH.get("html_entry", "") or ""

# ── Methodology ──────────────────────────────────────────────────────────────
METHODOLOGY_ENTIRE_FILES: frozenset[str] = frozenset(
    _get("methodology.entire_files", METHODOLOGY_ENTIRE_FILES_DEFAULT)
)
METHODOLOGY_HEADING_PATTERNS: tuple[str, ...] = tuple(_get("methodology.heading_patterns", []))
METHODOLOGY_INFER_KEYWORDS: tuple[str, ...] = tuple(
    _get("methodology.infer_keywords", METHODOLOGY_INFER_KEYWORDS_DEFAULT)
)

# ── Router (configurable por proyecto) ───────────────────────────────────────
ROUTER: dict = {**ROUTER_DEFAULT, **_get("router", {})}

_METHODOLOGY_BASENAMES = frozenset({"readme.md", "readme", "agents.md", "agents", "claude.md", "cursorrules"})


def _rel_key(rel: str) -> str:
    return rel.replace("\\", "/").lstrip("./")


def is_methodology_file(rel: str) -> bool:
    """¿El archivo completo es 'alma' del proyecto (entire files + README/AGENTS)?"""
    key = _rel_key(rel)
    if key in METHODOLOGY_ENTIRE_FILES:
        return True
    return key.rsplit("/", 1)[-1].lower() in _METHODOLOGY_BASENAMES


def is_methodology_heading(rel: str, heading_text: str, heading_full: str = "") -> bool:
    """¿Una sección concreta es methodology? Patterns explícitos > inferencia."""
    key = _rel_key(rel)
    if key in METHODOLOGY_ENTIRE_FILES:
        return True
    t = heading_text.strip().lower()
    full = heading_full.lower()
    if METHODOLOGY_HEADING_PATTERNS:
        return any(p.lower() in full or p.lower() in t for p in METHODOLOGY_HEADING_PATTERNS)
    return any(kw in t for kw in METHODOLOGY_INFER_KEYWORDS)


# ── Colección Chroma metadata (whitelist) ────────────────────────────────────
META_KEYS: tuple[str, ...] = (
    "file",
    "ext",
    "kind",         # css-rule | js-fn | py-def | md-section | html-block | json | raw ...
    "category",     # categoría del grafo si aplica (antes "theme"), vacío si general
    "section",      # título / nombre simbólico del bloque
    "start_line",
    "end_line",
    "tokens",        # lista CSV de tokens referenciados
    "methodology",   # bool
)