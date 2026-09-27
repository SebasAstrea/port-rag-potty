#!/usr/bin/env python3
"""Orquestador: ejecuta todos los benchmarks y genera las figuras.

Uso:
  python3 run_all.py               # todo (≈3-4 min)
  python3 run_all.py --skip-embedding   # sin el bench de Ollama (≈1 min)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["run_tokens.py", "run_retrieval.py", "run_latency.py", "run_embedding.py"]


def main() -> None:
    steps = [s for s in STEPS if not (s == "run_embedding.py" and "--skip-embedding" in sys.argv)]
    t0 = time.time()
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        r = subprocess.run([sys.executable, str(HERE / step)])
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\n=== figures.py ===", flush=True)
    r = subprocess.run([sys.executable, str(HERE / "figures.py")])
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"\nlisto en {time.time() - t0:.1f}s → docs/benchmarks/{{results,figs}}")


if __name__ == "__main__":
    main()
