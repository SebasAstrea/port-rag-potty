"""Tests del router RAG vs lectura directa (sin Ollama/Chroma).

Ejecutar desde la raíz del repo:
    python -m unittest discover -s tests -v
    pytest tests/test_router.py -q
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.router import Calibration, ProbeResult, QueryRouter


class RouterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.state = self.root / "state"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _router(self, **overrides) -> QueryRouter:
        cfg = {"decision_cost_tokens": 0, "probe_k": 8}
        cfg.update(overrides)
        return QueryRouter(
            self.root,
            state_dir=self.state,
            config_overrides=cfg,
        )

    def _file(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    # ── decisión ──
    def test_small_file_prefers_direct(self) -> None:
        self._file("small.md", "x" * 200)
        r = self._router()
        out = r.decide(
            "¿qué dice small?",
            candidate_files=["small.md"],
            probe=ProbeResult(source="none"),
        )
        self.assertEqual(out["action"], "direct")
        self.assertEqual(out["read"][0]["path"], "small.md")
        self.assertIn("save", out)

    def test_large_file_prefers_rag(self) -> None:
        self._file("big.md", "x" * 20000)
        r = self._router()
        out = r.decide(
            "¿qué dice big?",
            candidate_files=["big.md"],
            probe=ProbeResult(source="none"),
        )
        self.assertEqual(out["action"], "rag")
        self.assertEqual(out["rag"]["k"], 5)

    def test_loaded_file_is_free(self) -> None:
        self._file("small.md", "x" * 200)
        r = self._router()
        out = r.decide(
            "ya lo tengo",
            candidate_files=["small.md"],
            loaded_files=["small.md"],
            probe=ProbeResult(source="none"),
        )
        self.assertEqual(out["action"], "direct")
        self.assertEqual(out["read"], [])

    def test_path_traversal_is_ignored(self) -> None:
        outside = self.root.parent / "outside_router_test.txt"
        outside.write_text("secreto", encoding="utf-8")
        try:
            r = self._router()
            out = r.decide(
                "dame el secreto",
                candidate_files=["../outside_router_test.txt"],
                probe=ProbeResult(source="none"),
            )
            self.assertEqual(out["action"], "rag")
            self.assertNotIn("read", out)
            self.assertIn("warn", out)
        finally:
            outside.unlink(missing_ok=True)

    def test_lexical_probe_finds_file(self) -> None:
        self._file("docs/security.md", "# Seguridad\n\ncontroles y permisos")
        self._file("src/main.py", "print('hola')\n")
        r = self._router()
        probe = r.lexical_probe("security y permisos", limit=5)
        self.assertEqual(probe.source, "lexical")
        self.assertEqual(probe.chunks[0].file, "docs/security.md")
        out = r.decide("security y permisos", probe=probe)
        self.assertIn("read", out)
        self.assertEqual(out["read"][0]["path"], "docs/security.md")

    def test_no_candidates_warns_and_uses_rag(self) -> None:
        r = self._router()
        out = r.decide("consulta sin pistas", probe=ProbeResult(source="none"))
        self.assertEqual(out["action"], "rag")
        self.assertIn("warn", out)
        self.assertAlmostEqual(out["conf"], 0.3, places=2)

    def test_methodology_loaded_saves_tokens(self) -> None:
        self._file("big.md", "x" * 20000)
        r = self._router()
        with_m = r.decide(
            "¿qué dice big?",
            candidate_files=["big.md"],
            include_methodology=True,
            probe=ProbeResult(source="none"),
            compact=False,
        )
        without_m = r.decide(
            "¿qué dice big?",
            candidate_files=["big.md"],
            methodology_loaded=True,
            probe=ProbeResult(source="none"),
            compact=False,
        )
        self.assertLess(without_m["rag_tok"], with_m["rag_tok"])
        self.assertFalse(without_m["rag"]["methodology"])

    def test_compact_response_is_small(self) -> None:
        self._file("small.md", "x" * 200)
        r = self._router()
        out = r.decide(
            "x",
            candidate_files=["small.md"],
            probe=ProbeResult(source="none"),
        )
        self.assertLess(len(json.dumps(out, ensure_ascii=False)), 1200)

    def test_globs_resolve(self) -> None:
        self._file("docs/a.md", "a" * 100)
        self._file("docs/b.md", "b" * 100)
        r = self._router()
        files = r.resolve_files(["docs/*.md"])
        self.assertEqual(len(files), 2)

    # ── calibración / feedback ──
    def test_feedback_updates_calibration_and_persists(self) -> None:
        r = self._router()
        before = r.cal.estimate_rag(5, True)
        for _ in range(8):
            r.record_feedback(actual_rag_tokens=3000)
        after = r.cal.estimate_rag(5, True)
        self.assertGreater(after, before)
        self.assertGreaterEqual(after, 2500)
        self.assertTrue((self.state / "router_calibration.json").exists())

        r2 = self._router()
        self.assertAlmostEqual(r2.cal.base_tokens, r.cal.base_tokens, places=2)
        self.assertAlmostEqual(
            r2.cal.per_result_tokens, r.cal.per_result_tokens, places=2
        )

    def test_feedback_direct_ratio_and_miss(self) -> None:
        r = self._router()
        for _ in range(4):
            r.cal.observe_direct_ratio(0.5)
            r.cal.observe_miss(True)
        self.assertAlmostEqual(r.cal.direct_read_ratio, 0.5, places=2)
        self.assertGreater(r.cal.rag_miss_rate, 0.05)

    def test_stats_with_counterfactual_feedback(self) -> None:
        self._file("small.md", "x" * 200)
        r = self._router()
        out = r.decide(
            "x",
            candidate_files=["small.md"],
            probe=ProbeResult(source="none"),
        )
        r.record_feedback(
            decision_id=out["id"],
            actual_rag_tokens=1700,
            actual_direct_tokens=500,
            actual_total_tokens=500,
            needed_followup=False,
            answer_ok=True,
        )
        stats = r.stats()
        self.assertEqual(stats["evaluated"]["n"], 1)
        self.assertEqual(stats["evaluated"]["accuracy"], 1.0)
        self.assertLessEqual(stats["evaluated"]["total_regret_tokens"], 0.0)

    def test_calibration_fit_is_stable(self) -> None:
        cal = Calibration()
        for k in (3, 5, 8):
            for meth in (False, True):
                for _ in range(4):
                    actual = 100 + 200 * k + (450 if meth else 0)
                    cal.observe_rag(k, meth, actual)
        self.assertTrue(60 <= cal.per_result_tokens <= 320)
        self.assertTrue(250 <= cal.methodology_tokens <= 700)
        self.assertTrue(20 <= cal.base_tokens <= 300)


if __name__ == "__main__":
    unittest.main()
