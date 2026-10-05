"""Regression tests for the 2026-10-05 fleet-sweep fix of dinov2_feature_extraction_colab (SWP-A).

The sweep flagged three `assert` lines that compare a quality metric. Two were quality asserts (frozen policy and
selected policy against the majority floor) and are recorded verdicts now; the third is the reload-parity contract
and stays a hard check. The tests execute the notebook's own cell source with stand-in values; they need no torch
and claim no pretrained-inference evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook source lines in full

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "dinov2_feature_extraction_colab.ipynb"
METRIC = re.compile(r"\b(accuracy|macro_f1|log_loss|recall|precision|f1)\b")


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells() -> list[str]:
    return ["".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"]


def _cell(marker: str) -> str:
    found = [src for src in _code_cells() if marker in src]
    assert len(found) == 1, marker
    return found[0]


def test_swp_a_no_quality_assert_remains():
    code = "\n".join(_code_cells())
    assert "assert frozen_test['accuracy'] > floor['accuracy']" not in code
    assert "assert adapted_test['accuracy'] > floor['accuracy']" not in code
    quality = [
        line
        for line in code.splitlines()
        if line.strip().startswith("assert ") and METRIC.search(line) and re.search(r"[<>]", line) and "reloaded_test" not in line
    ]
    assert quality == []
    # Contract checks stay hard: the frozen-policy contract of Section 6 and the reload parity of Section 9.
    assert "assert probe_result['policy'].startswith('frozen')" in code
    assert "assert parity['probabilities_identical'] and parity['classes_identical'] and abs(adapted_test['accuracy'] - reloaded_test['accuracy']) < 1e-9" in code


def test_swp_a_verdicts_are_recorded_and_do_not_stop_the_notebook():
    s6 = _cell("frozen_test = pipe.evaluate(test_records)")
    frozen_line = next(line for line in s6.splitlines() if line.startswith("frozen_verdict = "))
    s8 = _cell("adapted_test = pipe.evaluate(test_records)")
    verdict_block = s8[s8.index("selected_verdict = ") : s8.index("for metric, row in comparison.items():")]
    cases = (
        (0.1, 0.1667, 0.1, "not above the majority floor", "not above the majority floor", "no gain"),
        (0.8333, 0.1667, 0.8333, "above the majority floor", "above the majority floor", "no gain"),
        (0.5, 0.1667, 0.75, "above the majority floor", "above the majority floor", "improved"),
        (0.5, 0.1667, 0.1, "above the majority floor", "not above the majority floor", "worse"),
    )
    for frozen, floor, adapted, frozen_expected, selected_expected, adaptation_expected in cases:
        ns = {"frozen_test": {"accuracy": frozen}, "floor": {"accuracy": floor}, "adapted_test": {"accuracy": adapted}, "comparison": {}}
        exec(frozen_line, ns)
        exec(verdict_block, ns)
        verdicts = ns["comparison"]["verdicts"]
        assert verdicts["frozen_vs_majority_floor"] == frozen_expected
        assert verdicts["selected_vs_majority_floor"] == selected_expected
        assert verdicts["selected_vs_frozen_test_accuracy"] == adaptation_expected
    # The verdicts are stored before the evaluation report is written, and the report feeds result.json.
    assert s8.index("comparison['verdicts'] = ") < s8.index("json.dump(evaluation_report_payload")
    s9 = _cell("result_payload = {")
    assert "'comparison': comparison," in s9


def test_swp_a_learner_text_describes_the_verdict_not_an_assert():
    markdown = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    assert "The cell asserts the selected model beats the majority floor" not in markdown
    assert "The cell records a verdict" in markdown
