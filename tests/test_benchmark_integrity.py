"""The benchmark must measure the skill, not recall of answers written into its prompt."""
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent.parent
KEYS = ROOT / "benchmarks" / "answer_keys"
SKILL_FILE = {"data_quality": "data_quality", "leakage_detector": "leakage_detector",
              "model_selector": "model_selector", "feature_engineering": "feature_engineering"}


def _hints(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "column_hint" and isinstance(v, str):
                yield v
            yield from _hints(v)
    elif isinstance(node, list):
        for v in node:
            yield from _hints(v)


def _planted_columns():
    """(skill, column) for every distinctive column an answer key checks."""
    for path in KEYS.rglob("*.json"):
        key = json.loads(path.read_text())
        skill = key["_meta"]["skill"]
        cols = set(_hints(key)) | set(key["_meta"].get("columns", {}))
        for col in cols:
            if "_" in col:  # generic words like "amount" can't be avoided in a prompt
                yield skill, col


@pytest.mark.parametrize("skill,column", sorted(set(_planted_columns())))
def test_prompt_does_not_name_benchmark_columns(skill, column):
    source = (ROOT / "skills" / f"{SKILL_FILE[skill]}.py").read_text()
    assert not re.search(rf"\b{re.escape(column)}\b", source), (
        f"skills/{SKILL_FILE[skill]}.py names '{column}', which the {skill} benchmark checks")
