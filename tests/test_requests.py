"""Request shape for every skill: valid on 5.x models, effort set, tools usable."""
import pytest

import databricks_client as dc
from config import config
from skills import (data_quality, eda, experiment_tracker, feature_engineering, join_auditor,
                    leakage_detector, mlops_reviewer, model_selector, nl_to_spark)

ECOM = dc._gen_ecommerce(200)

SKILL_RUNS = {
    "data_quality":        lambda: data_quality.assess(dc._gen_orders_dirty(200), "orders"),
    "eda":                 lambda: eda.analyze(ECOM, "ecommerce"),
    "join_auditor":        lambda: join_auditor.audit(dc._gen_sales_with_product_id(100), dc._gen_products(), ["product_id"]),
    "nl_to_spark":         lambda: nl_to_spark.generate("revenue by region", {"sales": dc._gen_sales_clean(50)}),
    "mlops_review":        lambda: mlops_reviewer.review("model.fit(X, y)"),
    "model_selector":      lambda: model_selector.select(ECOM, "converted"),
    "leakage_detector":    lambda: leakage_detector.check(dc._gen_ecommerce_leakage(200), "converted"),
    "feature_engineering": lambda: feature_engineering.suggest(ECOM, "converted"),
    "experiment_tracker":  lambda: experiment_tracker.interpret({"val_auc": 0.81}),
}


# Tools deliberately left non-strict, with the measured reason
NON_STRICT_OK = {
    "report_eda": "strict decoding ran away (max_tokens / 1,305-item list) in 2 of 4 runs; output is validated in code",
}


def _objects(schema):
    """Every object node in a JSON schema."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for v in schema.values():
            yield from _objects(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _objects(v)


@pytest.mark.parametrize("skill", SKILL_RUNS)
def test_skill_requests_are_valid_on_5x(fake_api, skill):
    SKILL_RUNS[skill]()
    assert fake_api.calls, f"{skill} made no API call"
    for call in fake_api.calls:
        model = call["model"]
        assert model.startswith("claude-") and "-4" not in model
        assert "budget_tokens" not in str(call.get("thinking")), "budget_tokens 400s on 5.x"
        assert "temperature" not in call and "top_p" not in call

        choice = (call.get("tool_choice") or {}).get("type")
        if choice in ("tool", "any"):
            # forced tool calls are accepted only by the structured-extraction model
            assert model == config.structured_model
        else:
            assert "effort" in (call.get("output_config") or {}), f"{skill}: no explicit effort on {model}"

        if choice == "auto":
            for tool in call["tools"]:
                if tool["name"] in NON_STRICT_OK:
                    continue
                assert tool.get("strict") is True, f"{skill}: {tool['name']} not strict under auto"
                for obj in _objects(tool["input_schema"]):
                    assert obj.get("additionalProperties") is False
                prompt = str(call["messages"]) + str(call.get("system", ""))
                assert tool["name"] in prompt, f"{skill}: prompt never names {tool['name']}"


def test_model_ids_come_from_config():
    """No model ID literals outside config.py (project rule 3)."""
    import pathlib
    import re
    root = pathlib.Path(__file__).parent.parent
    offenders = []
    for path in list(root.glob("skills/*.py")) + list(root.glob("agents/*.py")) \
            + list(root.glob("pages/*.py")) + list(root.glob("benchmarks/*.py")):
        if re.search(r"[\"']claude-(opus|sonnet|haiku|fable)-", path.read_text()):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"hardcoded model IDs in {offenders}"
