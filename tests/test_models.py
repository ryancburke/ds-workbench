"""Model roles come from config and can be overridden for sweeps."""
import databricks_client as dc
from config import config
from skills import leakage_detector
from skills.models import ROLES, override_roles, role


def test_roles_read_config():
    assert role("reasoning") == {"model": config.reasoning_model,
                                 "output_config": {"effort": config.reasoning_effort}}
    assert role("structured") == {"model": config.structured_model}  # default effort


def test_override_applies_inside_block_only(fake_api):
    with override_roles(leakage=("some-other-model", "low")):
        assert role("leakage") == {"model": "some-other-model", "output_config": {"effort": "low"}}
        leakage_detector.check(dc._gen_ecommerce_leakage(100), "converted")
    assert role("leakage")["model"] == config.leakage_model
    pass1 = fake_api.calls[0]
    assert pass1["model"] == "some-other-model" and pass1["output_config"] == {"effort": "low"}
    assert fake_api.calls[1]["model"] == config.structured_model  # Pass 2 untouched


def test_every_role_resolves():
    for name in ROLES:
        assert role(name)["model"]
