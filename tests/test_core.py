"""ds_workbench.core: deterministic, exact, and free of model/config dependencies."""
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from ds_workbench.core import join_stats, leakage_profile


def _tables(seed=0, n_left=500, n_right=300, n_keys=120):
    rng = np.random.default_rng(seed)
    left = pd.DataFrame({"k": rng.integers(0, n_keys, n_left), "a": rng.random(n_left)})
    right = pd.DataFrame({"k": rng.integers(n_keys // 3, n_keys + 40, n_right), "b": rng.random(n_right)})
    return left, right


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_row_counts_match_an_actual_merge(how):
    left, right = _tables()                     # duplicates on both sides, unmatched keys both ways
    assert join_stats(left, right, ["k"], how)["result_rows"] == len(left.merge(right, on="k", how=how))


def test_composite_keys():
    rng = np.random.default_rng(1)
    left = pd.DataFrame({"a": rng.integers(0, 5, 400), "b": rng.integers(0, 4, 400)})
    right = pd.DataFrame({"a": rng.integers(0, 6, 200), "b": rng.integers(0, 4, 200), "x": 1})
    assert join_stats(left, right, ["a", "b"])["result_rows"] == len(left.merge(right, on=["a", "b"]))


def test_null_keys_never_match_as_in_sql():
    left = pd.DataFrame({"k": [1.0, None, None]})
    right = pd.DataFrame({"k": [1.0, None]})
    s = join_stats(left, right, ["k"])
    assert s["result_rows"] == 1                 # pandas' merge would also pair the NaNs (3 rows)
    assert s["left_key_null_rows"] == 2 and s["left_rows_dropped"] == 2


def test_verdicts():
    one = pd.DataFrame({"k": range(100)})
    assert join_stats(one, one, ["k"])["verdict"] == "safe"
    doubled = pd.DataFrame({"k": list(range(100)) * 3})   # each left row matches 3 right rows
    s = join_stats(one, doubled, ["k"])
    assert s["fan_out_factor"] == 3.0 and s["verdict"] == "dangerous"
    lossy = pd.DataFrame({"k": range(70)})               # 30% of left rows dropped by an inner join
    assert join_stats(one, lossy, ["k"])["verdict"] == "dangerous"
    assert join_stats(one, lossy, ["k"], how="left")["verdict"] == "safe"   # a left join keeps them


def test_sample_join_scenarios_match_their_answer_keys():
    import databricks_client as dc
    from skills.answer_keys import JOIN_ANSWER_KEYS
    for name, key in JOIN_ANSWER_KEYS.items():
        left, right, k = dc.load_join_scenario(name)
        verdict = join_stats(left, right, [k])["verdict"]
        assert verdict in [key["expected_verdict"], *key.get("acceptable_verdicts", [])], name


def test_huge_many_to_many_join_is_measured_not_built():
    """200k × 200k rows on 10 keys: the real join would have 4 billion rows."""
    keys = np.arange(200_000) % 10
    left, right = pd.DataFrame({"k": keys}), pd.DataFrame({"k": keys})
    t0 = time.perf_counter()
    s = join_stats(left, right, ["k"])
    assert time.perf_counter() - t0 < 2
    assert s["result_rows"] == 4_000_000_000 and s["fan_out_factor"] == 20_000
    assert s["many_to_many_keys"] == 10 and s["verdict"] == "dangerous"


def test_leakage_profile_is_available_without_a_key():
    import databricks_client as dc
    prof = leakage_profile(dc._gen_ecommerce_leakage(), "converted")["features"]
    assert prof["conversion_score"]["single_feature_auc"] >= 0.99


def test_importing_core_does_not_load_the_sdk_or_config():
    code = ("import sys, ds_workbench.core, ds_workbench.core.causal_stats; "
            "bad = [m for m in ('anthropic', 'config', 'dotenv', 'skills') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(__import__('pathlib').Path(__file__).parent.parent))


def test_skills_still_import_from_the_package():
    from ds_workbench import WorkflowSession, causal_inference, data_quality  # noqa: F401
    assert callable(data_quality.assess) and callable(causal_inference.did)


def test_readme_quick_start_runs_without_an_api_key():
    """The README's no-key example, executed as written."""
    import pathlib
    import re
    readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
    section = readme.split("## Try it without an API key", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    env = {**__import__("os").environ, "ANTHROPIC_API_KEY": ""}
    out = subprocess.run([sys.executable, "-c", code + "\nimport sys; assert 'anthropic' not in sys.modules"],
                         cwd=str(pathlib.Path(__file__).parent.parent), env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert "dangerous" in out.stdout and "engagement_index_v2" in out.stdout
