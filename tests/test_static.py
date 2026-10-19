"""Static checks that catch errors only a click in the UI would otherwise reveal."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent
PATHS = ["app.py", "config.py", "databricks_client.py", "run_sim.py",
         "skills", "agents", "benchmarks", "pages", "components", "workbench", "ds_workbench"]


def test_no_undefined_names():
    """An undefined name inside a callback only fails when someone clicks the button
    (the Workbench Experiment Tracker raised NameError: json on every run)."""
    out = subprocess.run([sys.executable, "-m", "pyflakes", *PATHS], cwd=ROOT,
                         capture_output=True, text=True).stdout
    problems = [line for line in out.splitlines() if "undefined name" in line]
    assert not problems, "\n".join(problems)


def test_workbench_registers_all_callbacks():
    import app  # noqa: F401  (creates the Dash app and imports every page)
    from dash._callback import GLOBAL_CALLBACK_MAP
    modules = {cb["callback"].__module__ for cb in GLOBAL_CALLBACK_MAP.values()}
    for skill in ("dq", "eda", "ja", "nl", "ml", "et", "ms", "ld", "fe", "ci"):
        assert f"workbench.callbacks.{skill}" in modules


def test_every_module_compiles():
    import compileall
    import re
    ok = all(compileall.compile_dir(str(ROOT / p), quiet=2, force=True,
                                    rx=re.compile(r"[\\/]\.venv[\\/]"))
             if (ROOT / p).is_dir() else compileall.compile_file(str(ROOT / p), quiet=2, force=True)
             for p in PATHS)
    assert ok, "a module has a syntax error (see output above)"


def test_workbench_survives_runs_saved_by_older_skill_versions():
    """Reported crash: an EDA run saved before strict schemas had recommended_charts as strings."""
    import app  # noqa: F401
    from workbench.callbacks.eda import wb_render_eda
    from workbench.renderers.ms import _ms_render
    old_eda = {"id": "eda_1", "label": "ecommerce", "timestamp": "10:00", "status": "complete",
               "result": {"narrative": "n", "key_insights": ["i"], "recommended_charts": ["histogram of cart_value"]},
               "summary": {}}
    rendered = str(wb_render_eda({"eda": [old_eda]})[0].to_plotly_json())
    assert "earlier version" not in rendered      # old string charts are skipped, the rest renders
    garbage = _ms_render({"recommendations": "not a list", "decision": 3})
    assert "earlier version" in str(garbage.to_plotly_json())
