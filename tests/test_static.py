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


_THREADED_IMPORT = """
import importlib, sys, threading
{prelude}
mods = ["skills.join_auditor", "skills.leakage_detector", "skills.causal_inference",
        "ds_workbench.core.leakage", "ds_workbench.core.joins", "ds_workbench.core.causal_stats"]
errors, barrier = [], threading.Barrier(10)
def worker(i):
    barrier.wait()
    try:
        importlib.import_module(mods[i % len(mods)])
    except Exception as exc:
        errors.append(repr(exc))
threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
[t.start() for t in threads]; [t.join() for t in threads]
assert not errors, errors
"""


def test_concurrent_first_imports_do_not_deadlock():
    """Dash serves callbacks on several threads, and skills are imported lazily inside them.
    Before the fix this failed ~10/10 with _DeadlockError on ds_workbench.core.*."""
    for prelude in ("", "import app"):          # bare (root-cause fix) and as the app runs
        for _ in range(5):
            out = subprocess.run([sys.executable, "-c", _THREADED_IMPORT.format(prelude=prelude)],
                                 cwd=ROOT, capture_output=True, text=True)
            assert out.returncode == 0, f"prelude={prelude!r}: {out.stderr[-400:]}"


def test_button_callbacks_do_nothing_without_a_click(monkeypatch):
    """Dash Pages fired some run callbacks when a page mounted despite prevent_initial_call:
    every Workbench load ran data quality, EDA and a join audit (paid API calls). Every callback
    driven by a button must refuse to act when n_clicks is empty."""
    import inspect
    import app  # noqa: F401
    from dash._callback import GLOBAL_CALLBACK_MAP
    from dash.exceptions import PreventUpdate
    import skills.api_retry as api_retry

    def no_api(*a, **k):
        raise AssertionError("a button callback reached the API without a click")
    monkeypatch.setattr(api_retry, "get_client", no_api)

    checked, offenders = [], []
    for cb in GLOBAL_CALLBACK_MAP.values():
        if not any(i.get("property") == "n_clicks" for i in cb["inputs"]):
            continue
        fn = inspect.unwrap(cb["callback"])
        n_params = len(inspect.signature(fn).parameters)
        # pattern-matching ALL inputs arrive as lists, e.g. [None] for an unclicked button
        first = [None] if '"ALL"' in str(cb["inputs"][0]["id"]) else None
        try:
            fn(first, *([None] * (n_params - 1)))
        except PreventUpdate:
            checked.append(fn.__name__)
            continue
        except Exception as exc:              # anything else means it started doing work
            offenders.append(f"{fn.__module__}.{fn.__name__}: {type(exc).__name__}")
            continue
        offenders.append(f"{fn.__module__}.{fn.__name__}: returned")
    assert not offenders, "ran without a click:\n" + "\n".join(offenders)
    assert len(checked) >= 20, checked
