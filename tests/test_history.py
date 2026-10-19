"""Suite history summary used by the Benchmarks page."""
from benchmarks.history import suite_summary


def _run(ts, results, mean=None):
    return {"timestamp": ts, "results": results, "summary": {"mean_score": mean, "n_pass": 1, "n_total": 1}}


def test_delta_is_against_the_previous_run_that_scored_the_case():
    runs = [
        _run("20261001T000000", {"a/data_quality": {"overall": 80.0, "passed": True}}, 80),
        _run("20261002T000000", {"b/data_quality": {"overall": 50.0, "passed": False}}, 50),
        _run("20261003T000000", {"a/data_quality": {"overall": 90.0, "passed": True},
                                 "b/data_quality": {"overall": 0.0, "passed": False, "error": "boom"}}, 45),
    ]
    rows = {r["id"]: r for r in suite_summary(runs)["rows"]}
    assert rows["a/data_quality"]["delta"] == 10.0
    assert rows["b/data_quality"]["score"] is None  # a crash is not a score of 0
    assert rows["b/data_quality"]["error"] == "boom"
    assert rows["b/data_quality"]["skill"] == "data_quality"


def test_no_history():
    assert suite_summary([]) is None


def test_page_renders_the_panel():
    import app  # noqa: F401
    from pages.benchmarks import _suite_panel
    panel = _suite_panel(suite_summary([_run("20261003T000000", {"a/data_quality": {"overall": 90.0, "passed": True}}, 90)]))
    assert "a/data_quality" in str(panel.to_plotly_json())
