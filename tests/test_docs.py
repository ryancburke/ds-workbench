"""The docs describe the code as it is: these are the facts that drifted before."""
import dataclasses
import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
CLAUDE = (ROOT / "CLAUDE.md").read_text()
README = (ROOT / "README.md").read_text()
ENV = (ROOT / ".env.example").read_text()


def _settings():
    from config import Config
    return [f.name.upper() for f in dataclasses.fields(Config)]


def test_every_setting_is_in_env_example():
    assert [s for s in _settings() if s not in ENV] == []


def test_every_setting_is_in_the_readme_config_table():
    def documented(s):
        if s in README:
            return True
        # "| `LEAKAGE_MODEL` / `_EFFORT` |" documents LEAKAGE_EFFORT
        return s.endswith("_EFFORT") and f"{s[:-len('_EFFORT')]}_MODEL` / `_EFFORT" in README
    assert [s for s in _settings() if not documented(s)] == []


def test_every_notebook_skill_is_in_claude_md():
    import ds_workbench
    assert [k for k in ds_workbench._SKILLS if f"`{k}`" not in CLAUDE and f"{k}." not in CLAUDE] == []


def test_every_core_module_and_page_is_in_the_project_structure():
    import ds_workbench.core
    assert [m for m in ds_workbench.core._SUBMODULES if f"{m}.py" not in CLAUDE] == []
    pages = [p.name for p in (ROOT / "pages").glob("*.py") if p.name != "__init__.py"]
    assert [p for p in pages if p not in CLAUDE] == []


def test_the_benchmark_case_count_is_current():
    from benchmarks.runner import _get_benchmarks
    m = re.search(r"Benchmark cases \((\d+),", CLAUDE)
    assert m and int(m.group(1)) == len(_get_benchmarks())


def test_roadmap_milestones_name_their_prs():
    roadmap = (ROOT / "ROADMAP.md").read_text()
    for row in re.findall(r"^\| M\d .*$", roadmap, flags=re.M):
        assert "shipped (#" in row or "in progress" in row or "next" in row, row
