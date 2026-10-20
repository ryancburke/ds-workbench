"""Headless-Chrome checks of behaviour only a browser shows. Opt in: RUN_BROWSER_TESTS=1
(needs selenium from requirements-dev.txt and Chrome)."""
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).parent.parent
pytestmark = pytest.mark.skipif(os.getenv("RUN_BROWSER_TESTS") != "1", reason="set RUN_BROWSER_TESTS=1")
PAGES = ["/", "/sessions", "/feedback", "/benchmarks", "/agent-sim", "/costs", "/data-quality", "/eda",
         "/join-auditor", "/nl-to-spark", "/mlops-review"]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    log = tmp_path_factory.mktemp("browser") / "server.log"
    proc = subprocess.Popen([sys.executable, str(ROOT / "tests" / "browser_app.py"), str(port)], cwd=ROOT,
                            stdout=log.open("w"), stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT), "SIM_INTERVAL_HOURS": "0"})
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1); break
        except OSError:
            time.sleep(0.5)
    yield port, log
    proc.terminate(); proc.wait(timeout=10)


@pytest.fixture
def browser():
    from selenium import webdriver
    opts = webdriver.ChromeOptions(); opts.add_argument("--headless=new")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    driver = webdriver.Chrome(options=opts)
    yield driver
    driver.quit()


def _renderer_errors(driver):
    # Dash dev tools check plotly.com for a newer version; that network call isn't the app's
    ignore = ("favicon", "dash-version.plotly.com")
    return [l["message"] for l in driver.get_log("browser")
            if l["level"] == "SEVERE" and not any(i in l["message"] for i in ignore)]


def test_visiting_pages_runs_nothing_and_raises_no_errors(server, browser):
    port, log = server
    for path in PAGES:
        browser.get(f"http://127.0.0.1:{port}{path}"); time.sleep(2.5)
    assert "CALLED" not in log.read_text(), log.read_text()[-600:]
    assert not _renderer_errors(browser)


def test_leaving_the_workbench_mid_run_raises_no_errors(server, browser):
    from selenium.webdriver.common.by import By
    port, log = server
    browser.get(f"http://127.0.0.1:{port}/"); time.sleep(3)
    browser.execute_script("document.getElementById('wb-ja-run-btn').click()"); time.sleep(0.5)
    browser.find_element(By.CSS_SELECTOR, 'a[href="/sessions"]').click(); time.sleep(5)
    assert "CALLED skills.join_auditor.audit" in log.read_text()        # the click did run
    assert not [e for e in _renderer_errors(browser) if "nonexistent" in e.lower()]
