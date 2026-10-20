"""App for browser tests: every skill and paid action stubbed (2s, logged), nothing calls the API."""
import importlib
import sys
import time

import app as a
from skills.result import SkillResult

TARGETS = {"skills.data_quality": "assess", "skills.eda": "analyze", "skills.join_auditor": "audit",
           "skills.nl_to_spark": "generate", "skills.mlops_reviewer": "review", "skills.model_selector": "select",
           "skills.leakage_detector": "check", "skills.feature_engineering": "suggest",
           "skills.experiment_tracker": "interpret", "skills.causal_inference": "analyze",
           "agents.scheduler": ["trigger_bulk_run", "draft_code_change", "export_patch", "dismiss_stale"]}

for mod, fns in TARGETS.items():
    m = importlib.import_module(mod)
    for fn in [fns] if isinstance(fns, str) else fns:
        def stub(*args, _name=f"{mod}.{fn}", **kw):
            print("CALLED", _name, flush=True)
            time.sleep(2)
            return SkillResult({"error": "stubbed for browser test"}, skill=_name.split(".")[1])
        setattr(m, fn, stub)

a.app.run(debug=True, use_reloader=False, port=int(sys.argv[1]))
