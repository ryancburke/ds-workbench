"""
Pilot harness: does the Data Scout find better data than plain catalog search, on real questions?

    # 1. Questions from past projects, with the tables experts know were right (see the example)
    .venv/bin/python benchmarks/pilot.py prepare benchmarks/pilot_questions.example.json --out runs/pilot

    # 2. Experts fill runs/pilot/relevance_sheet.csv (blind: lists A/B, no method names) and
    #    runs/pilot/scout_notes_sheet.csv (is each suitability note correct?)

    # 3. Score
    .venv/bin/python benchmarks/pilot.py score runs/pilot

For each question the Scout and plain keyword search (the same terms a person would type, via the
same catalog) each return a top-5. Raters see both lists per question in random order, labelled A
and B, with only the table name and its catalog comment, and rate relevance 0 (no), 1 (useful
context), 2 (needed). The arm mapping stays in key.json. Scoring reports, per method: recall@5 of
the expert tables, precision@5 (relevance >= 1), mean relevance, and for the Scout also how often
it was confident and wrong (recommended "investigate", rated 0), how accurate its suitability
notes were, and time and cost per question. Outputs go to runs/ (git-ignored): real questions and
table names stay out of the repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

K = 5
SHEET = "relevance_sheet.csv"
NOTES = "scout_notes_sheet.csv"


def _scout_top(report: dict) -> list[dict]:
    order = {"investigate": 0, "maybe": 1}
    cands = sorted([c for c in report.get("candidates", []) if c["recommendation"] in order],
                   key=lambda c: (order[c["recommendation"]], c["rank"]))
    return cands[:K]


def prepare(questions: list[dict], catalog, out_dir: Path, scout_fn=None, frame_fn=None, seed: int = 7) -> dict:
    """Run both methods on every question and write the blinded sheets and the key."""
    from ds_workbench.core.scout import search_terms
    from skills.usage import load
    if scout_fn is None:
        from skills.data_scout import scout as scout_fn
    if frame_fn is None:
        from skills.problem_framer import frame as frame_fn
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key, sheet, notes = {"k": K, "questions": {}}, [], []
    for q in questions:
        t0 = time.time()
        spec = q.get("spec") or frame_fn(q["question"], context=q.get("context", ""))
        if spec.get("error"):
            key["questions"][q["id"]] = {"error": f"framing failed: {spec['error']}"}
            continue
        report = scout_fn(spec=spec, catalog=catalog)
        elapsed = round(time.time() - t0, 1)
        if report.get("error"):
            key["questions"][q["id"]] = {"error": f"scout failed: {report['error']}"}
            continue
        scout_list = _scout_top(report)
        baseline = catalog.search(search_terms(spec), limit=K)
        arms = ["scout", "baseline"]
        rng.shuffle(arms)
        lists = {"scout": [c["table"] for c in scout_list], "baseline": [h["name"] for h in baseline]}
        comments = {h["name"]: h.get("comment", "") for h in baseline}
        for label, arm in zip("AB", arms):
            for rank, table in enumerate(lists[arm], 1):
                comment = comments.get(table) or catalog.describe(table).get("comment", "")
                sheet.append({"question_id": q["id"], "question": q["question"], "list": label, "rank": rank,
                              "table": table, "catalog_comment": comment, "relevance_0_1_2": ""})
        for c in scout_list:
            note = c["why"] + ("" if not c["concerns"] else " | concerns: " + "; ".join(c["concerns"]))
            notes.append({"question_id": q["id"], "table": c["table"], "scout_recommendation": c["recommendation"],
                          "scout_note": note, "note_correct_y_n": ""})
        run_id = report.get("run_id")
        try:
            ledger = load()
            cost = round(float(ledger[ledger.run_id == run_id].cost_usd.sum()), 4) if run_id else None
        except Exception:
            cost = None
        key["questions"][q["id"]] = {
            "arms": dict(zip("AB", arms)), "lists": lists, "expert_tables": q.get("expert_tables", []),
            "scout_investigate": [c["table"] for c in scout_list if c["recommendation"] == "investigate"],
            "coverage_gaps": report.get("coverage_gaps", []), "seconds": elapsed, "cost_usd": cost,
            "ungrounded_claims_removed": len(report.get("validation", {}).get("dropped", []))}
    _write(out_dir / SHEET, sheet)
    _write(out_dir / NOTES, notes)
    (out_dir / "key.json").write_text(json.dumps(key, indent=1))
    return key


def _write(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def score(out_dir: Path) -> dict:
    """Per-method metrics from the filled sheets and the key."""
    key = json.loads((out_dir / "key.json").read_text())
    ratings = {}
    for r in _read(out_dir / SHEET):
        v = (r.get("relevance_0_1_2") or "").strip()
        if v in ("0", "1", "2"):
            ratings[(r["question_id"], r["table"])] = int(v)
    notes = [r for r in _read(out_dir / NOTES) if (r.get("note_correct_y_n") or "").strip().lower() in ("y", "n")]
    per = {"scout": [], "baseline": []}
    unrated = 0
    for qid, q in key["questions"].items():
        if "error" in q:
            continue
        for arm in per:
            tables = q["lists"][arm]
            rel = [ratings.get((qid, t)) for t in tables]
            unrated += sum(r is None for r in rel)
            got = [r for r in rel if r is not None]
            expert = q.get("expert_tables") or []
            per[arm].append({
                "recall": (sum(t in tables for t in expert) / len(expert)) if expert else None,
                "precision": (sum(r >= 1 for r in got) / len(got)) if got else None,
                "relevance": (sum(got) / len(got)) if got else None,
            })
    def mean(rows, k):
        vals = [r[k] for r in rows if r[k] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    qs = [q for q in key["questions"].values() if "error" not in q]
    confident_wrong = sum(1 for qid, q in key["questions"].items() if "error" not in q
                          for t in q["scout_investigate"] if ratings.get((qid, t)) == 0)
    out = {
        "questions": len(qs), "failed": sum("error" in q for q in key["questions"].values()), "unrated": unrated,
        **{arm: {f"recall@{K}": mean(rows, "recall"), f"precision@{K}": mean(rows, "precision"),
                 "mean_relevance": mean(rows, "relevance")} for arm, rows in per.items()},
        "scout_confident_but_wrong": confident_wrong,
        "scout_note_accuracy": round(sum(r["note_correct_y_n"].strip().lower() == "y" for r in notes) / len(notes), 3)
        if notes else None,
        "scout_seconds_per_question": round(sum(q["seconds"] for q in qs) / len(qs), 1) if qs else None,
        "scout_cost_per_question": (round(sum(q["cost_usd"] for q in qs if q["cost_usd"] is not None) / len(qs), 4)
                                    if qs and any(q["cost_usd"] is not None for q in qs) else None),
        "scout_questions_with_coverage_gaps": sum(bool(q["coverage_gaps"]) for q in qs),
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description="Data Scout pilot: Scout vs plain catalog search, rated blind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("questions")
    p.add_argument("--out", default="runs/pilot")
    s = sub.add_parser("score")
    s.add_argument("out", nargs="?", default="runs/pilot")
    args = ap.parse_args()
    if args.cmd == "prepare":
        from skills.data_scout import get_catalog
        qs = json.loads(Path(args.questions).read_text())
        key = prepare(qs, get_catalog(), Path(args.out))
        print(f"{len(key['questions'])} questions → {args.out}/{SHEET} and {NOTES}; fill them, then `score`.")
    else:
        print(json.dumps(score(Path(args.out)), indent=1))


if __name__ == "__main__":
    main()
