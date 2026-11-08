"""Rescore the calibration pilots, and total every dollar spent, into results/pilots.json.

Pilots are never part of any result."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vmr.matrix import build
from vmr.problems import load


def main() -> None:
    slugs = [m["slug"] for m in json.loads(Path("data/models.json").read_text())["models"]]
    out = {}
    for name in ("pilot1", "pilot2"):
        rows = build(load(Path("data") / name / "problems.jsonl"), slugs, Path("data") / name, allow_mock=False)
        cells = defaultdict(lambda: [0, 0])
        for r in rows:
            for key in (f"{r['family']}|{r['model']}", f"{r['family']}|T{r['tier']}", f"{r['family']}|all models"):
                cells[key][0] += r["status"] == "correct"
                cells[key][1] += 1
        out[name] = {k: {"correct": c, "n": n, "accuracy": round(c / n, 4)} for k, (c, n) in sorted(cells.items())}
    from vmr.collect import read_records, record_cost

    spend = {p.name: round(sum(record_cost(r) for f in sorted(p.glob("*.jsonl")) for r in read_records(f)), 4)
             for p in sorted(Path("data").iterdir()) if p.is_dir()}
    out["spend_usd_by_directory"] = {**spend, "all": round(sum(spend.values()), 4)}
    Path("results/pilots.json").write_text(json.dumps(out, indent=1) + "\n")
    print("results/pilots.json written")


if __name__ == "__main__":
    main()
