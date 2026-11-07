"""Rescore the calibration pilots into results/pilots.json. Pilots are never part of any result."""

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
    Path("results/pilots.json").write_text(json.dumps(out, indent=1) + "\n")
    print("results/pilots.json written")


if __name__ == "__main__":
    main()
