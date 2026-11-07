"""Call every model once per problem, keeping the full raw response.

Output is append-only JSONL with one file per model, so an interrupted run resumes where it stopped.
Spend is the sum of OpenRouter's reported per-call cost across every raw file, including earlier
runs, and the run stops as soon as that sum reaches the budget.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vmr.llm import MockClient, OpenRouterClient
from vmr.problems import load
from vmr.prompts import build_prompt

WORKERS = 4


def slug_file(slug: str) -> str:
    return slug.replace("/", "__").replace(":", "_") + ".jsonl"


def record_cost(record: dict) -> float:
    response = record.get("response") or {}
    return float((response.get("usage") or {}).get("cost") or 0.0)


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def total_spend(roots: list[Path]) -> float:
    return sum(record_cost(r) for root in roots for f in root.glob("*.jsonl") for r in read_records(f))


def collect(problems: list[dict], models: list[dict], out: Path, client, budget: float, ledger: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spent = total_spend(sorted(set(ledger + [out])))
    lock = threading.Lock()
    stop = threading.Event()
    tasks = []
    for model in models:
        path = out / slug_file(model["slug"])
        done = {r["problem_id"] for r in read_records(path) if r["error"] is None}
        tasks += [(model, p, path) for p in problems if p["id"] not in done]
    print(f"{len(tasks)} calls to make; ${spent:.4f} already spent; budget ${budget:.2f}", flush=True)

    def run(task):
        nonlocal spent
        model, problem, path = task
        if stop.is_set():
            return
        prompt = build_prompt(problem)
        outcome = client.complete(prompt, model, problem)
        record = {
            "problem_id": problem["id"], "model": model["slug"],
            "request": {"prompt": prompt, "max_tokens": model["max_tokens"],
                        "temperature": model.get("temperature", 0), "extra_body": model.get("extra_body", {})},
            "collected_at": int(time.time()), "mock": client.mock, **outcome,
        }
        with lock:
            with path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            spent += record_cost(record)
            if spent >= budget:
                stop.set()
            finished[0] += 1
            if finished[0] % 20 == 0:
                print(f"{finished[0]}/{len(tasks)} done, ${spent:.4f}", flush=True)

    finished = [0]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(run, tasks))
    if stop.is_set():
        raise SystemExit(f"Stopped: spend ${spent:.4f} reached the ${budget:.2f} budget")
    print(f"Finished. Total spend ${spent:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="data/problems.jsonl")
    parser.add_argument("--models", default="data/models.json")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--budget", type=float, default=4.0)
    parser.add_argument("--ledger", nargs="*", default=["data/raw", "data/pilot"])
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock and not args.out.startswith("build/"):
        raise SystemExit("Mock output must go under build/ so it can never be mistaken for a result")
    models = json.loads(Path(args.models).read_text())["models"]
    client = MockClient() if args.mock else OpenRouterClient()
    ledger = [] if args.mock else [Path(p) for p in args.ledger]
    collect(load(args.problems), models, Path(args.out), client, args.budget, ledger)


if __name__ == "__main__":
    main()
