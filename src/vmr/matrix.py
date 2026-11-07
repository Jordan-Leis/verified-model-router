"""Turn raw responses into the verdict matrix: one row per (problem, model)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vmr.collect import read_records, record_cost, slug_file
from vmr.problems import load
from vmr.verifier import CORRECT, NO_ANSWER, Verdict, extract_answer, verify_answer, verify_response

LENIENT_REWRITES = [("^", "**"), ("ln(", "log("), ("arctan(", "atan("), ("arcsin(", "asin("), ("arccos(", "acos(")]


def lenient_answer(text: str | None) -> str | None:
    """Sensitivity check only: recover answers a strict reader rejects for formatting.

    It takes the last line containing 'ANSWER:' anywhere, strips markdown and LaTeX dollar signs,
    applies common spellings (^, ln, arctan), and drops a trailing '+ C'. Strict scoring never uses it.
    """
    lines = [line for line in (text or "").splitlines() if "ANSWER:" in line]
    if not lines:
        return None
    answer = lines[-1].split("ANSWER:", 1)[1]
    for token in ("**ANSWER", "`", "$", "\\(", "\\)"):
        answer = answer.replace(token, "")
    answer = answer.strip().strip("*").strip()
    for old, new in LENIENT_REWRITES:
        answer = answer.replace(old, new)
    for suffix in ("+ C", "+C"):
        if answer.endswith(suffix):
            answer = answer[: -len(suffix)].strip()
    return answer or None


def judge(record: dict, problem: dict) -> dict:
    """Score one raw record. An API failure gives no answer and costs nothing."""
    response = record.get("response") or {}
    choices = response.get("choices") or []
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    message = (choices[0].get("message") or {}) if choices else {}
    finish = choices[0].get("finish_reason") if choices else None
    text = message.get("content")
    verdict = verify_response(text, finish, problem) if choices else Verdict(NO_ANSWER, "api_error")
    lenient = verdict
    if verdict.status not in ("correct", "wrong") and lenient_answer(text) is not None:
        lenient = verify_answer(lenient_answer(text), problem)
    return {
        "problem_id": problem["id"], "family": problem["family"], "tier": problem["tier"],
        "model": record["model"], **verdict.to_dict(), "lenient_status": lenient.status,
        "answer": extract_answer(text), "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens") or 0,
        "cost": round(record_cost(record), 8), "provider": response.get("provider"),
    }


def build(problems: list[dict], models: list[str], raw: Path, allow_mock: bool) -> list[dict]:
    by_id = {p["id"]: p for p in problems}
    rows = []
    for slug in models:
        latest = {}
        for record in read_records(raw / slug_file(slug)):
            if record.get("mock") and not allow_mock:
                raise SystemExit(f"Refusing mock record in {raw}: mock output is never a result")
            current = latest.get(record["problem_id"])
            if current is None or current["error"] is not None:
                latest[record["problem_id"]] = record
        missing = set(by_id) - set(latest)
        if missing:
            raise SystemExit(f"{slug}: {len(missing)} problems have no response in {raw}")
        rows += [judge(latest[p["id"]], p) for p in problems]
    return rows


@dataclass
class Matrix:
    problem_ids: list[str]
    families: np.ndarray  # (P,) index into family_names
    family_names: list[str]
    tiers: np.ndarray  # (P,)
    models: list[str]
    correct: np.ndarray  # (P, K) bool
    cost: np.ndarray  # (P, K) float, in dollars
    status: np.ndarray  # (P, K) str


def to_matrix(rows: list[dict]) -> Matrix:
    models = list(dict.fromkeys(r["model"] for r in rows))
    ids = list(dict.fromkeys(r["problem_id"] for r in rows))
    family_names = list(dict.fromkeys(r["family"] for r in rows))
    lookup = {(r["problem_id"], r["model"]): r for r in rows}
    first = {r["problem_id"]: r for r in reversed(rows)}
    grid = [[lookup[(p, m)] for m in models] for p in ids]
    return Matrix(
        problem_ids=ids,
        families=np.array([family_names.index(first[p]["family"]) for p in ids]),
        family_names=family_names,
        tiers=np.array([first[p]["tier"] for p in ids]),
        models=models,
        correct=np.array([[c["status"] == CORRECT for c in row] for row in grid]),
        cost=np.array([[c["cost"] for c in row] for row in grid], dtype=float),
        status=np.array([[c["status"] for c in row] for row in grid]),
    )


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="data/problems.jsonl")
    parser.add_argument("--models", default="data/models.json")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="results/verdicts.jsonl")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    slugs = [m["slug"] for m in json.loads(Path(args.models).read_text())["models"]]
    rows = build(load(args.problems), slugs, Path(args.raw), args.mock)
    write_rows(rows, Path(args.out))
    print(f"{args.out}: {len(rows)} verdicts")


if __name__ == "__main__":
    main()
