"""End-to-end run on the MOCK client. These numbers are synthetic and are never results."""
import json
from pathlib import Path

from vmr.analysis import run
from vmr.collect import collect
from vmr.llm import MockClient
from vmr.matrix import build, lenient_answer, write_rows
from vmr.problems import generate

SMALL = {"derivative": (2, 2, 2), "integral": (2, 2, 2), "dimension": (2, 2, 2)}
MODELS = [{"slug": "mock/a", "max_tokens": 10, "mock_cost": 0.001}, {"slug": "mock/b", "max_tokens": 10, "mock_cost": 0.01},
          {"slug": "mock/c", "max_tokens": 10, "mock_cost": 0.1}]


def test_mock_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("vmr.analysis.RUNS", 50)
    monkeypatch.setattr("vmr.analysis.PERMUTATIONS", 50)
    problems = generate(1, SMALL)
    raw = tmp_path / "build" / "mock"
    collect(problems, MODELS, raw, MockClient(), budget=10, ledger=[])
    rows = build(problems, [m["slug"] for m in MODELS], raw, allow_mock=True)
    assert {r["status"] for r in rows} <= {"correct", "wrong", "no_answer"}
    write_rows(rows, tmp_path / "verdicts.jsonl")
    results = run(tmp_path / "verdicts.jsonl", tmp_path / "out", source="MOCK")
    assert results["source"] == "MOCK"
    assert results["policies"]["per-problem oracle"]["accuracy"] >= results["policies"]["best single model (hindsight)"]["accuracy"]
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["n_problems"] == len(problems)


def test_matrix_refuses_mock_records(tmp_path):
    problems = generate(1, {"derivative": (1, 0, 0)})
    collect(problems, MODELS[:1], tmp_path, MockClient(), budget=10, ledger=[])
    try:
        build(problems, ["mock/a"], tmp_path, allow_mock=False)
    except SystemExit as error:
        assert "mock" in str(error)
    else:
        raise AssertionError("mock records were accepted as results")


def test_lenient_answer_only_fixes_formatting():
    assert lenient_answer("work\n**ANSWER:** x^2 + C") == "x**2"
    assert lenient_answer("ANSWER: ln(x)\nThat is all.") == "log(x)"
    assert lenient_answer("no answer here") is None
