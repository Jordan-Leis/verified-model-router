# Every result is rebuilt from committed raw responses. No API calls unless you run `make live`.
PY = PYTHONHASHSEED=0 uv run --frozen python

.PHONY: reproduce test live mock outbox-audit

reproduce: test
	$(PY) -m vmr.problems --check
	$(PY) -m vmr.matrix
	$(PY) -m vmr.pilots
	$(PY) -m vmr.analysis
	$(PY) -m vmr.audit
	$(PY) -m vmr.figure
	@git diff --quiet -- results ':!results/figure.png' \
	  && echo "reproduce: OK, every committed result was regenerated identically" \
	  || (git --no-pager diff --stat -- results; echo "reproduce: results differ from the committed ones"; exit 1)

test:
	$(PY) -m pytest -q

# Paid: calls OpenRouter for any (problem, model) pair without a saved response. Needs OPENROUTER_API_KEY.
live:
	$(PY) -m vmr.collect --out data/raw --budget 4

# Synthetic responses through the whole pipeline. Output goes to build/, which is never a result.
mock:
	$(PY) -m vmr.collect --mock --out build/mock
	$(PY) -m vmr.matrix --mock --raw build/mock --out build/mock_verdicts.jsonl
	$(PY) -m vmr.analysis --verdicts build/mock_verdicts.jsonl --out build/mock_results --source MOCK

outbox-audit:
	$(PY) -m vmr.audit --sample-markdown ../outbox/VERIFIER_AUDIT.md
