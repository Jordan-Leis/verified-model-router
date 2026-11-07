"""Model calls: the real OpenRouter client, and a mock used only for tests.

The mock's output is synthetic and is never a result. Every mock record carries `"mock": true`, it
is written under build/ (which git ignores), and `vmr.matrix` refuses it unless asked for a mock run.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ENV_PATH = Path.home() / "Follow_up_projs" / ".env"
RETRIES = 5


def read_api_key() -> str:
    """Read the key from the environment or the shared .env. It is never printed or stored."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"OPENROUTER_API_KEY is not set and is not in {ENV_PATH}")


class OpenRouterClient:
    mock = False

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(base_url=OPENROUTER_URL, api_key=read_api_key(), timeout=300, max_retries=0)

    def complete(self, prompt: str, model: dict, problem: dict) -> dict:
        extra = {"usage": {"include": True}, **model.get("extra_body", {})}
        last_error = "not attempted"
        for attempt in range(RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=model["slug"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=model["max_tokens"],
                    temperature=model.get("temperature", 0),
                    extra_body=extra,
                )
                data = response.model_dump()
                if data.get("choices"):
                    return {"response": data, "attempts": attempt + 1, "error": None}
                last_error = f"empty choices: {str(data.get('error'))[:300]}"
            except Exception as error:  # noqa: BLE001 - every API failure is retried, then recorded.
                last_error = f"{type(error).__name__}: {str(error)[:300]}"
            time.sleep(2**attempt)
        return {"response": None, "attempts": RETRIES, "error": last_error}


class MockClient:
    """Deterministic synthetic responses for exercising the pipeline. NOT A RESULT."""

    mock = True
    SKILL = {"derivative": 0.9, "integral": 0.6, "dimension": 0.75}

    def complete(self, prompt: str, model: dict, problem: dict) -> dict:
        digest = hashlib.sha256(f"{model['slug']}|{problem['id']}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        tilt = (digest[8] % 3 - 1) * 0.1
        roll = rng.random()
        kind = problem["answer_type"]
        if kind == "antiderivative":
            truth = problem["reference"]
        elif kind == "dimension":
            truth = "*".join(f"{u}**({e})" for u, e in zip(("kg", "m", "s", "A", "K"), problem["truth_dimension"]))
        elif kind == "monomial":
            truth = "*".join(f"{k}**({v})" for k, v in problem["reference_exponents"].items())
        else:
            truth = problem["truth"]
        if roll < self.SKILL[problem["family"]] + tilt:
            text = f"MOCK working.\nANSWER: {truth}"
        elif roll < 0.97:
            wrong = f"({truth})**2" if kind == "dimension" or kind == "monomial" else f"2*({truth}) + 1"
            text = f"MOCK working.\nANSWER: {wrong}"
        else:
            text = "MOCK response with no final answer line"
        usage = {"prompt_tokens": 200, "completion_tokens": 100, "cost": model.get("mock_cost", 0.001)}
        data = {"id": "mock", "model": model["slug"], "provider": "MOCK", "usage": usage,
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
        return {"response": data, "attempts": 1, "error": None}
