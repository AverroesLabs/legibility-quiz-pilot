"""OpenRouter client: one fixed provider per model, append-only call log, resumable cache."""
import hashlib, json, os, threading, time
from pathlib import Path

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"

# One provider per model, so a score never mixes hosts. First-party or a large host.
PROVIDERS = {
    "meta-llama/llama-3.2-3b-instruct": "Parasail",
    "ibm-granite/granite-4.2-8b": "CoreWeave",
    "mistralai/ministral-8b-2512": "Mistral",
    "qwen/qwen3.5-9b": "Together",
    "mistralai/ministral-14b-2512": "Mistral",
    "openai/gpt-oss-20b": "CoreWeave",
    "google/gemma-4-31b-it": "DeepInfra",
    "qwen/qwen3.8-27b": "DeepInfra",
    "anthropic/claude-opus-5.5": "Anthropic",
    "openai/gpt-5.6-sol": "OpenAI",
    "moonshotai/kimi-k3": "Moonshot AI",
    "z-ai/glm-5.3": "Z.AI",
}
# Concurrent requests per provider (Jason, 2026-11-03: limit concurrency per provider, not per paper).
PROVIDER_LIMIT = {"Moonshot AI": 8, "Anthropic": 8, "OpenAI": 12, "Z.AI": 24, "DeepInfra": 12, "Mistral": 12,
                  "Parasail": 12}
_SEMS = {}
_SEMS_LOCK = threading.Lock()


def provider_slot(model):
    prov = PROVIDERS[model]
    with _SEMS_LOCK:
        if prov not in _SEMS:
            _SEMS[prov] = threading.BoundedSemaphore(PROVIDER_LIMIT.get(prov, 8))
        return _SEMS[prov]


# Models that reject other sampling temperatures.
FIXED_TEMPERATURE = {"moonshotai/kimi-k3": 1.0}
READERS = [m for m in PROVIDERS if m.split("/")[0] in
           ("meta-llama", "ibm-granite", "mistralai", "qwen", "google") or m == "openai/gpt-oss-20b"]


def load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("OPENROUTER_API_KEY not found")


def account_usage():
    r = requests.get(KEY_URL, headers={"Authorization": f"Bearer {load_key()}"}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]["usage"]


class LengthError(RuntimeError):
    """The model used the whole token limit on hidden reasoning and gave no answer. Retrying unchanged is wasted spend."""


class Ledger:
    """Every successful call goes in calls.jsonl. A repeated key returns the logged result."""

    def __init__(self, path, spend_cap):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spend_cap = spend_cap
        self.lock = threading.Lock()
        self.cache = {}
        self.new_spend = 0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                if "content" in rec:
                    self.cache[rec["key"]] = rec

    def spent(self):
        return sum(r["cost"] or 0 for r in self.cache.values())

    def call(self, model, messages, *, tag, sample=0, max_tokens=16000, temperature=None, seed=None,
             reasoning_effort=None, timeout=600):
        """reasoning_effort: None keeps the model default; "minimal" is the lowest setting OpenRouter accepts for
        GLM-5.3, which refuses to turn reasoning off entirely."""
        if temperature is not None:
            temperature = FIXED_TEMPERATURE.get(model, temperature)
        params = {"max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        key = hashlib.sha256(json.dumps([model, messages, params, sample], sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            if self.new_spend >= self.spend_cap:
                raise RuntimeError(f"spend cap ${self.spend_cap} reached")
        body = {"model": model, "messages": messages, "usage": {"include": True},
                "provider": {"order": [PROVIDERS[model]], "allow_fallbacks": False},
                **{k: v for k, v in params.items() if v is not None and k != "reasoning_effort"}}
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        error, failed_cost = None, 0.0
        for attempt in range(4):
            try:
                with provider_slot(model):
                    r = requests.post(URL, json=body, timeout=timeout,
                                      headers={"Authorization": f"Bearer {load_key()}"})
                data = r.json()
                if r.status_code != 200 or "choices" not in data:
                    raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data)[:300]}")
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
                usage = data.get("usage") or {}
                cost = usage.get("cost") or 0
                with self.lock:
                    self.new_spend += cost
                if not content.strip():
                    failed_cost += cost
                    if choice.get("finish_reason") == "length":
                        error = "empty content, finish=length"
                        break
                    raise RuntimeError(f"empty content, finish={choice.get('finish_reason')}")
                rec = {"key": key, "tag": tag, "model": model, "model_returned": data.get("model"),
                       "provider": data.get("provider"), "generation": data.get("id"), "params": params,
                       "sample": sample, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "finish": choice.get("finish_reason"), "usage": usage, "cost": cost,
                       "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                       "content": content}
                with self.lock:
                    self.cache[key] = rec
                    with self.path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                return rec
            except (requests.RequestException, RuntimeError, ValueError) as e:
                error = str(e)
                time.sleep(20 * 2 ** attempt)
        with self.lock, self.path.open("a") as f:
            f.write(json.dumps({"key": key, "tag": tag, "model": model, "sample": sample,
                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "error": error, "cost": failed_cost}) + "\n")
        if error == "empty content, finish=length":
            raise LengthError(f"{model} failed: {error}")
        raise RuntimeError(f"{model} failed: {error}")
