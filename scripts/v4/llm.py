"""OpenAI client for v4: append-only call log per run, resumable cache, and one shared spend ledger with a hard cap.

Every call by any process is charged to data/v4/spend.jsonl under a file lock (earlier rows are in the frozen
spend-<n>.jsonl files, which count towards the cap too). Before a call, the worst-case cost
(prompt plus the full output allowance) must fit under the cap, so concurrent callers cannot overshoot it.
Costs use PRICES below, OpenAI's published list prices applied to each call's reported token counts. This is the
only spend record: the project key cannot read OpenAI's billing, and Jason has no access to the usage page.
"""
import fcntl, hashlib, json, os, threading, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
URL = "https://api.openai.com/v1/responses"
SPEND = Path(os.environ.get("V4_SPEND_FILE", ROOT / "data" / "v4" / "spend.jsonl"))
CAP = float(os.environ.get("V4_SPEND_CAP", "2500"))   # Jason raised the v4 cap from $500 to $2,000 on 2026-11-13, to $2,500 on 2026-11-17

# USD per million tokens (input, cached input, output), standard tier, input under 272K tokens. OpenAI's published
# list prices, read 2026-11-13 from https://developers.openai.com/api/docs/pricing. Reasoning tokens bill as output
# (https://developers.openai.com/api/docs/guides/reasoning). Before 2026-11-13 this table held v3's gateway rates,
# which were 2 to 50 times too high (see docs/decisions.md). Models not listed are charged at FALLBACK, the most
# expensive rate on the page, so the ledger errs high for them.
PRICES = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),        # promotional price until 2026-11-30
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-mini-2025-08-07": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-mini-2025-04-14": (0.40, 0.10, 1.60),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.5": (5.00, 0.50, 30.00),
}
FALLBACK = (5.00, 0.50, 30.00)
# The Batch API costs half of every rate above (https://developers.openai.com/api/docs/guides/batch).
NO_REASONING = ("gpt-4.1", "gpt-4o")      # model prefixes that reject the reasoning parameter

_SEM = threading.BoundedSemaphore(int(os.environ.get("V4_CONCURRENCY", "12")))
_LOCAL = threading.Lock()


def price(model):
    return PRICES.get(model, FALLBACK)


def load_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("OPENAI_API_KEY not found")


class CapReached(RuntimeError):
    pass


class LengthError(RuntimeError):
    """The model spent its whole output allowance (often on reasoning) and gave no answer."""


def _frozen_total():
    """Spend in frozen ledgers (spend-1.jsonl, spend-2.jsonl, ... next to SPEND). They are rotated out of SPEND to
    keep each file under GitHub's 100 MB limit, never written again, so their sum is read once per process."""
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = sum(json.loads(line)["usd"] for p in sorted(SPEND.parent.glob(SPEND.stem + "-*.jsonl"))
                      for line in p.open() if line.strip())
    return _FROZEN


_FROZEN = None


def _ledger(fn):
    """Run fn(total_spent, append) while holding an exclusive lock on the spend ledger. The total includes the
    frozen ledgers; new rows go to SPEND only."""
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    frozen = _frozen_total()
    with _LOCAL, open(SPEND, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            total = frozen + sum(json.loads(line)["usd"] for line in f if line.strip())

            def append(rec):
                f.seek(0, 2)
                f.write(json.dumps(rec) + "\n")
                f.flush()
            return fn(total, append)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def spent():
    return _ledger(lambda total, append: total)


def _cost(model, usage):
    pin, pcached, pout = price(model)
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    return ((usage.get("input_tokens", 0) - cached) * pin + cached * pcached + usage.get("output_tokens", 0) * pout) / 1e6


def _reserve(model, chars, max_tokens, tag, factor=1.0):
    """factor 0.5 for the Batch API."""
    pin, _, pout = price(model)
    worst = factor * (chars / 2.5 * pin + max_tokens * pout) / 1e6      # 2.5 characters per token is pessimistic
    rid = hashlib.sha256(f"{time.time_ns()}{tag}{os.getpid()}".encode()).hexdigest()[:16]

    def go(total, append):
        if total + worst > CAP:
            raise CapReached(f"spend cap ${CAP:.2f}: ${total:.2f} spent, this call could cost ${worst:.2f}")
        append({"id": rid, "kind": "reserve", "model": model, "tag": tag, "usd": worst,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _ledger(go)
    return rid, worst


def _settle(rid, worst, model, tag, usage, unknown=False):
    """Replace the reservation with the actual cost. If the request was lost in transit (timeout, dropped
    connection), OpenAI may still have billed it, so the full reservation stays charged."""
    actual = worst if unknown else (_cost(model, usage) if usage else 0.0)
    _ledger(lambda total, append: append({"id": rid, "kind": "settle", "model": model, "tag": tag,
                                          "usd": actual - worst, "actual": actual, "usage": usage or None,
                                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


def _params(model, max_tokens, effort):
    params = {"max_output_tokens": max_tokens}
    if effort and not model.startswith(NO_REASONING):
        params["reasoning"] = {"effort": effort}
    return params


def _key(model, system, user, params, sample):
    return hashlib.sha256(json.dumps([model, system, user, params, sample], sort_keys=True).encode()).hexdigest()


def _content(data):
    return "".join(c.get("text", "") for item in data.get("output", [])
                   if item.get("type") == "message" for c in item.get("content", []))


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Calls:
    """Every successful call goes in <run>/calls.jsonl. A repeated key returns the logged result without a new call.
    Batch API jobs (submit_batch / collect_batch) are listed in <run>/batches.jsonl and land in the same cache."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                if "content" in rec:
                    self.cache[rec["key"]] = rec

    def call(self, model, system, user, *, tag, sample=0, max_tokens=16000, effort=None, timeout=900):
        """effort: reasoning effort ("minimal", "low", "medium", "high") or None for the model default."""
        params = _params(model, max_tokens, effort)
        key = _key(model, system, user, params, sample)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        body = {"model": model, "instructions": system, "input": user, "store": False, **params}
        error = None
        for attempt in range(4):
            rid, worst = _reserve(model, len(system or "") + len(user), max_tokens, tag)
            usage, unknown = None, False
            try:
                with _SEM:
                    r = requests.post(URL, json=body, timeout=timeout,
                                      headers={"Authorization": f"Bearer {load_key()}"})
                data = r.json()
                usage = data.get("usage")
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data.get('error', data))[:300]}")
                content = _content(data)
                if not content.strip():
                    reason = (data.get("incomplete_details") or {}).get("reason")
                    if reason == "max_output_tokens":
                        error = "empty answer: output allowance used up"
                        break
                    raise RuntimeError(f"empty answer, status={data.get('status')}")
                rec = {"key": key, "tag": tag, "model": model, "model_returned": data.get("model"),
                       "response_id": data.get("id"), "params": params, "sample": sample,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": data.get("status"),
                       "usage": usage, "cost": _cost(model, usage or {}),
                       "prompt_sha256": hashlib.sha256(json.dumps([system, user]).encode()).hexdigest(),
                       "content": content}
                with self.lock:
                    self.cache[key] = rec
                    with self.path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                return rec
            except (requests.RequestException, RuntimeError, ValueError) as e:
                error = str(e)
                unknown = isinstance(e, requests.RequestException)
                if "HTTP 4" in error and "HTTP 429" not in error:
                    break                       # a bad request will not succeed on retry
                time.sleep(15 * 2 ** attempt)
            finally:
                _settle(rid, worst, model, tag, usage, unknown)
        with self.lock, self.path.open("a") as f:
            f.write(json.dumps({"key": key, "tag": tag, "model": model, "sample": sample, "error": error,
                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        if error and error.startswith("empty answer: output allowance"):
            raise LengthError(f"{model}: {error}")
        raise RuntimeError(f"{model} failed: {error}")

    # ---- Batch API: half price, results within 24 hours. Use only for runs nothing else waits on. ----

    def submit_batch(self, model, items, *, tag, max_tokens=16000, effort=None):
        """items: dicts with system, user and optional sample. Uncached items go to one batch (one model per batch,
        as the API requires). The worst case of the whole batch at the batch rate is reserved before submitting.
        Returns the batch id, or None if every item is already cached."""
        params = _params(model, max_tokens, effort)
        lines, samples, chars = [], {}, 0
        for it in items:
            key = _key(model, it["system"], it["user"], params, it.get("sample", 0))
            if key in self.cache or key in samples:
                continue
            samples[key] = it.get("sample", 0)
            chars += len(it["system"] or "") + len(it["user"])
            lines.append(json.dumps({"custom_id": key, "method": "POST", "url": "/v1/responses",
                                     "body": {"model": model, "instructions": it["system"], "input": it["user"],
                                              "store": False, **params}}))
        if not lines:
            return None
        assert len(lines) <= 50000, "the Batch API takes at most 50,000 requests per batch"
        rid, worst = _reserve(model, chars, max_tokens * len(lines), tag, factor=0.5)
        try:
            auth = {"Authorization": f"Bearer {load_key()}"}
            r = requests.post("https://api.openai.com/v1/files", headers=auth, timeout=600, data={"purpose": "batch"},
                              files={"file": (f"{tag}.jsonl", ("\n".join(lines) + "\n").encode())})
            r.raise_for_status()
            r = requests.post("https://api.openai.com/v1/batches", headers=auth, timeout=120,
                              json={"input_file_id": r.json()["id"], "endpoint": "/v1/responses",
                                    "completion_window": "24h", "metadata": {"tag": tag[:500]}})
            r.raise_for_status()
        except Exception:
            _settle(rid, worst, model, tag, None)      # nothing was submitted, so nothing is billed
            raise
        bid = r.json()["id"]
        with self.lock, self._batches().open("a") as f:
            f.write(json.dumps({"batch": bid, "tag": tag, "model": model, "params": params, "n": len(lines),
                                "reserve": rid, "worst": worst, "utc": _utc(),
                                "samples": samples}) + "\n")
        return bid

    def _batches(self):
        return self.path.with_name("batches.jsonl")

    def pending_batches(self):
        if not self._batches().exists():
            return []
        recs = [json.loads(l) for l in self._batches().read_text().splitlines() if l.strip()]
        done = {r["batch"] for r in recs if r.get("done")}
        return [r for r in recs if not r.get("done") and r["batch"] not in done]

    def collect_batch(self, bid):
        """Check one batch once (no waiting). If it has ended, log its results in calls.jsonl, settle the
        reservation at the batch rate from the returned usage, and return the final status; else the current status.
        Call from one process at a time: the done record in batches.jsonl is what prevents a second settlement."""
        sub = next((r for r in self.pending_batches() if r["batch"] == bid), None)
        if sub is None:
            raise ValueError(f"{bid} is not a pending batch in {self._batches()}")
        auth = {"Authorization": f"Bearer {load_key()}"}
        b = requests.get(f"https://api.openai.com/v1/batches/{bid}", headers=auth, timeout=120).json()
        if b.get("status") not in ("completed", "expired", "cancelled", "failed"):
            return b.get("status"), b.get("request_counts")
        model, tag, actual, ok, errors = sub["model"], sub["tag"], 0.0, 0, []
        tokens = {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 0}
        for fid in (b.get("output_file_id"), b.get("error_file_id")):
            if not fid:
                continue
            r = requests.get(f"https://api.openai.com/v1/files/{fid}/content", headers=auth, timeout=600)
            r.raise_for_status()
            for line in r.text.splitlines():
                if not line.strip():
                    continue
                out = json.loads(line)
                key, resp = out["custom_id"], out.get("response") or {}
                data = resp.get("body") or {}
                usage = data.get("usage")
                cost = 0.5 * _cost(model, usage or {})
                if usage:
                    tokens["input_tokens"] += usage.get("input_tokens", 0)
                    tokens["output_tokens"] += usage.get("output_tokens", 0)
                    tokens["input_tokens_details"]["cached_tokens"] += \
                        (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
                actual += cost
                content = _content(data) if resp.get("status_code") == 200 else ""
                if content.strip():
                    rec = {"key": key, "tag": tag, "model": model, "model_returned": data.get("model"),
                           "response_id": data.get("id"), "params": sub["params"],
                           "sample": sub["samples"].get(key, 0), "utc": _utc(), "status": data.get("status"),
                           "usage": usage, "cost": cost, "batch": bid, "content": content}
                    ok += 1
                else:
                    reason = (data.get("incomplete_details") or {}).get("reason")
                    rec = {"key": key, "tag": tag, "model": model, "sample": sub["samples"].get(key, 0),
                           "batch": bid, "utc": _utc(),
                           "error": "empty answer: output allowance used up" if reason == "max_output_tokens"
                           else json.dumps(out.get("error") or data.get("error") or resp)[:300]}
                    errors.append(rec["error"])
                with self.lock:
                    if "content" in rec:
                        self.cache[key] = rec
                    with self.path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
        _ledger(lambda total, append: append({"id": sub["reserve"], "kind": "settle", "model": model, "tag": tag,
                                              "usd": actual - sub["worst"], "actual": actual, "batch": bid,
                                              "rate": 0.5, "usage": tokens, "utc": _utc()}))
        with self.lock, self._batches().open("a") as f:
            f.write(json.dumps({"batch": bid, "done": True, "status": b["status"], "ok": ok, "failed": len(errors),
                                "missing": sub["n"] - ok - len(errors), "actual": actual, "utc": _utc()}) + "\n")
        return b["status"], {"ok": ok, "failed": len(errors), "missing": sub["n"] - ok - len(errors)}


if __name__ == "__main__":
    print(f"spent ${spent():.4f} of ${CAP:.2f} (estimated from token counts; see PRICES)")
