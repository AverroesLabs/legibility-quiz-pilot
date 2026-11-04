#!/usr/bin/env python3.12
"""Plan v2 per-paper pipeline: one command runs every stage built so far for one or more papers, in parallel.

Usage: python3.12 scripts/quiz/run_v2.py PAPER [PAPER ...] [--run-prefix v2] [--cap 25] [--stop-usage 639]
       python3.12 scripts/quiz/run_v2.py 2607.05615v2 --run v2dev-2607.05615v2   (one paper, named run dir)

Per paper, in data/runs/<prefix>-<paper>/:
  versions (versions.make)  ||  write (write_v2.write) -> filters in parallel (retrieval with each of
  filter_v2.RETRIEVERS, nopaper, withpaper) -> combine (kept-v2.json)
  -> takes (takes_v2.take: 4 readers x 5 conditions x 3 samples, graded as answers arrive; takes-v2.jsonl and
  takes-summary.json, with the information check)

All papers run in one process, so llm.py's per-provider concurrency limits hold across papers (Jason,
2026-11-03). Each paper has its own call log and spend cap. Before each stage, account usage is read from
OpenRouter, and no stage starts at or above --stop-usage (the plan v2 stop line in STATE.md). Writes
timing.jsonl (start, end and seconds per stage) and pipeline.json (status, error if any, and logged cost by
call tag) per paper. Resumable: rerunning repeats calls from each calls.jsonl cache.
"""
import argparse, json, sys, threading, time, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import filter_v2, llm, quiz, takes_v2, versions, write_v2


class Stopped(RuntimeError):
    pass


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Paper:
    def __init__(self, paper, d, cap, stop_usage):
        self.paper, self.d, self.stop_usage = paper, d, stop_usage
        d.mkdir(parents=True, exist_ok=True)
        self.led = llm.Ledger(d / "calls.jsonl", cap)
        self.lock = threading.Lock()

    def stage(self, name, fn, *args):
        usage = llm.account_usage()
        if usage >= self.stop_usage:
            raise Stopped(f"account usage ${usage:.2f} >= stop line ${self.stop_usage} before {name}")
        t0, start = time.time(), utc()
        print(f"{start} {self.paper} {name}: start (usage ${usage:.2f})", flush=True)
        out = fn(*args)
        rec = {"stage": name, "start": start, "end": utc(), "seconds": round(time.time() - t0)}
        with self.lock, (self.d / "timing.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{rec['end']} {self.paper} {name}: done in {rec['seconds']} s", flush=True)
        return out

    def quiz_and_filters(self):
        q = self.stage("write", write_v2.write, self.led, self.d, self.paper, write_v2.BATCHES)
        checks = [("retrieval", r) for r in filter_v2.RETRIEVERS] + [("nopaper", None), ("withpaper", None)]
        for attempt in range(2):
            with ThreadPoolExecutor(len(checks)) as ex:
                futs = [ex.submit(self.stage, f"filter_{c}" + (f":{r.split('/')[1]}" if r else ""),
                                  filter_v2.run_check, self.led, q, self.d, c, r or filter_v2.RETRIEVERS[0])
                        for c, r in checks]
                for f in futs:
                    f.result()
            try:
                return self.stage("combine", filter_v2.combine, q, self.d)
            except filter_v2.Ungraded as e:  # a failed grade: rerun the checks once; only failed calls are new
                if attempt:
                    raise
                print(f"{utc()} {self.paper} combine: {e}; rerunning the checks from cache", flush=True)

    def run(self):
        t0, status, error = time.time(), "done", None
        try:
            with ThreadPoolExecutor(2) as ex:  # versions do not depend on the quiz
                futs = [ex.submit(self.stage, "versions", versions.make, self.led, self.d, self.paper),
                        ex.submit(self.quiz_and_filters)]
                for f in futs:
                    f.result()
            self.stage("takes", takes_v2.take, self.led, self.d)
        except Stopped as e:
            status, error = "stopped", str(e)
        except Exception as e:  # recorded per paper so other papers continue
            status, error = "failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        cost = Counter()
        for r in self.led.cache.values():
            cost[r["tag"].split(":")[0]] += r["cost"] or 0
        out = {"paper": self.paper, "status": status, "error": error, "finished": utc(),
               "seconds": round(time.time() - t0), "logged_cost": {k: round(v, 4) for k, v in cost.items()},
               "note": "logged_cost excludes failed calls and includes any earlier runs in this directory"}
        (self.d / "pipeline.json").write_text(json.dumps(out, indent=1))
        print(f"{utc()} {self.paper}: {status}" + (f" ({error.splitlines()[0]})" if error else ""), flush=True)
        return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("papers", nargs="+")
    p.add_argument("--run", help="run directory name (one paper only)")
    p.add_argument("--run-prefix", default="v2")
    p.add_argument("--cap", type=float, default=25.0, help="new spend cap per paper, USD")
    p.add_argument("--stop-usage", type=float, default=639.0, help="no stage starts at this account usage, USD")
    a = p.parse_args()
    if a.run and len(a.papers) != 1:
        raise SystemExit("--run takes one paper")
    for paper in a.papers:
        if not (quiz.ROOT / "data/papers" / paper / "text.md").exists():
            raise SystemExit(f"no text for {paper}")
    jobs = [Paper(x, quiz.ROOT / "data/runs" / (a.run or f"{a.run_prefix}-{x}"), a.cap, a.stop_usage)
            for x in a.papers]
    with ThreadPoolExecutor(len(jobs)) as ex:
        statuses = list(ex.map(Paper.run, jobs))
    print(f"{utc()} all papers: {dict(Counter(statuses))}", flush=True)
    sys.exit(0 if set(statuses) == {"done"} else 1)


if __name__ == "__main__":
    main()
