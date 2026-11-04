"""Diagnostic: how many screened items passed the leakage screen on fewer graded answers than asked for.

The screen scores an item with no successful grade as 0, so it passes. This counts such items per pilot paper.
Usage: python3 scripts/quiz/screen_audit.py  (reads data/runs/pilot-*/quiz-*-screened.json)
"""
import glob
import json
from collections import Counter

for f in sorted(glob.glob("data/runs/pilot-*/quiz-*-screened.json")):
    q = json.load(open(f))
    final = Counter()
    for it in q["items"]:
        h = it["screen"]["history"][-1]
        final[len(h["answers"])] += 1
    n = len(q["items"])
    print(f"{q['paper']}: {n} items; graded screen answers in final round per item: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(final.items()))
          + f"; passed with no graded answer: {final[0]}")
