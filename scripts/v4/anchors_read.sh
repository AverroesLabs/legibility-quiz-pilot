#!/bin/bash
# Experiment 9 anchor addendum, step 3 (plan-v4.md): frozen gpt55-A reads on each anchor once its errors exist,
# then the Sol regrade pooled over anchors. Waits at most 4 h per anchor for anchors_build.sh.
# Usage: nohup bash scripts/v4/anchors_read.sh <paper>... > logs/anchors-read.log 2>&1 &   (done file at end)
cd "$(dirname "$0")/../.." || exit 1
for p in "$@"; do
  for i in $(seq 1440); do grep -q "\"paper\": \"$p\"" data/v4/err9/errors.jsonl && break; sleep 10; done
  grep -q "\"paper\": \"$p\"" data/v4/err9/errors.jsonl || { echo "$p: no errors after 4 h, stopping"; exit 1; }
  grep -q "\"paper\": \"$p\"" data/v4/err9/reader-gpt55-A-anchors.jsonl || python3 scripts/v4/err9.py read gpt55-A-anchors "$p"
done
python3 scripts/v4/err9.py regrade gpt55-A-anchors
touch logs/anchors-read.done
