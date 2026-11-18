#!/bin/bash
# Experiment 9 anchor addendum, steps 1 and 2 (plan-v4.md): versions, orig inventory, errors, one anchor at a time.
# Usage: nohup bash scripts/v4/anchors_build.sh <paper>... > logs/anchors-build.log 2>&1 &   (resumable; done file at end)
cd "$(dirname "$0")/../.." || exit 1
for p in "$@"; do
  echo "== $p $(date -u +%FT%TZ)"
  python3 scripts/v4/obf_max.py "$p" &
  [ -f "data/v4/err9/inventory-$p.jsonl" ] || python3 scripts/v4/err9.py inventory "$p"
  wait
  python3 scripts/v4/clarify.py "$p"
  grep -q "\"paper\": \"$p\"" data/v4/err9/errors.jsonl || python3 scripts/v4/err9.py construct "$p"
done
touch logs/anchors-build.done
