#!/bin/bash
# Experiment 9 held-out run (plan-v4.md, held-out preregistration): per paper, in the order given, clean versions
# (obf-max, explainer clarifications only), orig inventory, errors, then the frozen reader's reads; Sol regrade at end.
# Start only after the anchor addendum's rule has fixed the notes budget and decisions.md records it.
# Usage: nohup bash scripts/v4/heldout.sh <reader> <paper>... > logs/heldout.log 2>&1 &   (resumable; done file at end)
#   reader: gpt55-A-heldout, gpt55-A600-heldout or gpt55-A1200-heldout
cd "$(dirname "$0")/../.." || exit 1
rd=$1; shift
case "$rd" in gpt55-A-heldout|gpt55-A600-heldout|gpt55-A1200-heldout) ;; *) echo "unknown reader $rd"; exit 1;; esac
for p in "$@"; do
  echo "== $p $(date -u +%FT%TZ)"
  python3 scripts/v4/obf_max.py "$p" &
  [ -f "data/v4/err9/inventory-$p.jsonl" ] || python3 scripts/v4/err9.py inventory "$p"
  wait
  python3 scripts/v4/clarify.py "$p" --prompts explainer || { echo "$p: clarify failed, stopping"; exit 1; }
  grep -q "\"paper\": \"$p\"" data/v4/err9/errors.jsonl || python3 scripts/v4/err9.py construct "$p" || { echo "$p: construct failed, stopping"; exit 1; }
  grep -q "\"paper\": \"$p\"" "data/v4/err9/reader-$rd.jsonl" 2>/dev/null || python3 scripts/v4/err9.py read "$rd" "$p" || { echo "$p: read failed, stopping"; exit 1; }
done
python3 scripts/v4/err9.py regrade "$rd"
touch logs/heldout.done
