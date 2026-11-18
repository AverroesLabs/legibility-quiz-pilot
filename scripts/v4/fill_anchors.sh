#!/bin/bash
# Jason, INBOX 2026-11-17 ~15:30 UTC (items 1 and 2 of the ~02:00 note), exploratory fill-in, approved up to $80.
# Item 1: mini-high-A, mini54-A, luna-low-A (frozen variant A) on the 6 anchors, orig, obf-max, clar-explainer-orig.
# Item 2: gpt55-A1200 on the 2 anchors it never reached, resuming from reader-calls.jsonl.
# One process per reader (each rewrites its own reader file per paper). Guard: no paper starts once the fill-in has
# spent >= $65 since the ledger stood at $1,182.09 before the slice; up to 4 papers in flight (~$10 at most each) keeps
# it under $80. Usage: nohup bash scripts/v4/fill_anchors.sh > logs/fill.log 2>&1 &   (done file logs/fill.done)
cd "$(dirname "$0")/../.." || exit 1
spent() { python3 -c "import sys; sys.path.insert(0, 'scripts/v4'); from llm import spent; print(spent())"; }
run() {   # run <reader> <paper>...
  rd=$1; shift
  for p in "$@"; do
    used=$(python3 -c "print(round($(spent) - 1182.09, 2))")
    if python3 -c "import sys; sys.exit(0 if $used >= 65 else 1)"; then echo "guard: $rd stopping before $p, spent \$$used"; return; fi
    python3 scripts/v4/err9.py read "$rd" "$p"
  done
  python3 scripts/v4/err9.py regrade "$rd"
}
ANCHORS="2601.02896v4 2603.01214v2 2509.22621v3 2506.22982v1 2509.11379v3 2510.17247v2"
run mini-high-A-anchors $ANCHORS > logs/fill-mini-high.log 2>&1 &
run mini54-A-anchors $ANCHORS > logs/fill-mini54.log 2>&1 &
run luna-low-A-anchors $ANCHORS > logs/fill-luna.log 2>&1 &
( for p in 2506.22982v1 2510.17247v2; do
    used=$(python3 -c "print(round($(spent) - 1182.09, 2))")
    if python3 -c "import sys; sys.exit(0 if $used >= 65 else 1)"; then echo "guard: gpt55-A1200 stopping before $p"; break; fi
    python3 scripts/v4/err9.py read gpt55-A1200 "$p"
  done ) > logs/fill-a1200.log 2>&1 &
wait
python scripts/v4/llm.py
touch logs/fill.done
