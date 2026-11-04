#!/bin/bash
# Scaled-down main pilot (docs/preregistration.md). One paper per call: ./pilot.sh <paper>.
# Resumable: every API call is cached in data/runs/pilot-<paper>/calls.jsonl. Writes pilot.done there when finished.
# Run all six with scripts/runs/pilot-all.sh.
set -u
cd "$(dirname "$0")/../.."
P=$1; R=pilot-$P; D=data/runs/$R
Q_PY="python3.12 scripts/quiz/quiz.py --run $R"
WRITER=anthropic/claude-opus-5.5; SMART=openai/gpt-5.6-sol; LADDER=mistralai/ministral-8b-2512,google/gemma-4-31b-it
REWRITER=moonshotai/kimi-k3; SCREENER=moonshotai/kimi-k3; GRADER=z-ai/glm-5.3
STOP_AT=138   # preregistered stop is $140 account usage; checked before every step
mkdir -p $D
usage_ok() {
  u=$(python3.12 -c "import sys; sys.path.insert(0,'scripts/quiz'); import llm; print(llm.account_usage())")
  echo "$(date -u +%H:%M) usage \$$u before $1" >> $D/progress.log
  python3 -c "import sys; sys.exit(0 if $u < $STOP_AT else 1)" || { echo "STOPPED: usage \$$u >= \$$STOP_AT before $1" >> $D/progress.log; exit 3; }
}
step() { usage_ok "$1"; shift; "$@" >> $D/progress.log 2>&1 || { echo "FAILED step: $*" >> $D/progress.log; exit 4; }; }
Q0=$D/quiz-claude-opus-5.5-1.json; Q=$D/quiz-claude-opus-5.5-1-screened.json
[ -f $Q0 ] || step write $Q_PY --cap 3 write --paper $P --writer $WRITER --seed 1
[ -f $Q ] || step screen $Q_PY --cap 8 screen --quiz $Q0 --screener $SCREENER --grader $GRADER --heldout $SMART --samples 2 --rounds 2
[ -f $D/versions/meta.json ] || step obfuscate $Q_PY --cap 2 obfuscate --quiz $Q --rewriter $REWRITER --takers $SMART,$LADDER
step take $Q_PY --cap 4 take --quiz $Q --readers $SMART,$LADDER --conds none,paper,obf,rem --samples 2
step take-retrieved $Q_PY --cap 2 take --quiz $Q --readers $SMART --conds retrieved --samples 2
step grade $Q_PY --cap 12 grade --graders $GRADER
step summary $Q_PY summary --primary $GRADER
date -u > $D/pilot.done
