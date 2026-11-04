#!/bin/bash
# dev2: screened p2 quiz on reserve paper 2607.05615v2. Resumable: every call is cached in calls.jsonl.
# Writes data/runs/dev2-2607.05615v2/dev2.done when finished.
set -u
cd "$(dirname "$0")/../.."
R=dev2-2607.05615v2; D=data/runs/$R; Q=$D/quiz-claude-opus-5.5-1-screened.json
Q_PY="python3.12 scripts/quiz/quiz.py --run $R"
LADDER=ibm-granite/granite-4.2-8b,mistralai/ministral-8b-2512,qwen/qwen3.5-9b,mistralai/ministral-14b-2512,google/gemma-4-31b-it
SMART=openai/gpt-5.6-sol,z-ai/glm-5.3
[ -f $Q ] || { echo "no screened quiz"; exit 1; }
$Q_PY --cap 6 take --quiz $Q --readers $LADDER,$SMART --conds none,paper --samples 2
$Q_PY --cap 3 take --quiz $Q --readers google/gemma-4-31b-it,openai/gpt-5.6-sol --conds scrambled,retrieved --samples 2
$Q_PY --cap 8 grade --graders z-ai/glm-5.3
$Q_PY summary --primary z-ai/glm-5.3 > /dev/null
date -u > $D/dev2.done
