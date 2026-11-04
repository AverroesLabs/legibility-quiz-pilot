#!/bin/bash
# Runs the six preregistered pilot papers, three at a time, strata interleaved so a budget stop leaves strata balanced.
# Order: first core paper of A, B, C, then the second of each (data/selection/selection.json).
cd "$(dirname "$0")/../.."
printf '%s\n' 2609.00755v1 2608.25490v1 2607.20379v2 2607.17117v2 2607.28685v1 2608.19216v1 \
  | xargs -P 3 -I{} bash scripts/runs/pilot.sh {}
date -u > data/runs/pilot-all.done
